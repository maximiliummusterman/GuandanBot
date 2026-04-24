"""
Thin HTTP API for Guandan bot decisions.

This service is intended to run as a normal Python web service
(for example on Render), not as a serverless function bundle.

Supported decisions:
  - action:      choose the bot play for the current trick
  - tributeCard: choose the tribute card during the tribute phase
  - returnCard:  choose the return card during the tribute phase

Request shape:
  POST /decision
  {
    "requestType": "action" | "tributeCard" | "returnCard",
    "match": {...},              # alias: currentMatch
    "players": [...],            # alias: matchPlayers / allMatchPlayers
    "currentPlayer": {...},      # optional for tribute/return requests
    "seat": 1,                   # optional; defaults to match.currentSeat
    "sample": false,             # optional
    "checkpoint": "checkpoint.pt"  # optional override
  }

Response shape:
  {
    "decisionType": "action" | "tributeCard" | "returnCard" | "skipTribute",
    "seat": "4",
    "action": [...],
    "actionIdx": 123,
    "pass": false,
    "tributeCard": {...},
    "returnCard": {...},
    "tributeState": "BASIC",
    "checkpoint": "checkpoint.pt"
  }

Run locally:
  python bot_api.py --host 127.0.0.1 --port 8765

Run in production:
  gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 bot_api:app
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import threading
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from wsgiref.simple_server import make_server

import torch

from guandan_arena import (
    DEVICE,
    GuandanEnv,
    GuandanNet,
    RANKS,
    SUITS,
    _detect_from_hand,
    _upgrade_legacy_model_state,
    action_index_to_cards,
    critic_state_value,
)


RETURN_RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10"}


def _payload_value(payload: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def _extract_match(payload: Dict[str, Any]) -> Dict[str, Any]:
    match = _payload_value(payload, "match", "currentMatch")
    if not isinstance(match, dict):
        raise ValueError("Payload must include a 'match' or 'currentMatch' object.")
    return match


def _extract_players(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    players = _payload_value(payload, "players", "matchPlayers", "allMatchPlayers")
    if not isinstance(players, list) or not players:
        raise ValueError(
            "Payload must include a non-empty 'players'/'matchPlayers' list."
        )
    return [dict(player) for player in players]


def _normalize_level_rank(rank: Any) -> str:
    text = str(rank or "2").strip().upper()
    if text in RANKS:
        return text
    if text.startswith("A"):
        return "A"
    return "2"


def _level_rank_index(rank: Any) -> int:
    return RANKS.index(_normalize_level_rank(rank))


def _normalize_seat(value: Any, field_name: str = "seat") -> int:
    try:
        seat = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if seat not in {1, 2, 3, 4}:
        raise ValueError(f"{field_name} must be 1, 2, 3, or 4.")
    return seat


def _web_to_internal_seat(web_seat: int) -> int:
    return web_seat - 1


def _internal_to_web_seat(internal_seat: int) -> int:
    return internal_seat + 1


def _clean_card(card: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(card, dict):
        raise ValueError(f"Card must be an object, got {type(card).__name__}.")

    cleaned = dict(card)
    if cleaned.get("joker"):
        cleaned["joker"] = str(cleaned["joker"]).strip().lower()
        cleaned.pop("suit", None)
        cleaned.pop("rank", None)
        return cleaned

    suit = str(cleaned.get("suit", "")).strip().lower()
    rank = str(cleaned.get("rank", "")).strip().upper()
    if suit not in SUITS:
        raise ValueError(f"Invalid card suit: {card!r}")
    if rank not in RANKS:
        raise ValueError(f"Invalid card rank: {card!r}")
    cleaned["suit"] = suit
    cleaned["rank"] = rank
    cleaned.pop("joker", None)
    return cleaned


def _clean_cards(cards: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_clean_card(card) for card in cards or []]


def _card_signature(card: Dict[str, Any]) -> Tuple[str, ...]:
    card = _clean_card(card)
    if card.get("joker"):
        return ("joker", card["joker"])
    return ("normal", card["suit"], card["rank"])


def _card_from_signature(signature: Tuple[str, ...]) -> Dict[str, Any]:
    if signature[0] == "joker":
        return {"joker": signature[1]}
    return {"suit": signature[1], "rank": signature[2]}


def _remove_one_card(cards: List[Dict[str, Any]], target: Dict[str, Any]) -> Dict[str, Any]:
    target_sig = _card_signature(target)
    for index, card in enumerate(cards):
        if _card_signature(card) == target_sig:
            return cards.pop(index)
    raise ValueError(f"Card not found in hand: {target!r}")


def _count_red_jokers(hand: List[Dict[str, Any]]) -> int:
    return sum(1 for card in hand or [] if _clean_card(card).get("joker") == "red")


def _current_level_rank(match: Dict[str, Any]) -> str:
    caller = str(match.get("currentRoundLevelRank") or "Blue")
    if caller == "Red":
        return _normalize_level_rank(match.get("levelRankRed"))
    return _normalize_level_rank(match.get("levelRankBlue"))


def _player_by_seat(players: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_seat: Dict[int, Dict[str, Any]] = {}
    for player in players:
        seat = _normalize_seat(player.get("seat"), "player.seat")
        prepared = dict(player)
        prepared["seat"] = str(seat)
        prepared["hand"] = _clean_cards(player.get("hand") or [])
        by_seat[seat] = prepared
    missing = [seat for seat in range(1, 5) if seat not in by_seat]
    if missing:
        raise ValueError(f"Players payload is missing seat(s): {missing}")
    return by_seat


def _resolve_current_player(
    payload: Dict[str, Any],
    players_by_seat: Dict[int, Dict[str, Any]],
    match: Dict[str, Any],
    *,
    allow_match_current_seat: bool,
) -> Dict[str, Any]:
    player_value = _payload_value(payload, "currentPlayer", "player")
    if isinstance(player_value, dict):
        if player_value.get("seat") is not None:
            seat = _normalize_seat(player_value.get("seat"), "currentPlayer.seat")
            return players_by_seat[seat]
        player_id = player_value.get("id")
        user_id = player_value.get("user_id")
        for player in players_by_seat.values():
            if player_id is not None and player.get("id") == player_id:
                return player
            if user_id is not None and player.get("user_id") == user_id:
                return player

    seat_value = _payload_value(payload, "seat", "currentSeat")
    if seat_value is not None:
        return players_by_seat[_normalize_seat(seat_value)]

    if allow_match_current_seat and match.get("currentSeat") is not None:
        return players_by_seat[_normalize_seat(match.get("currentSeat"), "match.currentSeat")]

    raise ValueError(
        "Could not resolve the acting player. Provide 'currentPlayer' or 'seat'."
    )


def _infer_internal_team_mapping(
    players_by_seat: Dict[int, Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    seats_13 = {
        str(players_by_seat[seat].get("team")).strip()
        for seat in (1, 3)
        if players_by_seat[seat].get("team") is not None
    }
    seats_24 = {
        str(players_by_seat[seat].get("team")).strip()
        for seat in (2, 4)
        if players_by_seat[seat].get("team") is not None
    }

    if len(seats_13) == 1 and len(seats_24) == 1 and seats_13 != seats_24:
        web_for_internal = {
            "Blue": next(iter(seats_13)),
            "Red": next(iter(seats_24)),
        }
    else:
        web_for_internal = {"Blue": "Blue", "Red": "Red"}

    internal_for_web = {
        web_name: internal_name for internal_name, web_name in web_for_internal.items()
    }
    return web_for_internal, internal_for_web


def _all_finish_places(players_by_seat: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
    finishers: List[Tuple[int, int]] = []
    for web_seat, player in players_by_seat.items():
        finish_place = player.get("finishPlace")
        if finish_place in (None, ""):
            continue
        try:
            finishers.append((int(finish_place), web_seat))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid finishPlace for seat {web_seat}: {finish_place!r}"
            ) from exc
    finishers.sort(key=lambda item: item[0])
    return finishers


def _full_deck_counter() -> Counter[Tuple[str, ...]]:
    counter: Counter[Tuple[str, ...]] = Counter()
    for _ in range(2):
        for suit in SUITS:
            for rank in RANKS:
                counter[("normal", suit, rank)] += 1
    counter[("joker", "red")] += 2
    counter[("joker", "black")] += 2
    return counter


def _reconstruct_played_cards(hands: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    remaining = _full_deck_counter()
    for hand in hands:
        for card in hand:
            signature = _card_signature(card)
            if remaining[signature] <= 0:
                raise ValueError(
                    "Hands contain an impossible card count. "
                    f"Too many copies of {card!r}."
                )
            remaining[signature] -= 1

    played_cards: List[Dict[str, Any]] = []
    for signature, count in remaining.items():
        for _ in range(count):
            played_cards.append(_card_from_signature(signature))
    return played_cards


def _build_env_context(
    payload: Dict[str, Any],
    request_type: str,
    acting_player: Dict[str, Any],
) -> Dict[str, Any]:
    match = _extract_match(payload)
    players = _extract_players(payload)
    players_by_seat = _player_by_seat(players)
    web_for_internal, internal_for_web = _infer_internal_team_mapping(players_by_seat)

    env = GuandanEnv()
    env.hands = [[] for _ in range(4)]
    for web_seat, player in players_by_seat.items():
        env.hands[_web_to_internal_seat(web_seat)] = [dict(card) for card in player["hand"]]

    rank_by_web_team = {
        "Blue": _level_rank_index(match.get("levelRankBlue")),
        "Red": _level_rank_index(match.get("levelRankRed")),
    }
    env.level_ranks = {
        "Blue": rank_by_web_team.get(web_for_internal["Blue"], rank_by_web_team["Blue"]),
        "Red": rank_by_web_team.get(web_for_internal["Red"], rank_by_web_team["Red"]),
    }
    caller_web_team = str(match.get("currentRoundLevelRank") or "Blue")
    env.caller = internal_for_web.get(caller_web_team, "Blue")
    env.a_tries = {"Blue": 0, "Red": 0}
    env.round_num = int(match.get("roundNumber") or 1)
    env._match_winner = None
    env.tribute_seat_value_fn = None

    finish_order = [
        _web_to_internal_seat(web_seat)
        for _, web_seat in _all_finish_places(players_by_seat)
    ]
    env.finish_order = finish_order

    acting_web_seat = _normalize_seat(acting_player.get("seat"), "actingPlayer.seat")
    env.current_seat = _web_to_internal_seat(acting_web_seat)

    trick_cards = _clean_cards(match.get("trickLastPlay") or [])
    last_trick_seat = match.get("lastTrickSeat")
    should_clear = (
        request_type == "action"
        and last_trick_seat not in (None, "")
        and _normalize_seat(last_trick_seat, "match.lastTrickSeat") == acting_web_seat
    )
    if should_clear:
        env.trick_last = []
        env.trick_info = None
        env.trick_seat = None
    else:
        env.trick_last = trick_cards
        env.trick_info = (
            _detect_from_hand(trick_cards, env.active_level_rank()) if trick_cards else None
        )
        env.trick_seat = (
            _web_to_internal_seat(_normalize_seat(last_trick_seat, "match.lastTrickSeat"))
            if trick_cards and last_trick_seat not in (None, "")
            else None
        )

    explicit_played_cards = _payload_value(payload, "playedCards", "played_cards")
    if explicit_played_cards is not None:
        env.played_cards = _clean_cards(explicit_played_cards)
    else:
        env.played_cards = _reconstruct_played_cards(env.hands)

    env.pass_streak = 0

    return {
        "env": env,
        "match": match,
        "players_by_seat": players_by_seat,
        "web_for_internal": web_for_internal,
        "internal_for_web": internal_for_web,
    }


def _extract_episode_num(path: Path) -> Optional[int]:
    match = re.search(r"ep(\d+)", path.stem)
    return int(match.group(1)) if match else None


def resolve_checkpoint_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    env_path = os.environ.get("GUANDAN_CHECKPOINT")
    if env_path:
        return resolve_checkpoint_path(env_path)

    preferred_paths = [
        Path("checkpoint.pt"),
        Path("checkpoints") / "latest.pt",
    ]
    for checkpoint_path in preferred_paths:
        if checkpoint_path.exists():
            return str(checkpoint_path)

    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "Could not find checkpoint.pt or checkpoints/. "
            "Set GUANDAN_CHECKPOINT or pass --checkpoint."
        )

    checkpoint_paths = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: (
            _extract_episode_num(path) is None,
            _extract_episode_num(path) or 0,
            path.name,
        ),
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No checkpoint files found. Expected checkpoint.pt, checkpoints/latest.pt, "
            "or a .pt file in checkpoints/. Set GUANDAN_CHECKPOINT or pass --checkpoint."
        )
    return str(checkpoint_paths[-1])


class PolicyStore:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self._checkpoint_path = checkpoint_path
        self._loaded_path: Optional[str] = None
        self._net: Optional[GuandanNet] = None
        self._lock = threading.Lock()

    def get(self, override_path: Optional[str] = None) -> Tuple[GuandanNet, str]:
        resolved_path = resolve_checkpoint_path(override_path or self._checkpoint_path)
        with self._lock:
            if self._net is not None and resolved_path == self._loaded_path:
                return self._net, resolved_path

            checkpoint = torch.load(resolved_path, map_location=DEVICE)
            if "model_state" not in checkpoint:
                raise KeyError(f"Checkpoint is missing 'model_state': {resolved_path}")

            net = GuandanNet().to(DEVICE)
            model_state, _ = _upgrade_legacy_model_state(checkpoint["model_state"])
            net.load_state_dict(model_state)
            net.eval()

            self._net = net
            self._loaded_path = resolved_path
            return net, resolved_path


def _choose_action(
    env: GuandanEnv,
    seat: int,
    net: GuandanNet,
    *,
    sample: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    state_np = env.get_state(seat)[None, :]
    mask_np = env.get_legal_mask(seat)[None, :]

    states = torch.from_numpy(state_np).to(DEVICE, non_blocking=True)
    masks = torch.from_numpy(mask_np).to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        logits, _ = net(states, masks)

    hand = env.hands[seat]
    level_rank = env.active_level_rank()

    def try_action(action_idx: int) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
        cards = action_index_to_cards(action_idx, hand, level_rank)
        if action_idx == 0 or cards:
            return action_idx, cards
        return None

    if sample:
        dist = torch.distributions.Categorical(logits=logits)
        for _ in range(4):
            sampled_idx = int(dist.sample().item())
            choice = try_action(sampled_idx)
            if choice is not None:
                return choice
    else:
        greedy_idx = int(torch.argmax(logits, dim=1).item())
        choice = try_action(greedy_idx)
        if choice is not None:
            return choice

    for action_idx in torch.argsort(logits[0], descending=True).tolist():
        if not bool(mask_np[0, action_idx]):
            continue
        choice = try_action(int(action_idx))
        if choice is not None:
            return choice

    raise RuntimeError(f"No legal action could be decoded for seat {seat}.")


def _tribute_strength(card: Dict[str, Any], level_rank: str) -> Tuple[int, int]:
    card = _clean_card(card)
    if card.get("joker") == "red":
        return (4, 1000)
    if card.get("joker") == "black":
        return (3, 999)
    if card.get("rank") == level_rank and card.get("suit") == "hearts":
        return (-1, -1)
    if card.get("rank") == level_rank:
        return (2, 100)
    return (1, RANKS.index(card["rank"]) + 2)


def _exception1_compare_value(card: Dict[str, Any], level_rank: str) -> int:
    card = _clean_card(card)
    if card.get("joker") == "red":
        return 1000
    if card.get("joker") == "black":
        return 900
    if card.get("rank") == level_rank and card.get("suit") == "hearts":
        return 800
    if card.get("rank") == level_rank:
        return 700
    return max(0, RANKS.index(card["rank"]))


def _is_valid_return_card(card: Dict[str, Any], level_rank: str) -> bool:
    card = _clean_card(card)
    if card.get("joker"):
        return False
    if card.get("rank") == level_rank:
        return False
    return card.get("rank") in RETURN_RANKS


def _best_tribute_candidates(
    hand: List[Dict[str, Any]],
    level_rank: str,
) -> List[Dict[str, Any]]:
    scored = [(card, _tribute_strength(card, level_rank)) for card in hand]
    scored = [item for item in scored if item[1][0] >= 0]
    if not scored:
        raise ValueError("No valid tribute card found in the acting player's hand.")
    best_strength = max(score for _, score in scored)
    return [card for card, score in scored if score == best_strength]


def _best_return_candidates(
    hand: List[Dict[str, Any]],
    level_rank: str,
) -> List[Dict[str, Any]]:
    candidates = [card for card in hand if _is_valid_return_card(card, level_rank)]
    if not candidates:
        raise ValueError(
            "No valid return card found. The current hand has no card that is <= 10 and not the level rank."
        )
    return candidates


def _detect_tribute_state(
    players_by_seat: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    finishers = [
        players_by_seat[web_seat]
        for _, web_seat in _all_finish_places(players_by_seat)
    ]

    first = next((player for player in finishers if player.get("finishPlace") == "1"), None)
    second = next((player for player in finishers if player.get("finishPlace") == "2"), None)
    third = next((player for player in finishers if player.get("finishPlace") == "3"), None)
    fourth = next((player for player in finishers if player.get("finishPlace") == "4"), None)

    if fourth and _count_red_jokers(fourth["hand"]) == 2:
        return {
            "type": "EXCEPTION_2a",
            "description": "No tribute - loser has both red jokers",
            "loser": fourth,
        }

    if (
        third
        and fourth
        and third.get("team") == fourth.get("team")
        and _count_red_jokers(third["hand"]) + _count_red_jokers(fourth["hand"]) > 0
    ):
        third_reds = _count_red_jokers(third["hand"])
        fourth_reds = _count_red_jokers(fourth["hand"])
        if third_reds == 2 or fourth_reds == 2:
            return {
                "type": "EXCEPTION_2b_CASE_1",
                "description": "Tribute completely defended - one player has both red jokers",
                "loserWithJokers": third if third_reds == 2 else fourth,
            }
        if third_reds == 1 and fourth_reds == 1:
            return {
                "type": "EXCEPTION_2b_CASE_2",
                "description": "Both losers show their red joker, no tribute",
                "losers": [third, fourth],
            }

    if first and second and first.get("team") == second.get("team"):
        return {
            "type": "EXCEPTION_1",
            "description": "Both losers give highest card to both winners",
            "losers": [player for player in (third, fourth) if player is not None],
            "winners": [player for player in (first, second) if player is not None],
        }

    return {
        "type": "BASIC",
        "description": "Standard tribute: finishPlace 4 gives highest card to finishPlace 1",
        "losers": [fourth] if fourth else [],
        "winners": [first] if first else [],
    }


def _assigned_tributes(
    trick_cards: List[Dict[str, Any]],
    level_rank: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    tributes = [card for card in trick_cards if card.get("_type") == "tribute"]
    if len(tributes) != 2:
        return None

    first, second = tributes
    if _exception1_compare_value(first, level_rank) >= _exception1_compare_value(second, level_rank):
        higher, lower = first, second
    else:
        higher, lower = second, first
    return {"1": higher, "2": lower}


def _exception1_required_place(trick_cards: List[Dict[str, Any]]) -> str:
    tributes = [card for card in trick_cards if card.get("_type") == "tribute"]
    returns = [card for card in trick_cards if card.get("_type") == "return"]
    t_len = len(tributes)
    r_len = len(returns)

    if t_len == 2 and r_len == 2:
        return "take_returns"
    if t_len == 1 and r_len == 1:
        return "take_returns"
    if t_len == 2 and r_len == 1:
        return "2"
    if t_len == 2 and r_len == 0:
        return "1"
    if t_len == 1 and r_len == 0:
        return "4"
    if t_len == 0 and r_len == 0:
        return "3"
    return "take_returns"


def _score_return_candidate(
    base_env: GuandanEnv,
    *,
    winner_internal_seat: int,
    giver_internal_seat: int,
    tribute_card: Dict[str, Any],
    return_card: Dict[str, Any],
    net: GuandanNet,
) -> float:
    sim_env = copy.deepcopy(base_env)
    winner_hand = sim_env.hands[winner_internal_seat]
    giver_hand = sim_env.hands[giver_internal_seat]

    _remove_one_card(winner_hand, return_card)
    winner_hand.append(dict(_clean_card(tribute_card)))
    giver_hand.append(dict(_clean_card(return_card)))

    sim_env.trick_last = []
    sim_env.trick_info = None
    sim_env.trick_seat = None
    sim_env.played_cards = _reconstruct_played_cards(sim_env.hands)
    return critic_state_value(net, sim_env, winner_internal_seat)


def _choose_return_card_basic(
    payload: Dict[str, Any],
    acting_player: Dict[str, Any],
    tribute_state: Dict[str, Any],
    policy_store: PolicyStore,
) -> Dict[str, Any]:
    match = _extract_match(payload)
    level_rank = _current_level_rank(match)
    trick_cards = _clean_cards(match.get("trickLastPlay") or [])
    if not trick_cards:
        raise ValueError("Cannot choose a return card without a tribute card in trickLastPlay.")

    tribute_card = trick_cards[0]
    candidates = _best_return_candidates(acting_player["hand"], level_rank)
    if len(candidates) == 1:
        return candidates[0]

    context = _build_env_context(payload, "returnCard", acting_player)
    env = context["env"]
    winner_internal_seat = _web_to_internal_seat(_normalize_seat(acting_player["seat"]))
    losers = tribute_state.get("losers") or []
    if not losers:
        raise ValueError("Basic tribute state is missing the tributer.")
    giver_internal_seat = _web_to_internal_seat(_normalize_seat(losers[0]["seat"]))
    net, _ = policy_store.get(_payload_value(payload, "checkpoint"))

    best_card = candidates[0]
    best_score = None
    for candidate in candidates:
        score = _score_return_candidate(
            env,
            winner_internal_seat=winner_internal_seat,
            giver_internal_seat=giver_internal_seat,
            tribute_card=tribute_card,
            return_card=candidate,
            net=net,
        )
        if best_score is None or score > best_score:
            best_card = candidate
            best_score = score
    return best_card


def _choose_return_card_exception1(
    payload: Dict[str, Any],
    acting_player: Dict[str, Any],
    policy_store: PolicyStore,
) -> Dict[str, Any]:
    match = _extract_match(payload)
    level_rank = _current_level_rank(match)
    trick_cards = _clean_cards(match.get("trickLastPlay") or [])
    assignments = _assigned_tributes(trick_cards, level_rank)
    if not assignments:
        raise ValueError(
            "Cannot choose an EXCEPTION_1 return card before both tribute cards exist."
        )

    finish_place = str(acting_player.get("finishPlace") or "")
    if finish_place not in assignments:
        raise ValueError(
            f"Current player finishPlace must be '1' or '2' for EXCEPTION_1, got {finish_place!r}."
        )

    tribute_card = assignments[finish_place]
    giver_id = tribute_card.get("_giverId")
    if giver_id is None:
        raise ValueError("EXCEPTION_1 tribute cards must include '_giverId' metadata.")

    context = _build_env_context(payload, "returnCard", acting_player)
    players_by_seat = context["players_by_seat"]
    giver = next(
        (player for player in players_by_seat.values() if player.get("user_id") == giver_id),
        None,
    )
    if giver is None:
        raise ValueError("Could not resolve the tribute giver for the current EXCEPTION_1 return.")

    candidates = _best_return_candidates(acting_player["hand"], level_rank)
    if len(candidates) == 1:
        return candidates[0]

    env = context["env"]
    winner_internal_seat = _web_to_internal_seat(_normalize_seat(acting_player["seat"]))
    giver_internal_seat = _web_to_internal_seat(_normalize_seat(giver["seat"]))
    net, _ = policy_store.get(_payload_value(payload, "checkpoint"))

    best_card = candidates[0]
    best_score = None
    for candidate in candidates:
        score = _score_return_candidate(
            env,
            winner_internal_seat=winner_internal_seat,
            giver_internal_seat=giver_internal_seat,
            tribute_card=tribute_card,
            return_card=candidate,
            net=net,
        )
        if best_score is None or score > best_score:
            best_card = candidate
            best_score = score
    return best_card


def _resolve_request_type(payload: Dict[str, Any]) -> str:
    explicit = _payload_value(payload, "requestType", "request", "type")
    if explicit:
        normalized = str(explicit).strip().lower()
        lookup = {
            "action": "action",
            "tributecard": "tributeCard",
            "returncard": "returnCard",
        }
        if normalized not in lookup:
            raise ValueError(
                "requestType must be one of: action, tributeCard, returnCard."
            )
        return lookup[normalized]

    match = _extract_match(payload)
    game_status = str(match.get("gameStatus") or "").lower()
    if game_status == "playing":
        return "action"
    if game_status == "tribute":
        trick_cards = match.get("trickLastPlay") or []
        tribute_cards = [card for card in trick_cards if isinstance(card, dict) and card.get("_type") == "tribute"]
        return "tributeCard" if len(tribute_cards) < 1 and len(trick_cards) < 1 else "returnCard"
    raise ValueError("Could not infer requestType from the payload.")


def decide(payload: Dict[str, Any], policy_store: PolicyStore) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    request_type = _resolve_request_type(payload)
    match = _extract_match(payload)
    players = _extract_players(payload)
    players_by_seat = _player_by_seat(players)

    if request_type == "action":
        if str(match.get("gameStatus") or "playing").lower() not in {"", "playing"}:
            raise ValueError("Action requests are only valid while match.gameStatus is 'playing'.")
        acting_player = _resolve_current_player(
            payload,
            players_by_seat,
            match,
            allow_match_current_seat=True,
        )
        context = _build_env_context(payload, request_type, acting_player)
        env = context["env"]
        seat = env.current_seat
        net, checkpoint_path = policy_store.get(_payload_value(payload, "checkpoint"))
        action_idx, action_cards = _choose_action(
            env,
            seat,
            net,
            sample=bool(_payload_value(payload, "sample", default=False)),
        )
        return {
            "decisionType": "action",
            "seat": acting_player["seat"],
            "action": action_cards,
            "actionIdx": action_idx,
            "pass": action_idx == 0,
            "checkpoint": checkpoint_path,
        }

    if str(match.get("gameStatus") or "tribute").lower() not in {"", "tribute"}:
        raise ValueError(
            "Tribute requests are only valid while match.gameStatus is 'tribute'."
        )

    acting_player = _resolve_current_player(
        payload,
        players_by_seat,
        match,
        allow_match_current_seat=False,
    )
    level_rank = _current_level_rank(match)
    tribute_state = _detect_tribute_state(players_by_seat)
    trick_cards = _clean_cards(match.get("trickLastPlay") or [])

    if tribute_state["type"].startswith("EXCEPTION_2"):
        return {
            "decisionType": "skipTribute",
            "seat": acting_player["seat"],
            "tributeState": tribute_state["type"],
            "skipTribute": True,
        }

    if request_type == "tributeCard":
        finish_place = str(acting_player.get("finishPlace") or "")
        if tribute_state["type"] == "BASIC" and finish_place != "4":
            raise ValueError("Basic tribute can only be chosen by finishPlace 4.")
        if tribute_state["type"] == "EXCEPTION_1":
            required_place = _exception1_required_place(trick_cards)
            if required_place not in {"3", "4"}:
                raise ValueError(
                    "EXCEPTION_1 is not currently waiting for a tribute card."
                )
            if finish_place != required_place:
                raise ValueError(
                    f"EXCEPTION_1 is waiting for finishPlace {required_place} to tribute."
                )
        candidates = _best_tribute_candidates(acting_player["hand"], level_rank)
        chosen = candidates[0]
        return {
            "decisionType": "tributeCard",
            "seat": acting_player["seat"],
            "tributeState": tribute_state["type"],
            "tributeCard": chosen,
        }

    finish_place = str(acting_player.get("finishPlace") or "")
    if tribute_state["type"] == "BASIC":
        if finish_place != "1":
            raise ValueError("Basic return can only be chosen by finishPlace 1.")
        chosen = _choose_return_card_basic(payload, acting_player, tribute_state, policy_store)
    elif tribute_state["type"] == "EXCEPTION_1":
        required_place = _exception1_required_place(trick_cards)
        if required_place not in {"1", "2"}:
            raise ValueError(
                "EXCEPTION_1 is not currently waiting for a return card."
            )
        if finish_place != required_place:
            raise ValueError(
                f"EXCEPTION_1 is waiting for finishPlace {required_place} to return."
            )
        chosen = _choose_return_card_exception1(payload, acting_player, policy_store)
    else:
        raise ValueError(f"Unsupported tribute state for return card: {tribute_state['type']}")

    return {
        "decisionType": "returnCard",
        "seat": acting_player["seat"],
        "tributeState": tribute_state["type"],
        "returnCard": chosen,
    }


_STATUS_TEXT = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    500: "Internal Server Error",
}

_default_policy_store = PolicyStore()


def _current_policy_store() -> PolicyStore:
    return _default_policy_store


def _set_default_policy_store(checkpoint_path: Optional[str] = None) -> None:
    global _default_policy_store
    _default_policy_store = PolicyStore(checkpoint_path=checkpoint_path)


def _json_wsgi_response(
    start_response,
    status_code: int,
    payload: Dict[str, Any],
) -> List[bytes]:
    encoded = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(encoded))),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ]
    start_response(f"{status_code} {_STATUS_TEXT[status_code]}", headers)
    return [encoded]


def _health_payload(policy_store: PolicyStore) -> Dict[str, Any]:
    checkpoint = None
    try:
        _, checkpoint = policy_store.get()
    except Exception:
        checkpoint = None
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": checkpoint,
    }


def app(environ, start_response):
    policy_store = _current_policy_store()
    method = str(environ.get("REQUEST_METHOD") or "GET").upper()
    path = str(environ.get("PATH_INFO") or "/").rstrip("/") or "/"

    if method == "OPTIONS":
        return _json_wsgi_response(start_response, 200, {"ok": True})

    if method == "GET" and path in {"/", "/health"}:
        return _json_wsgi_response(start_response, 200, _health_payload(policy_store))

    if method == "POST" and path == "/decision":
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            content_length = 0

        try:
            raw_body = environ["wsgi.input"].read(content_length or 0)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            response = decide(payload, policy_store)
            return _json_wsgi_response(start_response, 200, response)
        except ValueError as exc:
            return _json_wsgi_response(start_response, 400, {"error": str(exc)})
        except FileNotFoundError as exc:
            return _json_wsgi_response(start_response, 500, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - last-resort handler
            traceback.print_exc()
            return _json_wsgi_response(
                start_response,
                500,
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    return _json_wsgi_response(start_response, 404, {"error": "Not found"})


def run_server(host: str, port: int, checkpoint_path: Optional[str] = None) -> None:
    _set_default_policy_store(checkpoint_path=checkpoint_path)
    print(f"Bot API listening on http://{host}:{port}")
    print("POST /decision")
    print("GET  /health")
    with make_server(host, port, app) as server:
        server.serve_forever()


def parse_args() -> argparse.Namespace:
    default_host = os.environ.get("HOST", "127.0.0.1")
    default_port = int(os.environ.get("PORT", "8765"))
    parser = argparse.ArgumentParser(description="Serve the Guandan bot as a thin HTTP API.")
    parser.add_argument("--host", type=str, default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to GUANDAN_CHECKPOINT, checkpoint.pt, "
             "checkpoints/latest.pt, or the latest file in checkpoints/.",
    )
    args = parser.parse_args()
    if not (1 <= args.port <= 65535):
        parser.error("--port must be between 1 and 65535.")
    return args


def main() -> None:
    args = parse_args()
    run_server(args.host, args.port, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
