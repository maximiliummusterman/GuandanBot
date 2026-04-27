"""
Thin HTTP API for Guandan bot decisions.

This service is intended to run as a normal Python web service
(for example on Render), not as a serverless function bundle.

Supported decisions:
  - action:      choose the bot play for the current trick
  - tributeCard: choose the tribute card during the tribute phase
  - returnCard:  choose the return card during the tribute phase
  - update:      advance transformer context after a confirmed play/pass

Request shape:
  POST /decision
  {
    "requestType": "action" | "tributeCard" | "returnCard" | "update",
    "match": {...},              # alias: currentMatch
    "players": [...],            # alias: matchPlayers / allMatchPlayers
    "currentPlayer": {...},      # optional for tribute/return requests
    "seat": 1,                   # optional; defaults to match.currentSeat
    "sample": false,             # optional
    "checkpoint": "checkpoint.pt",  # optional override
    "transformerContext": {         # optional rolling context
      "roundKey": "...",
      "roundNumber": 1,
      "observation": {...},        # previous match snapshot for update diffing
      "pendingHistory": [...],
      "memory": [...]
    }
  }

Response shape:
  {
    "decisionType": "action" | "tributeCard" | "returnCard" | "skipTribute" | "update",
    "seat": "4",
    "action": [...],
    "actionIdx": 123,
    "pass": false,
    "tributeCard": {...},
    "returnCard": {...},
    "tributeState": "BASIC",
    "checkpoint": "checkpoint.pt",
    "updateReason": "play" | "pass" | "trick_cleared" | "no_change",
    "transformerContext": {...}
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
import threading
import traceback
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
from wsgiref.simple_server import make_server

import numpy as np
import torch
import torch.nn as nn

from bot_transformer import (
    DEFAULT_CHECKPOINT_DIR,
    DEVICE,
    GuandanEnv,
    HISTORY_ENTRY_DIM,
    HISTORY_SEQ_LEN,
    PolicyMemorySnapshot,
    RANKS,
    STATE_DIM,
    SUITS,
    _detect_from_hand,
    action_index_to_cards,
    deserialize_policy_memory_snapshot,
    encode_history_entry,
    expected_policy_memory_shape,
    list_checkpoint_paths,
    load_policy_network_from_checkpoint,
    policy_critic_value,
    policy_forward,
    policy_update_history_memory,
    policy_uses_history_memory,
    resolve_checkpoint_file,
    serialize_policy_memory_snapshot,
    zero_policy_memory,
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


def _level_rank_a_tries(rank: Any) -> int:
    text = str(rank or "2").strip().upper()
    if text == "A2":
        return 1
    if text == "A3":
        return 2
    return 0


def _match_round_key(match: Dict[str, Any]) -> str:
    return ":".join(
        [
            str(match.get("roundNumber") or 1),
            str(match.get("currentRoundLevelRank") or "Blue"),
            _normalize_level_rank(match.get("levelRankBlue")),
            _normalize_level_rank(match.get("levelRankRed")),
        ]
    )


def _transformer_context_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = _payload_value(payload, "transformerContext", "policyContext", "context")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("transformerContext must be an object when provided.")
    return dict(raw)


def _transformer_round_number(payload: Dict[str, Any]) -> Optional[int]:
    context_payload = _transformer_context_payload(payload)
    round_value = _payload_value(
        payload,
        "roundNumber",
        default=context_payload.get("roundNumber"),
    )
    if round_value in (None, ""):
        match = _payload_value(payload, "match", "currentMatch")
        if isinstance(match, dict):
            round_value = match.get("roundNumber")
    if round_value in (None, ""):
        return None
    try:
        return int(round_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid roundNumber: {round_value!r}") from exc


def _transformer_round_key(payload: Dict[str, Any]) -> Optional[str]:
    explicit = _payload_value(payload, "roundKey")
    if explicit not in (None, ""):
        return str(explicit)
    match = _payload_value(payload, "match", "currentMatch")
    if isinstance(match, dict):
        return _match_round_key(match)
    return None


def _empty_transformer_context(
    net: nn.Module,
    *,
    round_key: Optional[str],
    round_number: Optional[int],
    checkpoint_path: str,
) -> Dict[str, Any]:
    return {
        "checkpoint": checkpoint_path,
        "roundKey": round_key,
        "roundNumber": round_number,
        "pendingHistory": [],
        "memorySnapshot": zero_policy_memory(net),
        "observation": None,
    }


def _normalize_history_entry(raw_entry: Any) -> np.ndarray:
    entry = np.asarray(raw_entry, dtype=np.float32)
    if entry.shape != (HISTORY_ENTRY_DIM,):
        raise ValueError(
            "Each transformer history entry must have shape "
            f"({HISTORY_ENTRY_DIM},), got {tuple(entry.shape)}."
        )
    return entry


def _normalize_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def _normalize_optional_seat_text(value: Any, field_name: str) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(_normalize_seat(value, field_name))


def _normalize_transformer_observation_match(match: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(match, dict):
        raise ValueError("Transformer observation match must be an object.")
    return {
        "gameStatus": str(match.get("gameStatus") or ""),
        "currentSeat": _normalize_optional_seat_text(
            match.get("currentSeat"),
            "match.currentSeat",
        ),
        "currentRoundLevelRank": str(match.get("currentRoundLevelRank") or "Blue"),
        "levelRankBlue": _normalize_level_rank(match.get("levelRankBlue")),
        "levelRankRed": _normalize_level_rank(match.get("levelRankRed")),
        "trickLastPlay": _clean_cards(match.get("trickLastPlay") or []),
        "lastTrickSeat": _normalize_optional_seat_text(
            match.get("lastTrickSeat"),
            "match.lastTrickSeat",
        ),
        "roundNumber": _normalize_optional_int(
            match.get("roundNumber"),
            "match.roundNumber",
        ) or 1,
    }


def _transformer_observation_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    match = _payload_value(payload, "match", "currentMatch")
    if not isinstance(match, dict):
        return None
    return _normalize_transformer_observation_match(match)


def _normalize_transformer_observation_value(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, ""):
        return None
    if not isinstance(value, dict):
        raise ValueError("transformerContext.observation must be an object when provided.")
    return _normalize_transformer_observation_match(value)


def _store_transformer_observation(
    context: Dict[str, Any],
    observation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context["observation"] = observation
    return context


def _reset_transformer_context(
    context: Dict[str, Any],
    net: nn.Module,
    *,
    round_key: Optional[str],
    round_number: Optional[int],
    checkpoint_path: str,
) -> Dict[str, Any]:
    context["checkpoint"] = checkpoint_path
    context["roundKey"] = round_key
    context["roundNumber"] = round_number
    context["pendingHistory"] = []
    context["memorySnapshot"] = zero_policy_memory(net)
    context["observation"] = None
    return context


def _resolve_transformer_context(
    payload: Dict[str, Any],
    net: nn.Module,
    checkpoint_path: str,
) -> Dict[str, Any]:
    context_payload = _transformer_context_payload(payload)
    round_key = _transformer_round_key(payload)
    round_number = _transformer_round_number(payload)

    if not context_payload:
        return _empty_transformer_context(
            net,
            round_key=round_key,
            round_number=round_number,
            checkpoint_path=checkpoint_path,
        )

    context = {
        "checkpoint": context_payload.get("checkpoint"),
        "roundKey": context_payload.get("roundKey"),
        "roundNumber": _normalize_optional_int(
            context_payload.get("roundNumber"),
            "transformerContext.roundNumber",
        ),
        "observation": _normalize_transformer_observation_value(
            context_payload.get("observation")
        ),
        "pendingHistory": [
            _normalize_history_entry(entry)
            for entry in (context_payload.get("pendingHistory") or context_payload.get("historyEntries") or [])
        ],
        "memorySnapshot": deserialize_policy_memory_snapshot(
            _payload_value(
                context_payload,
                "memory",
                "memorySnapshot",
            )
        ),
    }

    if context.get("checkpoint") not in (None, checkpoint_path):
        return _reset_transformer_context(
            context,
            net,
            round_key=round_key,
            round_number=round_number,
            checkpoint_path=checkpoint_path,
        )

    if round_key is not None and round_key != context.get("roundKey"):
        return _reset_transformer_context(
            context,
            net,
            round_key=round_key,
            round_number=round_number,
            checkpoint_path=checkpoint_path,
        )

    if (
        round_number is not None
        and context.get("roundNumber") not in (None, round_number)
    ):
        return _reset_transformer_context(
            context,
            net,
            round_key=round_key,
            round_number=round_number,
            checkpoint_path=checkpoint_path,
        )

    if context.get("checkpoint") is None:
        context["checkpoint"] = checkpoint_path
    if context.get("roundKey") is None:
        context["roundKey"] = round_key
    if context.get("roundNumber") is None:
        context["roundNumber"] = round_number

    expected_shape = expected_policy_memory_shape(net)
    snapshot = context.get("memorySnapshot")
    if snapshot is not None and snapshot.memory is not None and expected_shape is not None:
        actual_shape = tuple(int(value) for value in snapshot.memory.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                "transformerContext.memory does not match the loaded checkpoint. "
                f"Expected {expected_shape}, got {actual_shape}."
            )
    if snapshot is None and policy_uses_history_memory(net):
        context["memorySnapshot"] = zero_policy_memory(net)

    if len(context["pendingHistory"]) > HISTORY_SEQ_LEN and not policy_uses_history_memory(net):
        context["pendingHistory"] = context["pendingHistory"][-HISTORY_SEQ_LEN:]

    return context


def _serialize_transformer_context(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "checkpoint": context.get("checkpoint"),
        "roundKey": context.get("roundKey"),
        "roundNumber": context.get("roundNumber"),
        "observation": context.get("observation"),
        "pendingHistory": [
            np.asarray(entry, dtype=np.float32).tolist()
            for entry in context.get("pendingHistory", [])
        ],
        "pendingHistoryCount": len(context.get("pendingHistory", [])),
        "memory": serialize_policy_memory_snapshot(context.get("memorySnapshot")),
    }


def _transformer_history_entries(context: Optional[Dict[str, Any]]) -> List[np.ndarray]:
    if not context:
        return []
    return list(context.get("pendingHistory") or [])


def _transformer_memory_snapshot(
    context: Optional[Dict[str, Any]],
) -> Optional[PolicyMemorySnapshot]:
    if not context:
        return None
    return context.get("memorySnapshot")


def _advance_transformer_context(
    context: Dict[str, Any],
    net: nn.Module,
    history_entry: np.ndarray,
) -> Dict[str, Any]:
    pending_history = list(context.get("pendingHistory") or [])
    pending_history.append(_normalize_history_entry(history_entry))

    if policy_uses_history_memory(net):
        memory_snapshot = context.get("memorySnapshot")
        while len(pending_history) > HISTORY_SEQ_LEN:
            segment = np.asarray(pending_history[:HISTORY_SEQ_LEN], dtype=np.float32)[None, :, :]
            memory_snapshot = policy_update_history_memory(
                net,
                segment,
                memory_snapshot,
            )
            pending_history = pending_history[HISTORY_SEQ_LEN:]
        context["memorySnapshot"] = memory_snapshot
    else:
        pending_history = pending_history[-HISTORY_SEQ_LEN:]

    context["pendingHistory"] = pending_history
    return context


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


def _next_active_seat(env: GuandanEnv, seat: int) -> Optional[int]:
    next_seat = (seat + 1) % 4
    for _ in range(4):
        if next_seat not in env.finish_order:
            return next_seat
        next_seat = (next_seat + 1) % 4
    return None


def _infer_pass_streak(env: GuandanEnv, acting_internal_seat: int) -> int:
    if not env.trick_last or env.trick_seat is None:
        return 0
    if acting_internal_seat == env.trick_seat:
        return 0

    passes = 0
    seat = env.trick_seat
    for _ in range(4):
        next_seat = _next_active_seat(env, seat)
        if next_seat is None:
            break
        if next_seat == acting_internal_seat:
            return passes
        passes += 1
        seat = next_seat
    return 0


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
    a_tries_by_web_team = {
        "Blue": _level_rank_a_tries(match.get("levelRankBlue")),
        "Red": _level_rank_a_tries(match.get("levelRankRed")),
    }
    env.level_ranks = {
        "Blue": rank_by_web_team.get(web_for_internal["Blue"], rank_by_web_team["Blue"]),
        "Red": rank_by_web_team.get(web_for_internal["Red"], rank_by_web_team["Red"]),
    }
    caller_web_team = str(match.get("currentRoundLevelRank") or "Blue")
    env.caller = internal_for_web.get(caller_web_team, "Blue")
    env.a_tries = {
        "Blue": a_tries_by_web_team.get(web_for_internal["Blue"], a_tries_by_web_team["Blue"]),
        "Red": a_tries_by_web_team.get(web_for_internal["Red"], a_tries_by_web_team["Red"]),
    }
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

    explicit_pass_streak = _payload_value(
        payload,
        "passStreak",
        "pass_streak",
        default=match.get("passStreak"),
    )
    if explicit_pass_streak not in (None, ""):
        env.pass_streak = max(0, int(explicit_pass_streak))
    else:
        env.pass_streak = _infer_pass_streak(env, env.current_seat)

    return {
        "env": env,
        "match": match,
        "players_by_seat": players_by_seat,
        "web_for_internal": web_for_internal,
        "internal_for_web": internal_for_web,
    }


def _policy_state_value(
    net: nn.Module,
    env: GuandanEnv,
    seat: int,
    transformer_context: Optional[Dict[str, Any]],
) -> float:
    state_np = env.get_state(
        seat,
        history_entries=_transformer_history_entries(transformer_context),
    )
    return policy_critic_value(
        net,
        state_np,
        memory_snapshot=_transformer_memory_snapshot(transformer_context),
    )


def _same_card_sequence(
    cards_a: Iterable[Dict[str, Any]],
    cards_b: Iterable[Dict[str, Any]],
) -> bool:
    cards_a_list = list(cards_a or [])
    cards_b_list = list(cards_b or [])
    if len(cards_a_list) != len(cards_b_list):
        return False
    return all(
        _card_signature(card_a) == _card_signature(card_b)
        for card_a, card_b in zip(cards_a_list, cards_b_list)
    )


def _observation_level_rank(observation: Dict[str, Any]) -> str:
    caller = str(observation.get("currentRoundLevelRank") or "Blue")
    if caller == "Red":
        return _normalize_level_rank(observation.get("levelRankRed"))
    return _normalize_level_rank(observation.get("levelRankBlue"))


def _infer_update_history_entry(
    previous_observation: Optional[Dict[str, Any]],
    current_observation: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str]:
    if current_observation.get("gameStatus", "").lower() != "playing":
        return None, "non_playing"
    if previous_observation is None:
        return None, "missing_previous_observation"
    if previous_observation.get("gameStatus", "").lower() != "playing":
        return None, "previous_non_playing"
    if _match_round_key(previous_observation) != _match_round_key(current_observation):
        return None, "round_changed"

    previous_trick = previous_observation.get("trickLastPlay") or []
    current_trick = current_observation.get("trickLastPlay") or []
    previous_current_seat = previous_observation.get("currentSeat")
    current_current_seat = current_observation.get("currentSeat")
    previous_last_trick_seat = previous_observation.get("lastTrickSeat")
    current_last_trick_seat = current_observation.get("lastTrickSeat")

    trick_changed = not _same_card_sequence(previous_trick, current_trick)
    current_seat_changed = previous_current_seat != current_current_seat

    if trick_changed:
        if not current_trick:
            return None, "trick_cleared"
        if current_last_trick_seat is None:
            raise ValueError(
                "Update request cannot infer the acting player because trickLastPlay changed "
                "but lastTrickSeat is missing."
            )
        level_rank = _observation_level_rank(current_observation)
        combo_info = _detect_from_hand(current_trick, level_rank)
        if not combo_info or combo_info.get("type") == "INVALID":
            raise ValueError("Updated trickLastPlay does not form a valid combination.")
        history_entry = encode_history_entry(
            _web_to_internal_seat(
                _normalize_seat(current_last_trick_seat, "match.lastTrickSeat")
            ),
            current_trick,
            combo_info,
        )
        return history_entry, "play"

    if (
        current_seat_changed
        and previous_current_seat is not None
        and previous_trick
        and previous_last_trick_seat == current_last_trick_seat
    ):
        history_entry = encode_history_entry(
            _web_to_internal_seat(
                _normalize_seat(previous_current_seat, "match.currentSeat")
            ),
            [],
            {"type": "PASS", "strength": 0, "bomb_strength": 0},
        )
        return history_entry, "pass"

    return None, "no_change"


def _handle_update_request(
    payload: Dict[str, Any],
    policy_store: PolicyStore,
) -> Dict[str, Any]:
    net, checkpoint_path = policy_store.get(_payload_value(payload, "checkpoint"))
    match = _extract_match(payload)
    context = _resolve_transformer_context(payload, net, checkpoint_path)
    current_observation = _normalize_transformer_observation_match(match)
    previous_observation = context.get("observation")
    history_entry, update_reason = _infer_update_history_entry(
        previous_observation,
        current_observation,
    )
    if history_entry is not None:
        _advance_transformer_context(context, net, history_entry)
    _store_transformer_observation(context, current_observation)

    return {
        "decisionType": "update",
        "checkpoint": checkpoint_path,
        "historyUpdated": history_entry is not None,
        "updateReason": update_reason,
        "usesHistoryMemory": policy_uses_history_memory(net),
        "transformerContext": _serialize_transformer_context(context),
    }


def resolve_checkpoint_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return str(resolve_checkpoint_file(explicit_path))

    env_path = os.environ.get("GUANDAN_CHECKPOINT")
    if env_path:
        return resolve_checkpoint_path(env_path)

    preferred_paths = ["checkpoint.pt", "checkpoints/latest.pt"]
    for checkpoint_ref in preferred_paths:
        try:
            return str(resolve_checkpoint_file(checkpoint_ref))
        except FileNotFoundError:
            continue

    checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "Could not find checkpoint.pt or checkpoints/. "
            "Set GUANDAN_CHECKPOINT or pass --checkpoint."
        )

    checkpoint_paths = list_checkpoint_paths(checkpoint_dir)
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
        self._net: Optional[nn.Module] = None
        self._lock = threading.Lock()

    def get(self, override_path: Optional[str] = None) -> Tuple[nn.Module, str]:
        resolved_path = resolve_checkpoint_path(override_path or self._checkpoint_path)
        with self._lock:
            if self._net is not None and resolved_path == self._loaded_path:
                return self._net, resolved_path

            checkpoint = torch.load(resolved_path, map_location=DEVICE)
            if "model_state" not in checkpoint:
                raise KeyError(f"Checkpoint is missing 'model_state': {resolved_path}")

            net = load_policy_network_from_checkpoint(
                checkpoint,
                device=DEVICE,
            )
            net.eval()

            self._net = net
            self._loaded_path = resolved_path
            return net, resolved_path


def _choose_action(
    env: GuandanEnv,
    seat: int,
    net: nn.Module,
    *,
    transformer_context: Optional[Dict[str, Any]] = None,
    sample: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    history_entries = _transformer_history_entries(transformer_context)
    if not history_entries:
        history_entries = list(env.round_history_entries[-HISTORY_SEQ_LEN:])
    state_np = env.get_state(seat, history_entries=history_entries)[None, :]
    mask_np = env.get_legal_mask(seat)[None, :]
    logits, _ = policy_forward(
        net,
        state_np,
        masks_np=mask_np,
        memory_snapshot=_transformer_memory_snapshot(transformer_context),
    )

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
    net: nn.Module,
    transformer_context: Optional[Dict[str, Any]],
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
    return _policy_state_value(
        net,
        sim_env,
        winner_internal_seat,
        transformer_context,
    )


def _choose_return_card_basic(
    payload: Dict[str, Any],
    acting_player: Dict[str, Any],
    tribute_state: Dict[str, Any],
    policy_store: PolicyStore,
    transformer_context: Optional[Dict[str, Any]],
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
            transformer_context=transformer_context,
        )
        if best_score is None or score > best_score:
            best_card = candidate
            best_score = score
    return best_card


def _choose_tribute_card_basic(
    payload: Dict[str, Any],
    acting_player: Dict[str, Any],
    tribute_state: Dict[str, Any],
    policy_store: PolicyStore,
    transformer_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    match = _extract_match(payload)
    level_rank = _current_level_rank(match)
    candidates = _best_tribute_candidates(acting_player["hand"], level_rank)
    if len(candidates) == 1:
        return candidates[0]

    winners = tribute_state.get("winners") or []
    if not winners:
        return candidates[0]

    context = _build_env_context(payload, "tributeCard", acting_player)
    env = context["env"]
    giver_internal_seat = _web_to_internal_seat(_normalize_seat(acting_player["seat"]))
    winner_internal_seat = _web_to_internal_seat(
        _normalize_seat(winners[0]["seat"])
    )
    net, _ = policy_store.get(_payload_value(payload, "checkpoint"))

    best_card = candidates[0]
    best_score = None
    for candidate in candidates:
        sim_env = copy.deepcopy(env)
        giver_hand = sim_env.hands[giver_internal_seat]
        winner_hand = sim_env.hands[winner_internal_seat]

        _remove_one_card(giver_hand, candidate)
        winner_hand.append(dict(_clean_card(candidate)))

        return_candidates = _best_return_candidates(winner_hand, level_rank)
        if len(return_candidates) == 1:
            return_card = return_candidates[0]
        else:
            return_card = max(
                return_candidates,
                key=lambda return_candidate: _score_return_candidate(
                    sim_env,
                    winner_internal_seat=winner_internal_seat,
                    giver_internal_seat=giver_internal_seat,
                    tribute_card=candidate,
                    return_card=return_candidate,
                    net=net,
                    transformer_context=transformer_context,
                ),
            )

        _remove_one_card(winner_hand, return_card)
        giver_hand.append(dict(_clean_card(return_card)))

        score = _policy_state_value(
            net,
            sim_env,
            giver_internal_seat,
            transformer_context,
        )
        if best_score is None or score > best_score:
            best_card = candidate
            best_score = score

    return best_card


def _choose_return_card_exception1(
    payload: Dict[str, Any],
    acting_player: Dict[str, Any],
    policy_store: PolicyStore,
    transformer_context: Optional[Dict[str, Any]],
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
            transformer_context=transformer_context,
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
            "update": "update",
        }
        if normalized not in lookup:
            raise ValueError(
                "requestType must be one of: action, tributeCard, returnCard, update."
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
    if request_type == "update":
        return _handle_update_request(payload, policy_store)

    match = _extract_match(payload)
    players = _extract_players(payload)
    players_by_seat = _player_by_seat(players)
    net, checkpoint_path = policy_store.get(_payload_value(payload, "checkpoint"))
    transformer_context = _resolve_transformer_context(payload, net, checkpoint_path)
    _store_transformer_observation(
        transformer_context,
        _transformer_observation_from_payload(payload),
    )

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
        action_idx, action_cards = _choose_action(
            env,
            seat,
            net,
            transformer_context=transformer_context,
            sample=bool(_payload_value(payload, "sample", default=False)),
        )
        return {
            "decisionType": "action",
            "seat": acting_player["seat"],
            "action": action_cards,
            "actionIdx": action_idx,
            "pass": action_idx == 0,
            "checkpoint": checkpoint_path,
            "transformerContext": _serialize_transformer_context(transformer_context),
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
            "checkpoint": checkpoint_path,
            "transformerContext": _serialize_transformer_context(transformer_context),
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
        if tribute_state["type"] == "BASIC":
            chosen = _choose_tribute_card_basic(
                payload,
                acting_player,
                tribute_state,
                policy_store,
                transformer_context,
            )
        else:
            candidates = _best_tribute_candidates(acting_player["hand"], level_rank)
            chosen = candidates[0]
        return {
            "decisionType": "tributeCard",
            "seat": acting_player["seat"],
            "tributeState": tribute_state["type"],
            "tributeCard": chosen,
            "checkpoint": checkpoint_path,
            "transformerContext": _serialize_transformer_context(transformer_context),
        }

    finish_place = str(acting_player.get("finishPlace") or "")
    if tribute_state["type"] == "BASIC":
        if finish_place != "1":
            raise ValueError("Basic return can only be chosen by finishPlace 1.")
        chosen = _choose_return_card_basic(
            payload,
            acting_player,
            tribute_state,
            policy_store,
            transformer_context,
        )
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
        chosen = _choose_return_card_exception1(
            payload,
            acting_player,
            policy_store,
            transformer_context,
        )
    else:
        raise ValueError(f"Unsupported tribute state for return card: {tribute_state['type']}")

    return {
        "decisionType": "returnCard",
        "seat": acting_player["seat"],
        "tributeState": tribute_state["type"],
        "returnCard": chosen,
        "checkpoint": checkpoint_path,
        "transformerContext": _serialize_transformer_context(transformer_context),
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


def _read_json_request_body(environ: Dict[str, Any]) -> Dict[str, Any]:
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or "0")
    except ValueError:
        content_length = 0
    raw_body = environ["wsgi.input"].read(content_length or 0)
    payload = json.loads(raw_body.decode("utf-8") or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


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
        "state_dim": STATE_DIM,
        "history_seq_len": HISTORY_SEQ_LEN,
        "history_entry_dim": HISTORY_ENTRY_DIM,
        "supports_update": True,
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
            payload = _read_json_request_body(environ)
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
