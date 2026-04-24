"""
Thin website-to-arena tunnel for Guandan bots.

This module keeps only the bridge logic that the website needs:
  - normalize PocketBase match snapshots,
  - bootstrap / read tunnelState,
  - encode one website turn into the arena state vector,
  - decode one arena action back into website cards,
  - pick a play or tribute action for a bot.

All real card-combination, masking, action-catalogue, and model logic stays in
guandan_arena.py so we do not duplicate arena code here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from guandan_arena import (  # noqa: E402
    DEVICE,
    GuandanNet,
    RANKS,
    _detect_from_hand,
    _upgrade_legacy_model_state,
    action_index_to_cards,
    compute_legal_mask,
    encode_state,
)


TUNNEL_VERSION = 1
RETURN_RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10"}
RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
SUIT_ORDER = {
    "clubs": 0,
    "diamonds": 1,
    "hearts": 2,
    "spades": 3,
}


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_level_rank(level_rank: Optional[str]) -> Optional[str]:
    if level_rank is None:
        return None
    if level_rank in {"A1", "A2", "A3"}:
        return "A"
    return str(level_rank)


def _team_rank_progress(level_rank: Optional[str]) -> float:
    if level_rank in {"A1", "A2", "A3"}:
        return 1.0
    try:
        return RANKS.index(str(level_rank)) / 12.0
    except ValueError:
        return 0.0


def seat_to_index(seat: int) -> int:
    if not 1 <= seat <= 4:
        raise ValueError(f"Website seat must be 1-4, got {seat!r}.")
    return seat - 1


def _clean_card(card: Dict[str, Any], keep_id: bool = True) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    if keep_id and card.get("id") is not None:
        clean["id"] = card["id"]
    if card.get("joker") is not None:
        clean["joker"] = card["joker"]
        return clean
    if card.get("suit") is not None:
        clean["suit"] = card["suit"]
    if card.get("rank") is not None:
        clean["rank"] = card["rank"]
    return clean


def _clean_cards(cards: Sequence[Dict[str, Any]], keep_id: bool = True) -> List[Dict[str, Any]]:
    return [_clean_card(card, keep_id=keep_id) for card in cards]


def _cards_match(card_a: Dict[str, Any], card_b: Dict[str, Any]) -> bool:
    a_id = card_a.get("id")
    b_id = card_b.get("id")
    if a_id is not None and b_id is not None:
        return a_id == b_id
    return (
        card_a.get("joker") == card_b.get("joker")
        and card_a.get("suit") == card_b.get("suit")
        and card_a.get("rank") == card_b.get("rank")
    )


def remove_cards_from_hand(
    hand: Sequence[Dict[str, Any]],
    cards_to_remove: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    updated = [copy.deepcopy(card) for card in hand]
    for target in cards_to_remove:
        remove_index = None
        for index, card in enumerate(updated):
            if _cards_match(card, target):
                remove_index = index
                break
        if remove_index is None:
            raise ValueError(f"Card not found in hand: {target!r}")
        updated.pop(remove_index)
    return updated


def _round_key(match_record: Dict[str, Any]) -> str:
    return ":".join(
        [
            str(match_record.get("roundNumber") or 1),
            str(match_record.get("currentRoundLevelRank") or "Blue"),
            str(match_record.get("levelRankBlue") or "2"),
            str(match_record.get("levelRankRed") or "2"),
        ]
    )


def sync_tunnel_state(
    match_record: Dict[str, Any],
    match_players: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_state = copy.deepcopy(match_record.get("tunnelState") or {})
    round_key = _round_key(match_record)
    same_round = raw_state.get("roundKey") == round_key
    players = [
        player for player in match_players
        if _coerce_int(player.get("finishPlace")) is not None
    ]
    players.sort(key=lambda player: (_coerce_int(player.get("finishPlace")) or 99, int(player["seat"])))
    trick_cards = (
        _clean_cards(match_record.get("trickLastPlay") or [])
        if str(match_record.get("gameStatus") or "").lower() == "playing"
        else []
    )
    current_trick = raw_state.get("currentTrick") if same_round else {}
    current_cards = _clean_cards((current_trick or {}).get("cards") or [])
    pass_count = int((current_trick or {}).get("passCount") or 0) if current_cards == trick_cards else 0

    return {
        "version": TUNNEL_VERSION,
        "roundKey": round_key,
        "phase": match_record.get("gameStatus") or "dealing",
        "finishOrder": [str(player["seat"]) for player in players],
        "events": list(raw_state.get("events") or []) if same_round else [],
        "currentTrick": {
            "leaderSeat": str(match_record.get("lastTrickSeat")) if trick_cards and match_record.get("lastTrickSeat") is not None else None,
            "cards": trick_cards,
            "passCount": pass_count if trick_cards else 0,
        },
    }


@dataclass
class WebsitePlayer:
    seat: int
    team: str
    hand: List[Dict[str, Any]]
    finish_place: Optional[int]
    user_id: Optional[str]
    username: Optional[str]

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "WebsitePlayer":
        raw_user_id = record.get("user_id")
        if isinstance(raw_user_id, dict):
            raw_user_id = raw_user_id.get("id")
        seat = int(record["seat"])
        fallback_team = "Blue" if seat in {1, 3} else "Red"
        expand_user = ((record.get("expand") or {}).get("user_id") or {})
        return cls(
            seat=seat,
            team=str(record.get("team") or fallback_team),
            hand=[copy.deepcopy(card) for card in (record.get("hand") or [])],
            finish_place=_coerce_int(record.get("finishPlace")),
            user_id=str(raw_user_id) if raw_user_id is not None else None,
            username=expand_user.get("username"),
        )


@dataclass
class WebsiteSnapshot:
    match: Dict[str, Any]
    players: List[WebsitePlayer]
    players_by_seat: Dict[int, WebsitePlayer]
    players_by_user_id: Dict[str, WebsitePlayer]

    @classmethod
    def from_records(
        cls,
        match_record: Dict[str, Any],
        match_players: Sequence[Dict[str, Any]],
    ) -> "WebsiteSnapshot":
        players = [WebsitePlayer.from_record(record) for record in match_players]
        players.sort(key=lambda player: player.seat)
        return cls(
            match={**copy.deepcopy(match_record), "tunnelState": sync_tunnel_state(match_record, match_players)},
            players=players,
            players_by_seat={player.seat: player for player in players},
            players_by_user_id={
                player.user_id: player
                for player in players
                if player.user_id is not None
            },
        )

    @property
    def phase(self) -> str:
        return str(self.match.get("gameStatus") or "dealing")

    @property
    def current_seat(self) -> Optional[int]:
        return _coerce_int(self.match.get("currentSeat"))

    @property
    def current_level_rank_label(self) -> str:
        caller = str(self.match.get("currentRoundLevelRank") or "Blue")
        return str(self.match.get("levelRankBlue") or "2") if caller == "Blue" else str(self.match.get("levelRankRed") or "2")

    @property
    def current_level_rank(self) -> str:
        return normalize_level_rank(self.current_level_rank_label) or "2"

    @property
    def level_rank_progress(self) -> Tuple[float, float]:
        return (
            _team_rank_progress(self.match.get("levelRankBlue") or "2"),
            _team_rank_progress(self.match.get("levelRankRed") or "2"),
        )

    @property
    def tunnel_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.match["tunnelState"])

    @property
    def played_cards(self) -> List[Dict[str, Any]]:
        explicit = self.match.get("playedCards")
        if explicit:
            return _clean_cards(explicit)

        played: List[Dict[str, Any]] = []
        for event in (self.match.get("tunnelState") or {}).get("events") or []:
            if event.get("type") == "play":
                played.extend(_clean_cards(event.get("cards") or []))

        # Fallback for snapshots that were taken before the tunnel history existed.
        if not played and self.phase.lower() == "playing":
            played = _clean_cards(self.match.get("trickLastPlay") or [])
        return played

    @property
    def outstanding_trick(self) -> List[Dict[str, Any]]:
        trick = (self.match.get("tunnelState") or {}).get("currentTrick") or {}
        cards = _clean_cards(trick.get("cards") or [])
        if cards:
            return cards
        return _clean_cards(self.match.get("trickLastPlay") or [])

    def resolve_seat(
        self,
        seat: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> int:
        if seat is not None:
            resolved = _coerce_int(seat)
            if resolved not in self.players_by_seat:
                raise KeyError(f"Unknown seat {seat!r}.")
            return int(resolved)
        if user_id is not None:
            key = str(user_id)
            if key not in self.players_by_user_id:
                raise KeyError(f"Unknown user_id {user_id!r}.")
            return self.players_by_user_id[key].seat
        if self.current_seat is not None:
            return self.current_seat
        raise ValueError("No seat was provided and the match does not expose currentSeat.")

    def hand_sizes(self) -> List[int]:
        return [len(self.players_by_seat[seat].hand) for seat in range(1, 5)]

    def encode_turn(self, seat: int) -> np.ndarray:
        player = self.players_by_seat[seat]
        trick = self.outstanding_trick
        return encode_state(
            hand=player.hand,
            trick_last=trick,
            played_cards=self.played_cards,
            level_rank=self.current_level_rank,
            my_seat=seat_to_index(seat),
            hand_sizes=self.hand_sizes(),
            level_ranks=self.level_rank_progress,
            is_opener=(len(trick) == 0),
        )

    def legal_mask(self, seat: int) -> np.ndarray:
        player = self.players_by_seat[seat]
        trick = self.outstanding_trick
        return compute_legal_mask(
            hand=player.hand,
            trick_last=trick,
            level_rank=self.current_level_rank,
            is_opener=(len(trick) == 0),
        )

    def decode_action(self, seat: int, action_idx: int) -> Dict[str, Any]:
        cards = action_index_to_cards(action_idx, self.players_by_seat[seat].hand, self.current_level_rank)
        return {
            "phase": self.phase,
            "seat": seat,
            "seatIndex": seat_to_index(seat),
            "action": "pass" if action_idx == 0 else "play",
            "actionIndex": int(action_idx),
            "cards": _clean_cards(cards),
            "combo": _detect_from_hand(cards, self.current_level_rank) if cards else None,
            "handler": "handlePass" if action_idx == 0 else "handlePlayCards",
            "currentTrick": self.outstanding_trick,
            "isOpener": len(self.outstanding_trick) == 0,
        }


def load_policy(checkpoint_path: str) -> GuandanNet:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state': {checkpoint_path}")
    net = GuandanNet().to(DEVICE)
    model_state, _ = _upgrade_legacy_model_state(checkpoint["model_state"])
    net.load_state_dict(model_state)
    net.eval()
    return net


def _critic_value(
    net: GuandanNet,
    hand: Sequence[Dict[str, Any]],
    hand_sizes: Sequence[int],
    level_rank: str,
    level_rank_progress: Tuple[float, float],
    seat: int,
    played_cards: Optional[Sequence[Dict[str, Any]]] = None,
) -> float:
    state = encode_state(
        hand=list(hand),
        trick_last=[],
        played_cards=list(played_cards or []),
        level_rank=level_rank,
        my_seat=seat_to_index(seat),
        hand_sizes=list(hand_sizes),
        level_ranks=level_rank_progress,
        is_opener=True,
    )[None, :]
    states = torch.from_numpy(state).to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        _, value = net(states)
    return float(value.squeeze(-1).item())


def choose_model_action(
    snapshot: WebsiteSnapshot,
    seat: int,
    net: GuandanNet,
    checkpoint_path: Optional[str] = None,
    sample: bool = False,
) -> Dict[str, Any]:
    state_np = snapshot.encode_turn(seat)[None, :]
    mask_np = snapshot.legal_mask(seat)[None, :]
    states = torch.from_numpy(state_np).to(DEVICE, non_blocking=True)
    masks = torch.from_numpy(mask_np).to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        logits, value = net(states, masks)

    hand = snapshot.players_by_seat[seat].hand

    def try_action(action_idx: int) -> Optional[Dict[str, Any]]:
        cards = action_index_to_cards(action_idx, hand, snapshot.current_level_rank)
        if action_idx == 0 or cards:
            payload = snapshot.decode_action(seat, action_idx)
            payload["criticValue"] = float(value.squeeze(-1).item())
            payload["checkpoint"] = checkpoint_path
            payload["sampled"] = bool(sample)
            return payload
        return None

    if sample:
        dist = torch.distributions.Categorical(logits=logits)
        for _ in range(4):
            choice = try_action(int(dist.sample().item()))
            if choice is not None:
                return choice
    else:
        choice = try_action(int(torch.argmax(logits, dim=1).item()))
        if choice is not None:
            return choice

    for action_idx in torch.argsort(logits[0], descending=True).tolist():
        action_idx = int(action_idx)
        if not bool(mask_np[0, action_idx]):
            continue
        choice = try_action(action_idx)
        if choice is not None:
            return choice

    raise RuntimeError(f"No legal action could be decoded for seat {seat}.")


def _rank_value(rank: Optional[str]) -> int:
    return RANK_VALUE.get(str(rank), 0)


def choose_highest_tribute_card(
    hand: Sequence[Dict[str, Any]],
    level_rank: str,
) -> Optional[Dict[str, Any]]:
    eligible = [
        card
        for card in hand
        if not (card.get("suit") == "hearts" and card.get("rank") == level_rank)
    ]
    if not eligible:
        return None

    def score(card: Dict[str, Any]) -> int:
        if card.get("joker") == "red":
            return 1000
        if card.get("joker") == "black":
            return 999
        if card.get("rank") == level_rank:
            return 100
        return _rank_value(card.get("rank"))

    return copy.deepcopy(max(eligible, key=score))


def valid_return_cards(
    hand: Sequence[Dict[str, Any]],
    level_rank: str,
) -> List[Dict[str, Any]]:
    return [
        copy.deepcopy(card)
        for card in hand
        if card.get("joker") is None
        and card.get("rank") != level_rank
        and card.get("rank") in RETURN_RANKS
    ]


def choose_return_card(
    snapshot: WebsiteSnapshot,
    winner_seat: int,
    giver_seat: int,
    tribute_card: Dict[str, Any],
    net: Optional[GuandanNet] = None,
) -> Optional[Dict[str, Any]]:
    candidates = valid_return_cards(snapshot.players_by_seat[winner_seat].hand, snapshot.current_level_rank)
    if not candidates:
        return None
    if net is None or len(candidates) == 1:
        return copy.deepcopy(sorted(candidates, key=lambda card: (_rank_value(card.get("rank")), SUIT_ORDER.get(card.get("suit"), 99)))[0])

    hands = {
        player.seat: [copy.deepcopy(card) for card in player.hand]
        for player in snapshot.players
    }
    best_card = candidates[0]
    best_value = None
    for candidate in candidates:
        simulated = copy.deepcopy(hands)
        simulated[winner_seat] = remove_cards_from_hand(simulated[winner_seat], [candidate])
        simulated[winner_seat].append(copy.deepcopy(tribute_card))
        simulated[giver_seat].append(copy.deepcopy(candidate))
        critic = _critic_value(
            net,
            hand=simulated[winner_seat],
            hand_sizes=[len(simulated[i]) for i in range(1, 5)],
            level_rank=snapshot.current_level_rank,
            level_rank_progress=snapshot.level_rank_progress,
            seat=winner_seat,
            played_cards=[],
        )
        if best_value is None or critic > best_value:
            best_card = candidate
            best_value = critic
    return copy.deepcopy(best_card)


def detect_tribute_state(snapshot: WebsiteSnapshot) -> Dict[str, Any]:
    finished = [player for player in snapshot.players if player.finish_place is not None]
    finished.sort(key=lambda player: (player.finish_place, player.seat))
    first = next((player for player in finished if player.finish_place == 1), None)
    second = next((player for player in finished if player.finish_place == 2), None)
    third = next((player for player in finished if player.finish_place == 3), None)
    fourth = next((player for player in finished if player.finish_place == 4), None)

    def red_jokers(player: Optional[WebsitePlayer]) -> int:
        if player is None:
            return 0
        return sum(1 for card in player.hand if card.get("joker") == "red")

    if fourth and red_jokers(fourth) == 2:
        return {"type": "EXCEPTION_2a"}

    if third and fourth and third.team == fourth.team:
        third_red = red_jokers(third)
        fourth_red = red_jokers(fourth)
        if third_red == 2 or fourth_red == 2:
            return {"type": "EXCEPTION_2b_CASE_1"}
        if third_red == 1 and fourth_red == 1:
            return {"type": "EXCEPTION_2b_CASE_2"}

    if first and second and first.team == second.team:
        return {"type": "EXCEPTION_1"}

    return {"type": "BASIC"}


def _required_place_exception_1(trick_cards: Sequence[Dict[str, Any]]) -> str:
    tributes = [card for card in trick_cards if card.get("_type") == "tribute"]
    returns = [card for card in trick_cards if card.get("_type") == "return"]
    if len(tributes) == 2 and len(returns) == 2:
        return "take_returns"
    if len(tributes) == 1 and len(returns) == 1:
        return "take_returns"
    if len(tributes) == 2 and len(returns) == 1:
        return "2"
    if len(tributes) == 2 and len(returns) == 0:
        return "1"
    if len(tributes) == 1 and len(returns) == 0:
        return "4"
    return "3"


def _exception_1_assignments(
    trick_cards: Sequence[Dict[str, Any]],
    level_rank: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    tributes = [card for card in trick_cards if card.get("_type") == "tribute"]
    if len(tributes) != 2:
        return None

    def score(card: Dict[str, Any]) -> int:
        if card.get("joker") == "red":
            return 1000
        if card.get("joker") == "black":
            return 900
        if card.get("rank") == level_rank and card.get("suit") == "hearts":
            return 800
        if card.get("rank") == level_rank:
            return 700
        return _rank_value(card.get("rank"))

    higher, lower = tributes[0], tributes[1]
    if score(lower) > score(higher):
        higher, lower = lower, higher
    return {"1": copy.deepcopy(higher), "2": copy.deepcopy(lower)}


def decide_tribute_action(
    snapshot: WebsiteSnapshot,
    seat: int,
    net: Optional[GuandanNet] = None,
) -> Dict[str, Any]:
    player = snapshot.players_by_seat[seat]
    tribute_state = detect_tribute_state(snapshot)["type"]
    trick_cards = [copy.deepcopy(card) for card in (snapshot.match.get("trickLastPlay") or [])]

    if tribute_state in {"EXCEPTION_2a", "EXCEPTION_2b_CASE_1", "EXCEPTION_2b_CASE_2"}:
        if player.finish_place == 1:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "skip_tribute",
                "cards": [],
                "handler": "handleSkipTribute",
                "tributeType": tribute_state,
            }
        return {
            "phase": "tribute",
            "seat": seat,
            "action": "wait",
            "cards": [],
            "handler": None,
            "tributeType": tribute_state,
        }

    if tribute_state == "EXCEPTION_1":
        required_place = _required_place_exception_1(trick_cards)
        if required_place == "take_returns":
            my_return = next(
                (
                    card for card in trick_cards
                    if card.get("_type") == "return"
                    and str(card.get("_targetGiverId")) == str(player.user_id)
                ),
                None,
            )
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "take_return" if my_return else "wait",
                "cards": [_clean_card(my_return)] if my_return else [],
                "handler": "handleTakeReturn" if my_return else None,
                "tributeType": tribute_state,
            }

        if str(player.finish_place or "") != required_place:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "wait",
                "cards": [],
                "handler": None,
                "tributeType": tribute_state,
            }

        if required_place in {"3", "4"}:
            tribute_card = choose_highest_tribute_card(player.hand, snapshot.current_level_rank)
            if tribute_card is None:
                return {
                    "phase": "tribute",
                    "seat": seat,
                    "action": "error",
                    "cards": [],
                    "handler": None,
                    "error": "No valid tribute card available for this hand.",
                    "tributeType": tribute_state,
                }
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "play_tribute",
                "cards": [_clean_card(tribute_card)],
                "handler": "handlePlayTribute",
                "tributeType": tribute_state,
            }

        assignments = _exception_1_assignments(trick_cards, snapshot.current_level_rank)
        if assignments is None:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "wait",
                "cards": [],
                "handler": None,
                "tributeType": tribute_state,
            }
        assigned_tribute = assignments[str(player.finish_place)]
        giver_id = assigned_tribute.get("_giverId")
        giver = snapshot.players_by_user_id.get(str(giver_id)) if giver_id is not None else None
        if giver is None:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "error",
                "cards": [],
                "handler": None,
                "error": "Could not resolve the tributer seat for exception 1.",
                "tributeType": tribute_state,
            }
        return_card = choose_return_card(
            snapshot,
            winner_seat=seat,
            giver_seat=giver.seat,
            tribute_card=_clean_card(assigned_tribute, keep_id=False),
            net=net,
        )
        if return_card is None:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "error",
                "cards": [],
                "handler": None,
                "error": "Winner has no website-valid return card (<=10 and not level rank).",
                "tributeType": tribute_state,
            }
        return {
            "phase": "tribute",
            "seat": seat,
            "action": "play_return",
            "cards": [_clean_card(return_card)],
            "handler": "handlePlayReturn",
            "tributeType": tribute_state,
            "targetGiverSeat": giver.seat,
        }

    if player.finish_place == 4 and len(trick_cards) == 0:
        tribute_card = choose_highest_tribute_card(player.hand, snapshot.current_level_rank)
        if tribute_card is None:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "error",
                "cards": [],
                "handler": None,
                "error": "No valid tribute card available for this hand.",
                "tributeType": "BASIC",
            }
        return {
            "phase": "tribute",
            "seat": seat,
            "action": "play_tribute",
            "cards": [_clean_card(tribute_card)],
            "handler": "handlePlayTribute",
            "tributeType": "BASIC",
        }

    if player.finish_place == 1 and len(trick_cards) == 1:
        tributer = next((p for p in snapshot.players if p.finish_place == 4), None)
        if tributer is None:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "wait",
                "cards": [],
                "handler": None,
                "tributeType": "BASIC",
            }
        return_card = choose_return_card(
            snapshot,
            winner_seat=seat,
            giver_seat=tributer.seat,
            tribute_card=_clean_card(trick_cards[0], keep_id=False),
            net=net,
        )
        if return_card is None:
            return {
                "phase": "tribute",
                "seat": seat,
                "action": "error",
                "cards": [],
                "handler": None,
                "error": "Winner has no website-valid return card (<=10 and not level rank).",
                "tributeType": "BASIC",
            }
        return {
            "phase": "tribute",
            "seat": seat,
            "action": "play_return",
            "cards": [_clean_card(return_card)],
            "handler": "handlePlayReturn",
            "tributeType": "BASIC",
        }

    if player.finish_place == 4 and len(trick_cards) >= 2:
        return {
            "phase": "tribute",
            "seat": seat,
            "action": "take_return",
            "cards": [_clean_card(trick_cards[1])],
            "handler": "handleTakeReturn",
            "tributeType": "BASIC",
        }

    return {
        "phase": "tribute",
        "seat": seat,
        "action": "wait",
        "cards": [],
        "handler": None,
        "tributeType": tribute_state,
    }


def encode_jsx_snapshot(
    match_record: Dict[str, Any],
    match_players: Sequence[Dict[str, Any]],
    seat: Optional[int] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = WebsiteSnapshot.from_records(match_record, match_players)
    resolved_seat = snapshot.resolve_seat(seat=seat, user_id=user_id)
    mask = snapshot.legal_mask(resolved_seat)
    return {
        "phase": snapshot.phase,
        "seat": resolved_seat,
        "seatIndex": seat_to_index(resolved_seat),
        "team": snapshot.players_by_seat[resolved_seat].team,
        "currentSeat": snapshot.current_seat,
        "levelRankLabel": snapshot.current_level_rank_label,
        "normalizedLevelRank": snapshot.current_level_rank,
        "stateVector": snapshot.encode_turn(resolved_seat).tolist(),
        "legalMask": mask.astype(bool).tolist(),
        "legalActionCount": int(mask.sum()),
        "outstandingTrick": snapshot.outstanding_trick,
        "tunnelState": snapshot.tunnel_state,
    }


def decode_model_action(
    match_record: Dict[str, Any],
    match_players: Sequence[Dict[str, Any]],
    seat: int,
    action_idx: int,
) -> Dict[str, Any]:
    snapshot = WebsiteSnapshot.from_records(match_record, match_players)
    return snapshot.decode_action(snapshot.resolve_seat(seat=seat), action_idx)


def decide_for_snapshot(
    match_record: Dict[str, Any],
    match_players: Sequence[Dict[str, Any]],
    checkpoint_path: Optional[str] = None,
    seat: Optional[int] = None,
    user_id: Optional[str] = None,
    sample: bool = False,
) -> Dict[str, Any]:
    snapshot = WebsiteSnapshot.from_records(match_record, match_players)
    resolved_seat = snapshot.resolve_seat(seat=seat, user_id=user_id)

    if snapshot.phase.lower() == "playing":
        if checkpoint_path is None:
            raise ValueError("A checkpoint is required to decide a playing-phase action.")
        if snapshot.current_seat != resolved_seat:
            return {
                "phase": "playing",
                "seat": resolved_seat,
                "action": "wait",
                "cards": [],
                "handler": None,
                "reason": f"It is seat {snapshot.current_seat}'s turn.",
            }
        net = load_policy(checkpoint_path)
        return choose_model_action(snapshot, resolved_seat, net, checkpoint_path=checkpoint_path, sample=sample)

    if snapshot.phase.lower() == "tribute":
        net = load_policy(checkpoint_path) if checkpoint_path else None
        return decide_tribute_action(snapshot, resolved_seat, net=net)

    return {
        "phase": snapshot.phase,
        "seat": resolved_seat,
        "action": "wait",
        "cards": [],
        "handler": None,
        "reason": f"No bot action is needed during gameStatus={snapshot.phase!r}.",
    }


def _load_snapshot_file(path: Optional[str]) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.load(sys.stdin)


def _extract_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    if "match" not in payload:
        raise KeyError("Snapshot JSON must contain a 'match' object.")
    players = payload.get("matchPlayers")
    if players is None:
        raise KeyError("Snapshot JSON must contain 'matchPlayers'.")
    return payload["match"], players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin PocketBase-to-arena tunnel for Guandan bots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode one website seat for the arena.")
    encode_parser.add_argument("snapshot", nargs="?", help="JSON snapshot path. Reads stdin when omitted.")
    encode_parser.add_argument("--seat", type=int, default=None, help="Website seat number (1-4).")
    encode_parser.add_argument("--user-id", type=str, default=None, help="Resolve the acting seat by user_id.")

    decide_parser = subparsers.add_parser("decide", help="Choose a bot action for the snapshot.")
    decide_parser.add_argument("snapshot", nargs="?", help="JSON snapshot path. Reads stdin when omitted.")
    decide_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path.")
    decide_parser.add_argument("--seat", type=int, default=None, help="Website seat number (1-4).")
    decide_parser.add_argument("--user-id", type=str, default=None, help="Resolve the acting seat by user_id.")
    decide_parser.add_argument("--sample", action="store_true", help="Sample from the policy instead of argmax.")

    decode_parser = subparsers.add_parser("decode", help="Decode a known arena action index.")
    decode_parser.add_argument("snapshot", nargs="?", help="JSON snapshot path. Reads stdin when omitted.")
    decode_parser.add_argument("--seat", type=int, required=True, help="Website seat number (1-4).")
    decode_parser.add_argument("--action-idx", type=int, required=True, help="Arena action index.")

    sync_parser = subparsers.add_parser("sync", help="Emit the normalized tunnelState.")
    sync_parser.add_argument("snapshot", nargs="?", help="JSON snapshot path. Reads stdin when omitted.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_snapshot_file(getattr(args, "snapshot", None))
    match, players = _extract_payload(payload)

    if args.command == "encode":
        output = encode_jsx_snapshot(match, players, seat=args.seat, user_id=args.user_id)
    elif args.command == "decide":
        output = decide_for_snapshot(
            match,
            players,
            checkpoint_path=args.checkpoint or payload.get("checkpoint"),
            seat=args.seat or payload.get("seat"),
            user_id=args.user_id,
            sample=bool(args.sample),
        )
    elif args.command == "decode":
        output = decode_model_action(match, players, seat=args.seat, action_idx=args.action_idx)
    elif args.command == "sync":
        output = sync_tunnel_state(match, players)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
