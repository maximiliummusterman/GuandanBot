"""
Standalone policy loading and transformer forward-pass helpers for the bot API.

This module intentionally keeps the checkpoint/model path separate from the
training arena so lightweight service repos do not need arena internals just
to load a checkpoint, run a forward pass, or advance transformer memory.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN = 256
HISTORY_SEQ_LEN = 64
GTRXL_BLOCK_COUNT = 1
GTRXL_HEAD_COUNT = 4
GTRXL_MEMORY_WINDOWS = 2
GTRXL_MEMORY_LEN = HISTORY_SEQ_LEN * GTRXL_MEMORY_WINDOWS
MODEL_ARCH_GTRXL = "gtrxl_v1"
MODEL_ARCH_LEGACY_MLP = "legacy_mlp_v1"
MEMORY_SNAPSHOT_DTYPE = np.float16

SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_VAL = {rank: index + 2 for index, rank in enumerate(RANKS)}
RANK_TO_INDEX = {rank: index for index, rank in enumerate(RANKS)}

TYPE_SORT_ORDER = {
    "SINGLE": 0,
    "PAIR": 1,
    "TRIPLE": 2,
    "STRAIGHT": 3,
    "FULL_HOUSE": 4,
    "SEQ_3_PAIRS": 5,
    "SEQ_2_TRIPLES": 6,
    "FOUR_OF_A_KIND": 7,
    "FIVE_OF_A_KIND": 8,
    "STRAIGHT_FLUSH": 9,
    "SIX_OF_A_KIND": 10,
    "SEVEN_OF_A_KIND": 11,
    "EIGHT_OF_A_KIND": 12,
    "FOUR_JOKERS": 13,
}
HISTORY_TYPE_NAMES = ["PASS"] + [
    trick_type
    for trick_type, _ in sorted(TYPE_SORT_ORDER.items(), key=lambda item: item[1])
]
HISTORY_TYPE_TO_INDEX = {
    trick_type: index for index, trick_type in enumerate(HISTORY_TYPE_NAMES)
}

CANONICAL = []
for _ in range(2):
    for suit in SUITS:
        for rank in RANKS:
            CANONICAL.append(("normal", suit, rank))
CANONICAL.append(("joker", "red", None))
CANONICAL.append(("joker", "red", None))
CANONICAL.append(("joker", "black", None))
CANONICAL.append(("joker", "black", None))
CANON_LEN = len(CANONICAL)
CARD_SIGNATURE_TO_CANON_INDEX: Dict[tuple, int] = {}
for index, slot in enumerate(CANONICAL):
    CARD_SIGNATURE_TO_CANON_INDEX.setdefault(slot, index)

STRAIGHT_FLUSH_FEATURE_DIM = len(SUITS) * (1 + (len(RANKS) - 4))
STATE_CONTEXT_DIM = (
    108 + 108 + 108 + 15 + 4 + 4 + 2 + 2 + 1 + 4 + 2 + 1 + 1
    + STRAIGHT_FLUSH_FEATURE_DIM
)
HISTORY_ENTRY_DIM = 4 + len(HISTORY_TYPE_NAMES) + 2 + CANON_LEN
STATE_HISTORY_DIM = HISTORY_SEQ_LEN * HISTORY_ENTRY_DIM
STATE_DIM = STATE_CONTEXT_DIM + STATE_HISTORY_DIM

STATE_HAND_SLICE = slice(0, CANON_LEN)
STATE_TRICK_SLICE = slice(STATE_HAND_SLICE.stop, STATE_HAND_SLICE.stop + CANON_LEN)
STATE_PLAYED_SLICE = slice(STATE_TRICK_SLICE.stop, STATE_TRICK_SLICE.stop + CANON_LEN)
STATE_LEVEL_RANK_SLICE = slice(STATE_PLAYED_SLICE.stop, STATE_PLAYED_SLICE.stop + 15)
STATE_MY_SEAT_SLICE = slice(STATE_LEVEL_RANK_SLICE.stop, STATE_LEVEL_RANK_SLICE.stop + 4)
STATE_HAND_SIZES_SLICE = slice(STATE_MY_SEAT_SLICE.stop, STATE_MY_SEAT_SLICE.stop + 4)
STATE_TEAM_LEVELS_SLICE = slice(STATE_HAND_SIZES_SLICE.stop, STATE_HAND_SIZES_SLICE.stop + 2)
STATE_A_TRIES_SLICE = slice(STATE_TEAM_LEVELS_SLICE.stop, STATE_TEAM_LEVELS_SLICE.stop + 2)
STATE_IS_OPENER_SLICE = slice(STATE_A_TRIES_SLICE.stop, STATE_A_TRIES_SLICE.stop + 1)
STATE_TRICK_SEAT_SLICE = slice(STATE_IS_OPENER_SLICE.stop, STATE_IS_OPENER_SLICE.stop + 4)
STATE_CALLER_SLICE = slice(STATE_TRICK_SEAT_SLICE.stop, STATE_TRICK_SEAT_SLICE.stop + 2)
STATE_PASS_STREAK_SLICE = slice(STATE_CALLER_SLICE.stop, STATE_CALLER_SLICE.stop + 1)
STATE_ROUND_NUM_SLICE = slice(STATE_PASS_STREAK_SLICE.stop, STATE_PASS_STREAK_SLICE.stop + 1)
STATE_SF_SLICE = slice(STATE_ROUND_NUM_SLICE.stop, STATE_ROUND_NUM_SLICE.stop + STRAIGHT_FLUSH_FEATURE_DIM)
STATE_HISTORY_SLICE = slice(STATE_SF_SLICE.stop, STATE_SF_SLICE.stop + STATE_HISTORY_DIM)

HISTORY_ENTRY_SEAT_SLICE = slice(0, 4)
HISTORY_ENTRY_TYPE_SLICE = slice(HISTORY_ENTRY_SEAT_SLICE.stop, HISTORY_ENTRY_SEAT_SLICE.stop + len(HISTORY_TYPE_NAMES))
HISTORY_ENTRY_STRENGTH_SLICE = slice(HISTORY_ENTRY_TYPE_SLICE.stop, HISTORY_ENTRY_TYPE_SLICE.stop + 2)
HISTORY_ENTRY_CARDS_SLICE = slice(HISTORY_ENTRY_STRENGTH_SLICE.stop, HISTORY_ENTRY_STRENGTH_SLICE.stop + CANON_LEN)

MAX_A_TRIES_FEATURE = 2.0
MAX_PASS_STREAK_FEATURE = 3.0
MAX_ROUND_NUM_FEATURE = 20.0
MAX_HISTORY_STRENGTH = 1000.0

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LEGACY_CHECKPOINT_DIR = PROJECT_ROOT / "legacy_checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = (
    torch.bfloat16
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
    else (torch.float16 if DEVICE.type == "cuda" else None)
)

CHECKPOINT_ROUNDS_MATCHES_RE = re.compile(r"rounds(\d+)_matches(\d+)")
CHECKPOINT_EPISODE_RE = re.compile(r"ep(\d+)")


def _autocast_context(device_type: Optional[str] = None):
    active_device_type = device_type or DEVICE.type
    if AMP_DTYPE is None or active_device_type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=active_device_type, dtype=AMP_DTYPE)


def extract_checkpoint_round_match_counts(path_like) -> Tuple[Optional[int], Optional[int]]:
    stem = Path(path_like).stem if not isinstance(path_like, Path) else path_like.stem
    match = CHECKPOINT_ROUNDS_MATCHES_RE.search(stem)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def extract_checkpoint_episode_num(path_like) -> Optional[int]:
    stem = Path(path_like).stem if not isinstance(path_like, Path) else path_like.stem
    match = CHECKPOINT_EPISODE_RE.search(stem)
    return int(match.group(1)) if match else None


def checkpoint_progress_sort_key(path: Path) -> Tuple[int, int, int, str]:
    rounds, matches = extract_checkpoint_round_match_counts(path)
    if rounds is not None and matches is not None:
        return (2, rounds, matches, path.name)

    episode = extract_checkpoint_episode_num(path)
    if episode is not None:
        return (1, episode, 0, path.name)

    return (0, 0, 0, path.name)


def checkpoint_path_candidates(path_like) -> List[Path]:
    raw_path = Path(path_like).expanduser()
    candidates: List[Path] = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(raw_path)
    if not raw_path.is_absolute():
        add_candidate(PROJECT_ROOT / raw_path)
        if len(raw_path.parts) == 1:
            add_candidate(DEFAULT_CHECKPOINT_DIR / raw_path.name)
            add_candidate(LEGACY_CHECKPOINT_DIR / raw_path.name)

    return candidates


@lru_cache(maxsize=512)
def _resolve_checkpoint_file_cached(path_key: str) -> str:
    candidates = checkpoint_path_candidates(path_key)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    tried_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found: {path_key}. Tried: {tried_paths}")


def resolve_checkpoint_file(path_like) -> Path:
    path_key = str(Path(path_like).expanduser())
    return Path(_resolve_checkpoint_file_cached(path_key))


def list_checkpoint_paths(checkpoint_dir: Path) -> List[Path]:
    checkpoint_paths = list(checkpoint_dir.glob("*.pt"))
    parseable_paths = [
        path for path in checkpoint_paths if checkpoint_progress_sort_key(path)[0] > 0
    ]
    return sorted(parseable_paths or checkpoint_paths, key=checkpoint_progress_sort_key)


def make_card(suit: str, rank: str) -> dict:
    return {"suit": suit, "rank": rank}


def make_joker(color: str) -> dict:
    return {"joker": color}


def is_joker(card: dict) -> bool:
    return "joker" in card


def card_rank_val(card: dict, level_rank: Optional[str] = None) -> int:
    if is_joker(card):
        return 18 if card["joker"] == "red" else 17
    rank = card["rank"]
    if level_rank and rank == level_rank:
        if card.get("suit") == "hearts":
            return 15
        return 16
    return RANK_VAL.get(rank, 0)


def is_wildcard(card: dict, level_rank: Optional[str]) -> bool:
    return (
        not is_joker(card)
        and level_rank is not None
        and card.get("rank") == level_rank
        and card.get("suit") == "hearts"
    )


def _card_signature(card: dict) -> tuple:
    if "joker" in card:
        return ("joker", card["joker"], None)
    return ("normal", card["suit"], card["rank"])


def _fill_multihot(cards: list, out: np.ndarray) -> np.ndarray:
    out.fill(0.0)
    for card in cards:
        index = CARD_SIGNATURE_TO_CANON_INDEX[_card_signature(card)]
        next_value = out[index] + 1.0
        out[index] = 2.0 if next_value > 2.0 else next_value
    return out


def hand_to_multihot(cards: list) -> np.ndarray:
    vec = np.zeros(CANON_LEN, dtype=np.float32)
    return _fill_multihot(cards, vec)


def count_ranks(cards: list) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for card in cards:
        if not is_joker(card):
            counts[card["rank"]] += 1
    return counts


def _check_straight_ranks(rank_list: List[str]) -> Optional[int]:
    if len(rank_list) != 5 or len(set(rank_list)) != 5:
        return None
    ace_low_choices = [True, False] if "A" in rank_list else [False]
    for ace_low in ace_low_choices:
        def rank_value(rank: str) -> int:
            return 1 if rank == "A" and ace_low else RANK_VAL.get(rank, 0)

        values = sorted(rank_value(rank) for rank in rank_list)
        if values[-1] - values[0] == 4 and len(set(values)) == 5:
            if ace_low and values[-1] == 14:
                continue
            if not ace_low and values[0] == 1:
                continue
            return values[-1]
    return None


def _check_rank_straight_n(rank_list: List[str], count: int) -> Optional[int]:
    if len(rank_list) != count:
        return None
    ace_low_choices = [True, False] if "A" in rank_list else [False]
    for ace_low in ace_low_choices:
        def rank_value(rank: str) -> int:
            return 1 if rank == "A" and ace_low else RANK_VAL.get(rank, 0)

        values = sorted(rank_value(rank) for rank in rank_list)
        if all(values[index] + 1 == values[index + 1] for index in range(len(values) - 1)):
            if ace_low and values[-1] == 14:
                continue
            if not ace_low and values[0] == 1:
                continue
            return values[-1]
    return None


_CATALOGUE: List[dict] = []
ACTION_DIM = 0
CAT_CARD_COUNTS: np.ndarray
CAT_INVALID_WHEN_LEVEL_IS_2: np.ndarray
LR_PLACEHOLDER_SUITS = ["spades", "clubs", "diamonds"]
_SF_WINDOWS = [["A", "2", "3", "4", "5"]] + [
    RANKS[index:index + 5] for index in range(len(RANKS) - 4)
]


def _cards_multihot(cards: list) -> np.ndarray:
    return hand_to_multihot(cards)


def _make_lr_placeholder_cards(count: int) -> list:
    cards = []
    full_cycles, remainder = divmod(count, len(LR_PLACEHOLDER_SUITS))
    for _ in range(full_cycles):
        for suit in LR_PLACEHOLDER_SUITS:
            cards.append(make_card(suit, "2"))
    for suit in LR_PLACEHOLDER_SUITS[:remainder]:
        cards.append(make_card(suit, "2"))
    return cards


def _add(entry: dict, cards: list, forced_wildcards: int = 0) -> None:
    stored = dict(entry)
    stored["vec_template"] = _cards_multihot(cards)
    stored["forced_wildcards"] = forced_wildcards
    _CATALOGUE.append(stored)


def _build_catalogue() -> None:
    global ACTION_DIM, CAT_CARD_COUNTS, CAT_INVALID_WHEN_LEVEL_IS_2

    _CATALOGUE.clear()
    _CATALOGUE.append(
        {
            "type": "PASS",
            "rank_key": None,
            "pair_rank_key": None,
            "strength": 0,
            "is_bomb": False,
            "bomb_strength": 0,
            "vec_template": np.zeros(CANON_LEN, dtype=np.float32),
            "forced_wildcards": 0,
        }
    )

    red_joker = make_joker("red")
    black_joker = make_joker("black")

    for rank in RANKS:
        strength = RANK_VAL[rank]
        _add(
            {
                "type": "SINGLE",
                "rank_key": rank,
                "pair_rank_key": None,
                "strength": strength,
                "is_bomb": False,
                "bomb_strength": 0,
            },
            [make_card("spades", rank)],
        )
    lr_single = _make_lr_placeholder_cards(1)
    _add(
        {
            "type": "SINGLE",
            "rank_key": "LR",
            "pair_rank_key": None,
            "strength": 16,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        lr_single,
    )
    _add(
        {
            "type": "SINGLE",
            "rank_key": "BJ",
            "pair_rank_key": None,
            "strength": 17,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        [black_joker],
    )
    _add(
        {
            "type": "SINGLE",
            "rank_key": "RJ",
            "pair_rank_key": None,
            "strength": 18,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        [red_joker],
    )

    for rank in RANKS:
        strength = RANK_VAL[rank]
        _add(
            {
                "type": "PAIR",
                "rank_key": rank,
                "pair_rank_key": None,
                "strength": strength,
                "is_bomb": False,
                "bomb_strength": 0,
            },
            [make_card("spades", rank), make_card("hearts", rank)],
        )
    lr_pair = _make_lr_placeholder_cards(2)
    _add(
        {
            "type": "PAIR",
            "rank_key": "LR",
            "pair_rank_key": None,
            "strength": 16,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        lr_pair,
    )
    _add(
        {
            "type": "PAIR",
            "rank_key": "BJ",
            "pair_rank_key": None,
            "strength": 17,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        [black_joker, black_joker],
    )
    _add(
        {
            "type": "PAIR",
            "rank_key": "RJ",
            "pair_rank_key": None,
            "strength": 18,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        [red_joker, red_joker],
    )

    for rank in RANKS:
        strength = RANK_VAL[rank]
        _add(
            {
                "type": "TRIPLE",
                "rank_key": rank,
                "pair_rank_key": None,
                "strength": strength,
                "is_bomb": False,
                "bomb_strength": 0,
            },
            [make_card(suit, rank) for suit in SUITS[:3]],
        )
    lr_triple = _make_lr_placeholder_cards(3)
    _add(
        {
            "type": "TRIPLE",
            "rank_key": "LR",
            "pair_rank_key": None,
            "strength": 16,
            "is_bomb": False,
            "bomb_strength": 0,
        },
        lr_triple,
    )

    straight_windows = [(["A", "2", "3", "4", "5"], 5)]
    for start in range(len(RANKS) - 4):
        window = RANKS[start:start + 5]
        straight_windows.append((window, RANK_VAL[window[-1]]))
    for window, strength in straight_windows:
        _add(
            {
                "type": "STRAIGHT",
                "rank_key": window[-1] if strength != 5 else "5",
                "pair_rank_key": None,
                "strength": strength,
                "is_bomb": False,
                "bomb_strength": 0,
            },
            [make_card("spades", rank) for rank in window],
        )

    all_rank_keys = RANKS + ["LR"]
    for triple_rank in all_rank_keys:
        for pair_rank in all_rank_keys:
            if triple_rank == pair_rank:
                continue
            triple_cards = (
                _make_lr_placeholder_cards(3)
                if triple_rank == "LR"
                else [make_card(suit, triple_rank) for suit in SUITS[:3]]
            )
            pair_cards = (
                _make_lr_placeholder_cards(2)
                if pair_rank == "LR"
                else [make_card(suit, pair_rank) for suit in ["spades", "hearts"]]
            )
            strength = 16 if triple_rank == "LR" else RANK_VAL[triple_rank]
            _add(
                {
                    "type": "FULL_HOUSE",
                    "rank_key": triple_rank,
                    "pair_rank_key": pair_rank,
                    "strength": strength,
                    "is_bomb": False,
                    "bomb_strength": 0,
                },
                triple_cards + pair_cards,
            )

    for start in range(len(RANKS) - 2):
        window = RANKS[start:start + 3]
        strength = RANK_VAL[window[-1]]
        cards = []
        for rank in window:
            cards.extend([make_card("spades", rank), make_card("hearts", rank)])
        _add(
            {
                "type": "SEQ_3_PAIRS",
                "rank_key": window[-1],
                "pair_rank_key": None,
                "strength": strength,
                "is_bomb": False,
                "bomb_strength": 0,
            },
            cards,
        )

    for start in range(len(RANKS) - 1):
        window = RANKS[start:start + 2]
        strength = RANK_VAL[window[-1]]
        cards = []
        for rank in window:
            cards.extend([make_card(suit, rank) for suit in SUITS[:3]])
        _add(
            {
                "type": "SEQ_2_TRIPLES",
                "rank_key": window[-1],
                "pair_rank_key": None,
                "strength": strength,
                "is_bomb": False,
                "bomb_strength": 0,
            },
            cards,
        )

    bomb_types = {
        4: "FOUR_OF_A_KIND",
        5: "FIVE_OF_A_KIND",
        6: "SIX_OF_A_KIND",
        7: "SEVEN_OF_A_KIND",
        8: "EIGHT_OF_A_KIND",
    }
    for rank in RANKS:
        strength = RANK_VAL[rank]
        all_copies = [make_card(suit, rank) for suit in SUITS] * 2
        for size in range(4, 9):
            bomb_strength = size * 100 + strength
            _add(
                {
                    "type": bomb_types[size],
                    "rank_key": rank,
                    "pair_rank_key": None,
                    "strength": strength,
                    "is_bomb": True,
                    "bomb_strength": bomb_strength,
                },
                all_copies[:size],
            )

    for size in range(4, 9):
        bomb_strength = size * 100 + 16
        natural_copies = _make_lr_placeholder_cards(min(size, 6))
        _add(
            {
                "type": bomb_types[size],
                "rank_key": "LR",
                "pair_rank_key": None,
                "strength": 16,
                "is_bomb": True,
                "bomb_strength": bomb_strength,
            },
            natural_copies,
            forced_wildcards=max(0, size - len(natural_copies)),
        )

    straight_flush_windows = [(["A", "2", "3", "4", "5"], 5)]
    for start in range(len(RANKS) - 4):
        window = RANKS[start:start + 5]
        straight_flush_windows.append((window, RANK_VAL[window[-1]]))
    for suit in SUITS:
        for window, strength in straight_flush_windows:
            _add(
                {
                    "type": "STRAIGHT_FLUSH",
                    "rank_key": window[-1] if strength != 5 else "5",
                    "pair_rank_key": None,
                    "strength": strength,
                    "is_bomb": True,
                    "bomb_strength": 550 + strength,
                },
                [make_card(suit, rank) for rank in window],
            )

    _add(
        {
            "type": "FOUR_JOKERS",
            "rank_key": None,
            "pair_rank_key": None,
            "strength": 1000,
            "is_bomb": True,
            "bomb_strength": 1000,
        },
        [red_joker, red_joker, black_joker, black_joker],
    )

    ACTION_DIM = len(_CATALOGUE)
    CAT_CARD_COUNTS = np.array(
        [
            int(entry["vec_template"].sum()) + int(entry.get("forced_wildcards", 0))
            for entry in _CATALOGUE
        ],
        dtype=np.int32,
    )
    CAT_INVALID_WHEN_LEVEL_IS_2 = np.array(
        [
            (
                entry["type"] in {
                    "SINGLE",
                    "PAIR",
                    "TRIPLE",
                    "FULL_HOUSE",
                    "FOUR_OF_A_KIND",
                    "FIVE_OF_A_KIND",
                    "SIX_OF_A_KIND",
                    "SEVEN_OF_A_KIND",
                    "EIGHT_OF_A_KIND",
                }
                and (
                    entry.get("rank_key") == "2"
                    or entry.get("pair_rank_key") == "2"
                )
            )
            for entry in _CATALOGUE
        ],
        dtype=bool,
    )


def _build_hand_context(hand: list, level_rank: Optional[str]) -> dict:
    plain_ranks = defaultdict(list)
    plain_rank_counts = defaultdict(int)
    seq_ranks = defaultdict(list)
    seq_rank_counts = defaultdict(int)
    suit_ranks = defaultdict(lambda: defaultdict(list))
    suit_rank_counts = defaultdict(lambda: defaultdict(int))
    lr_cards = []
    black_jokers = []
    red_jokers = []
    wildcards = []

    for card in hand:
        if is_joker(card):
            (red_jokers if card["joker"] == "red" else black_jokers).append(card)
            continue
        if is_wildcard(card, level_rank):
            wildcards.append(card)
            continue
        seq_ranks[card["rank"]].append(card)
        seq_rank_counts[card["rank"]] += 1
        suit_ranks[card["suit"]][card["rank"]].append(card)
        suit_rank_counts[card["suit"]][card["rank"]] += 1
        if level_rank is not None and card["rank"] == level_rank and card["suit"] != "hearts":
            lr_cards.append(card)
        else:
            plain_ranks[card["rank"]].append(card)
            plain_rank_counts[card["rank"]] += 1

    return {
        "plain_ranks": plain_ranks,
        "plain_rank_counts": plain_rank_counts,
        "seq_ranks": seq_ranks,
        "seq_rank_counts": seq_rank_counts,
        "suit_ranks": suit_ranks,
        "suit_rank_counts": suit_rank_counts,
        "lr_cards": lr_cards,
        "lr_count": len(lr_cards),
        "black_jokers": black_jokers,
        "black_joker_count": len(black_jokers),
        "red_jokers": red_jokers,
        "red_joker_count": len(red_jokers),
        "wildcards": wildcards,
        "wildcard_count": len(wildcards),
    }


def _entry_window(entry: dict) -> list:
    cached = entry.get("window")
    if cached is not None:
        return cached
    if entry["type"] in {"STRAIGHT", "STRAIGHT_FLUSH"}:
        if entry["strength"] == 5:
            window = ["A", "2", "3", "4", "5"]
        else:
            high_index = RANK_TO_INDEX[entry["rank_key"]]
            window = RANKS[high_index - 4:high_index + 1]
    elif entry["type"] == "SEQ_3_PAIRS":
        high_index = RANK_TO_INDEX[entry["rank_key"]]
        window = RANKS[high_index - 2:high_index + 1]
    elif entry["type"] == "SEQ_2_TRIPLES":
        high_index = RANK_TO_INDEX[entry["rank_key"]]
        window = RANKS[high_index - 1:high_index + 1]
    else:
        window = []
    entry["window"] = window
    return window


def _entry_straight_flush_suit(entry: dict) -> Optional[str]:
    cached = entry.get("straight_flush_suit")
    if cached is not None:
        return cached
    suit = None
    for slot_index in np.where(entry["vec_template"] > 0)[0]:
        slot = CANONICAL[slot_index]
        if slot[0] == "normal":
            suit = slot[1]
            break
    entry["straight_flush_suit"] = suit
    return suit


def _entry_cards_from_context(entry: dict, ctx: dict) -> list:
    wildcards = list(ctx["wildcards"])
    result = []

    def take_from(pool: list, need: int) -> bool:
        take = min(len(pool), need)
        result.extend(pool[:take])
        missing = need - take
        if missing > len(wildcards):
            return False
        result.extend(wildcards[:missing])
        del wildcards[:missing]
        return True

    if entry["type"] == "PASS":
        return []

    if entry["type"] == "SINGLE":
        if entry["rank_key"] == "BJ":
            return result if take_from(ctx["black_jokers"], 1) else []
        if entry["rank_key"] == "RJ":
            return result if take_from(ctx["red_jokers"], 1) else []
        if entry["rank_key"] == "LR":
            return result if take_from(ctx["lr_cards"], 1) else []
        return result if take_from(ctx["plain_ranks"][entry["rank_key"]], 1) else []

    if entry["type"] == "PAIR":
        if entry["rank_key"] == "BJ":
            return result if take_from(ctx["black_jokers"], 2) else []
        if entry["rank_key"] == "RJ":
            return result if take_from(ctx["red_jokers"], 2) else []
        if entry["rank_key"] == "LR":
            return result if take_from(ctx["lr_cards"], 2) else []
        return result if take_from(ctx["plain_ranks"][entry["rank_key"]], 2) else []

    if entry["type"] == "TRIPLE":
        pool = ctx["lr_cards"] if entry["rank_key"] == "LR" else ctx["plain_ranks"][entry["rank_key"]]
        return result if take_from(pool, 3) else []

    if entry["type"] == "FULL_HOUSE":
        triple_pool = ctx["lr_cards"] if entry["rank_key"] == "LR" else ctx["plain_ranks"][entry["rank_key"]]
        pair_key = entry["pair_rank_key"]
        pair_pool = ctx["lr_cards"] if pair_key == "LR" else ctx["plain_ranks"][pair_key]
        if entry["rank_key"] == pair_key:
            return []
        if not take_from(triple_pool, 3):
            return []
        if not take_from(pair_pool, 2):
            return []
        return result

    if entry["type"] == "STRAIGHT":
        for rank in _entry_window(entry):
            if not take_from(ctx["seq_ranks"][rank], 1):
                return []
        return result

    if entry["type"] == "SEQ_3_PAIRS":
        for rank in _entry_window(entry):
            if not take_from(ctx["seq_ranks"][rank], 2):
                return []
        return result

    if entry["type"] == "SEQ_2_TRIPLES":
        for rank in _entry_window(entry):
            if not take_from(ctx["seq_ranks"][rank], 3):
                return []
        return result

    if entry["type"] in {
        "FOUR_OF_A_KIND",
        "FIVE_OF_A_KIND",
        "SIX_OF_A_KIND",
        "SEVEN_OF_A_KIND",
        "EIGHT_OF_A_KIND",
    }:
        need = {
            "FOUR_OF_A_KIND": 4,
            "FIVE_OF_A_KIND": 5,
            "SIX_OF_A_KIND": 6,
            "SEVEN_OF_A_KIND": 7,
            "EIGHT_OF_A_KIND": 8,
        }[entry["type"]]
        pool = ctx["lr_cards"] if entry["rank_key"] == "LR" else ctx["plain_ranks"][entry["rank_key"]]
        return result if take_from(pool, need) else []

    if entry["type"] == "STRAIGHT_FLUSH":
        suit = _entry_straight_flush_suit(entry)
        for rank in _entry_window(entry):
            if not take_from(ctx["suit_ranks"][suit][rank], 1):
                return []
        return result

    if entry["type"] == "FOUR_JOKERS":
        if not take_from(ctx["red_jokers"], 2):
            return []
        if not take_from(ctx["black_jokers"], 2):
            return []
        return result

    return []


def _entry_reachable_from_context(entry: dict, ctx: dict) -> bool:
    if entry["type"] == "PASS":
        return True

    plain_rank_counts = ctx.get("plain_rank_counts")
    seq_rank_counts = ctx.get("seq_rank_counts")
    suit_rank_counts = ctx.get("suit_rank_counts")
    wildcards_left = int(ctx.get("wildcard_count", len(ctx["wildcards"])))

    def consume(available: int, need: int) -> bool:
        nonlocal wildcards_left
        missing = need - int(available)
        if missing <= 0:
            return True
        if missing > wildcards_left:
            return False
        wildcards_left -= missing
        return True

    def plain_count(rank_key: str) -> int:
        if rank_key == "LR":
            return int(ctx.get("lr_count", len(ctx["lr_cards"])))
        if rank_key == "BJ":
            return int(ctx.get("black_joker_count", len(ctx["black_jokers"])))
        if rank_key == "RJ":
            return int(ctx.get("red_joker_count", len(ctx["red_jokers"])))
        if plain_rank_counts is not None:
            return int(plain_rank_counts.get(rank_key, 0))
        return len(ctx["plain_ranks"][rank_key])

    def seq_count(rank: str) -> int:
        if seq_rank_counts is not None:
            return int(seq_rank_counts.get(rank, 0))
        return len(ctx["seq_ranks"][rank])

    def suited_count(suit: Optional[str], rank: str) -> int:
        if suit is None:
            return 0
        if suit_rank_counts is not None:
            return int(suit_rank_counts[suit].get(rank, 0))
        return len(ctx["suit_ranks"][suit][rank])

    if entry["type"] == "SINGLE":
        return consume(plain_count(entry["rank_key"]), 1)
    if entry["type"] == "PAIR":
        return consume(plain_count(entry["rank_key"]), 2)
    if entry["type"] == "TRIPLE":
        return consume(plain_count(entry["rank_key"]), 3)
    if entry["type"] == "FULL_HOUSE":
        if entry["rank_key"] == entry["pair_rank_key"]:
            return False
        return (
            consume(plain_count(entry["rank_key"]), 3)
            and consume(plain_count(entry["pair_rank_key"]), 2)
        )
    if entry["type"] == "STRAIGHT":
        return all(consume(seq_count(rank), 1) for rank in _entry_window(entry))
    if entry["type"] == "SEQ_3_PAIRS":
        return all(consume(seq_count(rank), 2) for rank in _entry_window(entry))
    if entry["type"] == "SEQ_2_TRIPLES":
        return all(consume(seq_count(rank), 3) for rank in _entry_window(entry))
    if entry["type"] in {
        "FOUR_OF_A_KIND",
        "FIVE_OF_A_KIND",
        "SIX_OF_A_KIND",
        "SEVEN_OF_A_KIND",
        "EIGHT_OF_A_KIND",
    }:
        need = {
            "FOUR_OF_A_KIND": 4,
            "FIVE_OF_A_KIND": 5,
            "SIX_OF_A_KIND": 6,
            "SEVEN_OF_A_KIND": 7,
            "EIGHT_OF_A_KIND": 8,
        }[entry["type"]]
        return consume(plain_count(entry["rank_key"]), need)
    if entry["type"] == "STRAIGHT_FLUSH":
        suit = _entry_straight_flush_suit(entry)
        return all(consume(suited_count(suit, rank), 1) for rank in _entry_window(entry))
    if entry["type"] == "FOUR_JOKERS":
        return consume(plain_count("RJ"), 2) and consume(plain_count("BJ"), 2)
    return False


def _entry_runtime_info(entry: dict) -> dict:
    cached = entry.get("runtime_info")
    if cached is not None:
        return cached
    info = {
        "type": entry["type"],
        "strength": entry["strength"],
        "is_bomb": entry["is_bomb"],
        "bomb_strength": entry["bomb_strength"],
    }
    if entry["rank_key"] == "LR":
        info["strength"] = 16
        if entry["is_bomb"]:
            size = {
                "FOUR_OF_A_KIND": 4,
                "FIVE_OF_A_KIND": 5,
                "SIX_OF_A_KIND": 6,
                "SEVEN_OF_A_KIND": 7,
                "EIGHT_OF_A_KIND": 8,
            }.get(entry["type"])
            if size is not None:
                info["bomb_strength"] = size * 100 + 16
    entry["runtime_info"] = info
    return info


def _entry_beats_last(entry: dict, last_info: Optional[dict], level_rank: Optional[str]) -> bool:
    if last_info is None or last_info["type"] == "INVALID":
        return True
    if entry["type"] == "PASS":
        return False

    current = _entry_runtime_info(entry)
    entry_strength = current["strength"]
    entry_bomb_strength = current["bomb_strength"]
    entry_is_bomb = current["is_bomb"]
    last_is_bomb = last_info["is_bomb"]

    if entry_is_bomb and not last_is_bomb:
        return True
    if not entry_is_bomb and last_is_bomb:
        return False
    if entry_is_bomb and last_is_bomb:
        return entry_bomb_strength > last_info["bomb_strength"]
    if entry["type"] != last_info["type"]:
        return False
    return entry_strength > last_info["strength"]


def compute_legal_mask_from_context(
    ctx: dict,
    trick_last: list,
    level_rank: Optional[str],
    is_opener: bool,
    last_info: Optional[dict] = None,
) -> np.ndarray:
    reachable = np.zeros(ACTION_DIM, dtype=bool)
    for index, entry in enumerate(_CATALOGUE):
        if level_rank == "2" and CAT_INVALID_WHEN_LEVEL_IS_2[index]:
            continue
        reachable[index] = _entry_reachable_from_context(entry, ctx)

    if is_opener:
        mask = reachable.copy()
        mask[0] = False
        return mask

    if last_info is None:
        last_info = _detect_from_hand(trick_last, level_rank) if trick_last else None
    beat_mask = np.zeros(ACTION_DIM, dtype=bool)
    beat_mask[0] = True
    for index in np.where(reachable)[0]:
        if index == 0:
            continue
        if _entry_beats_last(_CATALOGUE[int(index)], last_info, level_rank):
            beat_mask[int(index)] = True
    return beat_mask


def compute_legal_mask(
    hand: list,
    trick_last: list,
    level_rank: Optional[str],
    is_opener: bool,
    last_info: Optional[dict] = None,
) -> np.ndarray:
    return compute_legal_mask_from_context(
        _build_hand_context(hand, level_rank),
        trick_last,
        level_rank,
        is_opener,
        last_info=last_info,
    )


def _detect_from_hand(cards: list, level_rank: Optional[str]) -> Optional[dict]:
    if not cards:
        return None
    target_vec = hand_to_multihot(cards)
    ctx = _build_hand_context(cards, level_rank)
    candidate_indices = np.where(CAT_CARD_COUNTS == len(cards))[0]
    for index in candidate_indices:
        if level_rank == "2" and CAT_INVALID_WHEN_LEVEL_IS_2[int(index)]:
            continue
        played = _entry_cards_from_context(_CATALOGUE[int(index)], ctx)
        if len(played) != len(cards):
            continue
        if np.array_equal(hand_to_multihot(played), target_vec):
            return _entry_runtime_info(_CATALOGUE[int(index)])
    return _detect_combo(cards, level_rank)


def _detect_combo(cards: list, level_rank: Optional[str]) -> dict:
    card_count = len(cards)
    jokers = [card for card in cards if is_joker(card)]
    wildcards = [card for card in cards if is_wildcard(card, level_rank)]
    normal_cards = [
        card for card in cards if not is_joker(card) and not is_wildcard(card, level_rank)
    ]

    if card_count == 4 and len(jokers) == 4:
        red_count = sum(1 for card in jokers if card["joker"] == "red")
        black_count = sum(1 for card in jokers if card["joker"] == "black")
        if red_count == 2 and black_count == 2:
            return {
                "type": "FOUR_JOKERS",
                "strength": 1000,
                "is_bomb": True,
                "bomb_strength": 1000,
            }

    if 4 <= card_count <= 8 and not jokers:
        rank_counts = count_ranks(normal_cards)
        if len(rank_counts) == 1:
            rank = next(iter(rank_counts))
            if level_rank and rank == level_rank:
                bomb_strength = card_count * 100 + 16
                strength = 16
            else:
                bomb_strength = card_count * 100 + RANK_VAL.get(rank, 0)
                strength = RANK_VAL.get(rank, 0)
            bomb_type = {
                4: "FOUR_OF_A_KIND",
                5: "FIVE_OF_A_KIND",
                6: "SIX_OF_A_KIND",
                7: "SEVEN_OF_A_KIND",
                8: "EIGHT_OF_A_KIND",
            }
            return {
                "type": bomb_type[card_count],
                "strength": strength,
                "is_bomb": True,
                "bomb_strength": bomb_strength,
            }

    if card_count == 5 and not jokers:
        if len({card["suit"] for card in normal_cards}) == 1 and not wildcards:
            straight_strength = _check_straight_ranks([card["rank"] for card in cards])
            if straight_strength is not None:
                return {
                    "type": "STRAIGHT_FLUSH",
                    "strength": straight_strength,
                    "is_bomb": True,
                    "bomb_strength": 550 + straight_strength,
                }

    if card_count == 1:
        strength = (
            18 if cards[0]["joker"] == "red"
            else 17
            if is_joker(cards[0])
            else card_rank_val(cards[0], level_rank)
        )
        return {"type": "SINGLE", "strength": strength, "is_bomb": False, "bomb_strength": 0}

    if card_count == 2:
        red_jokers = [card for card in cards if is_joker(card) and card["joker"] == "red"]
        black_jokers = [card for card in cards if is_joker(card) and card["joker"] == "black"]
        if len(red_jokers) == 2:
            return {"type": "PAIR", "strength": 18, "is_bomb": False, "bomb_strength": 0}
        if len(black_jokers) == 2:
            return {"type": "PAIR", "strength": 17, "is_bomb": False, "bomb_strength": 0}
        if len(normal_cards) == 2 and normal_cards[0]["rank"] == normal_cards[1]["rank"]:
            return {
                "type": "PAIR",
                "strength": card_rank_val(normal_cards[0], level_rank),
                "is_bomb": False,
                "bomb_strength": 0,
            }
        if len(normal_cards) == 1 and len(wildcards) == 1:
            return {
                "type": "PAIR",
                "strength": card_rank_val(normal_cards[0], level_rank),
                "is_bomb": False,
                "bomb_strength": 0,
            }

    if card_count == 3 and not jokers:
        rank_counts = count_ranks(normal_cards)
        if len(rank_counts) == 1:
            return {
                "type": "TRIPLE",
                "strength": card_rank_val(normal_cards[0], level_rank),
                "is_bomb": False,
                "bomb_strength": 0,
            }

    if card_count == 5:
        all_ranks = [card["rank"] for card in normal_cards + wildcards]
        straight_strength = _check_straight_ranks(all_ranks)
        if straight_strength is not None:
            return {
                "type": "STRAIGHT",
                "strength": straight_strength,
                "is_bomb": False,
                "bomb_strength": 0,
            }

    if card_count == 5 and not jokers:
        rank_counts = count_ranks(normal_cards + wildcards)
        counts = sorted(rank_counts.values(), reverse=True)
        if counts[:2] == [3, 2] and len(rank_counts) == 2:
            triple_rank = next(rank for rank, count in rank_counts.items() if count == 3)
            return {
                "type": "FULL_HOUSE",
                "strength": card_rank_val(make_card("spades", triple_rank), level_rank),
                "is_bomb": False,
                "bomb_strength": 0,
            }

    if card_count == 6 and not jokers:
        rank_counts = count_ranks(normal_cards)
        pairs = [rank for rank, count in rank_counts.items() if count == 2]
        if len(pairs) == 3:
            straight_strength = _check_rank_straight_n(pairs, 3)
            if straight_strength is not None:
                return {
                    "type": "SEQ_3_PAIRS",
                    "strength": straight_strength,
                    "is_bomb": False,
                    "bomb_strength": 0,
                }

    if card_count == 6 and not jokers:
        rank_counts = count_ranks(normal_cards)
        triples = [rank for rank, count in rank_counts.items() if count == 3]
        if len(triples) == 2:
            straight_strength = _check_rank_straight_n(triples, 2)
            if straight_strength is not None:
                return {
                    "type": "SEQ_2_TRIPLES",
                    "strength": straight_strength,
                    "is_bomb": False,
                    "bomb_strength": 0,
                }

    return {"type": "INVALID", "strength": 0, "is_bomb": False, "bomb_strength": 0}


def action_index_to_cards(
    action_idx: int,
    hand: list,
    level_rank: Optional[str],
    ctx: Optional[dict] = None,
) -> list:
    if action_idx == 0:
        return []
    if level_rank == "2" and CAT_INVALID_WHEN_LEVEL_IS_2[action_idx]:
        return []
    entry = _CATALOGUE[action_idx]
    cards = _entry_cards_from_context(
        entry,
        ctx if ctx is not None else _build_hand_context(hand, level_rank),
    )
    return cards if len(cards) == CAT_CARD_COUNTS[action_idx] else []


def _straight_flush_features_into(
    hand: list,
    out: np.ndarray,
    hand_ctx: Optional[dict] = None,
) -> np.ndarray:
    out.fill(0.0)
    index = 0
    if hand_ctx is not None:
        suit_ranks = hand_ctx["suit_ranks"]
        for suit in SUITS:
            ranks_in_suit = suit_ranks[suit]
            for window in _SF_WINDOWS:
                if all(rank in ranks_in_suit for rank in window):
                    out[index] = 1.0
                index += 1
    else:
        rank_set_by_suit = defaultdict(set)
        for card in hand:
            if not is_joker(card):
                rank_set_by_suit[card["suit"]].add(card["rank"])
        for suit in SUITS:
            for window in _SF_WINDOWS:
                if all(rank in rank_set_by_suit[suit] for rank in window):
                    out[index] = 1.0
                index += 1
    return out


def _history_type_name(combo_info: Optional[dict]) -> str:
    if not combo_info:
        return "PASS"
    combo_type = str(combo_info.get("type", "PASS") or "PASS")
    return combo_type if combo_type in HISTORY_TYPE_TO_INDEX else "PASS"


def encode_history_entry(
    seat: int,
    action_cards: list,
    combo_info: Optional[dict],
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    entry = out if out is not None else np.zeros(HISTORY_ENTRY_DIM, dtype=np.float32)
    entry.fill(0.0)
    if 0 <= int(seat) < 4:
        entry[HISTORY_ENTRY_SEAT_SLICE.start + int(seat)] = 1.0

    history_type = "PASS" if not action_cards else _history_type_name(combo_info)
    entry[HISTORY_ENTRY_TYPE_SLICE.start + HISTORY_TYPE_TO_INDEX[history_type]] = 1.0
    if action_cards and combo_info:
        entry[HISTORY_ENTRY_STRENGTH_SLICE.start] = min(
            max(float(combo_info.get("strength", 0.0)), 0.0),
            MAX_HISTORY_STRENGTH,
        ) / MAX_HISTORY_STRENGTH
        entry[HISTORY_ENTRY_STRENGTH_SLICE.start + 1] = min(
            max(float(combo_info.get("bomb_strength", 0.0)), 0.0),
            MAX_HISTORY_STRENGTH,
        ) / MAX_HISTORY_STRENGTH
        _fill_multihot(action_cards, entry[HISTORY_ENTRY_CARDS_SLICE])
    return entry


def _history_entries_into(history_entries: Sequence[np.ndarray], out: np.ndarray) -> np.ndarray:
    out.fill(0.0)
    if not history_entries:
        return out
    limit = min(len(history_entries), HISTORY_SEQ_LEN)
    for index in range(limit):
        start = index * HISTORY_ENTRY_DIM
        stop = start + HISTORY_ENTRY_DIM
        out[start:stop] = np.asarray(history_entries[index], dtype=np.float32)
    return out


def encode_state(
    hand: list,
    trick_last: list,
    played_cards: list,
    level_rank: Optional[str],
    my_seat: int,
    hand_sizes: list,
    level_ranks: Tuple[float, float],
    a_tries: Tuple[float, float],
    is_opener: bool,
    trick_seat: Optional[int],
    caller: str,
    pass_streak: int,
    round_num: int,
    history_entries: Optional[Sequence[np.ndarray]] = None,
    *,
    hand_ctx: Optional[dict] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    vec = out if out is not None else np.zeros(STATE_DIM, dtype=np.float32)
    vec.fill(0.0)

    _fill_multihot(hand, vec[STATE_HAND_SLICE])
    _fill_multihot(trick_last, vec[STATE_TRICK_SLICE])
    _fill_multihot(played_cards, vec[STATE_PLAYED_SLICE])

    if level_rank in RANK_TO_INDEX:
        vec[STATE_LEVEL_RANK_SLICE.start + RANK_TO_INDEX[level_rank]] = 1.0
    vec[STATE_MY_SEAT_SLICE.start + my_seat] = 1.0

    for index, size in enumerate(hand_sizes):
        vec[STATE_HAND_SIZES_SLICE.start + index] = size / 27.0

    vec[STATE_TEAM_LEVELS_SLICE.start] = level_ranks[0]
    vec[STATE_TEAM_LEVELS_SLICE.start + 1] = level_ranks[1]
    vec[STATE_A_TRIES_SLICE.start] = (
        min(max(float(a_tries[0]), 0.0), MAX_A_TRIES_FEATURE) / MAX_A_TRIES_FEATURE
    )
    vec[STATE_A_TRIES_SLICE.start + 1] = (
        min(max(float(a_tries[1]), 0.0), MAX_A_TRIES_FEATURE) / MAX_A_TRIES_FEATURE
    )
    vec[STATE_IS_OPENER_SLICE.start] = float(is_opener)
    if trick_seat is not None:
        vec[STATE_TRICK_SEAT_SLICE.start + int(trick_seat)] = 1.0
    vec[STATE_CALLER_SLICE.start + (1 if caller == "Red" else 0)] = 1.0
    vec[STATE_PASS_STREAK_SLICE.start] = (
        min(max(float(pass_streak), 0.0), MAX_PASS_STREAK_FEATURE) / MAX_PASS_STREAK_FEATURE
    )
    vec[STATE_ROUND_NUM_SLICE.start] = (
        min(max(float(round_num), 1.0), MAX_ROUND_NUM_FEATURE) / MAX_ROUND_NUM_FEATURE
    )
    _straight_flush_features_into(hand, vec[STATE_SF_SLICE], hand_ctx=hand_ctx)
    _history_entries_into(history_entries or (), vec[STATE_HISTORY_SLICE])
    return vec


class GuandanEnv:
    def __init__(self):
        self.hands = [[] for _ in range(4)]
        self.level_ranks = {"Blue": 0, "Red": 0}
        self.a_tries = {"Blue": 0, "Red": 0}
        self.caller = "Blue"
        self.round_num = 1
        self._match_winner = None
        self.tribute_seat_value_fn = None
        self.finish_order = []
        self.round_history_entries = []
        self.played_cards = []
        self.trick_last = []
        self.trick_info = None
        self.trick_seat = None
        self.pass_streak = 0
        self.current_seat = 0

    def active_level_rank(self) -> str:
        return RANKS[self.level_ranks[self.caller]]

    def get_state(
        self,
        seat: int,
        out: Optional[np.ndarray] = None,
        hand_ctx: Optional[dict] = None,
        history_entries: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        return encode_state(
            hand=self.hands[seat],
            trick_last=self.trick_last,
            played_cards=self.played_cards,
            level_rank=self.active_level_rank(),
            my_seat=seat,
            hand_sizes=[len(self.hands[index]) for index in range(4)],
            level_ranks=(
                self.level_ranks["Blue"] / 12.0,
                self.level_ranks["Red"] / 12.0,
            ),
            a_tries=(self.a_tries["Blue"], self.a_tries["Red"]),
            is_opener=(not self.trick_last),
            trick_seat=self.trick_seat,
            caller=self.caller,
            pass_streak=self.pass_streak,
            round_num=self.round_num,
            history_entries=history_entries,
            hand_ctx=hand_ctx,
            out=out,
        )

    def get_legal_mask(self, seat: int) -> np.ndarray:
        return compute_legal_mask(
            hand=self.hands[seat],
            trick_last=self.trick_last,
            level_rank=self.active_level_rank(),
            is_opener=(not self.trick_last),
            last_info=self.trick_info,
        )


_build_catalogue()


def _upgrade_legacy_model_state(model_state: dict) -> Tuple[dict, bool]:
    upgraded = False
    upgraded_state = dict(model_state)

    if any(key.startswith("backbone.") for key in upgraded_state):
        converted_state = {}
        for key, value in upgraded_state.items():
            if key.startswith("backbone."):
                suffix = key[len("backbone.") :]
                converted_state[f"actor_backbone.{suffix}"] = value.clone()
                converted_state[f"critic_backbone.{suffix}"] = value.clone()
            elif key.startswith("policy_head."):
                suffix = key[len("policy_head.") :]
                converted_state[f"actor_head.{suffix}"] = value
            elif key.startswith("value_head."):
                suffix = key[len("value_head.") :]
                converted_state[f"critic_head.{suffix}"] = value
            else:
                converted_state[key] = value
        upgraded_state = converted_state
        upgraded = True

    for first_layer_key in ("actor_backbone.0.weight", "critic_backbone.0.weight"):
        weight = upgraded_state.get(first_layer_key)
        if weight is None or weight.shape[1] == STATE_CONTEXT_DIM:
            continue
        if weight.shape[1] > STATE_CONTEXT_DIM:
            raise ValueError(
                f"Checkpoint expects {weight.shape[1]} input features, "
                f"but the legacy policy uses {STATE_CONTEXT_DIM}."
            )

        padded = weight.new_zeros((weight.shape[0], STATE_CONTEXT_DIM))
        padded[:, : weight.shape[1]] = weight
        upgraded_state[first_layer_key] = padded
        upgraded = True

    return upgraded_state, upgraded


class LegacyGuandanNet(nn.Module):
    architecture_name = MODEL_ARCH_LEGACY_MLP
    uses_history_memory = False

    def __init__(self, input_size=STATE_CONTEXT_DIM, action_size=0, hidden=HIDDEN):
        super().__init__()
        feature_dim = hidden // 2
        self.actor_backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_size),
        )
        self.critic_backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def actor_parameters(self):
        return itertools.chain(
            self.actor_backbone.parameters(),
            self.actor_head.parameters(),
        )

    def critic_parameters(self):
        return itertools.chain(
            self.critic_backbone.parameters(),
            self.critic_head.parameters(),
        )

    def _context_only(self, state):
        return state[..., :STATE_CONTEXT_DIM]

    def _resolve_memories(self, memories=None, actor_memories=None, critic_memories=None):
        if memories is not None:
            return memories
        if actor_memories is not None:
            return actor_memories
        return critic_memories

    def zero_memory(self, batch_size: int, *, device=None):
        return None

    def update_history_memory(
        self,
        history_segments,
        memories=None,
        actor_memories=None,
        critic_memories=None,
    ):
        resolved = self._resolve_memories(
            memories=memories,
            actor_memories=actor_memories,
            critic_memories=critic_memories,
        )
        if memories is not None:
            return resolved
        return resolved, resolved

    def actor_logits(self, state, mask=None, memories=None, actor_memories=None):
        actor_features = self.actor_backbone(self._context_only(state))
        logits = self.actor_head(actor_features)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def critic_value(self, state, memories=None, critic_memories=None):
        critic_features = self.critic_backbone(self._context_only(state))
        return self.critic_head(critic_features)

    def forward(
        self,
        state,
        mask=None,
        memories=None,
        actor_memories=None,
        critic_memories=None,
    ):
        resolved_memories = self._resolve_memories(
            memories=memories,
            actor_memories=actor_memories,
            critic_memories=critic_memories,
        )
        logits = self.actor_logits(state, mask, memories=resolved_memories)
        value = self.critic_value(state, memories=resolved_memories)
        return logits, value


class _GTrXLGate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_z = nn.Linear(hidden_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g = nn.Parameter(torch.full((hidden_size,), 2.0))

    def forward(self, residual, update):
        reset = torch.sigmoid(self.W_r(update) + self.U_r(residual))
        mix = torch.sigmoid(self.W_z(update) + self.U_z(residual) - self.g)
        candidate = torch.tanh(self.W_h(update) + self.U_h(reset * residual))
        return (1.0 - mix) * residual + mix * candidate


def _causal_attention_mask(seq_len: int, mem_len: int, device) -> torch.Tensor:
    mask = torch.zeros((seq_len, mem_len + seq_len), dtype=torch.bool, device=device)
    if seq_len > 0:
        mask[:, mem_len:] = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
    return mask


class GTrXLBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attn_gate = _GTrXLGate(hidden_size)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ff_gate = _GTrXLGate(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        memory: Optional[torch.Tensor] = None,
        current_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_x = self.attn_norm(x)
        if memory is not None and memory.shape[1] > 0:
            attn_kv = torch.cat([memory, norm_x], dim=1)
            mem_len = int(memory.shape[1])
        else:
            attn_kv = norm_x
            mem_len = 0

        key_padding_mask = None
        if memory_padding_mask is not None or current_padding_mask is not None:
            if memory_padding_mask is None:
                memory_padding_mask = torch.zeros(
                    (x.shape[0], mem_len),
                    dtype=torch.bool,
                    device=x.device,
                )
            if current_padding_mask is None:
                current_padding_mask = torch.zeros(
                    (x.shape[0], x.shape[1]),
                    dtype=torch.bool,
                    device=x.device,
                )
            key_padding_mask = torch.cat(
                [memory_padding_mask, current_padding_mask],
                dim=1,
            )

        attn_mask = _causal_attention_mask(x.shape[1], mem_len, x.device)
        attn_out, _ = self.attn(
            norm_x,
            attn_kv,
            attn_kv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.attn_gate(x, attn_out)
        ff_out = self.ff(self.ff_norm(x))
        x = self.ff_gate(x, ff_out)
        return x


class GuandanNet(nn.Module):
    architecture_name = MODEL_ARCH_GTRXL
    uses_history_memory = True

    def __init__(self, input_size=STATE_DIM, action_size=0, hidden=HIDDEN):
        super().__init__()
        if hidden % GTRXL_HEAD_COUNT != 0:
            raise ValueError(
                f"HIDDEN ({hidden}) must be divisible by "
                f"GTRXL_HEAD_COUNT ({GTRXL_HEAD_COUNT})."
            )
        feature_dim = hidden // 2
        self.hidden_size = hidden
        self.feature_dim = feature_dim
        self.memory_len = GTRXL_MEMORY_LEN
        self.block_count = GTRXL_BLOCK_COUNT

        self.history_proj = nn.Linear(HISTORY_ENTRY_DIM, hidden)
        self.context_proj = nn.Sequential(
            nn.Linear(STATE_CONTEXT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(HISTORY_SEQ_LEN + 1, hidden))
        self.blocks = nn.ModuleList(
            [GTrXLBlock(hidden, GTRXL_HEAD_COUNT) for _ in range(GTRXL_BLOCK_COUNT)]
        )
        self.actor_body = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_size),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _resolve_memories(self, memories=None, actor_memories=None, critic_memories=None):
        if memories is not None:
            return memories
        if actor_memories is not None:
            return actor_memories
        return critic_memories

    def zero_memory(self, batch_size: int, *, device=None):
        active_device = device or next(self.parameters()).device
        return torch.zeros(
            (batch_size, self.block_count, self.memory_len, self.hidden_size),
            dtype=torch.float32,
            device=active_device,
        )

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = state[..., :STATE_CONTEXT_DIM]
        history_flat = state[..., STATE_CONTEXT_DIM:]
        history = history_flat.reshape(state.shape[0], HISTORY_SEQ_LEN, HISTORY_ENTRY_DIM)
        return context, history

    def _history_valid_mask(self, history_tokens: torch.Tensor) -> torch.Tensor:
        return history_tokens.abs().sum(dim=-1) > 0.0

    def _memory_padding_mask(self, memories: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if memories is None:
            return None
        return memories.abs().sum(dim=-1) <= 1e-8

    def _encode_stream(
        self,
        history_tokens: torch.Tensor,
        context: torch.Tensor,
        memories: Optional[torch.Tensor],
        include_decision_token: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = history_tokens.shape[0]
        history_valid = self._history_valid_mask(history_tokens)
        history_emb = self.history_proj(history_tokens)
        if include_decision_token:
            decision_token = self.context_proj(context).unsqueeze(1)
            x = torch.cat([history_emb, decision_token], dim=1)
            current_padding_mask = torch.cat(
                [
                    ~history_valid,
                    torch.zeros((batch_size, 1), dtype=torch.bool, device=history_tokens.device),
                ],
                dim=1,
            )
            x = x + self.position[: x.shape[1]].unsqueeze(0)
        else:
            x = history_emb + self.position[: history_emb.shape[1]].unsqueeze(0)
            current_padding_mask = ~history_valid

        next_memories: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            block_memory = None
            block_memory_padding = None
            if memories is not None:
                block_memory = memories[:, block_index]
                block_memory_padding = self._memory_padding_mask(block_memory)
            x = block(
                x,
                memory=block_memory,
                current_padding_mask=current_padding_mask,
                memory_padding_mask=block_memory_padding,
            )
            current_memory_source = x[:, :HISTORY_SEQ_LEN] if include_decision_token else x
            if block_memory is not None and block_memory.shape[1] > 0:
                next_memory_source = torch.cat([block_memory, current_memory_source], dim=1)
            else:
                next_memory_source = current_memory_source
            next_memory = next_memory_source[:, -self.memory_len :]
            if next_memory.shape[1] < self.memory_len:
                pad_len = self.memory_len - next_memory.shape[1]
                next_memory = F.pad(next_memory, (0, 0, pad_len, 0))
            next_memories.append(next_memory)

        next_memory_tensor = torch.stack(next_memories, dim=1)
        return x, next_memory_tensor

    def _encode_decision_state(self, state, memories=None):
        context, history = self._split_state(state)
        stream, _ = self._encode_stream(
            history,
            context,
            memories,
            include_decision_token=True,
        )
        return stream[:, -1, :]

    def actor_logits(self, state, mask=None, memories=None, actor_memories=None):
        resolved_memories = self._resolve_memories(
            memories=memories,
            actor_memories=actor_memories,
        )
        stream_last = self._encode_decision_state(state, memories=resolved_memories)
        actor_features = self.actor_body(stream_last)
        logits = self.actor_head(actor_features)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def critic_value(self, state, memories=None, critic_memories=None):
        resolved_memories = self._resolve_memories(
            memories=memories,
            critic_memories=critic_memories,
        )
        stream_last = self._encode_decision_state(state, memories=resolved_memories)
        critic_features = self.critic_body(stream_last)
        return self.critic_head(critic_features)

    def update_history_memory(
        self,
        history_segments,
        memories=None,
        actor_memories=None,
        critic_memories=None,
    ):
        resolved_memories = self._resolve_memories(
            memories=memories,
            actor_memories=actor_memories,
            critic_memories=critic_memories,
        )
        if history_segments is None:
            if memories is not None:
                return resolved_memories
            return resolved_memories, resolved_memories
        history_segments = history_segments.float()
        dummy_context = torch.zeros(
            (history_segments.shape[0], STATE_CONTEXT_DIM),
            dtype=history_segments.dtype,
            device=history_segments.device,
        )
        _, next_memory = self._encode_stream(
            history_segments,
            dummy_context,
            resolved_memories,
            include_decision_token=False,
        )
        if memories is not None:
            return next_memory
        return next_memory, next_memory

    def forward(
        self,
        state,
        mask=None,
        memories=None,
        actor_memories=None,
        critic_memories=None,
    ):
        resolved_memories = self._resolve_memories(
            memories=memories,
            actor_memories=actor_memories,
            critic_memories=critic_memories,
        )
        stream_last = self._encode_decision_state(state, memories=resolved_memories)
        actor_features = self.actor_body(stream_last)
        logits = self.actor_head(actor_features)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        critic_features = self.critic_body(stream_last)
        value = self.critic_head(critic_features)
        return logits, value


def checkpoint_model_architecture(checkpoint: dict) -> str:
    arch = str(checkpoint.get("model_architecture", "") or "").strip()
    if arch:
        return arch
    model_state = checkpoint.get("model_state", {})
    if isinstance(model_state, dict):
        if any(
            key.startswith("history_proj.") or key.startswith("actor_history_proj.")
            for key in model_state
        ):
            return MODEL_ARCH_GTRXL
    return MODEL_ARCH_LEGACY_MLP


def _infer_legacy_hidden_size(model_state: dict) -> int:
    for key in ("actor_backbone.0.weight", "critic_backbone.0.weight"):
        weight = model_state.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return HIDDEN


def _average_checkpoint_tensors(
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            raise ValueError(
                "Cannot merge split GTrXL checkpoint tensors with different shapes: "
                f"{tuple(left.shape)} vs {tuple(right.shape)}."
            )
        return (left + right) * 0.5
    if isinstance(left, torch.Tensor):
        return left.clone()
    if isinstance(right, torch.Tensor):
        return right.clone()
    return None


def _upgrade_gtrxl_model_state(model_state: dict) -> Tuple[dict, bool]:
    if not isinstance(model_state, dict):
        raise TypeError("GTrXL model_state must be a dict.")
    if any(key.startswith("history_proj.") for key in model_state):
        return model_state, False
    if not any(key.startswith("actor_history_proj.") for key in model_state):
        return model_state, False

    upgraded = {}
    skipped_prefixes = (
        "actor_history_proj.",
        "critic_history_proj.",
        "actor_context_proj.",
        "critic_context_proj.",
        "actor_blocks.",
        "critic_blocks.",
    )
    for key, value in model_state.items():
        if key in {"actor_position", "critic_position"}:
            continue
        if key.startswith(skipped_prefixes):
            continue
        upgraded[key] = value

    def merge_prefix(shared_prefix: str, actor_prefix: str, critic_prefix: str) -> None:
        suffixes = {
            key[len(actor_prefix) :]
            for key in model_state
            if key.startswith(actor_prefix)
        }
        suffixes.update(
            key[len(critic_prefix) :]
            for key in model_state
            if key.startswith(critic_prefix)
        )
        for suffix in suffixes:
            merged = _average_checkpoint_tensors(
                model_state.get(actor_prefix + suffix),
                model_state.get(critic_prefix + suffix),
            )
            if merged is not None:
                upgraded[shared_prefix + suffix] = merged

    merge_prefix("history_proj.", "actor_history_proj.", "critic_history_proj.")
    merge_prefix("context_proj.", "actor_context_proj.", "critic_context_proj.")
    merge_prefix("blocks.", "actor_blocks.", "critic_blocks.")
    position = _average_checkpoint_tensors(
        model_state.get("actor_position"),
        model_state.get("critic_position"),
    )
    if position is not None:
        upgraded["position"] = position
    return upgraded, True


def build_policy_network_for_architecture(
    architecture_name: str,
    *,
    model_state: Optional[dict] = None,
) -> nn.Module:
    if architecture_name == MODEL_ARCH_GTRXL:
        action_size = _infer_action_size(model_state or {})
        return GuandanNet(action_size=action_size)
    if architecture_name == MODEL_ARCH_LEGACY_MLP:
        legacy_hidden = _infer_legacy_hidden_size(model_state or {})
        action_size = _infer_action_size(model_state or {})
        return LegacyGuandanNet(hidden=legacy_hidden, action_size=action_size)
    raise ValueError(f"Unsupported checkpoint architecture: {architecture_name}")


def _infer_action_size(model_state: dict) -> int:
    for key in ("actor_head.2.weight", "actor_head.2.bias"):
        tensor = model_state.get(key)
        if isinstance(tensor, torch.Tensor):
            return int(tensor.shape[0])
    raise KeyError("Could not infer action dimension from checkpoint model_state.")


def load_policy_network_from_checkpoint(
    checkpoint: dict,
    *,
    device: Optional[torch.device] = None,
) -> nn.Module:
    architecture_name = checkpoint_model_architecture(checkpoint)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint is missing 'model_state'.")
    if architecture_name == MODEL_ARCH_GTRXL:
        model_state, _ = _upgrade_gtrxl_model_state(model_state)
    elif architecture_name == MODEL_ARCH_LEGACY_MLP:
        model_state, _ = _upgrade_legacy_model_state(model_state)
    net = build_policy_network_for_architecture(
        architecture_name,
        model_state=model_state,
    )
    target_device = device or DEVICE
    net = net.to(target_device)
    net.load_state_dict(model_state)
    return net


@dataclass(eq=False)
class PolicyMemorySnapshot:
    memory: Optional[np.ndarray]


def policy_uses_history_memory(policy) -> bool:
    return bool(getattr(policy, "uses_history_memory", False))


def snapshot_memory_array(memory: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if memory is None:
        return None
    array = np.asarray(memory, dtype=MEMORY_SNAPSHOT_DTYPE)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                "Policy memory snapshots must contain exactly one batch item; "
                f"got shape {tuple(array.shape)}."
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            "Policy memory snapshots must be rank-3 "
            f"(blocks, mem_len, hidden); got shape {tuple(array.shape)}."
        )
    return array.copy()


def make_policy_memory_snapshot(
    memory: Optional[np.ndarray],
) -> Optional[PolicyMemorySnapshot]:
    if memory is None:
        return None
    return PolicyMemorySnapshot(memory=snapshot_memory_array(memory))


def zero_policy_memory(policy) -> Optional[PolicyMemorySnapshot]:
    if policy is None or not policy_uses_history_memory(policy):
        return None
    memory = policy.zero_memory(1, device=torch.device("cpu"))
    return make_policy_memory_snapshot(
        memory.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    )


def expected_policy_memory_shape(policy) -> Optional[Tuple[int, int, int]]:
    zero_snapshot = zero_policy_memory(policy)
    if zero_snapshot is None or zero_snapshot.memory is None:
        return None
    return tuple(int(value) for value in zero_snapshot.memory.shape)


def serialize_policy_memory_snapshot(
    snapshot: Optional[PolicyMemorySnapshot],
) -> Optional[list]:
    if snapshot is None or snapshot.memory is None:
        return None
    return np.asarray(snapshot.memory, dtype=np.float32).tolist()


def deserialize_policy_memory_snapshot(
    raw_memory,
) -> Optional[PolicyMemorySnapshot]:
    if raw_memory in (None, ""):
        return None
    return make_policy_memory_snapshot(np.asarray(raw_memory, dtype=np.float32))


def _memory_tensor_from_snapshot(
    policy,
    memory_snapshot: Optional[PolicyMemorySnapshot],
) -> Optional[torch.Tensor]:
    if memory_snapshot is None or memory_snapshot.memory is None:
        return None
    model_device = next(policy.parameters()).device
    memory_t = torch.from_numpy(memory_snapshot.memory).to(
        model_device,
        non_blocking=True,
    ).float()
    if memory_t.ndim == 3:
        memory_t = memory_t.unsqueeze(0)
    return memory_t


def policy_forward(
    policy,
    states_np: np.ndarray,
    masks_np: Optional[np.ndarray] = None,
    memory_snapshot: Optional[PolicyMemorySnapshot] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model_device = next(policy.parameters()).device
    states = torch.from_numpy(np.asarray(states_np, dtype=np.float32)).to(
        model_device,
        non_blocking=True,
    )
    masks = None
    if masks_np is not None:
        masks = torch.from_numpy(np.asarray(masks_np, dtype=bool)).to(
            model_device,
            non_blocking=True,
        )
    memories = _memory_tensor_from_snapshot(policy, memory_snapshot)
    with torch.inference_mode():
        with _autocast_context(model_device.type):
            logits, values = policy(states, masks, memories=memories)
    return logits.float().cpu(), values.float().cpu()


def policy_critic_values(
    policy,
    states_np: np.ndarray,
    memory_snapshot: Optional[PolicyMemorySnapshot] = None,
) -> np.ndarray:
    model_device = next(policy.parameters()).device
    states = torch.from_numpy(np.asarray(states_np, dtype=np.float32)).to(
        model_device,
        non_blocking=True,
    )
    memories = _memory_tensor_from_snapshot(policy, memory_snapshot)
    with torch.inference_mode():
        with _autocast_context(model_device.type):
            value = policy.critic_value(states, memories=memories)
    return value.squeeze(-1).float().cpu().numpy()


def policy_critic_value(
    policy,
    state_np: np.ndarray,
    memory_snapshot: Optional[PolicyMemorySnapshot] = None,
) -> float:
    value = policy_critic_values(
        policy,
        np.asarray(state_np, dtype=np.float32)[None, :],
        memory_snapshot=memory_snapshot,
    )
    return float(value[0])


def policy_update_history_memory(
    policy,
    history_segment: np.ndarray,
    memory_snapshot: Optional[PolicyMemorySnapshot],
) -> Optional[PolicyMemorySnapshot]:
    if policy is None or not policy_uses_history_memory(policy):
        return memory_snapshot

    model_device = next(policy.parameters()).device
    history_t = torch.from_numpy(np.asarray(history_segment, dtype=np.float32)).to(
        model_device,
        non_blocking=True,
    )
    memory_t = _memory_tensor_from_snapshot(policy, memory_snapshot)
    with torch.inference_mode():
        with _autocast_context(model_device.type):
            next_memory_t = policy.update_history_memory(history_t, memories=memory_t)
    if next_memory_t is None:
        return None
    next_memory = next_memory_t.float().cpu().numpy().astype(np.float32, copy=False)
    return make_policy_memory_snapshot(next_memory)
