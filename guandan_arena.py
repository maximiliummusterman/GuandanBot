"""
Guandan Training Arena  ─  Rewrite v2
======================================
Key changes vs v1:
  • Static action catalogue (permanent indices)
  • Level-rank slots that remap at runtime without changing indices
  • Rank-aware legal-action masking for suitless combos, with straight flushes
    still checked suit-by-suit
  • Vectorised environment pool (Windows-safe, spawn-based worker pool)
  • Batched inference: all active seats across all envs in one GPU forward pass
  • Straight-flush auxiliary features (40 bits) added to state vector
  • Larger default batch size (2048) and non_blocking GPU transfers

Requirements:
    pip install torch numpy

Run:
    python guandan_arena.py
"""

import random, itertools, os, time, copy
from collections import defaultdict
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp

# ── Hyperparameters ────────────────────────────────────────────────────────────
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.01
VALUE_COEF    = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS    = 4
BATCH_SIZE    = 2048
HIDDEN        = 256
SAVE_INTERVAL = 4000
LOG_INTERVAL  = 400
NUM_ENVS      = 48          # parallel game environments
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Card constants ─────────────────────────────────────────────────────────────
SUITS    = ['hearts', 'diamonds', 'clubs', 'spades']
RANKS    = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
RANK_VAL = {r: i+2 for i, r in enumerate(RANKS)}   # 2→2 … A→14

# Canonical 108-slot card list (2 decks + 4 jokers)
CANONICAL = []
for _ in range(2):
    for s in SUITS:
        for r in RANKS:
            CANONICAL.append(('normal', s, r))
CANONICAL.append(('joker', 'red',   None))
CANONICAL.append(('joker', 'red',   None))
CANONICAL.append(('joker', 'black', None))
CANONICAL.append(('joker', 'black', None))
CANON_LEN = len(CANONICAL)   # 108

# ── State vector layout ────────────────────────────────────────────────────────
#   108  my hand (multi-hot)
#   108  trick last play (multi-hot)
#   108  played-card history for the round (multi-hot)
#    15  trump rank one-hot  (13 ranks + black joker + red joker)
#     4  my seat one-hot
#     4  hand sizes (normalised)
#     2  team level ranks (normalised)
#     1  is trick opener
#    40  straight-flush auxiliary (4 suits × 10 windows)
# ─────────────────────────────────────────────────────────────────────────────
STRAIGHT_FLUSH_FEATURE_DIM = len(SUITS) * (1 + (len(RANKS) - 4))
STATE_DIM = 108 + 108 + 108 + 15 + 4 + 4 + 2 + 1 + STRAIGHT_FLUSH_FEATURE_DIM   # = 390


# ══════════════════════════════════════════════════════════════════════════════
#  Card utilities
# ══════════════════════════════════════════════════════════════════════════════

def make_card(suit, rank):
    return {'suit': suit, 'rank': rank}

def make_joker(color):
    return {'joker': color}

def is_joker(c):
    return 'joker' in c

def card_rank_val(c, level_rank=None):
    """
    Numeric strength of a single card.
    non-heart level_rank → 16 (between A=14 and black joker=17)
    heart level_rank     → wildcard, but if asked for its "value" return 15
                           (below non-heart level_rank; never used in combination
                           strength because wildcards are invisible in combos)
    """
    if is_joker(c):
        return 18 if c['joker'] == 'red' else 17
    r = c['rank']
    if level_rank and r == level_rank:
        if c.get('suit') == 'hearts':
            return 15   # wildcard — strength only relevant for tribute ordering
        return 16       # elevated non-heart trump
    return RANK_VAL.get(r, 0)

def is_wildcard(c, level_rank):
    """Hearts level-rank card is the wildcard."""
    return (not is_joker(c)
            and level_rank is not None
            and c.get('rank') == level_rank
            and c.get('suit') == 'hearts')

def build_deck():
    deck = []
    for _ in range(2):
        for s in SUITS:
            for r in RANKS:
                deck.append(make_card(s, r))
    deck += [make_joker('red'), make_joker('red'),
             make_joker('black'), make_joker('black')]
    random.shuffle(deck)
    return deck

def hand_to_multihot(hand) -> np.ndarray:
    """108-dim multi-hot (values 0/1/2 for duplicates)."""
    vec  = np.zeros(CANON_LEN, dtype=np.float32)
    used = defaultdict(int)
    for c in hand:
        if is_joker(c):
            color = c['joker']
            for i, slot in enumerate(CANONICAL):
                if slot[0] == 'joker' and slot[1] == color and used[i] < 2:
                    used[i] += 1
                    vec[i] = min(vec[i] + 1, 2)
                    break
        else:
            for i, slot in enumerate(CANONICAL):
                if (slot[0] == 'normal' and slot[1] == c['suit']
                        and slot[2] == c['rank'] and used[i] < 2):
                    used[i] += 1
                    vec[i] = min(vec[i] + 1, 2)
                    break
    return vec

def count_ranks(cards):
    counts = defaultdict(int)
    for c in cards:
        if not is_joker(c):
            counts[c['rank']] += 1
    return counts

def _check_straight_ranks(rank_list: List[str]) -> Optional[int]:
    """Returns highest RANK_VAL if rank strings are 5 consecutive, else None."""
    if len(rank_list) != 5 or len(set(rank_list)) != 5:
        return None
    for ace_low in ([True, False] if 'A' in rank_list else [False]):
        def rv(r):
            return 1 if (r == 'A' and ace_low) else RANK_VAL.get(r, 0)
        vals = sorted(rv(r) for r in rank_list)
        if vals[-1] - vals[0] == 4 and len(set(vals)) == 5:
            if ace_low and vals[-1] == 14:
                continue
            if not ace_low and vals[0] == 1:
                continue
            return vals[-1]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Static Action Catalogue
# ══════════════════════════════════════════════════════════════════════════════
#
#  Each entry is a dict:
#    type          : str  (combo type name)
#    rank_key      : str  (rank string or 'LR' for level-rank slot, 'BJ'/'RJ' for jokers)
#    pair_rank_key : str  (for FULL_HOUSE only — the pair's rank_key)
#    strength      : int  (static strength; runtime overrides LR slots to 16)
#    is_bomb       : bool
#    bomb_strength : int  (0 if not bomb)
#    vec_template  : np.ndarray shape (108,)  — canonical card requirements
#                    For LR slots the vector uses a PLACEHOLDER rank ('2') that
#                    the runtime mask replaces with the actual level_rank cards.
#
#  The catalogue is built ONCE at module load.  Its length = ACTION_DIM.
#  Index 0 = PASS.

_CATALOGUE: List[dict] = []   # filled by _build_catalogue()
CAT_VECS:   np.ndarray = None  # shape (ACTION_DIM, 108), filled at startup
ACTION_DIM: int        = 0
LR_PLACEHOLDER_SUITS   = ['spades', 'clubs', 'diamonds']

# ─── helpers used during catalogue construction ───────────────────────────────

def _cards_multihot(cards) -> np.ndarray:
    return hand_to_multihot(cards)

def _cards_slot_counts(cards) -> Dict[int, int]:
    vec = _cards_multihot(cards)
    return {int(i): int(v) for i, v in enumerate(vec) if v > 0}

def _make_lr_placeholder_cards(n: int) -> list:
    cards = []
    full_cycles, remainder = divmod(n, len(LR_PLACEHOLDER_SUITS))
    for _ in range(full_cycles):
        for suit in LR_PLACEHOLDER_SUITS:
            cards.append(make_card(suit, '2'))
    for suit in LR_PLACEHOLDER_SUITS[:remainder]:
        cards.append(make_card(suit, '2'))
    return cards

def _add(entry: dict, cards, lr_placeholder_cards=None, forced_wildcards: int = 0):
    stored = dict(entry)
    stored['vec_template'] = _cards_multihot(cards)
    stored['lr_placeholder_counts'] = _cards_slot_counts(lr_placeholder_cards or [])
    stored['forced_wildcards'] = forced_wildcards
    _CATALOGUE.append(stored)

def _build_catalogue():
    """
    Build the static catalogue.
    LR (level-rank) entries use a placeholder rank index 0 ('2') in their
    vec_template; the runtime mask function substitutes the real level_rank.
    """
    global _CATALOGUE, CAT_VECS, ACTION_DIM

    # Index 0: PASS
    _CATALOGUE.append({
        'type': 'PASS', 'rank_key': None, 'pair_rank_key': None,
        'strength': 0, 'is_bomb': False, 'bomb_strength': 0,
        'vec_template': np.zeros(CANON_LEN, dtype=np.float32),
        'lr_placeholder_counts': {},
        'forced_wildcards': 0,
    })

    std = [make_card(s, r) for s in SUITS for r in RANKS]
    rj  = make_joker('red')
    bj  = make_joker('black')

    # ── Singles ───────────────────────────────────────────────────────────────
    for r in RANKS:
        rv = RANK_VAL[r]
        c  = make_card('spades', r)   # canonical placeholder for a suitless single
        _add({'type':'SINGLE','rank_key':r,'pair_rank_key':None,
              'strength':rv,'is_bomb':False,'bomb_strength':0}, [c])
    # Single level-rank (elevated, non-heart; placeholder = spades '2' overwritten at runtime)
    lr_single = _make_lr_placeholder_cards(1)
    _add({'type':'SINGLE','rank_key':'LR','pair_rank_key':None,
          'strength':16,'is_bomb':False,'bomb_strength':0},
         lr_single, lr_placeholder_cards=lr_single)
    # Black joker single
    _add({'type':'SINGLE','rank_key':'BJ','pair_rank_key':None,
          'strength':17,'is_bomb':False,'bomb_strength':0}, [bj])
    # Red joker single
    _add({'type':'SINGLE','rank_key':'RJ','pair_rank_key':None,
          'strength':18,'is_bomb':False,'bomb_strength':0}, [rj])

    # ── Pairs ─────────────────────────────────────────────────────────────────
    for r in RANKS:
        rv = RANK_VAL[r]
        _add({'type':'PAIR','rank_key':r,'pair_rank_key':None,
              'strength':rv,'is_bomb':False,'bomb_strength':0},
             [make_card('spades',r), make_card('hearts',r)])
    lr_pair = _make_lr_placeholder_cards(2)
    _add({'type':'PAIR','rank_key':'LR','pair_rank_key':None,
          'strength':16,'is_bomb':False,'bomb_strength':0},
         lr_pair, lr_placeholder_cards=lr_pair)
    _add({'type':'PAIR','rank_key':'BJ','pair_rank_key':None,
          'strength':17,'is_bomb':False,'bomb_strength':0}, [bj, bj])
    _add({'type':'PAIR','rank_key':'RJ','pair_rank_key':None,
          'strength':18,'is_bomb':False,'bomb_strength':0}, [rj, rj])

    # ── Triples ───────────────────────────────────────────────────────────────
    for r in RANKS:
        rv = RANK_VAL[r]
        _add({'type':'TRIPLE','rank_key':r,'pair_rank_key':None,
              'strength':rv,'is_bomb':False,'bomb_strength':0},
             [make_card(s,r) for s in SUITS[:3]])
    lr_triple = _make_lr_placeholder_cards(3)
    _add({'type':'TRIPLE','rank_key':'LR','pair_rank_key':None,
          'strength':16,'is_bomb':False,'bomb_strength':0},
         lr_triple, lr_placeholder_cards=lr_triple)

    # ── Straights (treat level-rank as its normal rank) ───────────────────────
    # A-5 through 10-A  (10 windows)
    straight_windows = []
    # A-low: A,2,3,4,5  → highest normal val = 5
    straight_windows.append((['A','2','3','4','5'], 5))
    for start in range(len(RANKS)-4):
        window = RANKS[start:start+5]
        straight_windows.append((window, RANK_VAL[window[-1]]))
    for window, rv in straight_windows:
        _add({'type':'STRAIGHT','rank_key':window[-1] if rv != 5 else '5',
              'pair_rank_key':None,
              'strength':rv,'is_bomb':False,'bomb_strength':0},
             [make_card('spades', r) for r in window])

    # ── Full house (triple_rank × pair_rank, both normal + LR slots) ──────────
    # rank_key = triple rank, pair_rank_key = pair rank
    all_rank_keys = RANKS + ['LR']   # 14 slots
    for t_key in all_rank_keys:
        for p_key in all_rank_keys:
            if t_key == p_key:
                continue
            triple_cards = (_make_lr_placeholder_cards(3) if t_key == 'LR'
                            else [make_card(s, t_key) for s in SUITS[:3]])
            pair_cards   = (_make_lr_placeholder_cards(2) if p_key == 'LR'
                            else [make_card(s, p_key) for s in ['spades','hearts']])
            tv = 16 if t_key == 'LR' else RANK_VAL[t_key]
            _add({'type':'FULL_HOUSE','rank_key':t_key,'pair_rank_key':p_key,
                  'strength':tv,'is_bomb':False,'bomb_strength':0},
                 triple_cards + pair_cards,
                 lr_placeholder_cards=((triple_cards if t_key == 'LR' else [])
                                       + (pair_cards if p_key == 'LR' else [])))

    # ── Seq of 3 pairs (11 windows, level-rank treated as normal rank) ─────────
    for start in range(len(RANKS)-2):
        window = RANKS[start:start+3]
        rv = RANK_VAL[window[-1]]
        cards = []
        for r in window:
            cards += [make_card('spades',r), make_card('hearts',r)]
        _add({'type':'SEQ_3_PAIRS','rank_key':window[-1],'pair_rank_key':None,
              'strength':rv,'is_bomb':False,'bomb_strength':0}, cards)

    # ── Seq of 2 triples (12 windows) ─────────────────────────────────────────
    for start in range(len(RANKS)-1):
        window = RANKS[start:start+2]
        rv = RANK_VAL[window[-1]]
        cards = []
        for r in window:
            cards += [make_card(s,r) for s in SUITS[:3]]
        _add({'type':'SEQ_2_TRIPLES','rank_key':window[-1],'pair_rank_key':None,
              'strength':rv,'is_bomb':False,'bomb_strength':0}, cards)

    # ── Bombs: 4-8 of a kind (normal ranks only) ──────────────────────────────
    bomb_type = {4:'FOUR_OF_A_KIND',5:'FIVE_OF_A_KIND',6:'SIX_OF_A_KIND',
                 7:'SEVEN_OF_A_KIND',8:'EIGHT_OF_A_KIND'}
    for r in RANKS:
        rv = RANK_VAL[r]
        # Two decks → up to 8 copies
        all_copies = [make_card(s,r) for s in SUITS] * 2   # 8 cards
        for n in range(4, 9):
            bs = n * 100 + rv
            _add({'type':bomb_type[n],'rank_key':r,'pair_rank_key':None,
                  'strength':rv,'is_bomb':True,'bomb_strength':bs},
                 all_copies[:n])

    # ── Bombs: 4-8 of level-rank (elevated, strength = n*100+16) ──────────────
    for n in range(4, 9):
        bs = n * 100 + 16
        natural_copies = _make_lr_placeholder_cards(min(n, 6))
        _add({'type':bomb_type[n],'rank_key':'LR','pair_rank_key':None,
              'strength':16,'is_bomb':True,'bomb_strength':bs},
             natural_copies,
             lr_placeholder_cards=natural_copies,
             forced_wildcards=max(0, n - len(natural_copies)))

    # ── Straight flushes (4 suits × 10 windows) ───────────────────────────────
    # bomb_strength = between 5-of-a-kind(500+rv) and 6-of-a-kind(600+rv)
    # Use fixed 550 + rv to keep them in that band regardless of rv
    sf_windows = []
    sf_windows.append((['A','2','3','4','5'], 5))
    for start in range(len(RANKS)-4):
        window = RANKS[start:start+5]
        sf_windows.append((window, RANK_VAL[window[-1]]))
    for suit in SUITS:
        for window, rv in sf_windows:
            bs = 550 + rv
            cards = [make_card(suit, r) for r in window]
            _add({'type':'STRAIGHT_FLUSH','rank_key':window[-1] if rv!=5 else '5',
                  'pair_rank_key':None,
                  'strength':rv,'is_bomb':True,'bomb_strength':bs}, cards)

    # ── Four jokers ───────────────────────────────────────────────────────────
    _add({'type':'FOUR_JOKERS','rank_key':None,'pair_rank_key':None,
          'strength':1000,'is_bomb':True,'bomb_strength':1000},
         [rj, rj, bj, bj])

    # Freeze into numpy matrix
    ACTION_DIM = len(_CATALOGUE)
    CAT_VECS   = np.stack([e['vec_template'] for e in _CATALOGUE], axis=0).astype(np.float32)
    return ACTION_DIM, CAT_VECS

print("Building static action catalogue...", end=" ", flush=True)
ACTION_DIM, CAT_VECS = _build_catalogue()
CAT_FORCED_WILDCARDS = np.array(
    [entry.get('forced_wildcards', 0) for entry in _CATALOGUE],
    dtype=np.float32,
)
CAT_CARD_COUNTS = np.array(
    [int(entry['vec_template'].sum()) + entry.get('forced_wildcards', 0) for entry in _CATALOGUE],
    dtype=np.int32,
)
CAT_INVALID_WHEN_LEVEL_IS_2 = np.array([
    (
        entry['type'] in {
            'SINGLE', 'PAIR', 'TRIPLE', 'FULL_HOUSE',
            'FOUR_OF_A_KIND', 'FIVE_OF_A_KIND', 'SIX_OF_A_KIND',
            'SEVEN_OF_A_KIND', 'EIGHT_OF_A_KIND',
        }
        and (
            entry.get('rank_key') == '2'
            or entry.get('pair_rank_key') == '2'
        )
    )
    for entry in _CATALOGUE
], dtype=bool)
print(f"done. {ACTION_DIM} entries (index 0 = PASS).")


# ══════════════════════════════════════════════════════════════════════════════
#  Runtime mask computation
# ══════════════════════════════════════════════════════════════════════════════

def _get_lr_adjusted_vecs(level_rank: Optional[str]) -> np.ndarray:
    """
    Return a copy of CAT_VECS where every LR-slot placeholder ('2' cards)
    is replaced with the actual level_rank cards.

    Each catalogue entry tracks exactly which placeholder counts belong to the
    level-rank side, so normal '2' cards in the same entry stay untouched.

    This is called once per turn per environment — it returns a (ACTION_DIM, 108)
    float32 array.
    """
    if level_rank is None or level_rank == '2':
        return CAT_VECS

    vecs = CAT_VECS.copy()
    lr_slot_by_suit = {}
    for i, slot in enumerate(CANONICAL):
        if slot[0] == 'normal' and slot[2] == level_rank and slot[1] in LR_PLACEHOLDER_SUITS:
            lr_slot_by_suit.setdefault(slot[1], i)

    for entry_idx, entry in enumerate(_CATALOGUE):
        ph_counts = entry.get('lr_placeholder_counts', {})
        if not ph_counts:
            continue
        for ph_idx, count in ph_counts.items():
            suit = CANONICAL[ph_idx][1]
            lr_idx = lr_slot_by_suit.get(suit)
            if lr_idx is None:
                continue
            vecs[entry_idx, ph_idx] -= count
            vecs[entry_idx, lr_idx] += count

    return vecs


def _build_hand_context(hand: list, level_rank: Optional[str]) -> dict:
    plain_ranks = defaultdict(list)
    seq_ranks   = defaultdict(list)
    suit_ranks  = defaultdict(lambda: defaultdict(list))
    lr_cards    = []
    black_jokers, red_jokers, wildcards = [], [], []

    for c in hand:
        if is_joker(c):
            (red_jokers if c['joker'] == 'red' else black_jokers).append(c)
            continue
        if is_wildcard(c, level_rank):
            wildcards.append(c)
            continue
        seq_ranks[c['rank']].append(c)
        suit_ranks[c['suit']][c['rank']].append(c)
        if level_rank is not None and c['rank'] == level_rank and c['suit'] != 'hearts':
            lr_cards.append(c)
        else:
            plain_ranks[c['rank']].append(c)

    return {
        'plain_ranks': plain_ranks,
        'seq_ranks': seq_ranks,
        'suit_ranks': suit_ranks,
        'lr_cards': lr_cards,
        'black_jokers': black_jokers,
        'red_jokers': red_jokers,
        'wildcards': wildcards,
    }


def _entry_window(entry: dict) -> list:
    if entry['type'] in {'STRAIGHT', 'STRAIGHT_FLUSH'}:
        if entry['strength'] == 5:
            return ['A', '2', '3', '4', '5']
        hi = RANKS.index(entry['rank_key'])
        return RANKS[hi-4:hi+1]
    if entry['type'] == 'SEQ_3_PAIRS':
        hi = RANKS.index(entry['rank_key'])
        return RANKS[hi-2:hi+1]
    if entry['type'] == 'SEQ_2_TRIPLES':
        hi = RANKS.index(entry['rank_key'])
        return RANKS[hi-1:hi+1]
    return []


def _entry_straight_flush_suit(entry: dict) -> Optional[str]:
    for slot_idx in np.where(entry['vec_template'] > 0)[0]:
        slot = CANONICAL[slot_idx]
        if slot[0] == 'normal':
            return slot[1]
    return None


def _entry_cards_from_context(entry: dict, ctx: dict) -> list:
    wildcards = list(ctx['wildcards'])
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

    if entry['type'] == 'PASS':
        return []

    if entry['type'] == 'SINGLE':
        if entry['rank_key'] == 'BJ':
            return result if take_from(list(ctx['black_jokers']), 1) else []
        if entry['rank_key'] == 'RJ':
            return result if take_from(list(ctx['red_jokers']), 1) else []
        if entry['rank_key'] == 'LR':
            return result if take_from(list(ctx['lr_cards']), 1) else []
        return result if take_from(list(ctx['plain_ranks'][entry['rank_key']]), 1) else []

    if entry['type'] == 'PAIR':
        if entry['rank_key'] == 'BJ':
            return result if take_from(list(ctx['black_jokers']), 2) else []
        if entry['rank_key'] == 'RJ':
            return result if take_from(list(ctx['red_jokers']), 2) else []
        if entry['rank_key'] == 'LR':
            return result if take_from(list(ctx['lr_cards']), 2) else []
        return result if take_from(list(ctx['plain_ranks'][entry['rank_key']]), 2) else []

    if entry['type'] == 'TRIPLE':
        pool = list(ctx['lr_cards']) if entry['rank_key'] == 'LR' else list(ctx['plain_ranks'][entry['rank_key']])
        return result if take_from(pool, 3) else []

    if entry['type'] == 'FULL_HOUSE':
        triple_pool = list(ctx['lr_cards']) if entry['rank_key'] == 'LR' else list(ctx['plain_ranks'][entry['rank_key']])
        pair_key = entry['pair_rank_key']
        pair_pool = list(ctx['lr_cards']) if pair_key == 'LR' else list(ctx['plain_ranks'][pair_key])
        if entry['rank_key'] == pair_key:
            return []
        if not take_from(triple_pool, 3):
            return []
        if not take_from(pair_pool, 2):
            return []
        return result

    if entry['type'] == 'STRAIGHT':
        for rank in _entry_window(entry):
            if not take_from(list(ctx['seq_ranks'][rank]), 1):
                return []
        return result

    if entry['type'] == 'SEQ_3_PAIRS':
        for rank in _entry_window(entry):
            if not take_from(list(ctx['seq_ranks'][rank]), 2):
                return []
        return result

    if entry['type'] == 'SEQ_2_TRIPLES':
        for rank in _entry_window(entry):
            if not take_from(list(ctx['seq_ranks'][rank]), 3):
                return []
        return result

    if entry['type'] in {'FOUR_OF_A_KIND', 'FIVE_OF_A_KIND', 'SIX_OF_A_KIND',
                         'SEVEN_OF_A_KIND', 'EIGHT_OF_A_KIND'}:
        need = {'FOUR_OF_A_KIND': 4, 'FIVE_OF_A_KIND': 5, 'SIX_OF_A_KIND': 6,
                'SEVEN_OF_A_KIND': 7, 'EIGHT_OF_A_KIND': 8}[entry['type']]
        pool = list(ctx['lr_cards']) if entry['rank_key'] == 'LR' else list(ctx['plain_ranks'][entry['rank_key']])
        return result if take_from(pool, need) else []

    if entry['type'] == 'STRAIGHT_FLUSH':
        suit = _entry_straight_flush_suit(entry)
        for rank in _entry_window(entry):
            if not take_from(list(ctx['suit_ranks'][suit][rank]), 1):
                return []
        return result

    if entry['type'] == 'FOUR_JOKERS':
        if not take_from(list(ctx['red_jokers']), 2):
            return []
        if not take_from(list(ctx['black_jokers']), 2):
            return []
        return result

    return []


def compute_legal_mask(hand: list, trick_last: list, level_rank: Optional[str],
                       is_opener: bool, last_info: Optional[dict] = None) -> np.ndarray:
    """
    Returns a boolean array of shape (ACTION_DIM,) where True = legal action.

    Algorithm:
      1. Build per-hand rank/joker/wildcard buckets.
      2. Check each catalogue entry against those buckets.
      3. For non-openers: further filter to combos that beat trick_last.
      4. PASS (index 0) is legal for non-openers only.
    """
    ctx = _build_hand_context(hand, level_rank)
    reachable = np.zeros(ACTION_DIM, dtype=bool)
    for i, entry in enumerate(_CATALOGUE):
        if level_rank == '2' and CAT_INVALID_WHEN_LEVEL_IS_2[i]:
            continue
        cards = _entry_cards_from_context(entry, ctx)
        reachable[i] = (entry['type'] == 'PASS') or (len(cards) == CAT_CARD_COUNTS[i])

    if is_opener:
        # Openers can play any reachable combo; PASS is illegal
        mask = reachable.copy()
        mask[0] = False   # no PASS as opener
        return mask

    # Non-opener: must beat trick_last or PASS
    if last_info is None:
        last_info = _detect_from_hand(trick_last, level_rank) if trick_last else None
    beat_mask = np.zeros(ACTION_DIM, dtype=bool)
    beat_mask[0] = True   # PASS always legal for non-opener

    for i in np.where(reachable)[0]:
        if i == 0:
            continue
        entry = _CATALOGUE[i]
        if _entry_beats_last(entry, last_info, level_rank):
            beat_mask[i] = True

    return beat_mask


def _detect_from_hand(cards: list, level_rank: Optional[str]) -> Optional[dict]:
    """
    Quick combo detection for trick_last — returns the catalogue entry dict
    (or a synthetic one for unusual combos).  Used only for 'can_beat' checks.
    """
    if not cards:
        return None
    target_vec = hand_to_multihot(cards)
    ctx = _build_hand_context(cards, level_rank)
    candidate_idx = np.where(CAT_CARD_COUNTS == len(cards))[0]
    for idx in candidate_idx:
        if level_rank == '2' and CAT_INVALID_WHEN_LEVEL_IS_2[int(idx)]:
            continue
        played = _entry_cards_from_context(_CATALOGUE[int(idx)], ctx)
        if len(played) != len(cards):
            continue
        if np.array_equal(hand_to_multihot(played), target_vec):
            return _entry_runtime_info(_CATALOGUE[int(idx)])

    return _detect_combo(cards, level_rank)


def _detect_combo(cards: list, level_rank: Optional[str]) -> dict:
    """
    Minimal combo detection returning a dict with type/strength/is_bomb/bomb_strength.
    Used for trick_last comparison only — not used for action generation.
    """
    n      = len(cards)
    jokers = [c for c in cards if is_joker(c)]
    wcs    = [c for c in cards if is_wildcard(c, level_rank)]
    normal = [c for c in cards if not is_joker(c) and not is_wildcard(c, level_rank)]

    # Four jokers
    if n == 4 and len(jokers) == 4:
        reds   = sum(1 for j in jokers if j['joker'] == 'red')
        blacks = sum(1 for j in jokers if j['joker'] == 'black')
        if reds == 2 and blacks == 2:
            return {'type':'FOUR_JOKERS','strength':1000,'is_bomb':True,'bomb_strength':1000}

    # Bombs: n-of-a-kind (4-8); wildcards count toward the n
    if 4 <= n <= 8 and not jokers:
        rc = count_ranks(normal)
        if len(rc) == 1:
            r     = list(rc.keys())[0]
            # Is it a level-rank bomb?
            if level_rank and r == level_rank:
                bs = n * 100 + 16
            else:
                bs = n * 100 + RANK_VAL.get(r, 0)
            bomb_type = {4:'FOUR_OF_A_KIND',5:'FIVE_OF_A_KIND',6:'SIX_OF_A_KIND',
                         7:'SEVEN_OF_A_KIND',8:'EIGHT_OF_A_KIND'}
            return {'type':bomb_type[n],'strength':16 if (level_rank and r==level_rank) else RANK_VAL.get(r,0),
                    'is_bomb':True,'bomb_strength':bs}
        # could be achieved via wildcards — check
        if len(wcs) > 0 and len(rc) + len(wcs) >= 1:
            pass  # handled via mask; for trick detection we accept approximate

    # Straight flush
    if n == 5 and not jokers:
        suits_set = {c['suit'] for c in normal + wcs}
        if len({c['suit'] for c in normal}) == 1 and not wcs:
            rv = _check_straight_ranks([c['rank'] for c in cards])
            if rv is not None:
                bs = 550 + rv
                return {'type':'STRAIGHT_FLUSH','strength':rv,'is_bomb':True,'bomb_strength':bs}

    # Single
    if n == 1:
        c = cards[0]
        if is_joker(c):
            s = 18 if c['joker'] == 'red' else 17
        else:
            s = card_rank_val(c, level_rank)
        return {'type':'SINGLE','strength':s,'is_bomb':False,'bomb_strength':0}

    # Pair
    if n == 2:
        reds   = [c for c in cards if is_joker(c) and c['joker'] == 'red']
        blacks = [c for c in cards if is_joker(c) and c['joker'] == 'black']
        if len(reds) == 2:
            return {'type':'PAIR','strength':18,'is_bomb':False,'bomb_strength':0}
        if len(blacks) == 2:
            return {'type':'PAIR','strength':17,'is_bomb':False,'bomb_strength':0}
        if len(normal) == 2 and normal[0]['rank'] == normal[1]['rank']:
            rv = card_rank_val(normal[0], level_rank)
            return {'type':'PAIR','strength':rv,'is_bomb':False,'bomb_strength':0}
        # wildcard pair
        if len(normal) == 1 and len(wcs) == 1:
            rv = card_rank_val(normal[0], level_rank)
            return {'type':'PAIR','strength':rv,'is_bomb':False,'bomb_strength':0}

    # Triple
    if n == 3 and not jokers:
        all_cards = normal + wcs
        rc = count_ranks(normal)
        if len(rc) == 1:
            rv = card_rank_val(normal[0], level_rank)
            return {'type':'TRIPLE','strength':rv,'is_bomb':False,'bomb_strength':0}

    # Straight
    if n == 5:
        all_ranks = [c['rank'] for c in normal + wcs]
        rv = _check_straight_ranks(all_ranks)
        if rv is not None:
            return {'type':'STRAIGHT','strength':rv,'is_bomb':False,'bomb_strength':0}

    # Full house
    if n == 5 and not jokers:
        rc     = count_ranks(normal + wcs)
        counts = sorted(rc.values(), reverse=True)
        if counts[:2] == [3, 2] and len(rc) == 2:
            t_rank = next(r for r, cnt in rc.items() if cnt == 3)
            rv = card_rank_val(make_card('spades', t_rank), level_rank)
            return {'type':'FULL_HOUSE','strength':rv,'is_bomb':False,'bomb_strength':0}

    # Seq 3 pairs
    if n == 6 and not jokers:
        rc    = count_ranks(normal)
        pairs = [r for r, cnt in rc.items() if cnt == 2]
        if len(pairs) == 3:
            rv = _check_rank_straight_n(pairs, 3)
            if rv:
                return {'type':'SEQ_3_PAIRS','strength':rv,'is_bomb':False,'bomb_strength':0}

    # Seq 2 triples
    if n == 6 and not jokers:
        rc    = count_ranks(normal)
        trips = [r for r, cnt in rc.items() if cnt == 3]
        if len(trips) == 2:
            rv = _check_rank_straight_n(trips, 2)
            if rv:
                return {'type':'SEQ_2_TRIPLES','strength':rv,'is_bomb':False,'bomb_strength':0}

    return {'type':'INVALID','strength':0,'is_bomb':False,'bomb_strength':0}


def _check_rank_straight_n(rank_list: List[str], n: int) -> Optional[int]:
    """Returns highest RANK_VAL if rank_list forms n consecutive ranks, else None."""
    if len(rank_list) != n:
        return None
    for ace_low in ([True, False] if 'A' in rank_list else [False]):
        def rv(r):
            return 1 if (r == 'A' and ace_low) else RANK_VAL.get(r, 0)
        vals = sorted(rv(r) for r in rank_list)
        if all(vals[i]+1 == vals[i+1] for i in range(len(vals)-1)):
            if ace_low and vals[-1] == 14:
                continue
            if not ace_low and vals[0] == 1:
                continue
            return vals[-1]
    return None


def _entry_runtime_info(entry: dict) -> dict:
    info = {
        'type': entry['type'],
        'strength': entry['strength'],
        'is_bomb': entry['is_bomb'],
        'bomb_strength': entry['bomb_strength'],
    }
    if entry['rank_key'] == 'LR':
        info['strength'] = 16
        if entry['is_bomb']:
            n = {'FOUR_OF_A_KIND':4,'FIVE_OF_A_KIND':5,'SIX_OF_A_KIND':6,
                 'SEVEN_OF_A_KIND':7,'EIGHT_OF_A_KIND':8}.get(entry['type'])
            if n is not None:
                info['bomb_strength'] = n * 100 + 16
    return info


def _entry_beats_last(entry: dict, last_info: Optional[dict],
                      level_rank: Optional[str]) -> bool:
    """Can catalogue entry 'entry' beat the last played combination?"""
    if last_info is None or last_info['type'] == 'INVALID':
        return True
    if entry['type'] == 'PASS':
        return False

    current = _entry_runtime_info(entry)
    e_strength = current['strength']
    e_bomb_str = current['bomb_strength']
    e_is_bomb = current['is_bomb']
    l_is_bomb = last_info['is_bomb']

    if e_is_bomb and not l_is_bomb:
        return True
    if not e_is_bomb and l_is_bomb:
        return False
    if e_is_bomb and l_is_bomb:
        return e_bomb_str > last_info['bomb_strength']

    # Neither is bomb: must be same type AND same card count
    if entry['type'] != last_info['type']:
        return False

    return e_strength > last_info['strength']


def action_index_to_cards(action_idx: int, hand: list,
                           level_rank: Optional[str]) -> list:
    """
    Given a catalogue action index and the current hand, return the actual
    card objects to play (with wildcards filling any shortfall).
    Returns [] for PASS (index 0).
    """
    if action_idx == 0:
        return []

    if level_rank == '2' and CAT_INVALID_WHEN_LEVEL_IS_2[action_idx]:
        return []

    entry = _CATALOGUE[action_idx]
    cards = _entry_cards_from_context(entry, _build_hand_context(hand, level_rank))
    return cards if len(cards) == CAT_CARD_COUNTS[action_idx] else []


# ══════════════════════════════════════════════════════════════════════════════
#  State encoder  (282-dim)
# ══════════════════════════════════════════════════════════════════════════════

_SF_WINDOWS = [['A','2','3','4','5']] + [RANKS[i:i+5] for i in range(len(RANKS)-4)]

def _straight_flush_features(hand: list) -> np.ndarray:
    """40-dim auxiliary: one bit per (suit, 10-window) indicating complete SF."""
    feat = np.zeros(STRAIGHT_FLUSH_FEATURE_DIM, dtype=np.float32)
    idx  = 0
    rank_set_by_suit = defaultdict(set)
    for c in hand:
        if not is_joker(c):
            rank_set_by_suit[c['suit']].add(c['rank'])
    for suit in SUITS:
        for window in _SF_WINDOWS:
            if all(r in rank_set_by_suit[suit] for r in window):
                feat[idx] = 1.0
            idx += 1
    assert idx == STRAIGHT_FLUSH_FEATURE_DIM
    return feat


def encode_state(hand: list, trick_last: list, played_cards: list, level_rank: Optional[str],
                 my_seat: int, hand_sizes: list,
                 level_ranks: Tuple[float, float], is_opener: bool) -> np.ndarray:
    vec    = np.zeros(STATE_DIM, dtype=np.float32)
    offset = 0

    vec[offset:offset+108] = hand_to_multihot(hand);       offset += 108
    vec[offset:offset+108] = hand_to_multihot(trick_last); offset += 108
    vec[offset:offset+108] = hand_to_multihot(played_cards); offset += 108

    if level_rank in RANKS:
        vec[offset + RANKS.index(level_rank)] = 1.0
    offset += 15

    vec[offset + my_seat] = 1.0;  offset += 4

    for i, sz in enumerate(hand_sizes):
        vec[offset + i] = sz / 27.0
    offset += 4

    vec[offset]   = level_ranks[0]
    vec[offset+1] = level_ranks[1]
    offset += 2

    vec[offset] = float(is_opener); offset += 1

    vec[offset:offset+STRAIGHT_FLUSH_FEATURE_DIM] = _straight_flush_features(hand)

    return vec


def _upgrade_legacy_model_state(model_state: dict) -> Tuple[dict, bool]:
    """Zero-pad the first layer if loading a checkpoint with an older state size."""
    first_layer_key = 'backbone.0.weight'
    weight = model_state.get(first_layer_key)
    if weight is None or weight.shape[1] == STATE_DIM:
        return model_state, False
    if weight.shape[1] > STATE_DIM:
        raise ValueError(
            f"Checkpoint expects {weight.shape[1]} input features, "
            f"but current model uses {STATE_DIM}."
        )

    upgraded_state = dict(model_state)
    padded = weight.new_zeros((weight.shape[0], STATE_DIM))
    padded[:, :weight.shape[1]] = weight
    upgraded_state[first_layer_key] = padded
    return upgraded_state, True


# ══════════════════════════════════════════════════════════════════════════════
#  Neural network
# ══════════════════════════════════════════════════════════════════════════════

class GuandanNet(nn.Module):
    def __init__(self, input_size=STATE_DIM, action_size=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, state, mask=None):
        f      = self.backbone(state)
        logits = self.policy_head(f)
        value  = self.value_head(f)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits, value

    def batch_act(self, states_np: np.ndarray,
                  masks_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        states_np : (B, STATE_DIM)  float32
        masks_np  : (B, ACTION_DIM) bool
        Returns actions (B,), log_probs (B,), values (B,) — all numpy
        """
        s = torch.from_numpy(states_np).to(DEVICE, non_blocking=True)
        m = torch.from_numpy(masks_np).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            logits, values = self.forward(s, m)
        dist     = torch.distributions.Categorical(logits=logits)
        actions  = dist.sample()
        log_probs = dist.log_prob(actions)
        return (actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.squeeze(-1).cpu().numpy())

    def evaluate_actions(self, states, masks, actions):
        logits, values = self.forward(states, masks)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy().mean()
        return log_probs, values.squeeze(-1), entropy


def critic_state_value(net: GuandanNet, env, seat: int) -> float:
    """Estimate the critic value for one seat in the given environment state."""
    state_np = env.get_state(seat)[None, :]
    states = torch.from_numpy(state_np).to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        _, value = net(states)
    return float(value.squeeze(-1).item())


# ══════════════════════════════════════════════════════════════════════════════
#  PPO Memory
# ══════════════════════════════════════════════════════════════════════════════

class PPOMemory:
    def __init__(self):
        self.states, self.masks, self.actions = [], [], []
        self.log_probs, self.rewards, self.values, self.dones = [], [], [], []

    def push(self, state, mask, action, log_prob, reward, value, done):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def clear(self):
        self.__init__()

    def compute_returns(self, last_value=0.0):
        returns, gae = [], 0.0
        vals = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * vals[t+1] * (1-self.dones[t]) - vals[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1-self.dones[t]) * gae
            returns.insert(0, gae + vals[t])
        return returns

    def __len__(self):
        return len(self.states)


# ══════════════════════════════════════════════════════════════════════════════
#  Game Environment
# ══════════════════════════════════════════════════════════════════════════════

class GuandanEnv:
    def __init__(self):
        self.level_ranks   = {'Blue': 0, 'Red': 0}
        self.a_tries       = {'Blue': 0, 'Red': 0}
        self.caller        = 'Blue'
        self.round_num     = 0
        self._match_winner = None
        self.tribute_seat_value_fn = None
        self.reset_round()

    def team(self, seat): return 'Blue' if seat % 2 == 0 else 'Red'
    def teammate(self, seat): return (seat + 2) % 4

    def active_level_rank(self) -> str:
        return RANKS[self.level_ranks[self.caller]]

    def _resolved_finish_order(self, finish_order: Optional[list] = None) -> List[int]:
        """
        Expand a decisive round result into the full 4-seat order.
        Guandan rounds can end as soon as one team takes first and second,
        so tribute/reward logic needs the remaining seats filled in.
        """
        source = self.finish_order if finish_order is None else finish_order
        if isinstance(source, (int, np.integer)):
            source = [int(source)]

        ordered: List[int] = []
        for seat in source or []:
            try:
                seat = int(seat)
            except (TypeError, ValueError):
                continue
            if 0 <= seat < 4 and seat not in ordered:
                ordered.append(seat)

        if len(ordered) < 2:
            return ordered

        ordered.extend(seat for seat in range(4) if seat not in ordered)
        return ordered

    def reset_round(self):
        prev_finish_order = self._resolved_finish_order(
            getattr(self, 'finish_order', [])
        )
        deck              = build_deck()
        self.hands        = [deck[i*27:(i+1)*27] for i in range(4)]
        self.finish_order = []
        self.played_cards = []
        self.trick_last   = []
        self.trick_info   = None
        self.trick_seat   = None
        self.pass_streak  = 0
        self.current_seat = 0
        self.round_num   += 1
        if len(prev_finish_order) >= 2:
            self.current_seat = self._do_tribute(prev_finish_order)

    def _do_tribute(self, finish_order: Optional[list] = None):
        """Tribute phase; modifies hands in-place. Returns starting seat."""
        lr = self.active_level_rank()
        fo = self._resolved_finish_order(finish_order)
        if len(fo) < 2:
            return self.current_seat

        def strength(c):
            if is_joker(c):
                return (3, 1 if c['joker'] == 'red' else 0)
            if c['rank'] == lr and c['suit'] != 'hearts':
                return (2, 0)
            return (1, RANK_VAL.get(c['rank'], 0))

        def highest_candidates(hand):
            top = max((strength(c) for c in hand), default=None)
            return [c for c in hand if strength(c) == top]

        def valid_return(c):
            if is_joker(c): return False
            if c['rank'] == lr: return False
            return RANK_VAL.get(c['rank'], 0) <= 10

        def return_candidates(hand):
            opts = [c for c in hand if valid_return(c)]
            if opts:
                return opts
            opts = [c for c in hand if not is_joker(c) and c['rank'] != lr]
            if opts:
                return opts
            weakest = min((strength(c) for c in hand), default=None)
            return [c for c in hand if strength(c) == weakest]

        def score_seat(env_state, seat):
            scorer = getattr(self, 'tribute_seat_value_fn', None)
            if not callable(scorer):
                return None
            try:
                score = scorer(env_state, seat)
            except Exception:
                return None
            if score is None or not np.isfinite(score):
                return None
            return float(score)

        def choose_option(env_state, seat, options, simulate_choice):
            if not options:
                return None
            if len(options) == 1:
                return options[0]

            best = options[0]
            best_score = None
            for option in options:
                sim_env = copy.deepcopy(env_state)
                simulate_choice(sim_env, option)
                score = score_seat(sim_env, seat)
                if score is None:
                    return options[0]
                if best_score is None or score > best_score:
                    best = option
                    best_score = score
            return best

        def choose_return_card(env_state, winner, giver):
            candidates = return_candidates(env_state.hands[winner])
            def simulate_return(sim_env, card):
                sim_env.hands[winner].remove(card)
                sim_env.hands[giver].append(copy.deepcopy(card))
            return choose_option(
                env_state,
                winner,
                candidates,
                simulate_return,
            )

        def choose_tribute_card(env_state, giver, winner):
            candidates = highest_candidates(env_state.hands[giver])
            def simulate_tribute(sim_env, card):
                sim_env.hands[giver].remove(card)
                sim_env.hands[winner].append(copy.deepcopy(card))
                ret = choose_return_card(sim_env, winner, giver)
                sim_env.hands[winner].remove(ret)
                sim_env.hands[giver].append(copy.deepcopy(ret))
            return choose_option(
                env_state,
                giver,
                candidates,
                simulate_tribute,
            )

        def resolve_exchange(winner, giver):
            tribute = choose_tribute_card(self, giver, winner)
            self.hands[giver].remove(tribute)
            self.hands[winner].append(tribute)
            ret = choose_return_card(self, winner, giver)
            self.hands[winner].remove(ret)
            self.hands[giver].append(ret)

        def tribute_value(hand):
            candidates = highest_candidates(hand)
            return card_rank_val(candidates[0], lr) if candidates else -1

        def has_two_red_jokers(seats):
            return sum(
                sum(1 for c in self.hands[s] if is_joker(c) and c['joker'] == 'red')
                for s in seats
            ) >= 2

        both_same = len(fo) >= 2 and self.team(fo[0]) == self.team(fo[1])

        if both_same:
            tributers = [fo[2], fo[3]]
            if has_two_red_jokers(tributers):
                return fo[0]
            s0 = tribute_value(self.hands[tributers[0]])
            s1 = tribute_value(self.hands[tributers[1]])
            if s0 >= s1:
                pairs   = [(fo[0], fo[2]), (fo[1], fo[3])]
                starter = fo[2]
            else:
                pairs   = [(fo[0], fo[3]), (fo[1], fo[2])]
                starter = fo[3]
            for winner, giver in pairs:
                resolve_exchange(winner, giver)
            return starter
        else:
            tributer = fo[3]
            if has_two_red_jokers([tributer]):
                return fo[0]
            resolve_exchange(fo[0], tributer)
            return tributer   # tributer starts

    def apply_action(self, seat: int, action_cards: list, action_idx: Optional[int] = None):
        """Returns (rewards_dict, done)."""
        if action_cards:
            for c in action_cards:
                self.hands[seat].remove(c)
            self.trick_last = action_cards
            self.played_cards.extend(copy.deepcopy(c) for c in action_cards)
            if action_idx is not None and action_idx != 0:
                self.trick_info = _entry_runtime_info(_CATALOGUE[action_idx])
            else:
                self.trick_info = _detect_from_hand(action_cards, self.active_level_rank())
            self.trick_seat = seat
            self.pass_streak = 0
        else:
            self.pass_streak += 1

        if len(self.hands[seat]) == 0 and seat not in self.finish_order:
            self.finish_order.append(seat)

        done = self._check_round_done()
        if done:
            return self._compute_rewards(), True

        next_seat = (seat + 1) % 4
        for _ in range(4):
            if next_seat not in self.finish_order:
                break
            next_seat = (next_seat + 1) % 4

        active_players = [s for s in range(4) if s not in self.finish_order]
        passes_needed = len(active_players) - (0 if self.trick_seat in self.finish_order else 1)
        if self.trick_seat is not None and self.pass_streak >= max(passes_needed, 0):
            self.trick_last = []
            self.trick_info = None
            self.trick_seat = None
            self.pass_streak = 0

        self.current_seat = next_seat
        return {s: 0.0 for s in range(4)}, False

    def _check_round_done(self):
        if len(self.finish_order) < 2:
            return False
        if self.team(self.finish_order[0]) == self.team(self.finish_order[1]):
            return True
        return len(self.finish_order) == 4

    def _compute_rewards(self):
        fo = self._resolved_finish_order()
        if len(fo) < 2:
            raise RuntimeError("Cannot score a round without at least two finishers.")
        first_team  = self.team(fo[0])
        second_team = self.team(fo[1]) if len(fo) > 1 else None

        if first_team == second_team:
            rank_gain = 3
        elif len(fo) > 2 and self.team(fo[0]) == self.team(fo[2]):
            rank_gain = 2
        else:
            rank_gain = 1

        winning_team = first_team
        self._advance_rank(winning_team, rank_gain)
        self.caller = winning_team

        place_bonus  = [1.5, 0.5, 0.5, 1.5]
        match_bonus  = 5.0
        match_winner = self.is_match_won()
        rewards = {}
        for place, seat in enumerate(fo):
            sign  = 1.0 if self.team(seat) == winning_team else -1.0
            bonus = match_bonus if (match_winner and self.team(seat) == match_winner) else 0.0
            rewards[seat] = sign * place_bonus[place] + bonus
        return rewards

    def _advance_rank(self, winning_team, gain):
        losing_team = 'Red' if winning_team == 'Blue' else 'Blue'
        self._match_winner = None
        max_rank = len(RANKS) - 1

        at_A      = self.level_ranks[winning_team] == max_rank
        is_caller = self.caller == winning_team

        if at_A and is_caller:
            if gain >= 2:
                self._match_winner = winning_team
                return
            else:
                self.a_tries[winning_team] += 1
                if self.a_tries[winning_team] >= 3:
                    self.level_ranks[winning_team] = 0
                    self.a_tries[winning_team]     = 0
                return

        losing_at_A = self.level_ranks[losing_team] == max_rank
        if losing_at_A and self.caller == losing_team:
            self.a_tries[losing_team] += 1
            if self.a_tries[losing_team] >= 3:
                self.level_ranks[losing_team] = 0
                self.a_tries[losing_team]     = 0

        self.level_ranks[winning_team] = min(
            self.level_ranks[winning_team] + gain, max_rank)

    def is_match_won(self):
        return getattr(self, '_match_winner', None)

    def get_state(self, seat: int) -> np.ndarray:
        lr = self.active_level_rank()
        return encode_state(
            hand        = self.hands[seat],
            trick_last  = self.trick_last,
            played_cards = self.played_cards,
            level_rank  = lr,
            my_seat     = seat,
            hand_sizes  = [len(self.hands[s]) for s in range(4)],
            level_ranks = (self.level_ranks['Blue'] / 12.0,
                           self.level_ranks['Red']  / 12.0),
            is_opener   = (not self.trick_last),
        )

    def get_legal_mask(self, seat: int) -> np.ndarray:
        return compute_legal_mask(
            hand       = self.hands[seat],
            trick_last = self.trick_last,
            level_rank = self.active_level_rank(),
            is_opener  = (not self.trick_last),
            last_info  = self.trick_info,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorised environment pool  (Windows-safe: no shared memory, pipe-based)
# ══════════════════════════════════════════════════════════════════════════════

class EnvPool:
    """
    Manages NUM_ENVS GuandanEnv instances on the main process.
    On Windows, spawning subprocesses for RL envs with complex Python objects
    is slow and error-prone due to pickling.  Instead we keep all envs on the
    main process and parallelise the CPU-bound mask computation using a thread
    pool (GIL is released during numpy operations).

    The key throughput gain comes from BATCHED GPU inference: we collect
    states+masks from all active seats across all envs, do one forward pass,
    then distribute actions back.
    """
    def __init__(self, n: int = NUM_ENVS):
        self.envs     = [GuandanEnv() for _ in range(n)]
        self.n        = n
        # per-env step buffers  {seat: [(state, mask, action_idx, log_p, value)]}
        self.step_buf = [{s: [] for s in range(4)} for _ in range(n)]
        # per-env per-seat accumulated reward (for final assignment)
        self.ep_rewards = [{s: 0.0 for s in range(4)} for _ in range(n)]

    def reset_all(self):
        for i, env in enumerate(self.envs):
            env.reset_round()
            self.step_buf[i]    = {s: [] for s in range(4)}
            self.ep_rewards[i]  = {s: 0.0 for s in range(4)}

    def collect_batch(self):
        """
        Gather (state, mask) for every active seat in every env.
        Returns:
          states  : (B, STATE_DIM) float32
          masks   : (B, ACTION_DIM) bool
          indices : list of (env_idx, seat) — same order as rows
        """
        states_list, masks_list, indices = [], [], []
        for ei, env in enumerate(self.envs):
            seat      = env.current_seat
            state_vec = env.get_state(seat)
            mask      = env.get_legal_mask(seat)
            states_list.append(state_vec)
            masks_list.append(mask)
            indices.append((ei, seat))
        states = np.stack(states_list, axis=0).astype(np.float32)
        masks  = np.stack(masks_list,  axis=0)
        return states, masks, indices

    def step_batch(self, actions: np.ndarray, log_probs: np.ndarray,
                   values: np.ndarray, indices: list,
                   states_np: np.ndarray, masks_np: np.ndarray,
                   memory: PPOMemory) -> Tuple[int, float]:
        """
        Apply actions to all envs.
        Returns (n_done_rounds, total_ep_reward_across_done).
        """
        n_done   = 0
        total_r  = 0.0

        for k, (ei, seat) in enumerate(indices):
            env        = self.envs[ei]
            action_idx = int(actions[k])
            lr         = env.active_level_rank()
            cards      = action_index_to_cards(action_idx, env.hands[seat], lr)

            state_vec  = states_np[k]
            mask       = masks_np[k]

            self.step_buf[ei][seat].append(
                (state_vec, mask, action_idx, log_probs[k], values[k]))

            rewards, done = env.apply_action(seat, cards, action_idx=action_idx)

            if done:
                n_done += 1
                for s in range(4):
                    r = rewards.get(s, 0.0)
                    self.ep_rewards[ei][s] += r
                    for t, (sv, mk, ai, lp, vl) in enumerate(self.step_buf[ei][s]):
                        is_last = (t == len(self.step_buf[ei][s]) - 1)
                        memory.push(sv, mk, ai, lp,
                                    r if is_last else 0.0,
                                    vl, 1.0 if is_last else 0.0)
                total_r += sum(self.ep_rewards[ei].values()) / 4
                # Reset this env
                if env.is_match_won():
                    env.level_ranks   = {'Blue': 0, 'Red': 0}
                    env.a_tries       = {'Blue': 0, 'Red': 0}
                    env.caller        = 'Blue'
                    env.round_num     = 0
                    env._match_winner = None
                    env.finish_order  = []
                env.reset_round()
                self.step_buf[ei]   = {s: [] for s in range(4)}
                self.ep_rewards[ei] = {s: 0.0 for s in range(4)}

        return n_done, total_r


# ══════════════════════════════════════════════════════════════════════════════
#  PPO Trainer
# ══════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    def __init__(self, net: GuandanNet):
        self.net = net
        self.opt = optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    def update(self, memory: PPOMemory) -> float:
        returns    = memory.compute_returns()
        states     = torch.from_numpy(np.array(memory.states, dtype=np.float32)).to(DEVICE, non_blocking=True)
        masks      = torch.from_numpy(np.array(memory.masks,  dtype=bool)).to(DEVICE, non_blocking=True)
        actions    = torch.tensor(memory.actions,   dtype=torch.long,  device=DEVICE)
        old_lp     = torch.tensor(memory.log_probs, dtype=torch.float32, device=DEVICE)
        returns_t  = torch.tensor(returns,           dtype=torch.float32, device=DEVICE)
        values_t   = torch.tensor(memory.values,     dtype=torch.float32, device=DEVICE)
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n          = len(memory)
        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(n, device=DEVICE)
            for start in range(0, n, BATCH_SIZE):
                batch     = idx[start:start+BATCH_SIZE]
                lp, vals, entropy = self.net.evaluate_actions(
                    states[batch], masks[batch], actions[batch])
                ratio      = torch.exp(lp - old_lp[batch])
                adv        = advantages[batch]
                surr1      = ratio * adv
                surr2      = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(vals, returns_t[batch])
                loss        = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                self.opt.step()
                total_loss += loss.item()

        memory.clear()
        return total_loss


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs('checkpoints', exist_ok=True)
    print(f"Device      : {DEVICE}")
    print(f"State dim   : {STATE_DIM}")
    print(f"Action dim  : {ACTION_DIM}")
    print(f"Num envs    : {NUM_ENVS}")
    print(f"Batch size  : {BATCH_SIZE}")

    net     = GuandanNet().to(DEVICE)
    trainer = PPOTrainer(net)
    pool    = EnvPool(NUM_ENVS)
    memory  = PPOMemory()
    seat_value_fn = lambda env_state, seat: critic_state_value(net, env_state, seat)
    for env in pool.envs:
        env.tribute_seat_value_fn = seat_value_fn

    # ── Load checkpoint ────────────────────────────────────────────────────────
    start_episode = 0
    ckpt_dir      = 'checkpoints'
    ckpt_files    = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')],
        key=lambda f: int(f.replace('guandan_ep','').replace('.pt',''))
    ) if os.path.isdir(ckpt_dir) else []

    if ckpt_files:
        latest = os.path.join(ckpt_dir, ckpt_files[-1])
        ckpt   = torch.load(latest, map_location=DEVICE)
        model_state, upgraded_legacy_state = _upgrade_legacy_model_state(ckpt['model_state'])
        net.load_state_dict(model_state)
        if upgraded_legacy_state:
            print(f"Checkpoint {latest} uses the legacy 278-dim state; padded new features with zeros.")
            print("Optimizer state skipped because the input layer shape changed.")
        else:
            trainer.opt.load_state_dict(ckpt['optim_state'])
        start_episode = ckpt['episode']
        # Restore game state to the first env (representative)
        if 'level_ranks' in ckpt:
            for env in pool.envs:
                env.level_ranks = copy.deepcopy(ckpt['level_ranks'])
                env.a_tries     = copy.deepcopy(ckpt.get('a_tries', {'Blue':0,'Red':0}))
                env.caller      = ckpt.get('caller', 'Blue')
        print(f"Resumed from {latest}  (step {start_episode})")
    else:
        print("No checkpoint — starting fresh.")

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters  : {total_params:,}\n")

    pool.reset_all()

    ep_rewards_log = []
    rounds_done    = 0
    t0             = time.time()
    step           = start_episode

    # Each "step" = one batch of actions across all envs
    while step < 500_000:
        step += 1

        # ── Collect states & masks from all envs (one seat per env) ───────────
        states_np, masks_np, indices = pool.collect_batch()

        # ── Batched GPU inference ──────────────────────────────────────────────
        actions, log_probs, values = net.batch_act(states_np, masks_np)

        # ── Step all envs, record experience ──────────────────────────────────
        n_done, avg_r = pool.step_batch(
            actions, log_probs, values, indices, states_np, masks_np, memory)
        rounds_done  += n_done
        if n_done > 0:
            ep_rewards_log.append(avg_r / max(n_done, 1))

        # ── PPO update when buffer is large enough ─────────────────────────────
        if len(memory) >= BATCH_SIZE:
            trainer.update(memory)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % (LOG_INTERVAL) == 0 and ep_rewards_log:
            recent = ep_rewards_log[-LOG_INTERVAL:]
            avg    = np.mean(recent)
            elapsed = time.time() - t0
            # Use first env as representative for rank display
            env0   = pool.envs[0]
            lr_b   = RANKS[env0.level_ranks['Blue']]
            lr_r   = RANKS[env0.level_ranks['Red']]
            print(f"Step {step:7d} | rounds {rounds_done:6d} | "
                  f"avg_reward {avg:+.3f} | "
                  f"Blue:{lr_b} Red:{lr_r} | {elapsed:.0f}s")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % (SAVE_INTERVAL) == 0:
            env0 = pool.envs[0]
            path = f'checkpoints/guandan_ep{step}.pt'
            torch.save({
                'episode':     step,
                'model_state': net.state_dict(),
                'optim_state': trainer.opt.state_dict(),
                'level_ranks': env0.level_ranks,
                'a_tries':     env0.a_tries,
                'caller':      env0.caller,
            }, path)
            print(f"  ✓ Saved {path}")


if __name__ == '__main__':
    # Windows requires this guard for multiprocessing safety
    mp.freeze_support()
    main()
