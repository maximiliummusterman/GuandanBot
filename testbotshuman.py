"""
Interactive checkpoint tester for Guandan.

Play one seat yourself while a single checkpoint controls the other three seats.
The script shows sorted hands for readability, and it reveals each bot hand
before revealing the move that bot chooses.

Examples:
    python testbotshuman.py
    python testbotshuman.py --checkpoint checkpoints/guandan_ep48000.pt
    python testbotshuman.py --seat 1 --sample
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from guandan_arena import (  # noqa: E402
    CAT_CARD_COUNTS,
    DEVICE,
    GuandanEnv,
    GuandanNet,
    RANKS,
    _CATALOGUE,
    _upgrade_legacy_model_state,
    action_index_to_cards,
    critic_state_value,
    is_joker,
    is_wildcard,
)


TYPE_LABELS = {
    "PASS": "Pass",
    "SINGLE": "Single",
    "PAIR": "Pair",
    "TRIPLE": "Triple",
    "STRAIGHT": "Straight",
    "FULL_HOUSE": "Full House",
    "SEQ_3_PAIRS": "3 Consecutive Pairs",
    "SEQ_2_TRIPLES": "2 Consecutive Triples",
    "FOUR_OF_A_KIND": "Four of a Kind Bomb",
    "FIVE_OF_A_KIND": "Five of a Kind Bomb",
    "SIX_OF_A_KIND": "Six of a Kind Bomb",
    "SEVEN_OF_A_KIND": "Seven of a Kind Bomb",
    "EIGHT_OF_A_KIND": "Eight of a Kind Bomb",
    "STRAIGHT_FLUSH": "Straight Flush Bomb",
    "FOUR_JOKERS": "Four Jokers Bomb",
}

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

SUIT_ABBREV = {
    "hearts": "H",
    "diamonds": "D",
    "clubs": "C",
    "spades": "S",
}

SUIT_SORT_ORDER = {
    "hearts": 0,
    "diamonds": 1,
    "clubs": 2,
    "spades": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Play an interactive Guandan match against three bots controlled "
            "by a checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to the latest .pt file in checkpoints/.",
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=0,
        help="Your seat number (0-3). Seats 0/2 are Blue, 1/3 are Red.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducible shuffles.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample bot moves from the policy instead of using greedy argmax.",
    )
    parser.add_argument(
        "--no-bot-pause",
        action="store_true",
        help="Do not pause after showing a bot hand before revealing its move.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Safety cap on total actions before aborting the match.",
    )
    args = parser.parse_args()
    if args.seat not in {0, 1, 2, 3}:
        parser.error("--seat must be 0, 1, 2, or 3.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be at least 1.")
    return args


def extract_episode_num(path: Path):
    match = re.search(r"ep(\d+)", path.stem)
    return int(match.group(1)) if match else None


def resolve_checkpoint_path(explicit_path):
    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            "Could not find checkpoints/. Pass --checkpoint explicitly."
        )

    checkpoint_paths = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda path: (
            extract_episode_num(path) is None,
            extract_episode_num(path) or 0,
            path.name,
        ),
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No checkpoint files found in checkpoints/. Pass --checkpoint explicitly."
        )
    return str(checkpoint_paths[-1])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state': {checkpoint_path}")

    net = GuandanNet().to(DEVICE)
    model_state, upgraded_legacy = _upgrade_legacy_model_state(
        checkpoint["model_state"]
    )
    net.load_state_dict(model_state)
    net.eval()
    return net, checkpoint, upgraded_legacy


def choose_action(env, seat, net, sample=False):
    state_np = env.get_state(seat)[None, :]
    mask_np = env.get_legal_mask(seat)[None, :]

    states = torch.from_numpy(state_np).to(DEVICE, non_blocking=True)
    masks = torch.from_numpy(mask_np).to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        logits, _ = net(states, masks)

    hand = env.hands[seat]
    level_rank = env.active_level_rank()

    def try_action(action_idx):
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


def team_name(seat):
    return "Blue" if seat % 2 == 0 else "Red"


def seat_label(seat, human_seat):
    if seat == human_seat:
        role = "You"
    elif seat == (human_seat + 2) % 4:
        role = "Bot teammate"
    else:
        role = "Bot opponent"
    return f"Seat {seat} ({role}, {team_name(seat)})"


def seat_count_label(seat, human_seat, env):
    owner = "You" if seat == human_seat else "Bot"
    suffix = " OUT" if seat in env.finish_order else ""
    return f"S{seat} {owner}/{team_name(seat)}: {len(env.hands[seat])}{suffix}"


def effective_card_sort_key(card, level_rank):
    if is_joker(card):
        joker_sort = 0 if card["joker"] == "black" else 1
        return (3, joker_sort, 0)
    if is_wildcard(card, level_rank):
        return (2, 0, 0)
    if level_rank is not None and card["rank"] == level_rank:
        return (1, SUIT_SORT_ORDER[card["suit"]], 0)
    return (0, RANKS.index(card["rank"]), SUIT_SORT_ORDER[card["suit"]])


def format_card(card, level_rank):
    if is_joker(card):
        return "RJ" if card["joker"] == "red" else "BJ"
    text = f"{SUIT_ABBREV[card['suit']]}{card['rank']}"
    if is_wildcard(card, level_rank):
        return f"{text}*"
    if level_rank is not None and card["rank"] == level_rank:
        return f"{text}!"
    return text


def format_cards(cards, level_rank):
    if not cards:
        return "-"
    ordered = sorted(cards, key=lambda card: effective_card_sort_key(card, level_rank))
    return " ".join(format_card(card, level_rank) for card in ordered)


def grouped_hand_lines(hand, level_rank):
    ordered = sorted(hand, key=lambda card: effective_card_sort_key(card, level_rank))

    normal_by_rank = {rank: [] for rank in RANKS}
    trumps = []
    wildcards = []
    jokers = []

    for card in ordered:
        text = format_card(card, level_rank)
        if is_joker(card):
            jokers.append(text)
        elif is_wildcard(card, level_rank):
            wildcards.append(text)
        elif level_rank is not None and card["rank"] == level_rank:
            trumps.append(text)
        else:
            normal_by_rank[card["rank"]].append(text)

    lines = []
    for rank in RANKS:
        cards = normal_by_rank[rank]
        if cards:
            lines.append(f"  {rank:<8}{' '.join(cards)}")
    if trumps:
        lines.append(f"  {'TRUMP':<8}{' '.join(trumps)}")
    if wildcards:
        lines.append(f"  {'WILD':<8}{' '.join(wildcards)}")
    if jokers:
        lines.append(f"  {'JOKERS':<8}{' '.join(jokers)}")
    return lines


def print_hand(title, hand, level_rank):
    print(f"{title} ({len(hand)} cards)")
    for line in grouped_hand_lines(hand, level_rank):
        print(line)


def combo_label(action_idx):
    combo_type = _CATALOGUE[action_idx]["type"]
    return TYPE_LABELS.get(combo_type, combo_type.replace("_", " ").title())


def action_strength_key(action_idx):
    entry = _CATALOGUE[action_idx]
    if entry["is_bomb"]:
        return entry["bomb_strength"]
    if entry.get("rank_key") == "LR":
        return 16
    return entry["strength"]


def legal_actions_for_human(env, seat):
    level_rank = env.active_level_rank()
    hand = env.hands[seat]
    mask = env.get_legal_mask(seat)

    pass_legal = bool(mask[0])
    actions = []
    for action_idx in np.flatnonzero(mask).tolist():
        if action_idx == 0:
            continue
        cards = action_index_to_cards(action_idx, hand, level_rank)
        if not cards:
            continue
        entry = _CATALOGUE[action_idx]
        actions.append(
            {
                "action_idx": action_idx,
                "cards": cards,
                "label": combo_label(action_idx),
                "sort_key": (
                    1 if entry["is_bomb"] else 0,
                    CAT_CARD_COUNTS[action_idx],
                    TYPE_SORT_ORDER.get(entry["type"], 99),
                    action_strength_key(action_idx),
                    format_cards(cards, level_rank),
                ),
            }
        )

    actions.sort(key=lambda item: item["sort_key"])
    return pass_legal, actions


def describe_action(action_idx, cards, level_rank):
    if action_idx == 0:
        return "Pass"
    return f"{combo_label(action_idx)} [{format_cards(cards, level_rank)}]"


def trick_summary(env, human_seat):
    if not env.trick_last:
        return "Open trick"
    trick_type = (
        TYPE_LABELS.get(env.trick_info["type"], env.trick_info["type"])
        if env.trick_info
        else "Play"
    )
    return (
        f"{seat_label(env.trick_seat, human_seat)} leads with "
        f"{trick_type} [{format_cards(env.trick_last, env.active_level_rank())}]"
    )


def finish_order_summary(env, human_seat):
    if not env.finish_order:
        return "-"
    return " -> ".join(seat_label(seat, human_seat) for seat in env.finish_order)


def print_board(env, human_seat):
    blue_level = RANKS[env.level_ranks["Blue"]]
    red_level = RANKS[env.level_ranks["Red"]]
    print("\n" + "=" * 88)
    print(
        f"Round {env.round_num} | Caller: {env.caller} | "
        f"Active level rank: {env.active_level_rank()}"
    )
    print(f"Team levels      : Blue {blue_level} | Red {red_level}")
    print(
        "Cards remaining  : "
        + " | ".join(seat_count_label(seat, human_seat, env) for seat in range(4))
    )
    print(f"Finish order     : {finish_order_summary(env, human_seat)}")
    print(f"Current trick    : {trick_summary(env, human_seat)}")
    print(f"Turn             : {seat_label(env.current_seat, human_seat)}")


def prompt_human_action(env, human_seat):
    level_rank = env.active_level_rank()
    print()
    print_hand("Your hand", env.hands[human_seat], level_rank)

    pass_legal, actions = legal_actions_for_human(env, human_seat)
    if not actions and not pass_legal:
        raise RuntimeError("No legal human action could be decoded.")

    print()
    print(f"Legal moves ({len(actions)} plays{' + pass' if pass_legal else ''})")
    if pass_legal:
        print("  p. Pass")
    for idx, action in enumerate(actions, start=1):
        print(f"  {idx:>2}. {action['label']:<24} {format_cards(action['cards'], level_rank)}")

    prompt = (
        "\nChoose a move number, 'p' to pass, or 'q' to quit: "
        if pass_legal
        else "\nChoose a move number or 'q' to quit: "
    )

    while True:
        raw = input(prompt).strip()
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            return None, None
        if lowered in {"p", "pass"}:
            if pass_legal:
                return 0, []
            print("Pass is not legal here.")
            continue
        if not raw:
            continue
        try:
            choice_idx = int(raw)
        except ValueError:
            print("Please enter a listed move number, 'p', or 'q'.")
            continue
        if 1 <= choice_idx <= len(actions):
            chosen = actions[choice_idx - 1]
            return chosen["action_idx"], chosen["cards"]
        print("That move number is out of range.")


def round_rank_gain(env, finish_order):
    if len(finish_order) < 2:
        return 0
    if env.team(finish_order[0]) == env.team(finish_order[1]):
        return 3
    if len(finish_order) > 2 and env.team(finish_order[0]) == env.team(finish_order[2]):
        return 2
    return 1


def print_round_result(env, human_seat):
    finish_order = env._resolved_finish_order()
    winner_team = env.team(finish_order[0])
    rank_gain = round_rank_gain(env, finish_order)

    print("\n" + "-" * 88)
    print(
        f"Round {env.round_num} complete | Winner: {winner_team} | "
        f"Rank gain: {rank_gain}"
    )
    print(
        "Finish order     : "
        + " -> ".join(seat_label(seat, human_seat) for seat in finish_order)
    )
    print(
        f"Updated levels    : Blue {RANKS[env.level_ranks['Blue']]} | "
        f"Red {RANKS[env.level_ranks['Red']]}"
    )
    print(
        f"Next caller       : {env.caller} | "
        f"Next active level rank: {env.active_level_rank()}"
    )
    match_winner = env.is_match_won()
    if match_winner:
        print(f"Match winner      : {match_winner}")


def print_checkpoint_info(path, checkpoint, upgraded_legacy):
    episode = checkpoint.get("episode")
    episode_note = f" (episode {episode})" if episode is not None else ""
    legacy_note = " [legacy state upgraded]" if upgraded_legacy else ""
    print(f"Checkpoint        : {path}{episode_note}{legacy_note}")


def pause(message):
    try:
        input(message)
    except EOFError:
        pass


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    seed_everything(args.seed)

    net, checkpoint, upgraded_legacy = load_policy(checkpoint_path)
    env = GuandanEnv()
    human_seat = args.seat
    pause_before_bot = not args.no_bot_pause
    env.tribute_seat_value_fn = (
        lambda env_state, seat: (
            None if seat == human_seat else critic_state_value(net, env_state, seat)
        )
    )

    print(f"Device            : {DEVICE}")
    print_checkpoint_info(checkpoint_path, checkpoint, upgraded_legacy)
    print(f"Human seat        : {seat_label(human_seat, human_seat)}")
    print(f"Bot action mode   : {'sampled' if args.sample else 'greedy'}")
    print(f"Seed              : {args.seed}")
    print("Card markers      : ! = non-heart level-rank, * = wildcard heart level-rank")

    steps = 0
    while steps < args.max_steps:
        seat = env.current_seat
        level_rank = env.active_level_rank()
        print_board(env, human_seat)

        if seat == human_seat:
            action_idx, action_cards = prompt_human_action(env, human_seat)
            if action_idx is None:
                print("\nSession ended by user.")
                return
            print(f"\nYou chose        : {describe_action(action_idx, action_cards, level_rank)}")
        else:
            print()
            print_hand(f"{seat_label(seat, human_seat)} hand", env.hands[seat], level_rank)
            if pause_before_bot:
                pause("\nPress Enter to reveal the bot move...")
            action_idx, action_cards = choose_action(
                env=env,
                seat=seat,
                net=net,
                sample=args.sample,
            )
            print(
                f"\n{seat_label(seat, human_seat)} chose: "
                f"{describe_action(action_idx, action_cards, level_rank)}"
            )

        _, round_done = env.apply_action(seat, action_cards, action_idx=action_idx)
        steps += 1

        if not round_done:
            continue

        print_round_result(env, human_seat)
        match_winner = env.is_match_won()
        if match_winner:
            print("\nMatch finished.")
            return

        pause("\nPress Enter to start the next round...")
        env.reset_round()

    raise RuntimeError(
        f"Match exceeded the --max-steps limit of {args.max_steps} actions."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
