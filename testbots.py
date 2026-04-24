"""
Checkpoint-vs-checkpoint evaluator for Guandan bots.

Runs full matches between an earlier checkpoint and a later checkpoint.
Each checkpoint controls an entire team:
  - one model on seats 0 and 2
  - the other model on seats 1 and 3

By default, the script alternates which checkpoint gets the Blue seats so the
fixed Blue caller advantage does not skew the comparison.

Examples:
    python testbots.py
    python testbots.py --games 1000
    python testbots.py --later checkpoints/guandan_ep6400.pt --earlier checkpoints/guandan_ep3200.pt
"""

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from guandan_arena import (  # noqa: E402
    DEVICE,
    GuandanEnv,
    GuandanNet,
    _upgrade_legacy_model_state,
    action_index_to_cards,
    critic_state_value,
)

# Optional manual override:
# Edit these directly if you want to choose the exact checkpoint files here.
# CLI args still take priority over these values.
MANUAL_EARLIER_CHECKPOINT = "checkpoints/guandan_ep185600.pt"
MANUAL_LATER_CHECKPOINT = "checkpoints/guandan_ep224000.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate two checkpoint teams against each other over many "
            "full Guandan matches."
        )
    )
    parser.add_argument(
        "--later",
        type=str,
        default=None,
        help="Path to the later checkpoint. Overrides MANUAL_LATER_CHECKPOINT.",
    )
    parser.add_argument(
        "--earlier",
        type=str,
        default=None,
        help="Path to the earlier checkpoint. Overrides MANUAL_EARLIER_CHECKPOINT.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of full matches to run.",
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
        help="Sample actions from the policy instead of taking greedy argmax actions.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="How often to print progress updates. Use 0 to disable.",
    )
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be at least 1.")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative.")
    return args


def extract_episode_num(path: Path):
    match = re.search(r"ep(\d+)", path.stem)
    return int(match.group(1)) if match else None


def find_default_checkpoints():
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            "Could not find a checkpoints/ directory. "
            "Pass --later and --earlier explicitly."
        )

    checkpoint_paths = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda path: (
            extract_episode_num(path) is None,
            extract_episode_num(path) or 0,
            path.name,
        ),
    )

    if len(checkpoint_paths) < 2:
        raise ValueError(
            "Need at least two checkpoint files in checkpoints/ to infer "
            "earlier vs later automatically."
        )

    return str(checkpoint_paths[0]), str(checkpoint_paths[-1])


def resolve_checkpoint_paths(later_path, earlier_path):
    manual_earlier = MANUAL_EARLIER_CHECKPOINT or None
    manual_later = MANUAL_LATER_CHECKPOINT or None

    earlier = earlier_path or manual_earlier
    later = later_path or manual_later

    if earlier is None or later is None:
        default_earlier, default_later = find_default_checkpoints()
        earlier = earlier or default_earlier
        later = later or default_later

    if not os.path.exists(later):
        raise FileNotFoundError(f"Later checkpoint not found: {later}")
    if not os.path.exists(earlier):
        raise FileNotFoundError(f"Earlier checkpoint not found: {earlier}")

    return earlier, later


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


def play_match(blue_net, red_net, sample=False, max_steps=50000):
    env = GuandanEnv()
    env.tribute_seat_value_fn = (
        lambda env_state, seat: critic_state_value(
            blue_net if env_state.team(seat) == "Blue" else red_net,
            env_state,
            seat,
        )
    )
    steps = 0
    completed_rounds = 0

    while steps < max_steps:
        seat = env.current_seat
        net = blue_net if env.team(seat) == "Blue" else red_net
        action_idx, action_cards = choose_action(env, seat, net, sample=sample)
        _, round_done = env.apply_action(seat, action_cards, action_idx=action_idx)
        steps += 1

        if not round_done:
            continue

        completed_rounds += 1
        match_winner = env.is_match_won()
        if match_winner:
            return match_winner, completed_rounds, steps

        env.reset_round()

    raise RuntimeError(f"Match exceeded {max_steps} actions without finishing.")


def print_checkpoint_info(tag, path, checkpoint, upgraded_legacy):
    episode = checkpoint.get("episode")
    legacy_note = " [legacy state upgraded]" if upgraded_legacy else ""
    episode_note = f" (episode {episode})" if episode is not None else ""
    print(f"{tag:<18}: {path}{episode_note}{legacy_note}")


def main():
    args = parse_args()
    earlier_path, later_path = resolve_checkpoint_paths(args.later, args.earlier)
    seed_everything(args.seed)

    print(f"Device             : {DEVICE}")
    print(f"Games              : {args.games}")
    print(f"Action mode        : {'sampled' if args.sample else 'greedy'}")
    print(f"Seed               : {args.seed}")
    print("Seat assignment    : alternating Blue/Red each game")

    earlier_net, earlier_ckpt, earlier_upgraded = load_policy(earlier_path)
    later_net, later_ckpt, later_upgraded = load_policy(later_path)

    print_checkpoint_info(
        "Earlier checkpoint", earlier_path, earlier_ckpt, earlier_upgraded
    )
    print_checkpoint_info(
        "Later checkpoint", later_path, later_ckpt, later_upgraded
    )
    print()

    later_wins = 0
    earlier_wins = 0
    later_blue_wins = 0
    later_red_wins = 0
    blue_side_games = 0
    red_side_games = 0
    total_rounds = 0
    total_steps = 0

    start_time = time.time()

    for game_idx in range(1, args.games + 1):
        # Pair adjacent games to reuse the same base seed while swapping sides.
        game_seed = args.seed + ((game_idx - 1) // 2)
        seed_everything(game_seed)

        later_is_blue = (game_idx % 2 == 1)
        blue_net = later_net if later_is_blue else earlier_net
        red_net = earlier_net if later_is_blue else later_net

        winner_team, rounds, steps = play_match(
            blue_net=blue_net,
            red_net=red_net,
            sample=args.sample,
        )

        total_rounds += rounds
        total_steps += steps

        later_team = "Blue" if later_is_blue else "Red"
        later_won = winner_team == later_team

        if later_is_blue:
            blue_side_games += 1
            if later_won:
                later_blue_wins += 1
        else:
            red_side_games += 1
            if later_won:
                later_red_wins += 1

        if later_won:
            later_wins += 1
        else:
            earlier_wins += 1

        if args.progress_every and game_idx % args.progress_every == 0:
            elapsed = time.time() - start_time
            later_rate = later_wins / game_idx
            print(
                f"[{game_idx:4d}/{args.games}] "
                f"later wins: {later_wins:4d}  "
                f"earlier wins: {earlier_wins:4d}  "
                f"later win rate: {later_rate:.2%}  "
                f"elapsed: {elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    later_rate = later_wins / args.games
    earlier_rate = earlier_wins / args.games
    later_blue_rate = (
        later_blue_wins / blue_side_games if blue_side_games else 0.0
    )
    later_red_rate = later_red_wins / red_side_games if red_side_games else 0.0

    print("\nResults")
    print("-------")
    print(f"Later checkpoint wins   : {later_wins}/{args.games} ({later_rate:.2%})")
    print(
        f"Earlier checkpoint wins : {earlier_wins}/{args.games} ({earlier_rate:.2%})"
    )
    print(
        f"Later as Blue           : "
        f"{later_blue_wins}/{blue_side_games} ({later_blue_rate:.2%})"
    )
    print(
        f"Later as Red            : "
        f"{later_red_wins}/{red_side_games} ({later_red_rate:.2%})"
    )
    print(f"Average rounds/match    : {total_rounds / args.games:.2f}")
    print(f"Average actions/match   : {total_steps / args.games:.2f}")
    print(f"Elapsed time            : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
