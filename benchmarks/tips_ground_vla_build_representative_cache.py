"""Build broader, episode-safe TIPS-Ground-VLA windows from a local v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def letterbox_uint8(image: torch.Tensor) -> torch.Tensor:
    image = image[None].float()
    height, width = image.shape[-2:]
    scale = min(224 / height, 336 / width)
    resized = (round(height * scale), round(width * scale))
    image = F.interpolate(image, resized, mode="bilinear", align_corners=False)
    vertical, horizontal = 224 - resized[0], 336 - resized[1]
    image = F.pad(
        image,
        (horizontal // 2, horizontal - horizontal // 2, vertical // 2, vertical - vertical // 2),
    )
    return image.round().clamp(0, 255).to(torch.uint8)[0]


def select_train(rows: list[int], actions: torch.Tensor, count: int) -> list[int]:
    candidates = rows[::16]
    if len(candidates) <= count:
        return candidates
    scores = []
    for row in candidates:
        chunk = actions[row : row + 32]
        scores.append((chunk[1:] - chunk[:-1]).abs().mean().item())
    ranked = [row for _, row in sorted(zip(scores, candidates, strict=True))]
    chosen = ranked[: count // 3] + ranked[-(count // 3) :]
    remaining = [row for row in candidates if row not in chosen]
    needed = count - len(chosen)
    if needed:
        positions = torch.linspace(0, len(remaining) - 1, needed).round().long().tolist()
        chosen.extend(remaining[index] for index in positions)
    return sorted(set(chosen))[:count]


def select_validation(rows: list[int], pairs: int) -> list[int]:
    max_base = len(rows) - 36
    bases = torch.linspace(0, max_base, pairs + 2)[1:-1].round().long().tolist()
    selected = []
    for base in bases:
        selected.extend((rows[base], rows[base + 4]))
    return selected


def build_sample(dataset: LeRobotDataset, row: int, stats: dict, task: str) -> dict:
    item = dataset[row]
    state = (item["observation.state"] - stats["observation.state"]["mean"]) / stats["observation.state"][
        "std"
    ]
    action = (item["action"] - stats["action"]["mean"]) / stats["action"]["std"]
    return {
        "observation.state": state[None, None],
        "observation.language": [task],
        "action": action[None],
        "action_is_pad": item["action_is_pad"][None],
        "episode_index": int(item["episode_index"]),
        "frame_index": int(item["frame_index"]),
        "row": row,
        "observation.images.top": letterbox_uint8(item["observation.images.top"])[None, None],
        "observation.images.wrist": letterbox_uint8(item["observation.images.wrist"])[None, None],
    }


def run(args: argparse.Namespace) -> None:
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    stats = reference["normalization_stats"]
    dataset = LeRobotDataset(
        "local",
        root=args.dataset,
        delta_timestamps={"action": [index / 30 for index in range(32)]},
        video_backend="pyav",
        return_uint8=True,
    )
    episodes = [int(value) for value in dataset.hf_dataset["episode_index"]]
    frames = [int(value) for value in dataset.hf_dataset["frame_index"]]
    actions = torch.stack(list(dataset.hf_dataset["action"]))
    by_episode = {}
    for row, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
        by_episode.setdefault(episode, []).append((frame, row))

    selected = {"train": [], "validation": []}
    for split, episode_ids in (
        ("train", reference["train_episodes"]),
        ("validation", reference["validation_episodes"]),
    ):
        for episode in episode_ids:
            ordered = [row for _, row in sorted(by_episode[episode])]
            valid = ordered[: max(0, len(ordered) - 31)]
            rows = (
                select_train(valid, actions, args.train_windows_per_episode)
                if split == "train"
                else select_validation(valid, args.validation_pairs_per_episode)
            )
            for row in rows:
                selected[split].append(build_sample(dataset, row, stats, reference["episode_task"][episode]))
            print(split, episode, len(rows), flush=True)

    payload = {
        "source": str(args.dataset),
        "selection": {
            "train_windows_per_episode": args.train_windows_per_episode,
            "validation_pairs_per_episode": args.validation_pairs_per_episode,
            "candidate_stride": 16,
            "validation_replan_stride": 4,
            "episode_safe": True,
        },
        "train_episodes": reference["train_episodes"],
        "validation_episodes": reference["validation_episodes"],
        "episode_task": reference["episode_task"],
        "normalization_stats": stats,
        **selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": payload["source"],
                "selection": payload["selection"],
                "train_windows": len(payload["train"]),
                "validation_windows": len(payload["validation"]),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-windows-per-episode", type=int, default=8)
    parser.add_argument("--validation-pairs-per-episode", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
