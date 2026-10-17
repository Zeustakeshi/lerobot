"""Build broad episode-safe windows and TIPSv2-only targets for joint training.

The script never requests, imports, or computes a DINO teacher target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.tips_ground_vla.teacher_supervision import semantic_distribution


def letterbox_uint8(image: torch.Tensor) -> torch.Tensor:
    image = image[None].float()
    height, width = image.shape[-2:]
    scale = min(224 / height, 336 / width)
    size = (round(height * scale), round(width * scale))
    image = F.interpolate(image, size, mode="bilinear", align_corners=False)
    dh, dw = 224 - size[0], 336 - size[1]
    image = F.pad(image, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
    return image.round().clamp(0, 255).to(torch.uint8)[0]


def _window_motion(actions: torch.Tensor, rows: list[int]) -> torch.Tensor:
    scores = []
    for row in rows:
        chunk = actions[row : row + 32]
        delta = (chunk[1:] - chunk[:-1]).abs().mean().item()
        gripper = (chunk[1:, -1] - chunk[:-1, -1]).abs().mean().item()
        scores.append(delta + 0.25 * gripper)
    return torch.tensor(scores)


def select_spread(rows: list[int], actions: torch.Tensor, count: int) -> list[int]:
    """Use episode-wide spread plus motion extrema and gripper events."""
    if len(rows) <= count:
        return rows
    scores = _window_motion(actions, rows)
    selected: set[int] = set()
    # Every episode contributes temporal coverage, including early approach and
    # final release/stop phases.
    positions = torch.linspace(0, len(rows) - 1, count // 2).round().long().tolist()
    selected.update(rows[position] for position in positions)
    remaining_count = max(0, count - len(selected))
    order = torch.argsort(scores)
    low_n = remaining_count // 3
    high_n = remaining_count // 3
    selected.update(rows[index] for index in order[:low_n].tolist())
    selected.update(rows[index] for index in order[-high_n:].tolist())
    gripper_scores = []
    for row in rows:
        chunk = actions[row : row + 32, -1]
        gripper_scores.append((chunk[1:] - chunk[:-1]).abs().mean().item())
    remaining = [row for row in rows if row not in selected]
    if remaining and len(selected) < count:
        gripper_tensor = torch.tensor(gripper_scores)
        event_order = torch.argsort(gripper_tensor, descending=True).tolist()
        for idx in event_order:
            row = rows[idx]
            if row in selected:
                continue
            selected.add(row)
            if len(selected) >= count:
                break
    if len(selected) < count:
        available = [row for row in rows if row not in selected]
        needed = count - len(selected)
        positions = torch.linspace(0, len(available) - 1, needed).round().long().tolist()
        selected.update(available[position] for position in positions)
    return sorted(selected)[:count]


def build_sample(dataset: Dataset, row: int, stats: dict, task: str) -> dict:
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


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_tips_teacher(teacher_source: str):
    from transformers import AutoModel

    return (
        AutoModel.from_pretrained(
            teacher_source,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float16,
        )
        .cuda()
        .eval()
    )


def make_tips_targets(
    samples: list[dict],
    tasks: list[str],
    teacher,
    text_targets: dict[str, torch.Tensor],
    temperature: float,
    batch_size: int,
) -> None:
    with torch.inference_mode():
        for task in tasks:
            if task not in text_targets:
                text_targets[task] = teacher.encode_text([task]).float()[0].cpu().half()
        for start in tqdm(range(0, len(samples), batch_size), desc="TIPSv2 semantic targets"):
            group = samples[start : start + batch_size]
            images = (
                torch.stack(
                    [
                        torch.stack(
                            (
                                sample["observation.images.top"][0, 0],
                                sample["observation.images.wrist"][0, 0],
                            )
                        )
                        for sample in group
                    ]
                )
                .flatten(0, 1)
                .cuda()
                .half()
            )
            patches = F.normalize(teacher.encode_image(images).patch_tokens.float(), dim=-1)
            text = torch.stack([text_targets[sample["observation.language"][0]] for sample in group]).cuda()
            text = F.normalize(text.float(), dim=-1)
            similarity = torch.einsum("bcnd,bd->bcn", patches.view(len(group), 2, 384, -1), text)
            similarity = F.interpolate(
                similarity.reshape(len(group) * 2, 1, 16, 24),
                (14, 21),
                mode="bilinear",
                align_corners=False,
            ).view(len(group), 588)
            valid = torch.ones_like(similarity, dtype=torch.bool)
            targets = semantic_distribution(similarity, valid, temperature).half().cpu()
            for index, sample in enumerate(group):
                task = sample["observation.language"][0]
                sample["tips_teacher_semantic"] = targets[index]
                sample["tips_teacher_text"] = text_targets[task]


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
    by_episode: dict[int, list[tuple[int, int]]] = {}
    for row, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
        by_episode.setdefault(episode, []).append((frame, row))

    payload = {
        "source": str(args.dataset),
        "selection": {
            "train_windows_per_episode": args.train_windows_per_episode,
            "validation_windows_per_episode": args.validation_windows_per_episode,
            "candidate_stride": args.candidate_stride,
            "episode_safe": True,
            "teacher": args.teacher,
            "dino_enabled": False,
            "tips_temperature": args.tips_temperature,
        },
        "train_episodes": reference["train_episodes"],
        "validation_episodes": reference["validation_episodes"],
        "episode_task": reference["episode_task"],
        "normalization_stats": stats,
        "train_shards": [],
        "validation_shards": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = None
    text_targets: dict[str, torch.Tensor] = {}
    for split, episode_ids, count in (
        ("train", payload["train_episodes"], args.train_windows_per_episode),
        ("validation", payload["validation_episodes"], args.validation_windows_per_episode),
    ):
        for episode in episode_ids:
            ordered = [row for _, row in sorted(by_episode[episode])]
            valid_rows = ordered[: max(0, len(ordered) - 31)]
            rows = select_spread(valid_rows[:: args.candidate_stride], actions, count)
            task = payload["episode_task"][episode]
            shard_path = args.output / split / f"episode-{episode:06d}.pt"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            if shard_path.exists():
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            else:
                shard_samples = [build_sample(dataset, row, stats, task) for row in rows]
                if teacher is None:
                    teacher = load_tips_teacher(args.teacher)
                make_tips_targets(
                    shard_samples,
                    [task],
                    teacher,
                    text_targets,
                    args.tips_temperature,
                    args.teacher_batch_size,
                )
                motion = _window_motion(actions, rows).tolist()
                signatures = []
                for row in rows:
                    chunk = actions[row : row + 32]
                    velocity = chunk[1:] - chunk[:-1]
                    active = velocity.abs() >= 0.02
                    reversal = (
                        active[:-1] & active[1:] & (velocity[:-1].sign() != velocity[1:].sign())
                    ).sum()
                    signatures.append(
                        {
                            "reversals": int(reversal),
                            "stops": int((velocity.abs() < 0.02).sum()),
                            "gripper_motion": float(velocity[:, -1].abs().sum()),
                        }
                    )
                shard = {
                    "episode_index": episode,
                    "task": task,
                    "samples": shard_samples,
                    "motion": motion,
                    "events": signatures,
                }
                torch.save(shard, shard_path)
            payload[f"{split}_shards"].append(
                {
                    "path": str(shard_path.relative_to(args.output)),
                    "episode_index": episode,
                    "task": task,
                    "count": len(shard["samples"]),
                    "motion": shard["motion"],
                    "events": shard["events"],
                }
            )
            payload["train_windows"] = sum(item["count"] for item in payload["train_shards"])
            payload["validation_windows"] = sum(item["count"] for item in payload["validation_shards"])
            (args.output / "manifest.partial.json").write_text(json.dumps(jsonable(payload), indent=2))
            print(split, episode, len(rows), flush=True)
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()
    train_episode_ids = set(payload["train_episodes"])
    train_rows = torch.tensor(
        [row for row, episode in enumerate(episodes) if episode in train_episode_ids], dtype=torch.long
    )
    action_mean = torch.as_tensor(stats["action"]["mean"])
    action_std = torch.as_tensor(stats["action"]["std"])
    normalized_train_actions = (actions[train_rows] - action_mean) / action_std
    payload["train_action_min"] = normalized_train_actions.amin(0)
    payload["train_action_max"] = normalized_train_actions.amax(0)
    payload["normalization_source"] = "training episodes only; inherited from validated split cache"
    payload["train_episodes_count"] = len(payload["train_episodes"])
    payload["validation_episodes_count"] = len(payload["validation_episodes"])
    payload["training_windows"] = payload["train_windows"]
    payload["validation_windows_count"] = payload["validation_windows"]
    (args.output / "manifest.json").write_text(json.dumps(jsonable(payload), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", default="google/tipsv2-b14")
    parser.add_argument("--tips-temperature", type=float, default=0.005065968260169029)
    parser.add_argument("--train-windows-per-episode", type=int, default=64)
    parser.add_argument("--validation-windows-per-episode", type=int, default=12)
    parser.add_argument("--candidate-stride", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
