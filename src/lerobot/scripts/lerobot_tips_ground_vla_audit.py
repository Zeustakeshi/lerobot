#!/usr/bin/env python
"""Audit TIPS teacher/student visual alignment on 30--50 LeRobot frames.

The audit is deliberately independent from training.  It checks patch indexing,
camera preprocessing, teacher attention, student distillation alignment, and
per-query (unaveraged) cross-attention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import _resolve_hub_cache_dir
from lerobot.scripts.lerobot_tips_ground_vla_viz import _label, _map_overlay, _tile, _to_bgr


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tips_ground_vla_alignment_audit"))
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Audit just one task (0, 1, or 2). Omit to audit all three collection bands.",
    )
    parser.add_argument("--num-frames", type=int, default=36, choices=range(30, 51))
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


class _TIPSTeacher:
    """TIPS-only inspector: avoids loading DINO because this audit concerns alignment."""

    def __init__(self, config: Any, device: torch.device) -> None:
        from transformers import AutoModel

        source = config.tips_teacher_path or config.tips_teacher_id
        cache_dir = _resolve_hub_cache_dir(config)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = (
            AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def maps(self, image: torch.Tensor, phrases: list[str]) -> torch.Tensor:
        """Return teacher patch distributions [P, 14, 14] for RGB images in [0, 1]."""
        patches = self.model.encode_image(image[None].to(self.device, dtype=self.dtype)).patch_tokens
        text = self.model.encode_text(phrases).to(self.device, dtype=patches.dtype)
        logits = F.normalize(patches, dim=-1) @ F.normalize(text, dim=-1).T  # [1, 256, P]
        logits = logits[0].T.reshape(len(phrases), 16, 16)
        return (
            F.interpolate(logits[:, None], size=(14, 14), mode="bilinear", align_corners=False)
            .squeeze(1)
            .softmax(-1)
        )


def _object_region(image: np.ndarray, phrase: str) -> tuple[np.ndarray, str]:
    """Return a visual validation mask; this is deliberately labelled heuristic, not GT."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if phrase == "red block":
        mask = cv2.inRange(hsv, (0, 70, 55), (12, 255, 255)) | cv2.inRange(
            hsv, (165, 70, 55), (180, 255, 255)
        )
        label = "red block region - HSV heuristic"
    elif phrase == "blue block":
        mask = cv2.inRange(hsv, (95, 70, 45), (135, 255, 255))
        label = "blue block region - HSV heuristic"
    elif phrase in {"green block", "T shape"}:
        mask = cv2.inRange(hsv, (35, 55, 45), (90, 255, 255))
        label = "green block/T-shape region - HSV heuristic"
    else:
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 105, 105))
        dark[int(image.shape[0] * 0.58) :] = 0  # reject the robot/gripper at the bottom of top view
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
        mask = np.zeros_like(dark)
        candidates = [
            (stats[index, cv2.CC_STAT_AREA], index)
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] > 80
        ]
        if candidates:
            _, best = max(candidates)
            mask[labels == best] = 255
        label = "pen holder region - dark-object heuristic"
    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)
    return _label(cv2.addWeighted(image, 0.55, overlay, 0.45, 0), label), label


def _sample_indices(
    dataset: LeRobotDataset,
    specs: list[tuple[int, int, str, list[str]]],
    count: int,
) -> list[tuple[str, int, int, list[str]]]:
    """Sample start/quarter/middle/three-quarter/end across explicit task bands."""
    per_task = [count // len(specs)] * len(specs)
    for index in range(count % len(specs)):
        per_task[index] += 1
    samples = []
    for wanted, (episode_start, episode_end, task, phrases) in zip(per_task, specs, strict=True):
        episode_ids = np.linspace(episode_start, episode_end, wanted, dtype=int)
        for number, episode_index in enumerate(episode_ids):
            record = dataset.meta.episodes[int(episode_index)]
            if task not in record.get("tasks", []):
                raise ValueError(
                    f"Episode {episode_index} was expected to contain {task!r}, got {record.get('tasks', [])!r}"
                )
            start, end = int(record["dataset_from_index"]), int(record["dataset_to_index"])
            fraction = (number % 5) / 4
            samples.append((task, int(episode_index), start + round((end - start - 1) * fraction), phrases))
    return samples


def _student_maps(
    policy: Any, preprocessor: Any, sample: dict[str, Any], phrases: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = preprocessor(dict(sample))
    with torch.inference_mode():
        images, state, instructions = policy._inputs(batch)
        _, outputs = policy.model(images, state, instructions, phrases=[phrases], debug=True)
    # Raw query attention: mean only the four MHA sub-heads, never the eight latent queries.
    raw_queries = outputs["alignment_cross_attention"][0].mean(0)[:, :196].reshape(8, 14, 14)
    return outputs["student_align_map"][0, 0, :, :].reshape(len(phrases), 14, 14), raw_queries


def _four_panel(
    image: np.ndarray, teacher: np.ndarray, student: np.ndarray, region: np.ndarray, phrase: str
) -> np.ndarray:
    base = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
    panels = [
        _label(base, "original RGB"),
        _map_overlay(base, teacher, f"TIPS teacher similarity: {phrase}"),
        _map_overlay(base, student, f"student alignment: {phrase}"),
        cv2.resize(region, (480, 270), interpolation=cv2.INTER_AREA),
    ]
    return _tile(panels, columns=2)


def _patch_coordinate_figure(output_dir: Path) -> None:
    image = np.full((700, 700, 3), (25, 25, 25), dtype=np.uint8)
    for row in range(14):
        for col in range(14):
            index = row * 14 + col
            x, y = col * 50, row * 50
            cv2.rectangle(image, (x, y), (x + 50, y + 50), (80, 80, 80), 1)
            if index in {0, 13, 182, 195, 97, 98}:
                cv2.putText(
                    image, str(index), (x + 7, y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
                )
    cv2.putText(
        image,
        "row-major patch grid: index = row * 14 + col",
        (12, 690),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )
    cv2.imwrite(str(output_dir / "patch_coordinate_check.png"), image)


def main() -> None:
    args = _args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    dataset = LeRobotDataset(args.dataset_root.name, root=args.dataset_root)
    task_names = [str(task) for task in dataset.meta.tasks.index.tolist()]
    if len(task_names) < 3:
        raise ValueError("Expected three dataset tasks for the collection-band audit")
    specs = [
        (0, 54, task_names[0], ["red block", "pen holder"]),
        (70, 100, task_names[1], ["red block", "blue block"]),
        (120, 150, task_names[2], ["green block", "T shape"]),
    ]
    if args.task_index is not None:
        if not 0 <= args.task_index < len(specs):
            raise ValueError(f"task-index must be in [0, {len(specs) - 1}]")
        specs = [specs[args.task_index]]
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    policy = (
        get_policy_class(config.type).from_pretrained(args.checkpoint, config=config).to(args.device).eval()
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    teacher = _TIPSTeacher(config, torch.device(args.device))
    samples = _sample_indices(dataset, specs, args.num_frames)
    representative_queries: list[np.ndarray] = []
    cards: list[np.ndarray] = []
    for number, (task, episode_index, index, phrases) in enumerate(samples):
        print(
            f"Auditing frame {number + 1}/{len(samples)} (task={task!r}, episode={episode_index}, dataset index={index})",
            flush=True,
        )
        sample: dict[str, Any] = dict(dataset[index])
        sample["task"] = task
        source = _to_bgr(sample[config.image_keys[0]])
        # Match the production policy path exactly: both student and TIPS see
        # the same 224x224 RGB view.  TIPS B/14 cannot consume the raw 720px
        # height because it is not divisible by its patch size.
        teacher_image = F.interpolate(
            sample[config.image_keys[0]][None], size=(224, 224), mode="bilinear", align_corners=False
        )[0]
        teacher_maps = teacher.maps(teacher_image, phrases).float().cpu()
        student_maps, raw_queries = _student_maps(policy, preprocessor, sample, phrases)
        if number in {0, len(samples) // 2, len(samples) - 1}:
            query_panels = [
                _map_overlay(
                    cv2.resize(source, (360, 202), interpolation=cv2.INTER_AREA),
                    query.cpu().numpy(),
                    f"raw query {query_index + 1}",
                )
                for query_index, query in enumerate(raw_queries)
            ]
            representative_queries.append(_tile(query_panels, columns=4))
        for phrase_index, phrase in enumerate(phrases):
            region, _ = _object_region(source, phrase)
            panel = _four_panel(
                source,
                teacher_maps[phrase_index].numpy(),
                student_maps[phrase_index].cpu().numpy(),
                region,
                phrase,
            )
            cv2.imwrite(
                str(frames_dir / f"frame_{number:02d}_ep_{episode_index}_{phrase.replace(' ', '_')}.png"),
                panel,
            )
            if len(cards) < 12:
                cards.append(cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(args.output_dir / "teacher_student_overview.png"), _tile(cards, columns=4))
    cv2.imwrite(
        str(args.output_dir / "raw_query_attention_examples.png"), _tile(representative_queries, columns=1)
    )
    _patch_coordinate_figure(args.output_dir)
    report = {
        "tasks": [
            {"episode_band": f"{start}..{end}", "task": task, "phrases": phrases}
            for start, end, task, phrases in specs
        ],
        "frames": len(samples),
        "checks": {
            "patch_coordinates": "row-major [row, col] = row * 14 + col; CLS is removed by model before 196-patch reshape",
            "camera_preprocessing": "student resizes every configured camera to 224x224 without crop/padding; camera order follows config.image_keys",
            "tips_preprocessing": "audit sends the same 224x224 RGB [0,1] resize used by the policy to TIPS encode_image; no ImageNet normalization",
            "object_regions": "red block uses HSV heuristic; pen holder uses dark-object heuristic, not annotated ground truth",
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(samples)}-frame TIPS/student alignment audit to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
