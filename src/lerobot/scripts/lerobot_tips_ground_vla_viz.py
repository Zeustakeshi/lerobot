#!/usr/bin/env python
"""Export a one-frame OpenCV diagnostic report for a TIPS-Ground-VLA checkpoint.

The tool runs the *student* model on one frame from a local LeRobot dataset and
writes PNGs plus ``report.json``.  It never loads the training-only TIPS/DINO
teachers, so it is safe to use while debugging inference.

Example:
    conda run -n lerobot_arm lerobot-tips-ground-vla-viz \
      --dataset-root /path/to/lerobot_dataset \
      --checkpoint /path/to/pretrained_model --episode-index 0 --frame-offset 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory containing config.json and model.safetensors.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tips_ground_vla_viz"))
    parser.add_argument("--episode-index", type=int, default=0, help="Episode to sample.")
    parser.add_argument(
        "--frame-offset", type=int, default=0, help="0-based frame within the chosen episode."
    )
    parser.add_argument(
        "--instruction", type=str, default=None, help="Override the task text stored in the dataset."
    )
    parser.add_argument(
        "--scenario-suite",
        action="store_true",
        help="Run visual A/B tests for instruction, camera availability, and episode changes.",
    )
    parser.add_argument(
        "--comparison-episode-index",
        type=int,
        default=None,
        help="Second episode for --scenario-suite (defaults to the next episode).",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _episode_frame_index(dataset: LeRobotDataset, episode_index: int, frame_offset: int) -> int:
    if not 0 <= episode_index < dataset.meta.total_episodes:
        raise ValueError(f"episode-index must be in [0, {dataset.meta.total_episodes - 1}]")
    episode = dataset.meta.episodes[episode_index]
    start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    if not 0 <= frame_offset < end - start:
        raise ValueError(f"frame-offset must be in [0, {end - start - 1}] for episode {episode_index}")
    return start + frame_offset


def _label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_NEAREST)


def _grid_overlay(image: np.ndarray, grid_size: int = 14) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for x in range(1, grid_size):
        cv2.line(
            output,
            (round(x * width / grid_size), 0),
            (round(x * width / grid_size), height),
            (0, 255, 255),
            1,
        )
    for y in range(1, grid_size):
        cv2.line(
            output,
            (0, round(y * height / grid_size)),
            (width, round(y * height / grid_size)),
            (0, 255, 255),
            1,
        )
    return output


def _map_overlay(image: np.ndarray, patch_map: np.ndarray, title: str) -> np.ndarray:
    values = patch_map.astype(np.float32)
    values -= values.min()
    values /= max(float(values.max()), 1e-8)
    heat = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    output = cv2.addWeighted(image, 0.52, heat, 0.48, 0)
    # A compact contour makes the most attended region immediately visible,
    # without asking the user to infer it from a colour scale.
    focus = cv2.resize(values, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask = (focus >= np.quantile(focus, 0.90)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 64:
            cv2.drawContours(output, [contour], -1, (255, 255, 255), 3, cv2.LINE_AA)
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + width, y + height), (20, 20, 20), 5)
            cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 255), 2)
    return _label(output, title)


def _pca_rgb(features: np.ndarray) -> np.ndarray:
    """Project the 14x14xD grounding tensor to a stable-per-frame RGB view."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    flat -= flat.mean(0, keepdims=True)
    _, _, vectors = np.linalg.svd(flat, full_matrices=False)
    rgb = flat @ vectors[:3].T
    lo, hi = np.percentile(rgb, 1, axis=0), np.percentile(rgb, 99, axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return cv2.cvtColor((rgb.reshape(14, 14, 3) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _tile(images: list[np.ndarray], columns: int = 2) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        canvas = np.full((height, width, 3), 25, dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        padded.append(canvas)
    while len(padded) % columns:
        padded.append(np.full((height, width, 3), 25, dtype=np.uint8))
    return np.vstack([np.hstack(padded[i : i + columns]) for i in range(0, len(padded), columns)])


def _flow_header(width: int, task: str) -> np.ndarray:
    """A human-readable architecture legend above the visual report."""
    header = np.full((145, width, 3), (19, 20, 24), dtype=np.uint8)
    cv2.putText(
        header,
        "TIPS-Ground-VLA: one-frame vision trace",
        (28, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    task_text = f"Instruction: {task}"
    cv2.putText(
        header, task_text[:150], (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 218, 255), 1, cv2.LINE_AA
    )
    stages = [
        "1. RGB camera",
        "2. ViT-S/16 patches",
        "3. Alignment: WHAT",
        "4. Grounding: WHERE",
        "5. Cross-attention focus",
    ]
    x_positions = np.linspace(30, max(30, width - 450), len(stages)).astype(int)
    for index, (x, stage) in enumerate(zip(x_positions, stages, strict=True)):
        cv2.rectangle(header, (x, 95), (x + 250, 130), (43, 64, 94), -1)
        cv2.putText(
            header, stage, (x + 10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA
        )
        if index < len(stages) - 1:
            cv2.arrowedLine(header, (x + 252, 112), (x + 292, 112), (100, 190, 255), 2, tipLength=0.28)
    return header


def _write_camera_report(
    output_dir: Path,
    camera_name: str,
    source: np.ndarray,
    outputs: dict[str, torch.Tensor],
    camera_index: int,
) -> list[np.ndarray]:
    align = outputs["student_align_map"][0, camera_index].mean(0).reshape(14, 14).cpu().numpy()
    energy = outputs["dense_grounding"][0, camera_index].norm(dim=-1).reshape(14, 14).cpu().numpy()
    visual_energy = outputs["visual_patch_tokens"][0, camera_index].norm(dim=-1).reshape(14, 14).cpu().numpy()
    pca = _pca_rgb(outputs["dense_grounding"][0, camera_index].reshape(14, 14, -1).cpu().numpy())
    panels = [
        _label(source, f"{camera_name}: dataset RGB"),
        _label(_grid_overlay(source), f"{camera_name}: ViT-S/16 patch grid (14x14)"),
        _map_overlay(source, visual_energy, f"{camera_name}: projected ViT patch-token norm"),
        _map_overlay(source, align, f"{camera_name}: alignment soft map"),
        _map_overlay(source, energy, f"{camera_name}: grounding feature norm"),
        _label(_resize_like(pca, source), f"{camera_name}: grounding feature PCA (RGB)"),
    ]
    for number, panel in enumerate(panels):
        cv2.imwrite(str(output_dir / f"{camera_name.replace('.', '_')}_{number:02d}.png"), panel)
    return panels


def _write_cross_attention(
    output_dir: Path, source_images: list[np.ndarray], outputs: dict[str, torch.Tensor], name: str
) -> list[np.ndarray]:
    # [B, 4 heads, 8 latent queries, C*196] -> mean heads/queries -> Cx14x14.
    attention = outputs[f"{name}_cross_attention"][0].mean(dim=(0, 1)).cpu().numpy()
    cameras = len(source_images)
    panels = []
    for camera_index, source in enumerate(source_images):
        patch_map = attention[camera_index * 196 : (camera_index + 1) * 196].reshape(14, 14)
        panels.append(_map_overlay(source, patch_map, f"{name}: mean 4 heads x 8 latent queries"))
    cv2.imwrite(str(output_dir / f"{name}_cross_attention.png"), _tile(panels, columns=cameras))
    return panels


def _write_query_atlas(
    output_dir: Path, source_images: list[np.ndarray], outputs: dict[str, torch.Tensor], name: str
) -> None:
    """Show every latent query separately so collapsed queries are easy to spot."""
    attention = outputs[f"{name}_cross_attention"][0].mean(dim=0).cpu().numpy()  # [8, C*196]
    for camera_index, source in enumerate(source_images):
        cells = []
        for query_index, query_map in enumerate(attention):
            patch_map = query_map[camera_index * 196 : (camera_index + 1) * 196].reshape(14, 14)
            small = cv2.resize(source, (480, 270), interpolation=cv2.INTER_AREA)
            cells.append(_map_overlay(small, patch_map, f"{name} latent query {query_index + 1}"))
        cv2.imwrite(
            str(output_dir / f"{name}_query_atlas_camera_{camera_index + 1}.png"),
            _tile(cells, columns=4),
        )
    # This is a visual diagnostic for query collapse: bright off-diagonal
    # cells mean different latent queries are attending to the same patches.
    normalized = attention / np.maximum(np.linalg.norm(attention, axis=1, keepdims=True), 1e-8)
    similarity = normalized @ normalized.T
    similarity = np.clip((similarity + 1) / 2, 0, 1)
    matrix = cv2.applyColorMap((similarity * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    matrix = cv2.resize(matrix, (560, 560), interpolation=cv2.INTER_NEAREST)
    matrix = _label(matrix, f"{name}: latent-query similarity (yellow off-diagonal = collapse)")
    for index in range(8):
        coordinate = 35 + round((index + 0.5) * 560 / 8)
        cv2.putText(
            matrix, str(index + 1), (coordinate - 5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
        )
        cv2.putText(
            matrix, str(index + 1), (8, coordinate + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
        )
    cv2.imwrite(str(output_dir / f"{name}_query_similarity.png"), matrix)


def _write_role_maps(
    output_dir: Path, source_images: list[np.ndarray], outputs: dict[str, torch.Tensor]
) -> None:
    """Show V2 source/destination readouts without recreating a shared prior."""
    panels = []
    for index, source in enumerate(source_images):
        for head, key in (("alignment", "alignment_role_map"), ("grounding", "grounding_role_map")):
            for role_index, role in enumerate(("source", "destination")):
                role_map = outputs[key][0, index, role_index].reshape(14, 14).cpu().numpy()
                panels.append(_map_overlay(source, role_map, f"{head} {role} readout"))
    cv2.imwrite(str(output_dir / "role_specific_maps.png"), _tile(panels, columns=4))


def _run_debug_forward(
    policy: Any, preprocessor: Any, sample: dict[str, Any], camera_valid: list[bool] | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one model trace while retaining the vision-only debug tensors."""
    model_batch = preprocessor(dict(sample))
    with torch.inference_mode():
        images, state, instruction = policy._inputs(model_batch)
        valid = None if camera_valid is None else torch.tensor([camera_valid], device=images.device)
        return policy.model(images, state, instruction, camera_valid=valid, debug=True)


def _scenario_card(
    title: str,
    instruction: str,
    source_images: list[np.ndarray],
    outputs: dict[str, torch.Tensor],
    camera_index: int,
    camera_enabled: bool = True,
) -> np.ndarray:
    """One compact before/after card; it is meant to be read without tensors."""
    source = cv2.resize(source_images[camera_index], (480, 270), interpolation=cv2.INTER_AREA)
    align = outputs["student_align_map"][0, camera_index].mean(0).reshape(14, 14).cpu().numpy()
    heatmap = _map_overlay(source, align, "alignment focus")
    if not camera_enabled:
        source = cv2.addWeighted(source, 0.25, np.zeros_like(source), 0.75, 0)
        cv2.putText(source, "CAMERA MASKED", (170, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)
    source = _label(source, "input camera" if camera_enabled else "input camera (masked)")
    card = np.hstack([source, heatmap])
    header = np.full((82, card.shape[1], 3), (24, 27, 33), dtype=np.uint8)
    cv2.putText(header, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        header, instruction[:145], (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (184, 220, 255), 1, cv2.LINE_AA
    )
    return np.vstack([header, card])


def _latent_image(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    """Render the three specialized latent groups and fused latent as coloured strips."""
    groups = (
        ("Za - Alignment", outputs["Za"]),
        ("Zg - Grounding", outputs["Zg"]),
        ("Zs - Robot state", outputs["Zs"]),
        ("Z - Fused", outputs["Z"]),
    )
    strips = []
    for label, values in groups:
        matrix = values[0].detach().float().cpu().numpy()
        matrix = (matrix - matrix.mean(1, keepdims=True)) / np.maximum(matrix.std(1, keepdims=True), 1e-6)
        matrix = np.clip((matrix + 2.5) / 5.0, 0, 1)
        image = cv2.applyColorMap((matrix * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image = cv2.resize(image, (720, 72), interpolation=cv2.INTER_NEAREST)
        strips.append(_label(image, label))
    return np.vstack(strips)


def _action_trace_image(action_chunk: torch.Tensor) -> np.ndarray:
    """Plot action-chunk shape only; no raw values are needed for a visual diagnostic."""
    width, height = 720, 260
    image = np.full((height, width, 3), (25, 25, 28), dtype=np.uint8)
    cv2.putText(
        image,
        "Action decoder: 12-step chunk shape",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    values = action_chunk[0].detach().float().cpu().numpy()
    colors = [
        (255, 170, 50),
        (80, 210, 255),
        (105, 235, 135),
        (255, 105, 185),
        (230, 225, 80),
        (180, 120, 255),
    ]
    for dim in range(values.shape[1]):
        series = values[:, dim]
        span = max(float(series.max() - series.min()), 1e-6)
        normalized = (series - series.min()) / span
        points = np.array(
            [
                [50 + round(i * (width - 80) / 11), height - 35 - round(value * (height - 80))]
                for i, value in enumerate(normalized)
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, colors[dim % len(colors)], 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "each colour = one action dimension; vertical scale is normalized per dimension",
        (12, 248),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )
    return image


def _layer_walkthrough(
    output_dir: Path,
    source_images: list[np.ndarray],
    outputs: dict[str, torch.Tensor],
    action_chunk: torch.Tensor,
) -> None:
    """Render the student execution order from camera pixels to action chunk."""
    source = cv2.resize(source_images[0], (480, 270), interpolation=cv2.INTER_AREA)
    visual_energy = outputs["visual_patch_tokens"][0, 0].norm(dim=-1).reshape(14, 14).cpu().numpy()
    alignment = outputs["student_align_map"][0, 0].mean(0).reshape(14, 14).cpu().numpy()
    grounding = outputs["dense_grounding"][0, 0].norm(dim=-1).reshape(14, 14).cpu().numpy()
    align_cross = outputs["alignment_cross_attention"][0].mean(dim=(0, 1))[:196].reshape(14, 14).cpu().numpy()
    ground_cross = (
        outputs["grounding_cross_attention"][0].mean(dim=(0, 1))[:196].reshape(14, 14).cpu().numpy()
    )
    stages = [
        _label(source, "1. RGB frame"),
        _label(_grid_overlay(source), "2. ViT-S/16 patches"),
        _map_overlay(source, visual_energy, "3. visual projection token strength"),
        _map_overlay(source, alignment, "4. Alignment head: WHAT"),
        _map_overlay(source, grounding, "5. Grounding spatial block: WHERE"),
        _map_overlay(source, align_cross, "6. Alignment cross-attention"),
        _map_overlay(source, ground_cross, "7. Grounding cross-attention"),
        _latent_image(outputs),
        _action_trace_image(action_chunk),
    ]
    flow = _tile(stages, columns=3)
    header = np.full((78, flow.shape[1], 3), (18, 22, 29), dtype=np.uint8)
    cv2.putText(
        header,
        "Student model layer-by-layer: input -> vision tokens -> specialized heads -> fused latent -> action chunk",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Stages 1-7 are spatial overlays; stage 8 is latent-token activity; stage 9 is action-decoder trajectory shape.",
        (16, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (182, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "layer_by_layer_walkthrough.png"), np.vstack([header, flow]))


def _run_scenario_suite(
    args: argparse.Namespace,
    dataset: LeRobotDataset,
    policy: Any,
    preprocessor: Any,
) -> None:
    """Create an image-first regression suite covering the likely failure modes."""
    base_record = dataset.meta.episodes[args.episode_index]
    base_index = (int(base_record["dataset_from_index"]) + int(base_record["dataset_to_index"]) - 1) // 2
    base_sample: dict[str, Any] = dict(dataset[base_index])
    base_instruction = args.instruction or str(base_sample["task"])
    base_sample["task"] = base_instruction
    source_images = [_to_bgr(base_sample[key]) for key in policy.config.image_keys]
    camera_count = len(source_images)
    if camera_count < 2:
        raise ValueError("--scenario-suite needs at least two camera inputs")

    _, native = _run_debug_forward(policy, preprocessor, base_sample)
    task_names = [str(task) for task in dataset.meta.tasks.index.tolist()]
    task_cards = []
    for task_number, task_name in enumerate(task_names[:3], start=1):
        task_sample = dict(base_sample)
        task_sample["task"] = task_name
        _, task_outputs = _run_debug_forward(policy, preprocessor, task_sample)
        task_cards.append(
            _scenario_card(
                f"A{task_number}. Counterfactual query {task_number} on task-1 image",
                task_name,
                source_images,
                task_outputs,
                0,
            )
        )

    top_only = [True, *([False] * (camera_count - 1))]
    wrist_only = [False, True, *([False] * (camera_count - 2))]
    _, top_outputs = _run_debug_forward(policy, preprocessor, base_sample, top_only)
    _, wrist_outputs = _run_debug_forward(policy, preprocessor, base_sample, wrist_only)
    camera_cards = [
        _scenario_card("C. Top camera only", base_instruction, source_images, top_outputs, 0),
        _scenario_card("D. Wrist camera only", base_instruction, source_images, wrist_outputs, 1),
    ]

    temporal_cards = []
    start, end = int(base_record["dataset_from_index"]), int(base_record["dataset_to_index"])
    for label, frame_index in (("start", start), ("middle", base_index), ("end", end - 1)):
        temporal_sample: dict[str, Any] = dict(dataset[frame_index])
        temporal_sample["task"] = base_instruction
        temporal_sources = [_to_bgr(temporal_sample[key]) for key in policy.config.image_keys]
        _, temporal_outputs = _run_debug_forward(policy, preprocessor, temporal_sample)
        temporal_cards.append(
            _scenario_card(
                f"E. Episode {args.episode_index}: {label}",
                base_instruction,
                temporal_sources,
                temporal_outputs,
                0,
            )
        )

    task_episode_cards = []
    # Dataset collection is ordered by task. Use representative episodes from
    # the requested collection bands rather than the first matching episode.
    task_episode_bands = ((0, 54), (70, 100), (120, 150))
    for task_number, (task_name, (episode_start, episode_end)) in enumerate(
        zip(task_names[:3], task_episode_bands, strict=True), start=1
    ):
        episode_index = (episode_start + episode_end) // 2
        episode = dataset.meta.episodes[episode_index]
        if task_name not in episode.get("tasks", []):
            raise ValueError(
                f"Episode {episode_index} in configured band {episode_start}..{episode_end} does not match {task_name!r}"
            )
        middle = (int(episode["dataset_from_index"]) + int(episode["dataset_to_index"]) - 1) // 2
        task_sample = dict(dataset[middle])
        task_sample["task"] = task_name
        task_sources = [_to_bgr(task_sample[key]) for key in policy.config.image_keys]
        _, task_outputs = _run_debug_forward(policy, preprocessor, task_sample)
        task_episode_cards.append(
            _scenario_card(
                f"F{task_number}. Task {task_number}, episode {episode_index} middle",
                task_name,
                task_sources,
                task_outputs,
                0,
            )
        )

    rows = []
    for label, cards in (
        ("TEST 1 - Counterfactual: task-1 image with all three dataset instructions", task_cards),
        ("TEST 2 - Does the model behave correctly when a camera is unavailable?", camera_cards),
        ("TEST 3 - Start, middle and end of one episode (robot movement overview)", temporal_cards),
        ("TEST 4 - Middle frame from each of the three real dataset tasks", task_episode_cards),
    ):
        row = np.hstack(cards)
        row_header = np.full((42, row.shape[1], 3), (48, 66, 86), dtype=np.uint8)
        cv2.putText(
            row_header, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA
        )
        rows.append(np.vstack([row_header, row]))
    max_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        canvas = np.full((row.shape[0], max_width, 3), (18, 20, 24), dtype=np.uint8)
        canvas[:, : row.shape[1]] = row
        padded_rows.append(canvas)
    dashboard = np.vstack(padded_rows)
    cv2.imwrite(str(args.output_dir / "scenario_comparison.png"), dashboard)
    readme = np.full((110, dashboard.shape[1], 3), (18, 20, 24), dtype=np.uint8)
    cv2.putText(
        readme,
        "White contour = strongest visual focus. Compare the right half of cards in the same test row.",
        (18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        readme,
        "Healthy model: focus follows instruction, selected camera, and moving objects across start/middle/end frames.",
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.output_dir / "scenario_readme.png"), readme)


def main() -> None:
    args = _parse_args()
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in checkpoint: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_root}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(args.dataset_root.name, root=args.dataset_root)
    dataset_index = _episode_frame_index(dataset, args.episode_index, args.frame_offset)
    sample: dict[str, Any] = dict(dataset[dataset_index])
    if args.instruction is not None:
        sample["task"] = args.instruction

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(args.checkpoint, config=config).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    action_chunk, outputs = _run_debug_forward(policy, preprocessor, sample)
    del postprocessor  # Makes it explicit that action values below are model-space diagnostics.

    source_images = [_to_bgr(sample[key]) for key in config.image_keys]
    panels = []
    for index, (camera_name, source) in enumerate(zip(config.image_keys, source_images, strict=True)):
        panels.extend(_write_camera_report(args.output_dir, camera_name, source, outputs, index))
    panels.extend(_write_cross_attention(args.output_dir, source_images, outputs, "alignment"))
    panels.extend(_write_cross_attention(args.output_dir, source_images, outputs, "grounding"))
    _write_query_atlas(args.output_dir, source_images, outputs, "alignment")
    _write_query_atlas(args.output_dir, source_images, outputs, "grounding")
    _write_role_maps(args.output_dir, source_images, outputs)
    _layer_walkthrough(args.output_dir, source_images, outputs, action_chunk)
    body = _tile(panels, columns=2)
    report_image = np.vstack([_flow_header(body.shape[1], str(sample["task"])), body])
    cv2.imwrite(str(args.output_dir / "report.png"), report_image)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_index": dataset_index,
        "episode_index": args.episode_index,
        "frame_offset": args.frame_offset,
        "task": sample["task"],
        "camera_keys": list(config.image_keys),
        "input_shapes": {key: list(sample[key].shape) for key in config.image_keys},
        "tensor_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "action_chunk_model_space": action_chunk[0].cpu().tolist(),
        "notes": [
            "Alignment and cross-attention maps are meaningful only with trained checkpoint weights.",
            "Grounding PCA colours are a per-frame projection: compare structure within one report, not colours across reports.",
            "The action chunk is before the policy postprocessor; use it for debugging, not robot execution.",
            "The white outline on an overlay encloses its strongest 10 percent of patches.",
            "The query-atlas files reveal whether the eight latent queries focus on distinct regions or collapse together.",
            "V2 has no shared visual attention prior; role_specific_maps.png shows source/destination readouts.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.scenario_suite:
        _run_scenario_suite(args, dataset, policy, preprocessor)
    print(f"Wrote TIPS-Ground-VLA vision debug report to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
