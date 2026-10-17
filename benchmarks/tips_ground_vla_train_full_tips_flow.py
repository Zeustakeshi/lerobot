"""Jointly fine-tune the complete student with TIPSv2 targets and flow actions.

This experiment is offline-only.  DINO is fixed to zero and this runner has no
DINO import, target generation, or DINO feature path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tips_ground_vla.configuration_tips_ground_vla import TIPSGroundVLAConfig
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import TIPSGroundVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE

JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class ShardedSamples:
    """Episode-local LRU reader so training does not load the image cache into RAM."""

    def __init__(self, root: Path, records: list[dict], capacity: int = 4):
        self.root = root
        self.records = records
        self.capacity = capacity
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.offsets = [0]
        for record in records:
            self.offsets.append(self.offsets[-1] + record["count"])

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        shard_index = bisect_right(self.offsets, index) - 1
        if shard_index not in self.cache:
            path = self.root / self.records[shard_index]["path"]
            self.cache[shard_index] = torch.load(path, map_location="cpu", weights_only=False)
            self.cache.move_to_end(shard_index)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(shard_index)
        shard = self.cache[shard_index]
        return shard["samples"][index - self.offsets[shard_index]]


def make_config(args: argparse.Namespace, head: str = "architecture_aware_flow") -> TIPSGroundVLAConfig:
    images = {
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 224, 336)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 224, 336)),
    }
    return TIPSGroundVLAConfig(
        input_features={
            **images,
            OBS_STATE: PolicyFeature(FeatureType.STATE, (6,)),
            OBS_LANGUAGE: PolicyFeature(FeatureType.LANGUAGE, (1,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (6,))},
        infer_from_dataset=False,
        image_keys=tuple(images),
        state_dim=6,
        action_dim=6,
        action_head=head,
        action_delta_from_state=True,
        flow_num_steps=args.flow_steps,
        flow_num_layers=4,
        flow_reconstruction_weight=args.reconstruction_weight,
        flow_motion_weight=args.motion_weight,
        first_four_loss_weight=args.first_four_weight,
        lambda_tips=args.lambda_tips,
        tips_semantic_target="alignment_attention",
        tips_semantic_action_scale=0.0,
        lambda_tips_text=args.lambda_tips_text,
        tips_text_relational_weight=1.0,
        lambda_dino=0.0,
        teacher_prepare="off",
        local_files_only=True,
    )


def current(sample: dict, patch_content_fraction: Tensor) -> dict:
    return {
        "observation.images.top": sample["observation.images.top"][:, -1],
        "observation.images.wrist": sample["observation.images.wrist"][:, -1],
        OBS_STATE: sample[OBS_STATE][:, -1],
        OBS_LANGUAGE: sample[OBS_LANGUAGE],
        ACTION: sample[ACTION],
        "action_is_pad": sample["action_is_pad"],
        # Preserve the original-source content mask after caching letterboxed images.
        "patch_content_fraction": patch_content_fraction,
        "tips_teacher_semantic": sample["tips_teacher_semantic"][None],
        "tips_teacher_text": sample["tips_teacher_text"][None],
        "episode_index": sample["episode_index"],
        "frame_index": sample["frame_index"],
        "row": sample["row"],
    }


def merge(items: list[dict]) -> dict:
    merged = {}
    for key, first in items[0].items():
        values = [item[key] for item in items]
        if isinstance(first, list):
            merged[key] = sum(values, [])
        elif isinstance(first, Tensor):
            merged[key] = torch.cat(values)
        else:
            merged[key] = values
    return merged


def source_content_fraction(height: int, width: int) -> Tensor:
    """Recreate exact ViT-patch content fractions for the source video shape."""
    scale = min(224 / height, 336 / width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    top, left = (224 - resized_height) // 2, (336 - resized_width) // 2
    content = torch.zeros(1, 1, 224, 336)
    content[..., top : top + resized_height, left : left + resized_width] = 1
    fractions = torch.nn.functional.avg_pool2d(content, kernel_size=16, stride=16).squeeze(1)
    return fractions[:, None].expand(-1, 2, -1, -1).contiguous()


def cache_content_fraction(cache: dict) -> Tensor:
    info = json.loads((Path(cache["source"]) / "meta" / "info.json").read_text())
    camera = info["features"]["observation.images.top"]["info"]
    return source_content_fraction(int(camera["video.height"]), int(camera["video.width"]))


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def derivative(value: Tensor, order: int) -> Tensor:
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
    return value


def event_masks(value: Tensor, threshold: float = 0.02) -> tuple[Tensor, Tensor]:
    velocity = derivative(value, 1)
    stopped = velocity.abs() < threshold
    before, after = velocity[:, :-1], velocity[:, 1:]
    reversed_motion = (
        (before.abs() >= threshold) & (after.abs() >= threshold) & (before.sign() != after.sign())
    )
    return stopped, reversed_motion


def metrics(
    prediction: Tensor, target: Tensor, tasks: list[str], train_min: Tensor, train_max: Tensor
) -> dict:
    error = (prediction - target).abs()
    target_speed = derivative(target, 1).abs().mean((1, 2))
    high = target_speed >= target_speed.quantile(0.75)
    stop_pred, reverse_pred = event_masks(prediction)
    stop_true, reverse_true = event_masks(target)
    reversal_intersection = (reverse_pred & reverse_true).sum().float()
    result = {
        "first4_mae": error[:, :4].mean().item(),
        "full32_mae": error.mean().item(),
        "high_motion_first4_mae": error[high, :4].mean().item(),
        "high_motion_full32_mae": error[high].mean().item(),
        "action_delta_mae": (derivative(prediction, 1) - derivative(target, 1)).abs().mean().item(),
        "velocity_mae": (derivative(prediction, 1) - derivative(target, 1)).abs().mean().item(),
        "acceleration_mae": (derivative(prediction, 2) - derivative(target, 2)).abs().mean().item(),
        "jerk_mae": (derivative(prediction, 3) - derivative(target, 3)).abs().mean().item(),
        "temporal_variance_ratio": (
            derivative(prediction, 1).square().mean() / derivative(target, 1).square().mean().clamp_min(1e-8)
        ).item(),
        "stopping_rate_prediction": stop_pred.float().mean().item(),
        "stopping_rate_expert": stop_true.float().mean().item(),
        "stopping_agreement": (stop_pred == stop_true).float().mean().item(),
        "reversal_rate_prediction": reverse_pred.float().mean().item(),
        "reversal_rate_expert": reverse_true.float().mean().item(),
        "reversal_f1": (
            2 * reversal_intersection / (reverse_pred.sum() + reverse_true.sum()).clamp_min(1)
        ).item(),
        "direction_agreement": (derivative(prediction, 1).sign() == derivative(target, 1).sign())
        .float()
        .mean()
        .item(),
        "action_range_violation_rate": (
            ((prediction < train_min[None, None]) | (prediction > train_max[None, None]))
            .float()
            .mean()
            .item()
        ),
        "high_motion_count": int(high.sum()),
        "per_joint": {},
        "per_task": {},
    }
    for dim, name in enumerate(JOINTS):
        result["per_joint"][name] = {
            "first4_mae": error[:, :4, dim].mean().item(),
            "full32_mae": error[:, :, dim].mean().item(),
            "high_motion_first4_mae": error[high, :4, dim].mean().item(),
        }
    for task in sorted(set(tasks)):
        selected = torch.tensor([value == task for value in tasks])
        result["per_task"][task] = {
            "first4_mae": error[selected, :4].mean().item(),
            "full32_mae": error[selected].mean().item(),
        }
    result["per_horizon_mae"] = error.mean((0, 2)).tolist()
    return result


@torch.inference_mode()
def evaluate(
    policy: TIPSGroundVLAPolicy,
    samples: list[dict],
    train_min: Tensor,
    train_max: Tensor,
    *,
    batch_size: int,
    seed: int,
    patch_content_fraction: Tensor,
) -> tuple[dict, Tensor, Tensor, list[str]]:
    policy.eval()
    predictions, targets, tasks = [], [], []
    device = policy.config.device
    for start in range(0, len(samples), batch_size):
        group = [current(item, patch_content_fraction) for item in samples[start : start + batch_size]]
        batch = to_device(merge(group), device)
        noise_gen = torch.Generator(device=device).manual_seed(seed + start)
        noise = torch.randn(
            len(group),
            policy.config.chunk_size,
            policy.config.action_dim,
            device=device,
            generator=noise_gen,
        )
        model_batch = {
            key: value
            for key, value in batch.items()
            if key not in {ACTION, "tips_teacher_semantic", "tips_teacher_text"}
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = policy.predict_action_chunk(model_batch, flow_noise=noise)
        predictions.append(prediction.float().cpu())
        targets.append(batch[ACTION].float().cpu())
        tasks.extend(batch[OBS_LANGUAGE])
    pred, target = torch.cat(predictions), torch.cat(targets)
    return metrics(pred, target, tasks, train_min, train_max), pred, target, tasks


def set_learning_rates(optimizer: torch.optim.Optimizer, step: int, total: int, warmup: int) -> None:
    factor = min(1.0, step / max(1, warmup))
    progress = max(0.0, (step - warmup) / max(1, total - warmup))
    factor *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def save_checkpoint(path: Path, policy, optimizer, step, curve, best_score, best_step, args, seed):
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "curve": curve,
            "best_score": best_score,
            "best_step": best_step,
            "seed": seed,
            "configuration": _serializable_args(args),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def _serializable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def load_reference(policy: TIPSGroundVLAPolicy, source: Path) -> None:
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    state = artifact["model"]
    missing, unexpected = policy.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("model.flow_head")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"flow checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def run_one(args: argparse.Namespace, seed: int, cache: dict, output: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy = TIPSGroundVLAPolicy(make_config(args)).cuda()
    load_reference(policy, args.initial_checkpoint)
    policy.config.lambda_dino = 0.0
    train_min = cache["train_action_min"].float()
    train_max = cache["train_action_max"].float()
    optimizer_groups = policy.get_optim_params()
    for group in optimizer_groups:
        original_lr = group["lr"]
        group["base_lr"] = (
            args.text_lr if original_lr <= 2e-6 else args.vision_lr if original_lr <= 5e-6 else args.head_lr
        )
        group["lr"] = group["base_lr"]
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.95))
    start_step, curve, best_score, best_step = 0, [], float("inf"), -1
    if args.resume and (output / "latest.pt").exists():
        state = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, curve = state["step"], state["curve"]
        best_score, best_step = state["best_score"], state["best_step"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    samples = ShardedSamples(args.cache.parent, cache["train_shards"], capacity=4)
    validation = ShardedSamples(args.cache.parent, cache["validation_shards"], capacity=4)
    patch_content_fraction = cache["patch_content_fraction"]
    steps_per_epoch = math.ceil(len(samples) / args.batch_size)
    train_steps = args.max_steps
    output.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, dict] = {}
    snapshots_path = output / "trajectory_snapshots.pt"
    if args.resume and start_step and snapshots_path.exists():
        snapshots = torch.load(snapshots_path, map_location="cpu", weights_only=False)
    # Fixed validation examples represent low/median/high motion and a gripper
    # event; the exact indices are selected once and then reused for every plot.
    vmotion = torch.tensor([value for shard in cache["validation_shards"] for value in shard["motion"]])
    order = torch.argsort(vmotion)
    validation_events = [value for shard in cache["validation_shards"] for value in shard["events"]]
    reversal_idx = max(range(len(validation_events)), key=lambda i: validation_events[i]["reversals"])
    gripper_idx = max(range(len(validation_events)), key=lambda i: validation_events[i]["gripper_motion"])
    snapshot_indices = sorted(
        {
            int(order[0]),
            int(order[len(order) // 3]),
            int(order[2 * len(order) // 3]),
            int(order[-1]),
            reversal_idx,
            gripper_idx,
        }
    )
    started = time.monotonic()
    best_predictions = None
    stale_evals = 0
    if args.resume and start_step:
        stale_evals = int(state.get("stale_evals", 0))
    order_gen = torch.Generator().manual_seed(seed + 12_345)
    train_order = []
    episode_indices = []
    for shard_index, _record in enumerate(cache["train_shards"]):
        episode_indices.append(list(range(samples.offsets[shard_index], samples.offsets[shard_index + 1])))
    episode_order = torch.randperm(len(episode_indices), generator=order_gen).tolist()
    for episode_position in episode_order:
        local = episode_indices[episode_position]
        permutation = torch.randperm(len(local), generator=order_gen).tolist()
        train_order.extend(local[position] for position in permutation)
    cursor = 0
    if args.resume and start_step:
        train_order = state.get("train_order", train_order)
        cursor = state.get("train_cursor", 0)
        if "order_rng_state" in state:
            order_gen.set_state(state["order_rng_state"])

    def validate(step: int):
        nonlocal best_score, best_step, stale_evals, best_predictions
        result, pred, target, tasks = evaluate(
            policy,
            validation,
            train_min,
            train_max,
            batch_size=args.eval_batch_size,
            seed=args.sampling_seed,
            patch_content_fraction=patch_content_fraction,
        )
        eval_snapshot = {
            str(index): {
                "prediction": pred[index].tolist(),
                "target": target[index].tolist(),
                "task": tasks[index],
                "episode_index": validation[index]["episode_index"],
                "frame_index": validation[index]["frame_index"],
                "row": validation[index]["row"],
            }
            for index in snapshot_indices
        }
        snapshots[str(step)] = eval_snapshot
        score = result["first4_mae"] + 0.25 * result["full32_mae"]
        row = {"step": step, "validation_score": score, **result}
        print(seed, json.dumps(row, sort_keys=True), flush=True)
        curve.append(row)
        if score < best_score - args.min_improvement:
            best_score, best_step, stale_evals = score, step, 0
            best_predictions = {"prediction": pred, "target": target, "tasks": tasks}
            torch.save(
                {"model": policy.state_dict(), "step": step, "metrics": result, "seed": seed},
                output / "best.pt",
            )
        else:
            stale_evals += 1
        return result

    if start_step == 0:
        validate(0)
    policy.train()
    for step in range(start_step + 1, train_steps + 1):
        policy.train()
        if cursor + args.batch_size > len(train_order):
            episode_order = torch.randperm(len(episode_indices), generator=order_gen).tolist()
            train_order = []
            for episode_position in episode_order:
                local = episode_indices[episode_position]
                permutation = torch.randperm(len(local), generator=order_gen).tolist()
                train_order.extend(local[position] for position in permutation)
            cursor = 0
        indices = train_order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = to_device(
            merge([current(samples[index], patch_content_fraction) for index in indices]),
            torch.device("cuda"),
        )
        for key in ("episode_index", "frame_index", "row"):
            batch.pop(key, None)
        set_learning_rates(optimizer, step, train_steps, args.warmup_steps)
        if start_step:
            # Continue from the prior schedule's low-LR tail without the jump
            # that would occur if a longer cosine schedule were restarted.
            for group in optimizer.param_groups:
                group["lr"] *= args.resume_lr_multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, losses = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite full-student loss at step {step}: {losses}")
        loss.backward()
        params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        if step % args.eval_every == 0 or step == train_steps:
            validate(step)
            curve[-1]["train_objectives"] = losses
            curve[-1]["gradient_norm"] = float(grad_norm)
            torch.cuda.synchronize()
            checkpoint = {
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "curve": curve,
                "best_score": best_score,
                "best_step": best_step,
                "seed": seed,
                "configuration": _serializable_args(args),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "train_order": train_order,
                "train_cursor": cursor,
                "order_rng_state": order_gen.get_state(),
                "stale_evals": stale_evals,
                "last_train_losses": losses,
                "grad_norm": float(grad_norm),
            }
            torch.save(checkpoint, output / "latest.pt")
            (output / "metrics.json").write_text(json.dumps({"curve": curve}, indent=2))
            torch.save(snapshots, output / "trajectory_snapshots.pt")
            if stale_evals >= args.patience:
                print(
                    f"early stop at {step}; best step {best_step}; plateau patience={args.patience}",
                    flush=True,
                )
                break
        if step % 25 == 0:
            torch.cuda.synchronize()
            print(
                f"seed={seed} step={step}/{train_steps} epoch={step / steps_per_epoch:.2f} "
                f"loss={loss.item():.5f} grad={float(grad_norm):.3f} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )

    final_step = curve[-1]["step"]
    best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    policy.load_state_dict(best_state["model"])
    final_metrics, final_pred, final_target, final_tasks = evaluate(
        policy,
        validation,
        train_min,
        train_max,
        batch_size=args.eval_batch_size,
        seed=args.sampling_seed,
        patch_content_fraction=patch_content_fraction,
    )
    variability = []
    for sample_seed in args.variability_seeds:
        _, sampled, _, _ = evaluate(
            policy,
            validation,
            train_min,
            train_max,
            batch_size=args.eval_batch_size,
            seed=sample_seed,
            patch_content_fraction=patch_content_fraction,
        )
        variability.append(sampled)
    if variability:
        variability_tensor = torch.stack(variability)
        sampling_std = variability_tensor.std(0).mean().item()
        sampling_first4_std = variability_tensor[:, :, :4].std(0).mean().item()
    else:
        sampling_std = sampling_first4_std = 0.0
    torch.save(
        {
            "model": policy.state_dict(),
            "prediction": final_pred,
            "target": final_target,
            "tasks": final_tasks,
            "metrics": final_metrics,
            "step": best_state["step"],
            "seed": seed,
            "trajectory_snapshots": snapshots,
            "sample_motion": vmotion.tolist(),
            "sample_events": validation_events,
            "sample_metadata": [
                {
                    "episode_index": sample["episode_index"],
                    "frame_index": sample["frame_index"],
                    "row": sample["row"],
                }
                for sample in validation
            ],
        },
        output / "selected_checkpoint.pt",
    )
    summary = {
        "seed": seed,
        "best_step": best_state["step"],
        "last_step": final_step,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": final_metrics,
        "sampling_std_full": sampling_std,
        "sampling_std_first4": sampling_first4_std,
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "train_windows": len(samples),
        "validation_windows": len(validation),
        "training_config": _serializable_args(args),
        "curve": curve,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(args: argparse.Namespace) -> None:
    cache = json.loads(args.cache.read_text())
    cache["train_action_min"] = torch.as_tensor(cache["train_action_min"])
    cache["train_action_max"] = torch.as_tensor(cache["train_action_max"])
    cache["patch_content_fraction"] = cache_content_fraction(cache)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        seed_output = output / f"seed{seed}"
        summaries.append(run_one(args, seed, cache, seed_output))
    if args.direct_reference is not None:
        validation = ShardedSamples(args.cache.parent, cache["validation_shards"], capacity=4)
        direct = TIPSGroundVLAPolicy(make_config(args, "direct")).cuda().eval()
        reference = torch.load(args.direct_reference, map_location="cpu", weights_only=False)
        missing, unexpected = direct.load_state_dict(reference["model"], strict=False)
        if unexpected or missing:
            raise RuntimeError(
                f"direct reference checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        direct_metrics, prediction, target, tasks = evaluate(
            direct,
            validation,
            torch.as_tensor(cache["train_action_min"]).float(),
            torch.as_tensor(cache["train_action_max"]).float(),
            batch_size=args.eval_batch_size,
            seed=args.sampling_seed,
            patch_content_fraction=cache["patch_content_fraction"],
        )
        direct_result = {
            "source_checkpoint": str(args.direct_reference),
            "evaluation_cache": str(args.cache),
            "metrics": direct_metrics,
        }
        torch.save(
            {
                "prediction": prediction,
                "target": target,
                "tasks": tasks,
                "metrics": direct_metrics,
                "source_checkpoint": str(args.direct_reference),
            },
            output / "historical_direct_on_final_validation.pt",
        )
        (output / "historical_direct_on_final_validation.json").write_text(
            json.dumps(direct_result, indent=2)
        )
        del direct
    (output / "summary_all_seeds.json").write_text(json.dumps(summaries, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261018])
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-improvement", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--vision-lr", type=float, default=1e-6)
    parser.add_argument("--text-lr", type=float, default=1e-6)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=0.1)
    parser.add_argument("--first-four-weight", type=float, default=2.0)
    parser.add_argument("--lambda-tips", type=float, default=0.0002)
    parser.add_argument("--lambda-tips-text", type=float, default=0.01)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=314159)
    parser.add_argument("--variability-seeds", nargs="*", type=int, default=[314160, 314161, 314162])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-lr-multiplier", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
