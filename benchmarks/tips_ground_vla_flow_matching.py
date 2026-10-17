"""Bounded, matched Action-Head experiments for TIPS-Ground-VLA.

The runner is intentionally offline-only.  It consumes an existing episode-
disjoint cache, initializes every candidate from the same student checkpoint,
disables teacher objectives, and never opens a robot connection.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tips_ground_vla.configuration_tips_ground_vla import TIPSGroundVLAConfig
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import TIPSGroundVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE

HEADS = ("direct", "flow", "architecture_aware_flow")
HEAD_PREFIXES = (
    "model.action_queries",
    "model.action_context",
    "model.decoder",
    "model.action_out",
    "model.flow_head",
)


def make_config(head: str, flow_steps: int = 4) -> TIPSGroundVLAConfig:
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
        flow_num_steps=flow_steps,
        lambda_tips=0.0,
        lambda_tips_text=0.0,
        lambda_dino=0.0,
        local_files_only=True,
    )


def current(sample: dict) -> dict:
    return {
        "observation.images.top": sample["observation.images.top"][:, -1],
        "observation.images.wrist": sample["observation.images.wrist"][:, -1],
        OBS_STATE: sample[OBS_STATE][:, -1],
        OBS_LANGUAGE: sample[OBS_LANGUAGE],
        ACTION: sample[ACTION],
        "action_is_pad": sample["action_is_pad"],
    }


def merge(items: list[dict]) -> dict:
    merged = {}
    for key in items[0]:
        values = [item[key] for item in items]
        merged[key] = sum(values, []) if isinstance(values[0], list) else torch.cat(values)
    return merged


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def derivative(value: Tensor, order: int) -> Tensor:
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
    return value


def reversal_rate(value: Tensor, threshold: float = 0.02) -> Tensor:
    velocity = derivative(value, 1)
    before, after = velocity[:, :-1], velocity[:, 1:]
    valid = (before.abs() >= threshold) & (after.abs() >= threshold)
    return ((before.sign() != after.sign()) & valid).float().mean()


def stopping_rate(value: Tensor, threshold: float = 0.02) -> Tensor:
    return (derivative(value, 1).abs() < threshold).float().mean()


def metrics(
    prediction: Tensor, target: Tensor, tasks: list[str], train_min: Tensor, train_max: Tensor
) -> dict:
    error = (prediction - target).abs()
    motion = derivative(target, 1).abs().mean((1, 2))
    high = motion >= motion.quantile(0.75)
    velocity_error = (derivative(prediction, 1) - derivative(target, 1)).abs().mean()
    acceleration_error = (derivative(prediction, 2) - derivative(target, 2)).abs().mean()
    jerk_error = (derivative(prediction, 3) - derivative(target, 3)).abs().mean()
    result = {
        "full32_mae": error.mean().item(),
        "first4_mae": error[:, :4].mean().item(),
        "high_motion_full32_mae": error[high].mean().item(),
        "high_motion_first4_mae": error[high, :4].mean().item(),
        "action_delta_mae": velocity_error.item(),
        "velocity_mae": velocity_error.item(),
        "acceleration_mae": acceleration_error.item(),
        "jerk_mae": jerk_error.item(),
        "temporal_variance_ratio": (
            derivative(prediction, 1).square().mean() / derivative(target, 1).square().mean().clamp_min(1e-8)
        ).item(),
        "reversal_rate_prediction": reversal_rate(prediction).item(),
        "reversal_rate_expert": reversal_rate(target).item(),
        "stopping_rate_prediction": stopping_rate(prediction).item(),
        "stopping_rate_expert": stopping_rate(target).item(),
        "action_range_violation_rate": (
            ((prediction < train_min[None, None]) | (prediction > train_max[None, None]))
            .float()
            .mean()
            .item()
        ),
        "per_task": {},
    }
    for task in sorted(set(tasks)):
        selected = torch.tensor([item == task for item in tasks])
        result["per_task"][task] = {
            "full32_mae": error[selected].mean().item(),
            "first4_mae": error[selected, :4].mean().item(),
        }
    return result


def load_matched_state(policy: TIPSGroundVLAPolicy, reference: dict[str, Tensor], head: str) -> None:
    own = policy.state_dict()
    copied = {}
    for key, value in reference.items():
        is_old_head = key.startswith(HEAD_PREFIXES)
        if key in own and own[key].shape == value.shape and (head == "direct" or not is_old_head):
            copied[key] = value
    missing, unexpected = policy.load_state_dict(copied, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    non_head_missing = [
        key
        for key in missing
        if not key.startswith(HEAD_PREFIXES) and not key.startswith("model.tips_text_projection")
    ]
    if non_head_missing:
        raise RuntimeError(f"missing shared student keys: {non_head_missing}")


def freeze_for_head_comparison(policy: TIPSGroundVLAPolicy, head: str) -> list[Tensor]:
    trainable = []
    for name, parameter in policy.named_parameters():
        enabled = name.startswith(HEAD_PREFIXES)
        # The identity state anchor is part of the action representation, not a tuned upstream feature.
        if name.startswith("model.action_anchor"):
            enabled = False
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError(f"no trainable parameters selected for {head}")
    return trainable


@torch.inference_mode()
def evaluate(
    policy: TIPSGroundVLAPolicy,
    samples: list[dict],
    *,
    sampling_seed: int,
    train_min: Tensor,
    train_max: Tensor,
    inference_noise: str = "random",
) -> tuple[dict, Tensor, Tensor, list[str]]:
    policy.eval()
    predictions, targets, tasks = [], [], []
    for start in range(0, len(samples), 3):
        batch = to_device(
            merge([current(sample) for sample in samples[start : start + 3]]), policy.config.device
        )
        torch.manual_seed(sampling_seed + start)
        model_batch = {key: value for key, value in batch.items() if key != ACTION}
        flow_noise = None
        if policy.config.action_head != "direct" and inference_noise == "zero":
            flow_noise = torch.zeros(
                len(batch[OBS_LANGUAGE]),
                policy.config.chunk_size,
                policy.config.action_dim,
                device=policy.config.device,
            )
        prediction = policy.predict_action_chunk(model_batch, flow_noise=flow_noise)
        predictions.append(prediction.cpu())
        targets.append(batch[ACTION].cpu())
        tasks.extend(batch[OBS_LANGUAGE])
    prediction, target = torch.cat(predictions), torch.cat(targets)
    return metrics(prediction, target, tasks, train_min, train_max), prediction, target, tasks


@torch.inference_mode()
def benchmark_latency(policy: TIPSGroundVLAPolicy, sample: dict, repeats: int) -> dict:
    batch = to_device(
        {key: value for key, value in current(sample).items() if key != ACTION}, policy.config.device
    )
    torch.cuda.reset_peak_memory_stats()
    for index in range(5):
        torch.manual_seed(10_000 + index)
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(repeats):
        torch.manual_seed(20_000 + index)
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    return {
        "full_student_latency_ms": 1000 * (time.perf_counter() - started) / repeats,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    reference = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    train_actions = torch.cat([sample[ACTION] for sample in cache["train"]])
    train_min, train_max = train_actions.amin((0, 1)), train_actions.amax((0, 1))
    tasks = sorted({sample[OBS_LANGUAGE][0] for sample in cache["train"]})
    by_task = {
        task: [index for index, sample in enumerate(cache["train"]) if sample[OBS_LANGUAGE][0] == task]
        for task in tasks
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for head in args.heads:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        policy = TIPSGroundVLAPolicy(make_config(head, args.flow_steps)).to(device)
        load_matched_state(policy, reference, head)
        initial_state = copy.deepcopy(policy.state_dict())
        trainable = freeze_for_head_comparison(policy, head)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
        curve, started = [], time.monotonic()
        for step in range(1, args.steps + 1):
            indices = [random.choice(by_task[task]) for task in tasks]
            batch = to_device(merge([current(cache["train"][index]) for index in indices]), device)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, losses = policy(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation, _, _, _ = evaluate(
                    policy,
                    cache["validation"],
                    sampling_seed=args.sampling_seed,
                    train_min=train_min,
                    train_max=train_max,
                    inference_noise=args.inference_noise,
                )
                point = {"step": step, "grad_norm": float(grad_norm), **losses, **validation}
                curve.append(point)
                print(head, json.dumps(point, sort_keys=True), flush=True)
        validation, prediction, target, validation_tasks = evaluate(
            policy,
            cache["validation"],
            sampling_seed=args.sampling_seed,
            train_min=train_min,
            train_max=train_max,
            inference_noise=args.inference_noise,
        )
        stability = []
        if head != "direct":
            for offset in range(args.stability_seeds):
                _, sampled, _, _ = evaluate(
                    policy,
                    cache["validation"],
                    sampling_seed=args.sampling_seed + offset,
                    train_min=train_min,
                    train_max=train_max,
                    inference_noise=args.inference_noise,
                )
                stability.append(sampled)
        payload = {
            "head": head,
            "seed": args.seed,
            "sampling_seed": args.sampling_seed,
            "inference_noise": args.inference_noise,
            "steps": args.steps,
            "flow_steps": args.flow_steps if head != "direct" else 0,
            "head_only_training": True,
            "teacher_weights": {"lambda_tips": 0.0, "lambda_tips_text": 0.0, "lambda_dino": 0.0},
            "elapsed_seconds": time.monotonic() - started,
            "metrics": validation,
            "sampling_stability_mean_std": (
                torch.stack(stability).std(0).mean().item() if len(stability) > 1 else 0.0
            ),
            "runtime": benchmark_latency(policy, cache["validation"][0], args.latency_repeats),
            "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "curve": curve,
            "cache_protocol": {
                "train_episodes": len(cache["train_episodes"]),
                "validation_episodes": len(cache["validation_episodes"]),
                "train_windows": len(cache["train"]),
                "validation_windows": len(cache["validation"]),
            },
        }
        torch.save(
            {
                "model": policy.state_dict(),
                "initial_model": initial_state,
                "prediction": prediction,
                "target": target,
                "tasks": validation_tasks,
                "result": payload,
            },
            args.output / f"{head}_seed{args.seed}.pt",
        )
        (args.output / f"{head}_seed{args.seed}.json").write_text(json.dumps(payload, indent=2))
        summary[head] = payload
        del policy, optimizer
        torch.cuda.empty_cache()
    (args.output / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--sampling-seed", type=int, default=314159)
    parser.add_argument("--inference-noise", choices=("random", "zero"), default="random")
    parser.add_argument("--stability-seeds", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
