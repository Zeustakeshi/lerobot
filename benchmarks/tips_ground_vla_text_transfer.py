"""Bounded matched TIPSv2 text-transfer experiment.

This runner deliberately uses the representative episode-disjoint cache from
the preceding TIPS-Ground-VLA investigation.  It never launches a robot and it
never puts TIPSv2 in the deployable policy graph.
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

VARIANTS = {
    "action": (0.0, 0.0, False, "projection", "constant"),
    "semantic": (0.005, 0.0, False, "projection", "constant"),
    "text": (0.0, 0.01, False, "projection", "constant"),
    "semantic_text": (0.005, 0.01, False, "projection", "constant"),
    "text_shuffled": (0.0, 0.01, True, "projection", "constant"),
    # Action-relevant alternatives: supervise the Alignment attention that is
    # consumed by fusion instead of auxiliary-only semantic projections.
    "attention_0002": (0.0002, 0.0, False, "alignment_attention", "constant"),
    "attention_001": (0.001, 0.0, False, "alignment_attention", "constant"),
    "attention_warmup_001": (0.001, 0.0, False, "alignment_attention", "late"),
    "semantic_early": (0.005, 0.0, False, "projection", "early"),
    "semantic_decay": (0.005, 0.0, False, "projection", "decay"),
    "semantic_action_01": (0.005, 0.0, False, "projection", "constant"),
    "semantic_action_025": (0.005, 0.0, False, "projection", "constant"),
}

SEMANTIC_ACTION_SCALE = {"semantic_action_01": 0.1, "semantic_action_025": 0.25}


def _config(variant: str) -> TIPSGroundVLAConfig:
    semantic, text, shuffled, target, _ = VARIANTS[variant]
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
        lambda_tips=semantic,
        tips_semantic_target=target,
        tips_semantic_action_scale=SEMANTIC_ACTION_SCALE.get(variant, 0.0),
        lambda_tips_text=text,
        tips_text_shuffle_targets=shuffled,
        lambda_dino=0.0,
        local_files_only=True,
    )


def _current(sample: dict, teacher: Tensor | None = None, text: Tensor | None = None) -> dict:
    batch = {
        "observation.images.top": sample["observation.images.top"][:, -1],
        "observation.images.wrist": sample["observation.images.wrist"][:, -1],
        OBS_STATE: sample[OBS_STATE][:, -1],
        OBS_LANGUAGE: sample[OBS_LANGUAGE],
        ACTION: sample[ACTION],
        "action_is_pad": sample["action_is_pad"],
    }
    if teacher is not None:
        batch["tips_teacher_semantic"] = teacher[None]
    if text is not None:
        batch["tips_teacher_text"] = text[None]
    return batch


def _merge(items: list[dict]) -> dict:
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        result[key] = sum(values, []) if isinstance(values[0], list) else torch.cat(values)
    return result


def _to(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _metrics(prediction: Tensor, target: Tensor, tasks: list[str]) -> dict:
    error = (prediction - target).abs()
    motion = (target[:, 1:] - target[:, :-1]).abs().mean((1, 2))
    high = motion >= motion.quantile(0.75)
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    result = {
        "full32_mae": error.mean().item(),
        "first4_mae": error[:, :4].mean().item(),
        "high_motion_full32_mae": error[high].mean().item(),
        "high_motion_first4_mae": error[high, :4].mean().item(),
        "action_delta_mae": (pred_delta - target_delta).abs().mean().item(),
        "temporal_variance_ratio": (
            pred_delta.square().mean() / target_delta.square().mean().clamp_min(1e-8)
        ).item(),
        "per_task": {},
    }
    for task in sorted(set(tasks)):
        selected = torch.tensor([value == task for value in tasks])
        result["per_task"][task] = {
            "full32_mae": error[selected].mean().item(),
            "first4_mae": error[selected, :4].mean().item(),
        }
    return result


@torch.inference_mode()
def _evaluate(policy: TIPSGroundVLAPolicy, samples: list[dict], semantic: Tensor, text: dict[str, Tensor]):
    policy.eval()
    predictions, targets, tasks, projected = [], [], [], []
    for start in range(0, len(samples), 3):
        group = []
        for index, sample in enumerate(samples[start : start + 3], start):
            task = sample[OBS_LANGUAGE][0]
            group.append(_current(sample, semantic[index], text[task]))
            tasks.append(task)
        batch = _to(_merge(group), policy.config.device)
        images, state, language, content = policy._inputs(batch)
        pred, output = policy.model(images, state, language, patch_content_fraction=content)
        predictions.append(pred.cpu())
        targets.append(batch[ACTION].cpu())
        projected.append(policy.model.tips_text_projection(output["global_text"]).cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    projected = torch.nn.functional.normalize(torch.cat(projected).float(), dim=-1)
    teacher = torch.nn.functional.normalize(torch.stack([text[task] for task in tasks]).float(), dim=-1)
    pair_mask = ~torch.eye(len(tasks), dtype=torch.bool)
    language = {
        "teacher_student_cosine": (projected * teacher).sum(-1).mean().item(),
        "relational_mae": ((projected @ projected.T) - (teacher @ teacher.T))[pair_mask].abs().mean().item(),
        "student_off_diagonal_std": (projected @ projected.T)[pair_mask].std().item(),
    }
    return _metrics(prediction, target, tasks), language, prediction, target


def _teacher_text(path: Path, tasks: list[str]) -> dict[str, Tensor]:
    from transformers import AutoModel

    teacher = AutoModel.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True, dtype=torch.float16
    ).cuda().eval()
    with torch.inference_mode():
        features = teacher.encode_text(tasks).float().cpu()
    del teacher
    torch.cuda.empty_cache()
    return dict(zip(tasks, features, strict=True))


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    semantic = torch.load(args.semantic, map_location="cpu", weights_only=False)
    tasks = sorted({sample[OBS_LANGUAGE][0] for sample in cache["train"]})
    text = _teacher_text(args.tips, tasks)
    reference = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]

    # Instantiate once so every variant receives an identical adapter init too.
    torch.manual_seed(args.seed)
    initial = TIPSGroundVLAPolicy(_config("action"))
    missing, unexpected = initial.load_state_dict(reference, strict=False)
    allowed = {f"model.tips_text_projection.{suffix}" for suffix in ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias")}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    initial_state = copy.deepcopy(initial.state_dict())
    del initial

    by_task = {
        task: [index for index, sample in enumerate(cache["train"]) if sample[OBS_LANGUAGE][0] == task]
        for task in tasks
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in args.variants:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        policy = TIPSGroundVLAPolicy(_config(variant)).to(device)
        policy.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(policy.get_optim_params(), betas=(0.9, 0.95))
        curve, started = [], time.monotonic()
        for step in range(1, args.steps + 1):
            configured_tips, _, _, _, schedule = VARIANTS[variant]
            if schedule == "late":
                policy.config.lambda_tips = configured_tips if step > args.steps // 2 else 0.0
            elif schedule == "early":
                policy.config.lambda_tips = configured_tips if step <= args.steps // 2 else 0.0
            elif schedule == "decay":
                policy.config.lambda_tips = configured_tips * max(0.0, 1 - step / args.steps)
            else:
                policy.config.lambda_tips = configured_tips
            indices = [random.choice(by_task[task]) for task in tasks]
            group = [
                _current(cache["train"][index], semantic["train"]["tips"][index], text[tasks[pos]])
                for pos, index in enumerate(indices)
            ]
            batch = _to(_merge(group), device)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, losses = policy(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics, language, _, _ = _evaluate(
                    policy, cache["validation"], semantic["validation"]["tips"], text
                )
                curve.append({"step": step, "grad_norm": float(grad_norm), **losses, **metrics, **language})
                print(variant, json.dumps(curve[-1], sort_keys=True), flush=True)
        metrics, language, prediction, target = _evaluate(
            policy, cache["validation"], semantic["validation"]["tips"], text
        )
        payload = {
            "variant": variant,
            "seed": args.seed,
            "steps": args.steps,
            "elapsed_seconds": time.monotonic() - started,
            "weights": {
                "lambda_tips": policy.config.lambda_tips,
                "lambda_tips_text": policy.config.lambda_tips_text,
                "relational": policy.config.tips_text_relational_weight,
                "shuffled": policy.config.tips_text_shuffle_targets,
                "semantic_target": policy.config.tips_semantic_target,
                "semantic_schedule": VARIANTS[variant][4],
                "semantic_action_scale": policy.config.tips_semantic_action_scale,
            },
            "metrics": metrics,
            "language": language,
            "curve": curve,
            "student_parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "inference_parameters_without_text_adapter": sum(
                parameter.numel()
                for name, parameter in policy.named_parameters()
                if "tips_text_projection" not in name
            ),
        }
        (args.output / f"{variant}_seed{args.seed}.json").write_text(json.dumps(payload, indent=2))
        torch.save(
            {"model": policy.state_dict(), "prediction": prediction, "target": target, "result": payload},
            args.output / f"{variant}_seed{args.seed}.pt",
        )
        summary[variant] = payload
        del policy, optimizer
        torch.cuda.empty_cache()
    (args.output / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261017)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
