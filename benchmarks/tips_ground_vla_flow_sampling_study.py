"""Evaluate saved flow heads across Euler budgets and fixed inference policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.tips_ground_vla_flow_matching import (
    benchmark_latency,
    evaluate,
    make_config,
)
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import TIPSGroundVLAPolicy


def run(args: argparse.Namespace) -> None:
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    train_actions = torch.cat([sample["action"] for sample in cache["train"]])
    train_min, train_max = train_actions.amin((0, 1)), train_actions.amax((0, 1))
    rows = []
    predictions = {}
    for steps in args.flow_steps:
        config = make_config(args.head, steps)
        policy = TIPSGroundVLAPolicy(config).cuda().eval()
        policy.load_state_dict(artifact["model"])
        for noise in ("random", "zero"):
            values, prediction, target, tasks = evaluate(
                policy,
                cache["validation"],
                sampling_seed=args.sampling_seed,
                train_min=train_min,
                train_max=train_max,
                inference_noise=noise,
            )
            runtime = benchmark_latency(policy, cache["validation"][0], args.latency_repeats)
            row = {"flow_steps": steps, "inference_noise": noise, **values, **runtime}
            rows.append(row)
            predictions[f"{steps}_{noise}"] = prediction
        del policy
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    torch.save(
        {"rows": rows, "predictions": predictions, "target": target, "tasks": tasks},
        args.output.with_suffix(".pt"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=("flow", "architecture_aware_flow"), required=True)
    parser.add_argument("--flow-steps", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--sampling-seed", type=int, default=314159)
    parser.add_argument("--latency-repeats", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
