# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import importlib.util
import json
from pathlib import Path

import pytest

from lerobot.policies.turbovla.libero_benchmark import (
    TURBOVLA_LIBERO_EPISODE_LENGTHS,
    TURBOVLA_LIBERO_SUITES,
)
from lerobot.utils.eval_stats import success_summary


@pytest.fixture(scope="module")
def runner_module():
    path = Path(__file__).parents[3] / "benchmarks" / "run_turbovla_libero_benchmark.py"
    spec = importlib.util.spec_from_file_location("turbovla_benchmark_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_commands_use_the_official_individual_rollout_horizons(runner_module, tmp_path) -> None:
    for suite, episode_length in TURBOVLA_LIBERO_EPISODE_LENGTHS.items():
        command = runner_module.build_eval_command(tmp_path / "converted", tmp_path / suite, 50, suite)
        assert f"--env.task={suite}" in command
        assert f"--env.episode_length={episode_length}" in command
        assert "--env.observation_height=256" in command
        assert "--env.observation_width=256" in command


def test_parallel_suite_outputs_merge_into_a_complete_transparent_report(runner_module, tmp_path) -> None:
    eval_dir = tmp_path / "eval"
    for task_id, suite in enumerate(TURBOVLA_LIBERO_SUITES):
        successes = [task_id % 2 == 0] * 50
        stats = success_summary(successes)
        info = {
            "per_task": [
                {
                    "task_group": suite,
                    "task_id": 0,
                    "metrics": {
                        "successes": successes,
                        "sum_rewards": [float(value) for value in successes],
                        "max_rewards": [float(value) for value in successes],
                    },
                }
            ],
            "per_group": {
                suite: {
                    **stats,
                    "avg_sum_reward": sum(successes) / len(successes),
                    "avg_max_reward": sum(successes) / len(successes),
                    "video_paths": [],
                    "predicted_video_paths": [],
                }
            },
            "overall": stats,
        }
        suite_dir = eval_dir / suite
        suite_dir.mkdir(parents=True)
        (suite_dir / "eval_info.json").write_text(json.dumps(info))

    merged = runner_module.merge_parallel_suite_outputs(eval_dir, wall_time_s=12.5)

    assert set(merged["per_group"]) == set(TURBOVLA_LIBERO_SUITES)
    assert merged["overall"]["n_episodes"] == 200
    assert merged["overall"]["n_success"] == 100
    assert merged["overall"]["eval_s"] == 12.5
    assert (eval_dir / "eval_info.json").is_file()
    assert (eval_dir / "turbovla_libero_report.json").is_file()
