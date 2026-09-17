# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace

from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.libero_benchmark import (
    TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES,
    TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES,
    TURBOVLA_LIBERO_SUITES,
    check_turbovla_libero_eval_contract,
    summarize_turbovla_libero_results,
    turbovla_libero_benchmark_command,
)


def make_eval_cfg(**overrides):
    policy = TurboVLAConfig(device="cpu")
    policy.use_amp = True
    env = SimpleNamespace(
        type="libero",
        task=",".join(TURBOVLA_LIBERO_SUITES),
        task_ids=None,
        fps=20,
        obs_type="pixels_agent_pos",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        control_mode="relative",
        camera_name_mapping=None,
        observation_height=360,
        observation_width=360,
    )
    eval_cfg = SimpleNamespace(n_episodes=50, batch_size=1, use_async_envs=True)
    cfg = SimpleNamespace(policy=policy, env=env, eval=eval_cfg, seed=7)
    for path, value in overrides.items():
        target = cfg
        parts = path.split("__")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


def test_full_libero_contract_is_benchmark_comparable() -> None:
    check = check_turbovla_libero_eval_contract(make_eval_cfg(), strict=True)
    assert check.ok
    assert check.benchmark_comparable


def test_required_benchmark_values_are_declared_and_left_blank() -> None:
    assert "aggregate_success_rate" in TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES
    assert "model_only_latency_ms_p95" in TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES
    assert "vram_reserved_gb_peak" in TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES
    assert set(TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES) == set(
        TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES
    )
    assert all(value is None for value in TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES.values())


def test_smoke_eval_is_allowed_but_not_benchmark_comparable() -> None:
    cfg = make_eval_cfg(eval__n_episodes=1, env__task="libero_spatial", seed=123)
    check = check_turbovla_libero_eval_contract(cfg)
    assert check.ok
    assert not check.benchmark_comparable
    assert any("eval.n_episodes=50" in warning for warning in check.warnings)
    assert any("seed=7" in warning for warning in check.warnings)


def test_contract_rejects_wrong_control_or_camera_mapping() -> None:
    cfg = make_eval_cfg(env__control_mode="absolute", env__camera_name_mapping={"agentview_image": "camera1"})
    check = check_turbovla_libero_eval_contract(cfg)
    assert not check.ok
    assert any("control_mode" in error for error in check.errors)
    assert any("camera_name_mapping" in error for error in check.errors)


def test_benchmark_command_uses_libero_defaults() -> None:
    command = turbovla_libero_benchmark_command("org/turbovla-libero", "/tmp/out")
    assert "--env.type=libero" in command
    assert f"--env.task={','.join(TURBOVLA_LIBERO_SUITES)}" in command
    assert "--eval.n_episodes=50" in command
    assert "--policy.use_amp=true" in command


def test_summarize_turbovla_libero_results_reports_confidence_intervals() -> None:
    report = summarize_turbovla_libero_results(
        {
            "per_task": [
                {
                    "task_group": "libero_spatial",
                    "task_id": 0,
                    "metrics": {"successes": [True, False, True]},
                }
            ],
            "per_group": {
                "libero_spatial": {
                    "n_episodes": 3,
                    "n_success": 2,
                    "pc_success": 66.6666666667,
                    "pc_success_ci95": [20.8, 93.9],
                }
            },
            "overall": {
                "n_episodes": 3,
                "n_success": 2,
                "pc_success": 66.6666666667,
                "pc_success_ci95": [20.8, 93.9],
                "eval_s": 12.0,
                "eval_ep_s": 4.0,
            },
        }
    )
    assert report["contract"]["trials_per_task"] == 50
    assert report["pending_benchmark_values"]["aggregate_success_rate"] is None
    assert report["per_task"][0]["n_success"] == 2
    assert len(report["per_task"][0]["pc_success_ci95"]) == 2
    assert report["per_suite"]["libero_spatial"]["n_episodes"] == 3
