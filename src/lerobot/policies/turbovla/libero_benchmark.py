# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO benchmark contract for TurboVLA.

This module is intentionally lightweight: it imports no simulator packages, does
not construct the policy, and can run in CPU-only CI. The full benchmark still
requires converted weights plus a GPU LIBERO environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from lerobot.utils.eval_stats import success_summary

from .configuration_turbovla import LIBERO_IMAGE_KEYS, TurboVLAConfig

TURBOVLA_LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
TURBOVLA_LIBERO_TRIALS_PER_TASK = 50
TURBOVLA_LIBERO_CHUNK_SIZE = 12
TURBOVLA_LIBERO_OPEN_LOOP_ACTIONS = 12
TURBOVLA_LIBERO_ACTION_DIM = 7
TURBOVLA_LIBERO_STATE_DIM = 8
# TurboVLA's released VLA-Adapter rollout uses a distinct maximum action count
# for each suite. A single `--env.episode_length` cannot represent all suites.
TURBOVLA_LIBERO_EPISODE_LENGTHS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
TURBOVLA_LIBERO_SEED = 7
TURBOVLA_LIBERO_SUCCESS_TOLERANCE_POINTS = 2.0

# Values to fill only after running the full GPU LIBERO benchmark.
# Keep these blank in public checkpoint/model-card artifacts until measured with
# the command emitted by ``turbovla_libero_benchmark_command``.
TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES: tuple[str, ...] = (
    "source_checkpoint_sha256",
    "source_stats_sha256",
    "converted_checkpoint_revision",
    "lerobot_revision",
    "turbo_vla_upstream_revision",
    "libero_revision",
    "gpu_name",
    "gpu_driver",
    "cuda_version",
    "pytorch_version",
    "precision",
    "seed",
    "suites",
    "trials_per_task",
    "per_task_success_rate",
    "per_suite_success_rate",
    "aggregate_success_rate",
    "aggregate_success_ci95",
    "model_only_latency_ms_mean",
    "model_only_latency_ms_median",
    "model_only_latency_ms_p95",
    "end_to_end_latency_ms_mean",
    "end_to_end_latency_ms_median",
    "end_to_end_latency_ms_p95",
    "vram_allocated_gb_peak",
    "vram_reserved_gb_peak",
)
TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES: dict[str, Any] = dict.fromkeys(
    TURBOVLA_LIBERO_REQUIRED_BENCHMARK_VALUES
)


@dataclass(frozen=True)
class TurboVLALiberoEvalContract:
    """Serializable summary of the LIBERO settings required for benchmark parity."""

    suites: tuple[str, ...] = TURBOVLA_LIBERO_SUITES
    trials_per_task: int = TURBOVLA_LIBERO_TRIALS_PER_TASK
    seed: int = TURBOVLA_LIBERO_SEED
    state_order: tuple[str, ...] = (
        "eef_pos",
        "eef_axis_angle_from_quat",
        "gripper_qpos",
    )
    image_keys: tuple[str, ...] = LIBERO_IMAGE_KEYS
    camera_names: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image")
    observation_height: int = 256
    observation_width: int = 256
    control_mode: str = "relative"
    obs_type: str = "pixels_agent_pos"
    episode_lengths: dict[str, int] = field(default_factory=lambda: dict(TURBOVLA_LIBERO_EPISODE_LENGTHS))
    action_dim: int = TURBOVLA_LIBERO_ACTION_DIM
    state_dim: int = TURBOVLA_LIBERO_STATE_DIM
    chunk_size: int = TURBOVLA_LIBERO_CHUNK_SIZE
    open_loop_actions: int = TURBOVLA_LIBERO_OPEN_LOOP_ACTIONS
    precision: str = "bfloat16"
    success_tolerance_points: float = TURBOVLA_LIBERO_SUCCESS_TOLERANCE_POINTS


@dataclass
class TurboVLALiberoContractCheck:
    """Result of checking an eval config against the TurboVLA LIBERO contract."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def benchmark_comparable(self) -> bool:
        return self.ok and not self.warnings


def _as_suite_tuple(task: object) -> tuple[str, ...]:
    if isinstance(task, str):
        return tuple(item.strip() for item in task.split(",") if item.strip())
    return ()


def check_turbovla_libero_eval_contract(cfg: Any, *, strict: bool = False) -> TurboVLALiberoContractCheck:
    """Check whether an eval config matches the TurboVLA LIBERO benchmark contract.

    ``strict=False`` is used by ``lerobot-eval`` so smoke jobs with fewer trials can
    still run while logging why they are not benchmark-comparable. Unit tests and
    release automation can use ``strict=True`` to turn comparability warnings into
    hard failures.
    """

    contract = TurboVLALiberoEvalContract()
    result = TurboVLALiberoContractCheck()
    policy = getattr(cfg, "policy", None)
    env = getattr(cfg, "env", None)
    eval_cfg = getattr(cfg, "eval", None)

    if not isinstance(policy, TurboVLAConfig):
        result.errors.append("TurboVLA LIBERO eval requires a TurboVLAConfig policy.")
        return result
    if policy.variant != "libero":
        result.errors.append(
            f"TurboVLA LIBERO eval requires policy.variant='libero', got {policy.variant!r}."
        )
    if policy.image_keys != contract.image_keys:
        result.errors.append(f"TurboVLA LIBERO cameras must be ordered as {contract.image_keys!r}.")
    if policy.state_dim != contract.state_dim or policy.action_dim != contract.action_dim:
        result.errors.append(
            f"TurboVLA LIBERO requires state_dim={contract.state_dim} and action_dim={contract.action_dim}."
        )
    if policy.chunk_size != contract.chunk_size or policy.n_action_steps != contract.open_loop_actions:
        result.errors.append(
            "TurboVLA LIBERO requires "
            f"chunk_size={contract.chunk_size} and n_action_steps={contract.open_loop_actions}."
        )
    if policy.precision != contract.precision:
        result.warnings.append(
            f"Expected policy.precision={contract.precision!r} for published LIBERO parity, "
            f"got {policy.precision!r}."
        )

    if getattr(env, "type", None) != "libero":
        result.errors.append("TurboVLA LIBERO eval requires --env.type=libero.")
    else:
        if getattr(env, "obs_type", None) != contract.obs_type:
            result.errors.append(f"TurboVLA LIBERO requires env.obs_type={contract.obs_type!r}.")
        if getattr(env, "control_mode", None) != contract.control_mode:
            result.errors.append(f"TurboVLA LIBERO requires env.control_mode={contract.control_mode!r}.")
        if getattr(env, "init_states", None) is not True:
            result.errors.append("TurboVLA LIBERO requires env.init_states=true.")
        if getattr(env, "fps", None) != 20:
            result.errors.append("TurboVLA LIBERO requires env.fps=20.")
        if (
            getattr(env, "observation_height", None) != contract.observation_height
            or getattr(env, "observation_width", None) != contract.observation_width
        ):
            result.warnings.append(
                "Official TurboVLA LIBERO evaluation renders "
                f"{contract.observation_height}x{contract.observation_width} camera observations for DINOv3."
            )
        camera_names = tuple(item.strip() for item in str(getattr(env, "camera_name", "")).split(",") if item)
        if camera_names != contract.camera_names:
            result.errors.append(
                f"TurboVLA LIBERO requires env.camera_name={','.join(contract.camera_names)!r}."
            )
        if getattr(env, "camera_name_mapping", None) not in (None, {}):
            result.errors.append("TurboVLA LIBERO requires the default camera_name_mapping.")
        if getattr(env, "task_ids", None) is not None:
            result.warnings.append("task_ids restricts the benchmark; omit it for full-suite results.")
        suites = _as_suite_tuple(getattr(env, "task", None))
        if len(suites) == 1 and suites[0] in contract.episode_lengths:
            expected_episode_length = contract.episode_lengths[suites[0]]
            if getattr(env, "episode_length", None) != expected_episode_length:
                result.warnings.append(
                    f"TurboVLA {suites[0]} uses env.episode_length={expected_episode_length}."
                )
        elif suites == contract.suites:
            result.warnings.append(
                "TurboVLA's four LIBERO suites require separate evaluator invocations because their "
                "upstream episode lengths differ."
            )
        else:
            result.errors.append(
                f"TurboVLA LIBERO requires one of {tuple(contract.episode_lengths)!r}; got {suites!r}."
            )

    if eval_cfg is not None and getattr(eval_cfg, "n_episodes", None) != contract.trials_per_task:
        result.warnings.append(
            f"Full TurboVLA LIBERO benchmark uses eval.n_episodes={contract.trials_per_task} per task."
        )
    if eval_cfg is not None and getattr(eval_cfg, "batch_size", None) != 1:
        result.warnings.append("Full TurboVLA LIBERO benchmark uses eval.batch_size=1.")
    if getattr(cfg, "seed", None) != contract.seed:
        result.warnings.append(f"Full TurboVLA LIBERO benchmark uses seed={contract.seed}.")
    if getattr(policy, "use_amp", None) is not True:
        result.warnings.append("Full TurboVLA LIBERO benchmark uses AMP/bf16 inference.")

    if strict:
        result.errors.extend(result.warnings)
        result.warnings = []
    return result


def turbovla_libero_benchmark_command(policy_path: str, output_dir: str, suite: str) -> list[str]:
    """Return one canonical suite-specific ``lerobot-eval`` command.

    The public TurboVLA protocol requires a separate invocation per suite because
    LIBERO's rollout horizon differs by suite.
    """

    if suite not in TURBOVLA_LIBERO_EPISODE_LENGTHS:
        raise ValueError(f"Unsupported TurboVLA LIBERO suite {suite!r}")

    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--eval.n_episodes={TURBOVLA_LIBERO_TRIALS_PER_TASK}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=true",
        "--policy.device=cuda",
        "--policy.precision=bfloat16",
        "--policy.use_amp=true",
        f"--policy.n_action_steps={TURBOVLA_LIBERO_OPEN_LOOP_ACTIONS}",
        f"--env.episode_length={TURBOVLA_LIBERO_EPISODE_LENGTHS[suite]}",
        "--env.observation_height=256",
        "--env.observation_width=256",
        f"--seed={TURBOVLA_LIBERO_SEED}",
        f"--output_dir={output_dir}",
    ]


def summarize_turbovla_libero_results(info: dict[str, Any]) -> dict[str, Any]:
    """Build a compact per-task/suite/aggregate report with confidence intervals."""

    per_task = []
    for task_info in info.get("per_task", []):
        metrics = task_info.get("metrics", {})
        successes = metrics.get("successes", [])
        per_task.append(
            {
                "task_group": task_info.get("task_group"),
                "task_id": task_info.get("task_id"),
                **success_summary(successes),
            }
        )

    per_suite = {}
    for suite, suite_info in info.get("per_group", {}).items():
        per_suite[suite] = {
            "n_episodes": suite_info.get("n_episodes", 0),
            "n_success": suite_info.get("n_success", 0),
            "pc_success": suite_info.get("pc_success"),
            "pc_success_ci95": suite_info.get("pc_success_ci95"),
        }

    overall = info.get("overall", {})
    return {
        "contract": asdict(TurboVLALiberoEvalContract()),
        "pending_benchmark_values": dict(TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES),
        "per_task": per_task,
        "per_suite": per_suite,
        "overall": {
            "n_episodes": overall.get("n_episodes", 0),
            "n_success": overall.get("n_success", 0),
            "pc_success": overall.get("pc_success"),
            "pc_success_ci95": overall.get("pc_success_ci95"),
            "eval_s": overall.get("eval_s"),
            "eval_ep_s": overall.get("eval_ep_s"),
        },
    }
