from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.tips_ground_vla.configuration_tips_ground_vla import TIPSGroundVLAConfig
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import (
    TeacherTargetGenerator,
    TIPSGroundVLAPolicy,
    _resolve_hub_cache_dir,
)
from lerobot.rollout.inference.rtc import supports_rtc_inference
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images, state, instructions):
        self.calls += 1
        actions = torch.full((state.shape[0], 12, 6), float(self.calls), device=state.device)
        return actions, {}


class _StubTIPSTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dtype_anchor = nn.Parameter(torch.zeros(1))

    def encode_image(self, images: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(patch_tokens=torch.ones(images.shape[0], 256, 3, device=images.device))

    def encode_text(self, phrases: list[str]) -> torch.Tensor:
        return torch.ones(len(phrases), 3, device=self.dtype_anchor.device)


class _StubDINOProcessor:
    def __call__(self, images: torch.Tensor, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": images}


class _StubDINOTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dtype_anchor = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=torch.ones(pixel_values.shape[0], 197, 3, device=pixel_values.device)
        )


def _config() -> TIPSGroundVLAConfig:
    image = PolicyFeature(FeatureType.VISUAL, (3, 720, 1280))
    state = PolicyFeature(FeatureType.STATE, (6,))
    action = PolicyFeature(FeatureType.ACTION, (6,))
    return TIPSGroundVLAConfig(
        infer_from_dataset=False,
        image_keys=("observation.images.top", "observation.images.wrist"),
        state_dim=6,
        action_dim=6,
        input_features={
            "observation.images.top": image,
            "observation.images.wrist": image,
            OBS_STATE: state,
            OBS_LANGUAGE: PolicyFeature(FeatureType.LANGUAGE, (1,)),
        },
        output_features={ACTION: action},
        n_action_steps=4,
        device="cpu",
    )


def _batch(task: str = "put the red block in the pen holder") -> dict[str, object]:
    return {
        "observation.images.top": torch.zeros(1, 3, 32, 32),
        "observation.images.wrist": torch.zeros(1, 3, 32, 32),
        OBS_STATE: torch.zeros(1, 6),
        "task": task,
    }


def test_select_action_reuses_only_configured_chunk_prefix() -> None:
    model = _StubModel()
    policy = TIPSGroundVLAPolicy(_config(), model=model)

    actions = [policy.select_action(_batch()) for _ in range(5)]

    assert model.calls == 2
    assert [action.item() for action in (a[0, 0] for a in actions)] == [1, 1, 1, 1, 2]


def test_drop_queued_actions_refreshes_chunk_for_new_rollout_task() -> None:
    model = _StubModel()
    policy = TIPSGroundVLAPolicy(_config(), model=model)
    assert policy.select_action(_batch("old task"))[0, 0].item() == 1

    policy.drop_queued_actions()

    assert policy.select_action(_batch("new task"))[0, 0].item() == 2
    assert model.calls == 2


def test_instruction_batch_must_match_observation_batch() -> None:
    policy = TIPSGroundVLAPolicy(_config(), model=_StubModel())
    batch = _batch()
    batch[OBS_LANGUAGE] = ["first", "second"]

    with pytest.raises(ValueError, match=r"instruction batch size \(2\)"):
        policy.predict_action_chunk(batch)


def test_rtc_blends_previous_prefix_and_is_detected_by_rollout() -> None:
    config = _config()
    policy = TIPSGroundVLAPolicy(config, model=_StubModel())
    config.rtc_config = RTCConfig(execution_horizon=4)
    policy.init_rtc_processor()
    previous = torch.full((2, 6), 0.75)

    actions = policy.predict_action_chunk(_batch(), inference_delay=2, prev_chunk_left_over=previous)

    assert supports_rtc_inference(policy)
    torch.testing.assert_close(actions[:, :2], previous[None, :2])
    # A short prefix is not padded; the newly predicted suffix stays intact.
    torch.testing.assert_close(actions[:, 2:], torch.ones_like(actions[:, 2:]))
    with pytest.raises(AssertionError, match="RTC uses predict_action_chunk"):
        policy.select_action(_batch())


def test_rtc_arguments_require_rollout_configuration() -> None:
    policy = TIPSGroundVLAPolicy(_config(), model=_StubModel())
    with pytest.raises(RuntimeError, match="require --inference.type=rtc"):
        policy.predict_action_chunk(_batch(), inference_delay=1, prev_chunk_left_over=torch.zeros(1, 6))


def test_policy_optimizer_preset_honors_policy_level_overrides() -> None:
    config = _config()
    config.optimizer_lr = 5e-5
    config.optimizer_weight_decay = 0.01
    config.optimizer_grad_clip_norm = 1.0

    optimizer = config.get_optimizer_preset()

    assert optimizer.lr == 5e-5
    assert optimizer.weight_decay == 0.01
    assert optimizer.grad_clip_norm == 1.0


def test_teacher_repeats_each_instruction_for_every_camera() -> None:
    teachers = object.__new__(TeacherTargetGenerator)
    teachers.tips = _StubTIPSTeacher()
    teachers.dino = _StubDINOTeacher()
    teachers.dino_processor = _StubDINOProcessor()

    tips_target, dino_target = teachers(torch.zeros(2, 2, 3, 32, 32), ["first task", "second task"])

    assert tips_target.shape == (2, 2, 1, 196)
    assert dino_target.shape == (2, 2, 196, 196)


def test_unusable_checkpoint_cache_path_falls_back_to_user_cache(tmp_path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block cache directory creation")
    config = _config()
    config.hub_cache_dir = str(blocker / "hf-cache")

    assert _resolve_hub_cache_dir(config) is None

    config.local_files_only = True
    with pytest.raises(OSError, match="local_files_only=True"):
        _resolve_hub_cache_dir(config)
