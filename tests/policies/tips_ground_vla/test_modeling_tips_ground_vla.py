from __future__ import annotations

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.tips_ground_vla.configuration_tips_ground_vla import TIPSGroundVLAConfig
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import TIPSGroundVLAPolicy
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

    actions = policy.predict_action_chunk(
        _batch(), inference_delay=2, prev_chunk_left_over=previous
    )

    assert supports_rtc_inference(policy)
    torch.testing.assert_close(actions[:, :2], previous[None, :2])
    # A short prefix is not padded; the newly predicted suffix stays intact.
    torch.testing.assert_close(actions[:, 2:], torch.ones_like(actions[:, 2:]))
    with pytest.raises(AssertionError, match="RTC uses predict_action_chunk"):
        policy.select_action(_batch())


def test_rtc_arguments_require_rollout_configuration() -> None:
    policy = TIPSGroundVLAPolicy(_config(), model=_StubModel())
    with pytest.raises(RuntimeError, match="require --inference.type=rtc"):
        policy.predict_action_chunk(
            _batch(), inference_delay=1, prev_chunk_left_over=torch.zeros(1, 6)
        )
