"""Focused tests for the standalone TurboVLA-v2 policy."""

import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import TurboVLAV2Config, get_policy_class, make_policy_config
from lerobot.policies.turbovla_v2.modeling_turbovla_v2 import TurboVLAV2Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


def make_config() -> TurboVLAV2Config:
    images = {"observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 28, 28))}
    return TurboVLAV2Config(
        device="cpu",
        image_keys=tuple(images),
        image_size=(28, 28),
        resize_images=False,
        state_dim=6,
        action_dim=6,
        chunk_size=2,
        n_action_steps=1,
        hidden_dim=32,
        num_heads=4,
        fusion_layers=1,
        decoder_layers=1,
        feedforward_dim=64,
        input_features={
            **images,
            OBS_STATE: PolicyFeature(FeatureType.STATE, (6,)),
            OBS_LANGUAGE: PolicyFeature(FeatureType.LANGUAGE, (1,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (6,))},
    )


class TinyModel(nn.Module):
    def __init__(self, config: TurboVLAV2Config) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(config.action_dim))
        self.chunk_size = config.chunk_size

    def forward(self, instructions, images, state):
        assert instructions == ["pick cube"]
        return self.bias.view(1, 1, -1).expand(images.shape[0], self.chunk_size, -1)


def test_registry_and_policy_factory() -> None:
    assert "turbovla_v2" in PreTrainedConfig.get_known_choices()
    assert isinstance(make_policy_config("turbovla_v2", device="cpu"), TurboVLAV2Config)
    assert get_policy_class("turbovla_v2") is TurboVLAV2Policy


def test_training_and_rollout_interfaces() -> None:
    config = make_config()
    policy = TurboVLAV2Policy(config, model=TinyModel(config))
    batch = {
        "observation.images.top": torch.rand(1, 3, 28, 28),
        OBS_STATE: torch.rand(1, 6),
        OBS_LANGUAGE: ["pick cube"],
        ACTION: torch.rand(1, 2, 6),
    }
    loss, metrics = policy(batch)
    loss.backward()
    assert metrics["l1_loss"] > 0
    assert policy.predict_action_chunk(batch).shape == (1, 2, 6)
    assert policy.select_action(batch).shape == (1, 6)


def test_tips_geometry_is_enforced() -> None:
    try:
        TurboVLAV2Config(image_size=(256, 256))
    except ValueError as error:
        assert "patch size 14" in str(error)
    else:
        raise AssertionError("invalid TIPSv2 geometry was accepted")
