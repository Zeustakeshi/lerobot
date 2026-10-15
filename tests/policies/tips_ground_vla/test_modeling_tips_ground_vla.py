from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tips_ground_vla.configuration_tips_ground_vla import TIPSGroundVLAConfig
from lerobot.policies.tips_ground_vla.modeling_tips_ground_vla import (
    TIPSGroundVLA,
    TIPSGroundVLAPolicy,
    _resolve_hub_cache_dir,
    action_losses,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


class StubBatch(dict):
    def __getattr__(self, key):
        return self[key]

    def to(self, device):
        return StubBatch({key: value.to(device) for key, value in self.items()})


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        length = kwargs["max_length"]
        return StubBatch(
            input_ids=torch.ones(len(texts), length, dtype=torch.long),
            attention_mask=torch.ones(len(texts), length, dtype=torch.long),
        )


class StubText(nn.Module):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 384, device=input_ids.device))


class StubVision(nn.Module):
    blocks = ()

    def forward_features(self, images):
        return torch.ones(images.shape[0], 295, 384, device=images.device)


def config(cameras: int = 2) -> TIPSGroundVLAConfig:
    images = {
        f"observation.images.cam{i}": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)) for i in range(cameras)
    }
    state = PolicyFeature(FeatureType.STATE, (6,))
    action = PolicyFeature(FeatureType.ACTION, (7,))
    cfg = TIPSGroundVLAConfig(
        input_features={**images, OBS_STATE: state, OBS_LANGUAGE: PolicyFeature(FeatureType.LANGUAGE, (1,))},
        output_features={ACTION: action},
        infer_from_dataset=False,
        image_keys=tuple(images),
        state_dim=6,
        action_dim=7,
    )
    return cfg


def model(cameras: int = 2) -> TIPSGroundVLA:
    return TIPSGroundVLA(config(cameras), vision=StubVision(), text=StubText(), tokenizer=StubTokenizer())


def test_v3_config_is_locked() -> None:
    cfg = config()
    assert cfg.image_size == (224, 336)
    assert cfg.chunk_size == 32
    assert cfg.n_action_steps == 4
    assert cfg.max_phrases == 6
    with pytest.raises(ValueError):
        TIPSGroundVLAConfig(chunk_size=12)


def test_v3_forward_contract_and_camera_mask() -> None:
    net = model()
    spans = torch.zeros(1, 6, 64, dtype=torch.bool)
    spans[0, 0, 1:3] = True
    actions, out = net(
        torch.zeros(1, 2, 3, 224, 336),
        torch.zeros(1, 6),
        ["put the object in the tray"],
        camera_valid=torch.tensor([[True, False]]),
        phrase_spans=spans,
        debug=True,
    )
    assert actions.shape == (1, 32, 7)
    assert out["F"].shape == (1, 588, 384)
    assert out["T"].shape == (1, 64, 384)
    assert out["S"].shape == (1, 6, 588)
    assert out["Za"].shape == (1, 8, 384)
    assert out["Zg"].shape == (1, 8, 384)
    assert out["Zs"].shape == (1, 4, 384)
    assert out["region_maps"].shape == (1, 8, 2, 14, 21)
    assert out["geometry"].shape == (1, 8, 2, 6)
    assert out["fusion_context"].shape == (1, 24, 384)
    assert out["Z"].shape == (1, 20, 384)
    assert out["memory"].shape == (1, 4, 384)
    assert out["alignment_attention"].shape == (1, 6, 8, 588)
    assert out["slot_attention"].shape == (1, 6, 8, 588)
    assert out["region_maps"][:, :, 1].count_nonzero() == 0


def test_action_gradient_reaches_region_predictor() -> None:
    net = model(1)
    actions, _ = net(torch.zeros(1, 1, 3, 224, 336), torch.zeros(1, 6), ["pick object"])
    actions.square().mean().backward()
    assert net.region_slot.weight.grad is not None
    assert net.region_slot.weight.grad.abs().sum() > 0


def test_memory_is_causal_fifo_and_reset_by_policy() -> None:
    net = model(1)
    images, state, text = torch.zeros(1, 1, 3, 224, 336), torch.zeros(1, 6), ["move"]
    _, first = net(images, state, text)
    _, second = net(images, state, text, memory=first["memory"], memory_valid=first["memory_valid"])
    assert first["memory_valid"].tolist() == [[False, False, False, True]]
    assert second["memory_valid"].tolist() == [[False, False, True, True]]


def test_masked_smooth_l1_and_delta() -> None:
    pred = torch.zeros(1, 32, 2, requires_grad=True)
    target = torch.ones_like(pred)
    valid = torch.zeros(1, 32, dtype=torch.bool)
    valid[:, :4] = True
    action, delta = action_losses(pred, target, valid)
    (action + delta).backward()
    assert action > 0
    assert delta == 0
    assert pred.grad[:, 4:].count_nonzero() == 0


def test_plain_lerobot_batch_trains_without_teacher_targets() -> None:
    cfg = config(1)
    policy = TIPSGroundVLAPolicy(cfg, model=model(1))
    batch = {
        cfg.image_keys[0]: torch.zeros(1, 3, 64, 64),
        OBS_STATE: torch.zeros(1, 6),
        OBS_LANGUAGE: ["pick object"],
        ACTION: torch.zeros(1, 32, 7),
    }
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"action_loss", "delta_loss", "loss"}


def test_four_observation_sequence_training_is_causal() -> None:
    cfg = config(1)
    policy = TIPSGroundVLAPolicy(cfg, model=model(1))
    batch = {
        cfg.image_keys[0]: torch.zeros(1, 4, 3, 64, 64),
        OBS_STATE: torch.zeros(1, 4, 6),
        OBS_LANGUAGE: ["pick object"],
        ACTION: torch.zeros(1, 32, 7),
    }
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert cfg.observation_delta_indices == [-3, -2, -1, 0]


def test_unusable_checkpoint_cache_path_falls_back(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cfg = config(1)
    cfg.hub_cache_dir = str(blocker / "cache")
    assert _resolve_hub_cache_dir(cfg) is None
