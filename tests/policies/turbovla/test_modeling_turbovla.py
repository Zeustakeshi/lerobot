# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
import types

import torch
from torch import nn

from lerobot.policies import get_policy_class
from lerobot.policies.turbovla.bert import BertModelWarper
from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.modeling_turbovla import TurboVLAPolicy, _make_upstream_config
from lerobot.policies.turbovla.transformer import TransformerEncoderLayer
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


class SyntheticTurboVLA(nn.Module):
    def __init__(self, config: TurboVLAConfig) -> None:
        super().__init__()
        self.chunk_size = config.chunk_size
        self.action_dim = config.action_dim
        self.projection = nn.Linear(config.state_dim, config.action_dim)

    def forward(self, instructions, samples, state):
        assert len(instructions) == state.shape[0]
        assert samples["dinov3"].shape[:2] == (state.shape[0], 2)
        action = torch.tanh(self.projection(state[:, -1] if state.ndim == 3 else state))
        return action[:, None].expand(-1, self.chunk_size, -1)


def make_batch(config: TurboVLAConfig, batch_size: int = 2):
    batch = {key: torch.randn(batch_size, 3, *config.image_size) for key in config.image_keys}
    batch[OBS_STATE] = torch.randn(batch_size, config.state_dim)
    batch[OBS_LANGUAGE] = [f"instruction {index}" for index in range(batch_size)]
    batch[ACTION] = torch.randn(batch_size, config.chunk_size, config.action_dim)
    batch["action_is_pad"] = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
    return batch


class _MinimalBertEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(32, 8)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        return self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds


class _MinimalBertEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_head_mask = None
        self.last_attention_mask = None

    def forward(self, hidden_states, *, attention_mask=None, head_mask=None, **kwargs):
        del kwargs
        self.last_head_mask = head_mask
        self.last_attention_mask = attention_mask
        return (hidden_states,)


def _minimal_bert(api: str):
    encoder = _MinimalBertEncoder()
    config = types.SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        is_decoder=False,
        use_cache=False,
        num_hidden_layers=2,
    )

    class MinimalBert:
        def __init__(self) -> None:
            self.config = config
            self.embeddings = _MinimalBertEmbeddings()
            self.encoder = encoder
            self.pooler = None

        @staticmethod
        def invert_attention_mask(mask):
            return mask

        def get_head_mask(self, head_mask, num_hidden_layers):
            return ["legacy"] * num_hidden_layers

    if api == "legacy":

        def get_extended_attention_mask(self, attention_mask, input_shape, *, device):
            self.mask_kwargs = {"device": device}
            return attention_mask[:, None, None, :]

        MinimalBert.get_extended_attention_mask = get_extended_attention_mask
    else:
        delattr(MinimalBert, "get_head_mask")

        def get_extended_attention_mask(self, attention_mask, input_shape, *, dtype):
            self.mask_kwargs = {"dtype": dtype}
            return attention_mask[:, None, None, :]

        MinimalBert.get_extended_attention_mask = get_extended_attention_mask

    return MinimalBert(), encoder


def test_bert_wrapper_supports_legacy_and_current_transformers_apis() -> None:
    for api, expected_mask_keyword in (("legacy", "device"), ("current", "dtype")):
        bert, encoder = _minimal_bert(api)
        wrapped = BertModelWarper(bert)
        result = wrapped(input_ids=torch.ones((1, 3), dtype=torch.long), return_dict=False)

        assert result[0].shape == (1, 3, 8)
        assert expected_mask_keyword in bert.mask_kwargs
        expected_head_mask = ["legacy", "legacy"] if api == "legacy" else [None, None]
        assert encoder.last_head_mask == expected_head_mask


def test_bert_wrapper_supports_transformers_5_without_mask_helpers() -> None:
    bert, encoder = _minimal_bert("current")
    delattr(type(bert), "get_extended_attention_mask")
    delattr(type(bert), "invert_attention_mask")

    wrapped = BertModelWarper(bert)
    result = wrapped(
        input_ids=torch.ones((1, 3), dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 0]]),
        return_dict=False,
    )

    assert result[0].shape == (1, 3, 8)
    assert encoder.last_head_mask == [None, None]
    assert encoder.last_attention_mask.shape == (1, 1, 1, 3)
    assert encoder.last_attention_mask[0, 0, 0, 0] == 0
    assert encoder.last_attention_mask[0, 0, 0, 2] < -1e10


def test_synthetic_forward_backward_and_batch_shapes() -> None:
    config = TurboVLAConfig(device="cpu", precision="float32")
    policy = TurboVLAPolicy(config, model=SyntheticTurboVLA(config))
    for batch_size in (1, 3):
        batch = make_batch(config, batch_size)
        actions = policy.predict_action_chunk(batch)
        assert actions.shape == (batch_size, config.chunk_size, config.action_dim)
        loss, metrics = policy(batch)
        assert loss.ndim == 0
        assert metrics["l1_loss"] == loss.item()
        loss.backward()
        assert policy.model.projection.weight.grad is not None
        policy.zero_grad(set_to_none=True)


def test_backbone_revisions_are_forwarded_to_upstream_config() -> None:
    config = TurboVLAConfig(
        device="cpu",
        vision_encoder_revision="dinov3-commit",
        language_encoder_revision="bert-commit",
    )

    upstream_config = _make_upstream_config(config)

    assert upstream_config.vision.revision == "dinov3-commit"
    assert upstream_config.text.revision == "bert-commit"


def test_vendored_architecture_forward_backward(monkeypatch) -> None:
    class FakeDropPath(nn.Identity):
        def __init__(self, drop_prob=0.0) -> None:
            super().__init__()

    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.DropPath = FakeDropPath
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.models", timm_models)
    monkeypatch.setitem(sys.modules, "timm.models.layers", timm_layers)
    from lerobot.policies.turbovla import modeling_turbovla_architecture as architecture
    from lerobot.policies.turbovla.configuration_upstream import (
        ActionHeadConfig,
        InteractionConfig,
        TextEncoderConfig,
        TurboVLAConfig as ArchitectureConfig,
        VisionEncoderConfig,
    )

    class FakeVisionEncoder(nn.Module):
        hidden_size = 32
        num_patches = 4

        def __init__(self, config) -> None:
            super().__init__()

        def forward(self, pixels):
            batch_size, views = pixels.shape[:2]
            pooled = pixels.mean(dim=(-1, -2)).mean(dim=-1, keepdim=True)
            return pooled[..., None].expand(batch_size, views, self.num_patches, self.hidden_size)

    class FakeTextEncoder(nn.Module):
        def __init__(self, config, hidden_dim) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, hidden_dim)

        def forward(self, instructions, device):
            batch_size = len(instructions)
            ids = torch.arange(4, device=device).expand(batch_size, -1)
            tokens = self.embedding(ids)
            padding = torch.zeros(batch_size, 4, dtype=torch.bool, device=device)
            attention = torch.ones(batch_size, 4, 4, dtype=torch.bool, device=device)
            return tokens, padding, attention

    monkeypatch.setattr(architecture, "DINOv3VisionEncoder", FakeVisionEncoder)
    monkeypatch.setattr(architecture, "TurboVLATextEncoder", FakeTextEncoder)
    config = ArchitectureConfig(
        text=TextEncoderConfig(),
        vision=VisionEncoderConfig(image_size=16, num_views=2),
        interaction=InteractionConfig(
            hidden_dim=32, nheads=4, num_layers=2, dim_feedforward=64, enhancer_inner_dim=32
        ),
        action=ActionHeadConfig(
            action_dim=7,
            state_dim=8,
            horizon=12,
            num_layers=2,
            mlp_hidden_dim=32,
            state_hidden_dim=16,
        ),
    )
    model = architecture.TurboVLA(config)
    pixels = torch.randn(3, 2, 3, 16, 16)
    state = torch.randn(3, 8)
    actions = model(["task"] * 3, {"dinov3": pixels}, state)
    assert actions.shape == (3, 12, 7)
    actions.square().mean().backward()
    assert model.action_head.decoder.action_queries.weight.grad is not None


def test_text_attention_layer_matches_upstream_mask_layout() -> None:
    """Keep the vendored layer bit-for-bit compatible with TurboVLA's attention path."""
    torch.manual_seed(0)
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    layer.eval()
    source = torch.randn(4, 2, 8)
    attention_mask = torch.tensor(
        [
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ],
            [
                [False, False, True, True],
                [False, False, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ],
        ]
    )
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])

    actual = layer(source, src_mask=attention_mask, src_key_padding_mask=padding_mask)

    # `repeat`, rather than `repeat_interleave`, and the omission of a padding
    # mask are the semantics used by the published TurboVLA implementation.
    upstream_mask = attention_mask.repeat(layer.nhead, 1, 1)
    attended = layer.self_attn(source, source, value=source, attn_mask=upstream_mask)[0]
    expected = layer.norm1(source + layer.dropout1(attended))
    feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(expected))))
    expected = layer.norm2(expected + layer.dropout2(feedforward))

    torch.testing.assert_close(actual, expected)


def test_text_attention_layer_accepts_no_custom_attention_mask() -> None:
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    output = layer(torch.randn(4, 2, 8), src_mask=None, src_key_padding_mask=None)
    assert output.shape == (4, 2, 8)


def test_padded_targets_do_not_contribute_to_loss() -> None:
    config = TurboVLAConfig(device="cpu", precision="float32")
    policy = TurboVLAPolicy(config, model=SyntheticTurboVLA(config))
    batch = make_batch(config, 2)
    batch["action_is_pad"][:, -2:] = True
    loss, _ = policy(batch)
    batch[ACTION][:, -2:] = 1e6
    padded_loss, _ = policy(batch)
    torch.testing.assert_close(loss, padded_loss)


def test_action_queue_and_reset() -> None:
    config = TurboVLAConfig(device="cpu", precision="float32", n_action_steps=3)
    policy = TurboVLAPolicy(config, model=SyntheticTurboVLA(config))
    batch = make_batch(config, 2)
    expected = policy.predict_action_chunk(batch)
    first = policy.select_action(batch)
    second = policy.select_action(batch)
    torch.testing.assert_close(first, expected[:, 0])
    torch.testing.assert_close(second, expected[:, 1])
    assert len(policy._action_queue) == 1
    policy.reset()
    assert len(policy._action_queue) == 0


def test_save_load_round_trip_without_downloading(monkeypatch, tmp_path) -> None:
    config = TurboVLAConfig(device="cpu", precision="float32")
    monkeypatch.setattr(
        "lerobot.policies.turbovla.modeling_turbovla._build_model",
        lambda loaded_config: SyntheticTurboVLA(loaded_config),
    )
    policy = TurboVLAPolicy(config)
    policy.model.projection.weight.data.fill_(0.25)
    policy.save_pretrained(tmp_path)
    loaded = TurboVLAPolicy.from_pretrained(tmp_path)
    assert get_policy_class("turbovla") is TurboVLAPolicy
    torch.testing.assert_close(loaded.model.projection.weight, policy.model.projection.weight)
