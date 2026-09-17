# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""LeRobot adapter around the vendored upstream TurboVLA architecture."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_turbovla import TurboVLAConfig


def _make_upstream_config(config: TurboVLAConfig):
    from .configuration_upstream import (
        ActionHeadConfig,
        InteractionConfig,
        TextEncoderConfig,
        TurboVLAConfig as UpstreamTurboVLAConfig,
        VisionEncoderConfig,
    )

    return UpstreamTurboVLAConfig(
        text=TextEncoderConfig(
            model_name_or_path=config.language_encoder_id,
            max_length=config.max_text_length,
            frozen=config.freeze_text_encoder,
            local_files_only=config.local_files_only,
            attention_implementation=config.attention_implementation,
        ),
        vision=VisionEncoderConfig(
            model_name_or_path=config.vision_encoder_id,
            image_size=config.image_size[0],
            num_views=len(config.image_keys),
            frozen=config.freeze_vision_encoder,
            local_files_only=config.local_files_only,
            attention_implementation=config.attention_implementation,
            compute_precision="bf16_autocast" if config.precision == "bfloat16" else "fp32",
            dropout=config.dropout,
        ),
        interaction=InteractionConfig(
            hidden_dim=config.hidden_dim,
            nheads=config.num_attention_heads,
            num_layers=config.num_interaction_layers,
            dim_feedforward=config.interaction_feedforward_dim,
            enhancer_inner_dim=config.interaction_inner_dim,
            text_dropout=config.text_dropout,
            fusion_dropout=config.fusion_dropout,
            fusion_droppath=config.fusion_droppath,
            attention_backend=config.attention_implementation,
        ),
        action=ActionHeadConfig(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            horizon=config.chunk_size,
            num_state_tokens=config.num_state_tokens,
            num_layers=config.num_action_decoder_layers,
            mlp_hidden_dim=config.action_mlp_hidden_dim,
            state_hidden_dim=config.state_hidden_dim,
            dropout=config.dropout,
        ),
    )


def _build_model(config: TurboVLAConfig) -> nn.Module:
    require_package("transformers", extra="turbovla")
    require_package("timm", extra="turbovla")
    from .modeling_turbovla_architecture import TurboVLA

    return TurboVLA(_make_upstream_config(config))


class TurboVLAPolicy(PreTrainedPolicy):
    config_class = TurboVLAConfig
    name = "turbovla"
    _fsdp_wrap_modules = ["TransformerEncoderLayer", "TransformerDecoderLayer"]

    def __init__(self, config: TurboVLAConfig, *, model: nn.Module | None = None, **kwargs: object) -> None:
        del kwargs
        super().__init__(config)
        config.validate_features()
        self.config = config
        if model is None:
            model = _build_model(config)
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def _prepare_inputs(self, batch: dict[str, object]) -> tuple[list[str], Tensor, Tensor]:
        required = (*self.config.image_keys, OBS_STATE, OBS_LANGUAGE)
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"TurboVLA batch is missing required keys: {missing}")
        images = []
        for key in self.config.image_keys:
            image = batch[key]
            if not isinstance(image, Tensor):
                raise TypeError(f"{key!r} must be a torch.Tensor")
            if image.ndim == 5:
                image = image[:, -1]
            if image.ndim != 4:
                raise ValueError(f"{key!r} must have shape [B,3,H,W] or [B,T,3,H,W]")
            images.append(image)
        pixel_values = torch.stack(images, dim=1)
        state = batch[OBS_STATE]
        if not isinstance(state, Tensor):
            raise TypeError(f"{OBS_STATE!r} must be a torch.Tensor")
        language = batch[OBS_LANGUAGE]
        if isinstance(language, str):
            instructions = [language]
        elif isinstance(language, Sequence):
            instructions = [str(item) for item in language]
        else:
            raise TypeError(f"{OBS_LANGUAGE!r} must be a string or sequence of strings")
        if len(instructions) != pixel_values.shape[0]:
            raise ValueError("language batch size must match image batch size")
        return instructions, pixel_values, state

    def _predict(self, batch: dict[str, object]) -> Tensor:
        instructions, pixel_values, state = self._prepare_inputs(batch)
        actions = self.model(instructions, {"dinov3": pixel_values}, state)
        expected = (pixel_values.shape[0], self.config.chunk_size, self.config.action_dim)
        if tuple(actions.shape) != expected:
            raise RuntimeError(f"TurboVLA returned shape {tuple(actions.shape)}, expected {expected}")
        return actions

    def forward(self, batch: dict[str, object]) -> tuple[Tensor, dict[str, float]]:
        actions_hat = self._predict(batch)
        target = batch.get(ACTION)
        if not isinstance(target, Tensor):
            raise ValueError(f"training batch must contain tensor {ACTION!r}")
        if target.shape != actions_hat.shape:
            raise ValueError(
                f"target action shape {tuple(target.shape)} does not match prediction {tuple(actions_hat.shape)}"
            )
        element_loss = F.l1_loss(actions_hat, target.to(actions_hat), reduction="none")
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            loss = element_loss.mean()
        else:
            if not isinstance(action_is_pad, Tensor) or action_is_pad.shape != target.shape[:2]:
                raise ValueError("action_is_pad must be a [B, chunk_size] tensor")
            valid = (~action_is_pad.bool()).unsqueeze(-1)
            loss = (element_loss * valid).sum() / (valid.sum() * element_loss.shape[-1]).clamp_min(1)
        return loss, {"l1_loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, object], **kwargs: object) -> Tensor:
        del kwargs
        self.eval()
        return self._predict(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, object], **kwargs: object) -> Tensor:
        del kwargs
        self.eval()
        if not self._action_queue:
            actions = self._predict(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
