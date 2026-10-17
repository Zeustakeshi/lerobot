"""TIPS-Ground-VLA v3 Action-only policy."""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_tips_ground_vla import TIPSGroundVLAConfig
from .teacher_supervision import (
    semantic_distribution,
    semantic_kl,
    spatial_relation_loss,
    text_distillation_loss,
)

EPS, PATCHES = 1e-6, 14 * 21


def _letterbox_with_content(images: Tensor) -> tuple[Tensor, Tensor]:
    """Letterbox RGB and return exact per-patch real-content fractions."""
    height, width = images.shape[-2:]
    scale = min(224 / height, 336 / width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    images = F.interpolate(images, (resized_height, resized_width), mode="bilinear", align_corners=False)
    vertical, horizontal = 224 - resized_height, 336 - resized_width
    images = F.pad(
        images, (horizontal // 2, horizontal - horizontal // 2, vertical // 2, vertical - vertical // 2)
    )
    content = images.new_zeros(images.shape[0], 1, 224, 336)
    top, left = vertical // 2, horizontal // 2
    content[..., top : top + resized_height, left : left + resized_width] = 1
    fractions = F.avg_pool2d(content, kernel_size=16, stride=16).squeeze(1)
    return images, fractions


def _letterbox(images: Tensor) -> Tensor:
    return _letterbox_with_content(images)[0]


def _resolve_hub_cache_dir(config: TIPSGroundVLAConfig) -> str | None:
    if config.hub_cache_dir is None:
        return None
    path = Path(config.hub_cache_dir).expanduser()
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.is_dir() and os.access(ancestor, os.W_OK):
        return str(path)
    if config.local_files_only:
        raise OSError(f"hub_cache_dir {config.hub_cache_dir!r} is unavailable")
    logging.warning("Ignoring unavailable hub_cache_dir=%r", config.hub_cache_dir)
    return None


def _masked_mean(x: Tensor, valid: Tensor) -> Tensor:
    weight = valid.to(x).unsqueeze(-1)
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1)


def _masked_softmax(logits: Tensor, valid: Tensor, dim: int) -> Tensor:
    result = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(dim)
    return result.masked_fill(~valid, 0)


def action_losses(
    pred: Tensor, target: Tensor, valid: Tensor, horizon_weight: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    weight = valid.to(pred)
    if horizon_weight is not None:
        if horizon_weight.shape != pred.shape[:2]:
            raise ValueError("horizon_weight must match [batch, horizon]")
        weight = weight * horizon_weight.to(pred)
    point = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none")
    action = (point * weight[..., None]).sum() / (weight.sum() * pred.shape[-1]).clamp_min(1)
    dv = valid[:, 1:] & valid[:, :-1]
    delta_weight = weight[:, 1:].minimum(weight[:, :-1]) * dv
    delta = F.smooth_l1_loss(
        pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], beta=0.1, reduction="none"
    )
    delta = (delta * delta_weight[..., None]).sum() / (delta_weight.sum() * pred.shape[-1]).clamp_min(1)
    return action, delta


class CrossBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qn, self.kn, self.fn = nn.LayerNorm(384), nn.LayerNorm(384), nn.LayerNorm(384)
        self.attn = nn.MultiheadAttention(384, 6, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(384, 1536), nn.GELU(), nn.Dropout(0.1), nn.Linear(1536, 384))

    def forward(self, q: Tensor, kv: Tensor, invalid: Tensor | None = None, weights: bool = False):
        y, attention = self.attn(
            self.qn(q),
            self.kn(kv),
            self.kn(kv),
            key_padding_mask=invalid,
            need_weights=weights,
            average_attn_weights=False,
        )
        q = q + y
        q = q + self.ff(self.fn(q))
        return (q, attention) if weights else q


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(384), nn.LayerNorm(384)
        self.attn = nn.MultiheadAttention(384, 6, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(384, 1536), nn.GELU(), nn.Dropout(0.1), nn.Linear(1536, 384))

    def forward(self, x: Tensor, invalid: Tensor) -> Tensor:
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), key_padding_mask=invalid, need_weights=False)
        x = x + y
        return x + self.ff(self.n2(x))


class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sn, self.cn, self.mn, self.fn = (nn.LayerNorm(384) for _ in range(4))
        self.sa, self.ca = (
            nn.MultiheadAttention(384, 6, batch_first=True),
            nn.MultiheadAttention(384, 6, batch_first=True),
        )
        self.ff = nn.Sequential(nn.Linear(384, 1536), nn.GELU(), nn.Dropout(0.1), nn.Linear(1536, 384))

    def forward(self, x: Tensor, memory: Tensor, memory_invalid: Tensor | None = None) -> Tensor:
        y, _ = self.sa(self.sn(x), self.sn(x), self.sn(x), need_weights=False)
        x = x + y
        y, _ = self.ca(
            self.cn(x),
            self.mn(memory),
            self.mn(memory),
            key_padding_mask=memory_invalid,
            need_weights=False,
        )
        x = x + y
        return x + self.ff(self.fn(x))


class FlowActionHead(nn.Module):
    """Small conditional vector field over a complete normalized action chunk."""

    def __init__(self, config: TIPSGroundVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.action_in = nn.Linear(config.action_dim, 384)
        self.time = nn.Sequential(nn.Linear(16, 384), nn.SiLU(), nn.Linear(384, 384))
        self.position = nn.Parameter(torch.randn(config.chunk_size, 384) * 0.02)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(config.flow_num_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 384), nn.GELU(), nn.Linear(384, config.action_dim)
        )
        if config.action_head == "architecture_aware_flow":
            self.source_gate = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 4), nn.Sigmoid())
            self.source_adapters = nn.ModuleList(
                [
                    nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384), nn.GELU(), nn.Linear(384, 384))
                    for _ in range(4)
                ]
            )
        else:
            self.source_gate = None
            self.source_adapters = None

    def _time_embedding(self, time: Tensor) -> Tensor:
        frequencies = torch.arange(1, 9, device=time.device, dtype=torch.float32) * torch.pi
        angles = time.float()[:, None] * frequencies[None]
        return self.time(torch.cat((angles.sin(), angles.cos()), -1)).to(dtype=self.position.dtype)

    def _context(self, context: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.source_gate is None:
            # Credible generic control: condition on one fused summary without
            # exposing TIPS-Ground-VLA's source/token identities.
            return context.mean(1, keepdim=True), None
        # The context layout is the preserved TIPS-Ground-VLA contract:
        # Alignment(8), Grounding(8), State(4), dense visual bypass(8).
        pooled = context.mean(1)
        gates = self.source_gate(pooled)
        adapted = context.clone()
        for index, section in enumerate((slice(0, 8), slice(8, 16), slice(16, 20), slice(20, 28))):
            residual = self.source_adapters[index](context[:, section])
            adapted[:, section] = context[:, section] + gates[:, index, None, None] * residual
        return adapted, gates

    def forward(
        self, x_t: Tensor, time: Tensor, context: Tensor, invalid: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        context, gates = self._context(context)
        if self.source_gate is None:
            invalid = None
        tokens = self.action_in(x_t) + self.position[None] + self._time_embedding(time)[:, None]
        for block in self.blocks:
            tokens = block(tokens, context, invalid)
        return self.out(tokens), gates


class SpatialAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(384, 384, 3, padding=1), nn.Conv2d(384, 384, 3, padding=1)
        self.n1, self.n2 = nn.LayerNorm(384), nn.LayerNorm(384)

    def forward(self, vision: Tensor) -> Tensor:
        b, c, n, d = vision.shape
        grid = vision.view(b * c, 14, 21, d).permute(0, 3, 1, 2)
        y = self.n1(F.gelu(self.c1(grid)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.n2(F.gelu(self.c2(y)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (grid + y).permute(0, 2, 3, 1).reshape(b, c, n, d)


class CompetitiveSlots(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q, self.k, self.v = nn.Linear(384, 384), nn.Linear(384, 384), nn.Linear(384, 384)
        self.sn, self.xn, self.fn = nn.LayerNorm(384), nn.LayerNorm(384), nn.LayerNorm(384)
        self.gru = nn.GRUCell(384, 384)
        self.ff = nn.Sequential(nn.Linear(384, 1536), nn.GELU(), nn.Dropout(0.1), nn.Linear(1536, 384))

    def forward(self, slots: Tensor, x: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        b, n, _ = x.shape
        k = self.k(self.xn(x)).view(b, n, 6, 64).permute(0, 2, 1, 3)
        v = self.v(self.xn(x)).view(b, n, 6, 64).permute(0, 2, 1, 3)
        raw = None
        for _ in range(2):
            q = self.q(self.sn(slots)).view(b, 8, 6, 64).permute(0, 2, 1, 3)
            logits = torch.einsum("bhkd,bhnd->bhkn", q, k) / 8
            valid = content > 0
            logits = logits.masked_fill(~valid[:, None, None], torch.finfo(logits.dtype).min)
            raw = logits.softmax(2).masked_fill(~valid[:, None, None], 0)
            aggregate = raw * content[:, None, None].to(raw)
            aggregate = aggregate / aggregate.sum(-1, keepdim=True).clamp_min(EPS)
            update = torch.einsum("bhkn,bhnd->bhkd", aggregate, v).permute(0, 2, 1, 3).reshape(b, 8, 384)
            slots = self.gru(update.flatten(0, 1), slots.flatten(0, 1)).view(b, 8, 384)
            slots = slots + self.ff(self.fn(slots))
        return slots, raw


class TIPSGroundVLA(nn.Module):
    def __init__(self, config: TIPSGroundVLAConfig, *, vision=None, text=None, tokenizer=None) -> None:
        super().__init__()
        self.config = config
        if vision is None or text is None:
            require_package("timm", extra="tips_ground_vla")
            require_package("transformers", extra="tips_ground_vla")
            import timm
            from transformers import AutoModel, AutoTokenizer

            cache = _resolve_hub_cache_dir(config)
            vision = vision or timm.create_model(
                config.vision_encoder_id, pretrained=True, num_classes=0, dynamic_img_size=True
            )
            text = text or AutoModel.from_pretrained(
                config.text_encoder_id, cache_dir=cache, local_files_only=config.local_files_only
            )
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                config.text_encoder_id, cache_dir=cache, local_files_only=config.local_files_only
            )
        self.vision, self.text, self.tokenizer = vision, text, tokenizer
        self.vision_norm, self.camera_embedding = nn.LayerNorm(384), nn.Embedding(2, 384)
        pretrained = getattr(vision, "pretrained_cfg", {}) or {}
        self.register_buffer(
            "vision_mean", torch.tensor(pretrained.get("mean", (0.5, 0.5, 0.5))).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "vision_std", torch.tensor(pretrained.get("std", (0.5, 0.5, 0.5))).view(1, 1, 3, 1, 1)
        )

        self.alignment_queries = nn.Parameter(torch.randn(8, 384) * 0.02)
        self.aq, self.ap, self.ag, self.al = (nn.Linear(384, 384) for _ in range(4))
        self.alignment_block = CrossBlock()

        self.state_value = nn.Sequential(nn.Linear(1, 384), nn.GELU(), nn.Linear(384, 384))
        self.state_channels = nn.Parameter(torch.randn(max(config.state_dim, 1), 384) * 0.02)
        self.state_type = nn.Parameter(torch.randn(1, 384) * 0.02)
        self.state_queries = nn.Parameter(torch.randn(4, 384) * 0.02)
        self.state_block = CrossBlock()

        self.spatial_adapter = SpatialAdapter()
        # Lightweight student-side heads.  Frozen teachers and their weights
        # are deliberately absent from this module and from saved policies.
        # Keep the established 256-D checkpoint interface from v3.1/v3.2.
        self.semantic_visual = nn.Linear(384, 256)
        self.semantic_text = nn.Linear(384, 256)
        # Auxiliary-only adapter: excluded from the inference computation graph.
        self.tips_text_projection = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 768), nn.GELU(), nn.Linear(768, 768)
        )
        self.grounding_slots = nn.Parameter(torch.randn(8, 384) * 0.02)
        self.gg, self.ga, self.gs, self.gq, self.gp, self.gl = (nn.Linear(384, 384) for _ in range(6))
        self.slots = CompetitiveSlots()
        self.joint = nn.Linear(774, 384)
        self.camera_gate = nn.Sequential(nn.Linear(768, 384), nn.GELU(), nn.Linear(384, 1))
        self.ground_norm = nn.LayerNorm(384)

        self.residual_projection = nn.Linear(384, 384)
        self.residual_norm = nn.LayerNorm(384)
        self.anchor_embedding = nn.Parameter(torch.randn(4, 384) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(4, 384) * 0.02)
        self.fusion = nn.ModuleList([EncoderBlock(), EncoderBlock()])
        self.memory_projection = nn.Sequential(nn.Linear(1536, 384), nn.LayerNorm(384))
        self.action_queries = (
            nn.Parameter(torch.randn(32, 384) * 0.02) if config.action_head == "direct" else None
        )
        self.action_context = nn.Linear(384, 384) if config.action_head == "direct" else None
        self.decoder = (
            nn.ModuleList([DecoderBlock() for _ in range(6)])
            if config.action_head == "direct"
            else nn.ModuleList()
        )
        self.action_out = (
            nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Linear(384, config.action_dim))
            if config.action_head == "direct"
            else None
        )
        self.flow_head = FlowActionHead(config) if config.action_head != "direct" else None
        self.action_anchor = nn.Linear(config.state_dim, config.action_dim)
        nn.init.zeros_(self.action_anchor.weight)
        nn.init.zeros_(self.action_anchor.bias)
        with torch.no_grad():
            diagonal = min(config.state_dim, config.action_dim)
            self.action_anchor.weight[:diagonal, :diagonal].copy_(torch.eye(diagonal))
        if config.action_delta_from_state and self.action_out is not None:
            nn.init.zeros_(self.action_out[-1].weight)
            nn.init.zeros_(self.action_out[-1].bias)
        self._freeze_backbones()

    def _action_anchor(self, state: Tensor) -> Tensor:
        return self.action_anchor(state)[:, None]

    def _to_flow_space(self, action: Tensor, state: Tensor) -> Tensor:
        if self.config.action_head != "architecture_aware_flow":
            return action
        anchor = self._action_anchor(state)
        previous = torch.cat((anchor, action[:, :-1]), 1)
        return (action - previous) / self.config.flow_delta_scale

    def _from_flow_space(self, value: Tensor, state: Tensor) -> Tensor:
        if self.config.action_head != "architecture_aware_flow":
            return value
        return self._action_anchor(state) + (self.config.flow_delta_scale * value).cumsum(1)

    def _freeze_backbones(self) -> None:
        for parameter in self.vision.parameters():
            parameter.requires_grad_(False)
        for block in list(getattr(self.vision, "blocks", ()))[8:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.text.parameters():
            parameter.requires_grad_(False)
        layers = list(getattr(getattr(self.text, "encoder", None), "layer", ()))
        for layer in layers[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

    def _encode_vision(self, images: Tensor) -> Tensor:
        b, c = images.shape[:2]
        images = (images - self.vision_mean.to(images)) / self.vision_std.to(images)
        tokens = self.vision.forward_features(images.flatten(0, 1))
        if isinstance(tokens, dict):
            tokens = tokens.get("x_norm_patchtokens", tokens.get("x_prenorm"))
        if tokens.ndim == 4:
            tokens = tokens.flatten(1, 2)
        tokens = tokens[:, -PATCHES:]
        if tokens.shape[1:] != (PATCHES, 384):
            raise RuntimeError("ViT-S/16 must produce [B*C,294,384]")
        tokens = tokens.view(b, c, PATCHES, 384)
        camera = self.camera_embedding(torch.arange(c, device=images.device))[None, :, None]
        return self.vision_norm(tokens + camera)

    def _encode_text(self, instructions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(
            list(instructions), padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        ).to(device)
        text = self.text(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
        if text.shape[1:] != (64, 384):
            raise RuntimeError("MiniLM-L12-H384 must produce [B,64,384]")
        return text, encoded.attention_mask.bool()

    @staticmethod
    def _instruction_pool(text: Tensor, text_valid: Tensor) -> tuple[Tensor, Tensor]:
        """Return one masked instruction embedding per sample."""
        return _masked_mean(text, text_valid)[:, None], text_valid.any(-1, keepdim=True)

    @staticmethod
    def _mixture(seeds: Tensor, phrases: Tensor, valid: Tensor, q: nn.Module, p: nn.Module) -> Tensor:
        logits = torch.einsum("kd,bpd->bkp", q(seeds), p(phrases)) / 384**0.5
        exists, safe = valid.any(-1), valid.clone()
        safe[~exists, 0] = True
        mix = torch.einsum(
            "bkp,bpd->bkd", _masked_softmax(logits, safe[:, None].expand_as(logits), -1), phrases
        )
        return torch.where(exists[:, None, None], mix, torch.zeros_like(mix))

    def _ground(
        self,
        vision: Tensor,
        content: Tensor,
        phrase: Tensor,
        phrase_valid: Tensor,
        global_text: Tensor,
        za: Tensor,
        zs: Tensor,
    ):
        b, c, n, _ = vision.shape
        spatial = self.spatial_adapter(vision)
        mix = self._mixture(self.grounding_slots, phrase, phrase_valid, self.gq, self.gp)
        global_condition = self.gg(global_text)
        alignment_condition = self.ga(za.mean(1))
        state_condition = self.gs(zs.mean(1))
        phrase_condition = self.gl(mix)
        if self.config.grounding_conditioning == "additive":
            slots = (
                self.grounding_slots[None]
                + global_condition[:, None]
                + alignment_condition[:, None]
                + state_condition[:, None]
                + phrase_condition
            )
        else:
            # Preserve the learned slot identity while allowing bounded context modulation.
            # Each normalized component has a comparable scale; a shared context can no
            # longer overwhelm the much smaller randomly initialized slot parameters.
            width = self.grounding_slots.shape[-1] ** 0.5
            base = F.layer_norm(self.grounding_slots, (self.grounding_slots.shape[-1],))[None]
            shared = (
                F.normalize(global_condition, dim=-1)
                + F.normalize(alignment_condition, dim=-1)
                + F.normalize(state_condition, dim=-1)
            ) / 3**0.5
            phrase_condition = F.normalize(phrase_condition, dim=-1)
            slots = base + self.config.grounding_context_scale * width * (shared[:, None] + phrase_condition)
        conditioned_slots = slots
        slots, assignment = self.slots(slots, spatial.flatten(1, 2), content)
        masks = assignment.mean(1).view(b, 8, c, n)
        content_by_camera = content.view(b, c, n)
        normalized = masks * content_by_camera[:, None].to(masks)
        normalized = normalized / normalized.sum(-1, keepdim=True).clamp_min(EPS)
        pooled = torch.einsum("bkcn,bcnd->bkcd", normalized, spatial)
        yy, xx = torch.meshgrid(
            2 * (torch.arange(14, device=vision.device) + 0.5) / 14 - 1,
            2 * (torch.arange(21, device=vision.device) + 0.5) / 21 - 1,
            indexing="ij",
        )
        coords = torch.stack((xx, yy), -1).view(n, 2).to(vision)
        mean = torch.einsum("bkcn,nd->bkcd", normalized, coords)
        spread = (
            torch.einsum(
                "bkcn,bkcnd->bkcd", normalized, (coords[None, None, None] - mean[..., None, :]).square()
            )
            .clamp_min(EPS)
            .sqrt()
        )
        camera_valid = content_by_camera.sum(-1) > 0
        mass = (masks * content_by_camera[:, None]).sum(-1, keepdim=True) / content_by_camera.sum(-1)[
            :, None, :, None
        ].clamp_min(EPS)
        entropy = -(normalized.clamp_min(EPS) * normalized.clamp_min(EPS).log()).sum(
            -1, keepdim=True
        ) / torch.log(torch.tensor(float(n), device=vision.device))
        geometry = torch.cat((mean, spread, mass, entropy), -1) * camera_valid[:, None, :, None]
        camera = self.camera_embedding(torch.arange(c, device=vision.device))[None, None].expand(b, 8, -1, -1)
        joint = self.joint(torch.cat((pooled, geometry, camera), -1))
        gate_logits = self.camera_gate(torch.cat((slots[:, :, None].expand_as(joint), joint), -1)).squeeze(-1)
        gate = _masked_softmax(gate_logits, camera_valid[:, None].expand_as(gate_logits), -1)
        zg = self.ground_norm(slots + (gate[..., None] * joint).sum(2))
        return zg, spatial, masks.view(b, 8, c, 14, 21), geometry, assignment, conditioned_slots

    def _dense_bypass(self, vision: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        b, c, _, d = vision.shape
        grid, weights = vision.view(b, c, 14, 21, d), content.view(b, c, 14, 21)
        tokens, valid = [], []
        for camera in range(c):
            for anchor, (rows, cols) in enumerate(
                (
                    (slice(0, 7), slice(0, 10)),
                    (slice(0, 7), slice(10, 21)),
                    (slice(7, 14), slice(0, 10)),
                    (slice(7, 14), slice(10, 21)),
                )
            ):
                w = weights[:, camera, rows, cols]
                pooled = (grid[:, camera, rows, cols] * w[..., None]).sum((1, 2)) / w.sum(
                    (1, 2), keepdim=False
                )[:, None].clamp_min(EPS)
                token = self.residual_norm(
                    self.residual_projection(pooled)
                    + self.camera_embedding.weight[camera]
                    + self.anchor_embedding[anchor]
                )
                is_valid = w.sum((1, 2)) > 0
                tokens.append(token * is_valid[:, None])
                valid.append(is_valid)
        return torch.stack(tokens, 1), torch.stack(valid, 1)

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        instructions: Sequence[str],
        *,
        action_target: Tensor | None = None,
        flow_noise: Tensor | None = None,
        flow_time: Tensor | None = None,
        flow_num_steps: int | None = None,
        camera_valid: Tensor | None = None,
        patch_valid: Tensor | None = None,
        patch_content_fraction: Tensor | None = None,
        state_valid: Tensor | None = None,
        memory: Tensor | None = None,
        memory_valid: Tensor | None = None,
        debug: bool = False,
        **_: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        b, c = images.shape[:2]
        if images.shape[2:] != (3, 224, 336) or c > 2:
            raise ValueError("images must be [B,C<=2,3,224,336]")
        if c == 1:
            images = torch.cat((images, torch.zeros_like(images)), 1)
            if camera_valid is None:
                camera_valid = torch.tensor([[True, False]], device=images.device).expand(b, -1)
            else:
                camera_valid = torch.cat(
                    (
                        camera_valid.to(images.device).bool(),
                        torch.zeros(b, 1, dtype=torch.bool, device=images.device),
                    ),
                    1,
                )
            if patch_valid is not None:
                patch_valid = torch.cat(
                    (
                        patch_valid.to(images.device).bool(),
                        torch.zeros(b, 1, 14, 21, dtype=torch.bool, device=images.device),
                    ),
                    1,
                )
            if patch_content_fraction is not None:
                patch_content_fraction = torch.cat(
                    (patch_content_fraction.to(images), torch.zeros(b, 1, 14, 21, device=images.device)), 1
                )
            c = 2
        camera_valid = (
            torch.ones(b, c, dtype=torch.bool, device=images.device)
            if camera_valid is None
            else camera_valid.to(images.device).bool()
        )
        if not camera_valid.any(1).all():
            raise ValueError("each sample needs a valid camera")
        content = patch_content_fraction if patch_content_fraction is not None else patch_valid
        content = (
            camera_valid[:, :, None, None].expand(-1, -1, 14, 21).to(images)
            if content is None
            else content.to(images)
        )
        content = content * camera_valid[:, :, None, None]
        flat_content, flat_valid = content.flatten(1), content.flatten(1) > 0
        if not flat_valid.any(1).all():
            raise ValueError("each sample needs at least one non-padding visual patch")
        vision = self._encode_vision(images)
        text, text_valid = self._encode_text(instructions, images.device)
        global_text = _masked_mean(text, text_valid)
        phrase, phrase_valid = self._instruction_pool(text, text_valid)
        mix = self._mixture(self.alignment_queries, phrase, phrase_valid, self.aq, self.ap)
        aq = self.alignment_queries[None] + self.ag(global_text)[:, None] + self.al(mix)
        za, align_attention = self.alignment_block(aq, vision.flatten(1, 2), ~flat_valid, True)

        state_valid = (
            torch.ones_like(state, dtype=torch.bool)
            if state_valid is None
            else state_valid.to(state.device).bool()
        )
        state_tokens = (
            self.state_value(state[..., None]) + self.state_channels[: state.shape[1]] + self.state_type
        )
        zs = self.state_block(self.state_queries[None].expand(b, -1, -1), state_tokens, ~state_valid)
        zg, spatial, regions, geometry, slot_attention, conditioned_slots = self._ground(
            vision, flat_content, phrase, phrase_valid, global_text, za, zs
        )
        semantic_dense = F.normalize(self.semantic_visual(spatial), dim=-1)
        semantic_language = F.normalize(self.semantic_text(global_text), dim=-1)
        semantic_similarity = torch.einsum("bcnd,bd->bcn", semantic_dense, semantic_language)
        semantic_relevance = semantic_distribution(
            semantic_similarity.flatten(1), flat_valid, self.config.student_semantic_temperature
        )
        if self.config.tips_semantic_action_scale:
            # Make transferred semantic relevance useful to the deployed policy:
            # pool the original 384-D spatial tokens, then inject a bounded residual
            # into the action-consumed Alignment tokens. No teacher is used here.
            semantic_summary = torch.einsum(
                "bn,bnd->bd", semantic_relevance.to(spatial), spatial.flatten(1, 2)
            )
            za = za + self.config.tips_semantic_action_scale * semantic_summary[:, None]
        zr, residual_valid = self._dense_bypass(vision, content)
        current = torch.cat(
            (
                za + self.type_embedding[0],
                zg + self.type_embedding[1],
                zs + self.type_embedding[2],
                zr + self.type_embedding[3],
            ),
            1,
        )

        memory = (
            torch.zeros(b, 4, 384, device=images.device, dtype=current.dtype)
            if memory is None
            else memory.to(current)
        )
        memory_valid = (
            torch.zeros(b, 4, dtype=torch.bool, device=images.device)
            if memory_valid is None
            else memory_valid.to(images.device).bool()
        )
        fused = torch.cat((current, memory), 1)
        current_invalid = torch.cat(
            (torch.zeros(b, 20, dtype=torch.bool, device=images.device), ~residual_valid), 1
        )
        invalid = torch.cat((current_invalid, ~memory_valid), 1)
        for block in self.fusion:
            fused = block(fused, invalid)
        unified = fused[:, :28].masked_fill(current_invalid[..., None], 0)
        summary = self.memory_projection(torch.cat((za.mean(1), zg.mean(1), zs.mean(1), unified.mean(1)), -1))
        next_memory = torch.cat((memory[:, 1:], summary[:, None]), 1)
        next_valid = torch.cat(
            (memory_valid[:, 1:], torch.ones(b, 1, dtype=torch.bool, device=images.device)), 1
        )
        actions = None
        if self.flow_head is None:
            actions = self.action_queries[None] + self.action_context(unified.mean(1))[:, None]
            for block in self.decoder:
                actions = block(actions, unified, current_invalid)
        output = {
            "F": vision.flatten(1, 2),
            "T": text,
            "global_text": global_text,
            "Za": za,
            "Zs": zs,
            "Zg": zg,
            "Fg": spatial,
            "grounding_maps": regions,
            "geometry": geometry,
            "current_latents": current,
            "fusion_context": fused,
            "Z": unified,
            "memory": next_memory,
            "memory_valid": next_valid,
            "patch_valid": content > 0,
            "patch_content_fraction": content,
            "Zr": zr,
            "residual_valid": residual_valid,
            "alignment_attention": align_attention,
            "visual_patch_tokens": vision,
            "dense_grounding": spatial,
            "alignment_cross_attention": align_attention,
            "grounding_cross_attention": slot_attention,
            "conditioned_grounding_slots": conditioned_slots,
            "semantic_relevance": semantic_relevance,
        }
        if debug:
            output.update(alignment_attention=align_attention, slot_attention=slot_attention)
        if self.flow_head is None:
            assert actions is not None and self.action_out is not None
            action = self.action_out(actions)
            if self.config.action_residual_from_state or self.config.action_delta_from_state:
                anchor = self._action_anchor(state)
                action = anchor + (action.cumsum(1) if self.config.action_delta_from_state else action)
            return action, output

        if action_target is not None:
            data = self._to_flow_space(action_target.to(state), state)
            noise = torch.randn_like(data) if flow_noise is None else flow_noise.to(data)
            if flow_time is None:
                distribution = torch.distributions.Beta(
                    self.config.flow_time_alpha, self.config.flow_time_beta
                )
                flow_time = distribution.sample((b,)).to(data.device)
            flow_time = flow_time.to(device=data.device, dtype=torch.float32)
            scale = 1 - (1 - self.config.flow_sigma_min) * flow_time[:, None, None]
            x_t = scale * data + flow_time[:, None, None] * noise
            target_velocity = noise - (1 - self.config.flow_sigma_min) * data
            velocity, gates = self.flow_head(x_t, flow_time, unified, current_invalid)
            clean = self._from_flow_space(x_t - flow_time[:, None, None] * velocity, state)
            output.update(
                flow_velocity=velocity,
                flow_target_velocity=target_velocity,
                flow_time=flow_time,
                flow_clean_prediction=clean,
                action_source_gates=gates,
            )
            return clean, output

        sample = (
            torch.randn(
                b, self.config.chunk_size, self.config.action_dim, device=state.device, dtype=state.dtype
            )
            if flow_noise is None
            else flow_noise.to(state)
        )
        steps = flow_num_steps or self.config.flow_num_steps
        dt = -1.0 / steps
        gates = None
        for step in range(steps):
            time = torch.full((b,), 1.0 + step * dt, device=state.device, dtype=torch.float32)
            velocity, gates = self.flow_head(sample, time, unified, current_invalid)
            sample = sample + dt * velocity
        action = self._from_flow_space(sample, state)
        output["action_source_gates"] = gates
        return action, output


class TIPSGroundVLAPolicy(PreTrainedPolicy):
    config_class, name = TIPSGroundVLAConfig, "tips_ground_vla"
    _action_queue_attrs = (*PreTrainedPolicy._action_queue_attrs, "_queue")
    rtc_accepts_variable_prefix = True

    def __init__(self, config: TIPSGroundVLAConfig, *, model: nn.Module | None = None, **kwargs) -> None:
        del kwargs
        super().__init__(config)
        config.validate_features()
        self.config, self.model = config, model or TIPSGroundVLA(config)
        self.init_rtc_processor()
        self.reset()

    def supports_rtc(self) -> bool:
        return True

    def init_rtc_processor(self) -> None:
        self.rtc_processor = (
            RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None
        )

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def reset(self) -> None:
        self._queue: deque[Tensor] = deque(maxlen=4)
        self._memory = self._memory_valid = None

    def get_optim_params(self):
        groups: dict[tuple[float, float], list[Tensor]] = {}
        vision_lr = 5e-6
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = vision_lr if name.startswith("vision.") else 2e-6 if name.startswith("text.") else 1e-4
            no_decay = parameter.ndim <= 1 or any(
                marker in name.lower() for marker in ("bias", "norm", "embedding", "pos_embed")
            )
            groups.setdefault((lr, 0.0 if no_decay else 0.05), []).append(parameter)
        return [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()]

    def _inputs(self, batch: dict[str, object]):
        cameras, fractions = [], []
        for key in self.config.image_keys[:2]:
            image = batch[key]
            if not isinstance(image, Tensor):
                raise TypeError(f"{key!r} must be a tensor")
            image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
            if image.ndim == 5:
                b, t = image.shape[:2]
                image, fraction = _letterbox_with_content(image.flatten(0, 1))
                image = image.view(b, t, 3, 224, 336)
                fraction = fraction.view(b, t, 14, 21)
            else:
                image, fraction = _letterbox_with_content(image)
            cameras.append(image)
            fractions.append(fraction)
        images = torch.stack(cameras, 2 if cameras[0].ndim == 5 else 1)
        content = torch.stack(fractions, 2 if fractions[0].ndim == 4 else 1)
        language = batch.get(OBS_LANGUAGE, batch.get("task"))
        if language is None:
            raise ValueError("batch needs observation.language or task")
        language = [language] if isinstance(language, str) else list(language)
        state = batch[OBS_STATE]
        if not isinstance(state, Tensor) or len(language) != state.shape[0]:
            raise ValueError("state and language batch sizes must match")
        return images, state, language, content

    @staticmethod
    def _tensor(batch: dict[str, object], key: str, device: torch.device):
        value = batch.get(key)
        return (
            torch.as_tensor(value, device=device)
            if value is not None and not isinstance(value, Tensor)
            else value
        )

    def _kwargs(
        self, batch: dict[str, object], device: torch.device, recurrent: bool = False
    ) -> dict[str, Any]:
        result = {}
        for key in (
            "camera_valid",
            "patch_valid",
            "patch_content_fraction",
            "state_valid",
        ):
            value = self._tensor(batch, key, device)
            if value is not None:
                result[key] = value
        if recurrent:
            result.update(memory=self._memory, memory_valid=self._memory_valid)
        return result

    def forward(self, batch: dict[str, object]) -> tuple[Tensor, dict[str, float]]:
        images, state, language, content = self._inputs(batch)
        if images.ndim == 6:
            memory = memory_valid = None
            steps = images.shape[1]
            for index in range(steps):
                kwargs = self._kwargs(batch, images.device)
                kwargs.setdefault("patch_content_fraction", content[:, index])
                for key, value in tuple(kwargs.items()):
                    if isinstance(value, Tensor) and value.ndim > 1 and value.shape[1] == steps:
                        kwargs[key] = value[:, index]
                pred, output = self.model(
                    images[:, index],
                    state[:, index],
                    language,
                    action_target=batch[ACTION] if self.config.action_head != "direct" else None,
                    memory=memory,
                    memory_valid=memory_valid,
                    **kwargs,
                )
                memory, memory_valid = output["memory"], output["memory_valid"]
        else:
            kwargs = self._kwargs(batch, images.device)
            kwargs.setdefault("patch_content_fraction", content)
            pred, output = self.model(
                images,
                state,
                language,
                action_target=batch[ACTION] if self.config.action_head != "direct" else None,
                **kwargs,
            )
        target = batch[ACTION].to(pred)
        pad = self._tensor(batch, "action_is_pad", pred.device)
        valid = (
            torch.ones(target.shape[:2], dtype=torch.bool, device=pred.device) if pad is None else ~pad.bool()
        )
        horizon_weight = torch.ones_like(valid, dtype=pred.dtype)
        horizon_weight[:, :4] = self.config.first_four_loss_weight
        action, delta = action_losses(pred, target, valid, horizon_weight)
        if self.config.action_head == "direct":
            total = self.config.lambda_action * action + self.config.lambda_action_delta * delta
            losses = {"action_loss": action.item()}
            if self.config.lambda_action_delta:
                losses["action_delta_loss"] = delta.item()
        else:
            velocity_error = (output["flow_velocity"] - output["flow_target_velocity"]).square()
            flow = (velocity_error * valid[..., None]).sum()
            denominator = valid.sum() * target.shape[-1]
            flow = flow / denominator.clamp_min(1)
            motion_weight = (
                self.config.flow_motion_weight
                if self.config.action_head == "architecture_aware_flow"
                else 0.0
            )
            total = self.config.lambda_action * (
                flow + self.config.flow_reconstruction_weight * action + motion_weight * delta
            )
            losses = {
                "flow_matching_loss": flow.item(),
                "action_loss": action.item(),
                "action_delta_loss": delta.item(),
            }
        if self.config.lambda_tips:
            teacher = self._tensor(batch, "tips_teacher_semantic", pred.device)
            if teacher is None:
                raise ValueError("lambda_tips > 0 requires cached tips_teacher_semantic")
            patch_valid = output["patch_valid"].flatten(1)
            if self.config.tips_semantic_target == "alignment_attention":
                # Supervise the attention path whose Alignment tokens are consumed by
                # fusion/action decoding.  The legacy projection target can match the
                # teacher in auxiliary-only heads without making actions use the result.
                student = output["alignment_attention"].float().mean((1, 2))
                student = student.masked_fill(~patch_valid, 0)
                student = student / student.sum(-1, keepdim=True).clamp_min(EPS)
            else:
                student = output["semantic_relevance"]
            tips = semantic_kl(student, teacher.to(student), patch_valid)
            total = total + self.config.lambda_tips * tips
            losses["tips_loss"] = tips.item()
        if self.config.lambda_tips_text:
            teacher_text = self._tensor(batch, "tips_teacher_text", pred.device)
            if teacher_text is None:
                raise ValueError("lambda_tips_text > 0 requires cached tips_teacher_text")
            if self.config.tips_text_shuffle_targets and teacher_text.shape[0] > 1:
                teacher_text = teacher_text.roll(1, 0)
            projected = self.model.tips_text_projection(output["global_text"])
            text_loss, pointwise, relational = text_distillation_loss(
                projected, teacher_text.to(projected), self.config.tips_text_relational_weight
            )
            total = total + self.config.lambda_tips_text * text_loss
            losses.update(
                tips_text_loss=text_loss.item(),
                tips_text_pointwise_loss=pointwise.item(),
                tips_text_relational_loss=relational.item(),
            )
        if self.config.lambda_dino:
            teacher = self._tensor(batch, "dino_teacher_local_relations", pred.device)
            if teacher is None:
                raise ValueError("lambda_dino > 0 requires cached dino_teacher_local_relations")
            dino = spatial_relation_loss(
                output["dense_grounding"], teacher.to(pred.device), output["patch_valid"]
            )
            total = total + self.config.lambda_dino * dino
            losses["dino_loss"] = dino.item()
        losses["loss"] = total.item()
        return total, losses

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, object], **kwargs) -> Tensor:
        if (
            kwargs.get("inference_delay") is not None or kwargs.get("prev_chunk_left_over") is not None
        ) and not self._rtc_enabled():
            raise RuntimeError("RTC arguments require RTC inference")
        images, state, language, content = self._inputs(batch)
        if images.ndim == 6:
            images, state = images[:, -1], state[:, -1]
        model_kwargs = self._kwargs(batch, images.device, self.config.use_temporal_memory)
        model_kwargs.setdefault("patch_content_fraction", content[:, -1] if content.ndim == 5 else content)
        if kwargs.get("flow_noise") is not None:
            model_kwargs["flow_noise"] = kwargs["flow_noise"]
        actions, output = self.model(images, state, language, **model_kwargs)
        if self.config.use_temporal_memory:
            self._memory, self._memory_valid = output["memory"], output["memory_valid"]
        previous = kwargs.get("prev_chunk_left_over")
        if self._rtc_enabled() and previous is not None:
            previous = previous.unsqueeze(0) if previous.ndim == 2 else previous
            steps = min(previous.shape[1], 32)
            weights = self.rtc_processor.get_prefix_weights(
                max(0, kwargs.get("inference_delay") or 0), 4, 32
            ).to(actions)
            actions[:, :steps] = torch.lerp(
                actions[:, :steps], previous[:, :steps].to(actions), weights[:steps].view(1, -1, 1)
            )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, object], **kwargs) -> Tensor:
        del kwargs
        assert not self._rtc_enabled(), "RTC uses predict_action_chunk"
        if not self._queue:
            self._queue.extend(self.predict_action_chunk(batch)[:, :4].transpose(0, 1))
        return self._queue.popleft()
