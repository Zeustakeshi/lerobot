"""TIPS-Ground-VLA v3 student policy; teachers never enter the inference graph."""

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

EPS, PATCHES = 1e-6, 14 * 21


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


def semantic_kl(student: Tensor, teacher: Tensor, patch_valid: Tensor, reliability: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError("semantic targets must match [B,6,588]")
    teacher = teacher * patch_valid[:, None].to(teacher)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(EPS)
    values = (teacher * (teacher.clamp_min(EPS).log() - student.clamp_min(EPS).log())).sum(-1)
    weight = reliability.to(values)
    return (values * weight).sum() / weight.sum().clamp_min(EPS)


def spatial_relation_loss(student: Tensor, teacher: Tensor, patch_valid: Tensor) -> Tensor:
    b, c, n, _ = student.shape
    if teacher.shape != (b, c, n, n):
        raise ValueError(f"DINO relations must have shape {(b, c, n, n)}")
    relation = F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).transpose(-1, -2)
    valid = patch_valid.flatten(-2)
    pairs = valid[..., :, None] & valid[..., None, :]
    return ((relation - teacher.to(relation)).square() * pairs).sum() / pairs.sum().clamp_min(1)


def action_losses(pred: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    point = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none")
    action = (point * valid[..., None]).sum() / (valid.sum() * pred.shape[-1]).clamp_min(1)
    dv = valid[:, 1:] & valid[:, :-1]
    delta = F.smooth_l1_loss(
        pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], beta=0.1, reduction="none"
    )
    delta = (delta * dv[..., None]).sum() / (dv.sum() * pred.shape[-1]).clamp_min(1)
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

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        y, _ = self.sa(self.sn(x), self.sn(x), self.sn(x), need_weights=False)
        x = x + y
        y, _ = self.ca(self.cn(x), self.mn(memory), self.mn(memory), need_weights=False)
        x = x + y
        return x + self.ff(self.fn(x))


class SpatialAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(384, 384, 3, padding=1), nn.Conv2d(384, 384, 3, padding=1)
        self.n1, self.n2 = nn.LayerNorm(384), nn.LayerNorm(384)
        self.semantic = nn.Linear(6, 384, bias=False)
        nn.init.zeros_(self.semantic.weight)

    def forward(self, vision: Tensor, semantic: Tensor) -> Tensor:
        b, c, n, d = vision.shape
        x = vision + self.semantic(semantic.permute(0, 2, 3, 1)).view(b, c, n, d)
        grid = x.view(b * c, 14, 21, d).permute(0, 3, 1, 2)
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

    def forward(self, slots: Tensor, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        b, n, _ = x.shape
        k = self.k(self.xn(x)).view(b, n, 6, 64).permute(0, 2, 1, 3)
        v = self.v(self.xn(x)).view(b, n, 6, 64).permute(0, 2, 1, 3)
        raw = None
        for _ in range(2):
            q = self.q(self.sn(slots)).view(b, 8, 6, 64).permute(0, 2, 1, 3)
            logits = torch.einsum("bhkd,bhnd->bhkn", q, k) / 8
            logits = logits.masked_fill(~valid[:, None, None], torch.finfo(logits.dtype).min)
            raw = logits.softmax(2).masked_fill(~valid[:, None, None], 0)
            aggregate = raw / raw.sum(-1, keepdim=True).clamp_min(EPS)
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

        self.semantic_visual, self.semantic_text = nn.Linear(384, 256), nn.Linear(384, 256)
        self.alignment_queries = nn.Parameter(torch.randn(8, 384) * 0.02)
        self.aq, self.ap, self.ag, self.al = (nn.Linear(384, 384) for _ in range(4))
        self.alignment_block = CrossBlock()

        self.state_value = nn.Sequential(nn.Linear(1, 384), nn.GELU(), nn.Linear(384, 384))
        self.state_channels = nn.Parameter(torch.randn(max(config.state_dim, 1), 384) * 0.02)
        self.state_queries = nn.Parameter(torch.randn(4, 384) * 0.02)
        self.state_block = CrossBlock()

        self.spatial_adapter = SpatialAdapter()
        self.grounding_slots = nn.Parameter(torch.randn(8, 384) * 0.02)
        self.gg, self.ga, self.gs, self.gq, self.gp, self.gl = (nn.Linear(384, 384) for _ in range(6))
        self.slots = CompetitiveSlots()
        self.region_slot, self.region_feature = nn.Linear(384, 128), nn.Linear(384, 128)
        self.visibility = nn.Sequential(nn.Linear(386, 192), nn.GELU(), nn.Linear(192, 1))
        self.joint = nn.Linear(774, 384)
        self.camera_gate = nn.Sequential(nn.Linear(768, 384), nn.GELU(), nn.Linear(384, 1))
        self.ground_norm = nn.LayerNorm(384)

        self.type_embedding = nn.Parameter(torch.randn(3, 384) * 0.02)
        self.fusion = nn.ModuleList([EncoderBlock(), EncoderBlock()])
        self.memory_projection = nn.Sequential(nn.Linear(1536, 384), nn.LayerNorm(384))
        self.action_queries = nn.Parameter(torch.randn(32, 384) * 0.02)
        self.action_context = nn.Linear(384, 384)
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(6)])
        self.action_out = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Linear(384, config.action_dim))
        self._freeze_backbones()

    def _freeze_backbones(self) -> None:
        for parameter in self.vision.parameters():
            parameter.requires_grad_(False)
        for block in list(getattr(self.vision, "blocks", ()))[8:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.text.parameters():
            parameter.requires_grad_(False)
        layers = list(getattr(getattr(self.text, "encoder", None), "layer", ()))
        if self.config.training_phase == "joint":
            for layer in layers[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

    def _encode_vision(self, images: Tensor) -> Tensor:
        b, c = images.shape[:2]
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
    def _phrase_pool(
        text: Tensor, text_valid: Tensor, spans: Tensor | None, valid: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        b = text.shape[0]
        if spans is None:
            spans = torch.zeros(b, 6, 64, dtype=torch.bool, device=text.device)
            spans[:, 0] = text_valid
            valid = torch.zeros(b, 6, dtype=torch.bool, device=text.device)
            valid[:, 0] = True
        else:
            spans = spans.to(text.device).bool() & text_valid[:, None]
            valid = spans.any(-1) if valid is None else valid.to(text.device).bool()
        phrase = torch.einsum("bpl,bld->bpd", spans.to(text), text) / spans.sum(-1, keepdim=True).clamp_min(1)
        return phrase, valid

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
        semantic: Tensor,
        valid: Tensor,
        phrase: Tensor,
        phrase_valid: Tensor,
        global_text: Tensor,
        za: Tensor,
        zs: Tensor,
    ):
        b, c, n, _ = vision.shape
        spatial = self.spatial_adapter(vision, semantic.view(b, 6, c, n))
        mix = self._mixture(self.grounding_slots, phrase, phrase_valid, self.gq, self.gp)
        slots = (
            self.grounding_slots[None]
            + self.gg(global_text)[:, None]
            + self.ga(za.mean(1))[:, None]
            + self.gs(zs.mean(1))[:, None]
            + self.gl(mix)
        )
        slots, assignment = self.slots(slots, spatial.flatten(1, 2), valid)
        logits = (
            torch.einsum("bkd,bcnd->bkcn", self.region_slot(slots), self.region_feature(spatial)) / 128**0.5
        )
        masks = logits.sigmoid() * valid.view(b, c, n)[:, None]
        normalized = masks / masks.sum(-1, keepdim=True).clamp_min(EPS)
        pooled = torch.einsum("bkcn,bcnd->bkcd", normalized, spatial)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 14, device=vision.device),
            torch.linspace(-1, 1, 21, device=vision.device),
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
        camera_valid = valid.view(b, c, n).any(-1)
        mass = masks.sum(-1, keepdim=True) / valid.view(b, c, n).sum(-1)[:, None, :, None].clamp_min(1)
        entropy = (
            -(normalized.clamp_min(EPS) * normalized.clamp_min(EPS).log()).sum(-1, keepdim=True) / n**0.5
        )
        visibility = self.visibility(
            torch.cat((slots[:, :, None].expand(-1, -1, c, -1), mass, entropy), -1)
        ).sigmoid()
        geometry = torch.cat((mean, spread, mass, visibility), -1) * camera_valid[:, None, :, None]
        camera = self.camera_embedding(torch.arange(c, device=vision.device))[None, None].expand(b, 8, -1, -1)
        joint = self.joint(torch.cat((pooled, geometry, camera), -1))
        gate_logits = self.camera_gate(torch.cat((slots[:, :, None].expand_as(joint), joint), -1)).squeeze(-1)
        gate = _masked_softmax(gate_logits, camera_valid[:, None].expand_as(gate_logits), -1)
        zg = self.ground_norm(slots + (gate[..., None] * joint).sum(2))
        return zg, spatial, masks.view(b, 8, c, 14, 21), geometry, assignment

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        instructions: Sequence[str],
        *,
        camera_valid: Tensor | None = None,
        patch_valid: Tensor | None = None,
        phrase_spans: Tensor | None = None,
        phrase_valid: Tensor | None = None,
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
            c = 2
        camera_valid = (
            torch.ones(b, c, dtype=torch.bool, device=images.device)
            if camera_valid is None
            else camera_valid.to(images.device).bool()
        )
        if not camera_valid.any(1).all():
            raise ValueError("each sample needs a valid camera")
        patch_valid = (
            camera_valid[:, :, None, None].expand(-1, -1, 14, 21)
            if patch_valid is None
            else patch_valid.to(images.device).bool() & camera_valid[:, :, None, None]
        )
        flat_valid = patch_valid.flatten(1)
        vision = self._encode_vision(images)
        text, text_valid = self._encode_text(instructions, images.device)
        global_text = _masked_mean(text, text_valid)
        phrase, phrase_valid = self._phrase_pool(text, text_valid, phrase_spans, phrase_valid)

        sv, st = (
            F.normalize(self.semantic_visual(vision.flatten(1, 2)), dim=-1),
            F.normalize(self.semantic_text(phrase), dim=-1),
        )
        semantic = (
            _masked_softmax(
                torch.einsum("bpd,bnd->bpn", st, sv) / 0.07, flat_valid[:, None].expand(b, 6, c * PATCHES), -1
            )
            * phrase_valid[..., None]
        )
        mix = self._mixture(self.alignment_queries, phrase, phrase_valid, self.aq, self.ap)
        aq = self.alignment_queries[None] + self.ag(global_text)[:, None] + self.al(mix)
        za, align_attention = self.alignment_block(aq, vision.flatten(1, 2), ~flat_valid, True)

        state_valid = (
            torch.ones_like(state, dtype=torch.bool)
            if state_valid is None
            else state_valid.to(state.device).bool()
        )
        state_tokens = self.state_value(state[..., None]) + self.state_channels[: state.shape[1]]
        zs = self.state_block(self.state_queries[None].expand(b, -1, -1), state_tokens, ~state_valid)
        zg, spatial, regions, geometry, slot_attention = self._ground(
            vision, semantic, flat_valid, phrase, phrase_valid, global_text, za, zs
        )
        current = torch.cat(
            (za + self.type_embedding[0], zg + self.type_embedding[1], zs + self.type_embedding[2]), 1
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
        invalid = torch.cat((torch.zeros(b, 20, dtype=torch.bool, device=images.device), ~memory_valid), 1)
        for block in self.fusion:
            fused = block(fused, invalid)
        unified = fused[:, :20]
        summary = self.memory_projection(torch.cat((za.mean(1), zg.mean(1), zs.mean(1), unified.mean(1)), -1))
        next_memory = torch.cat((memory[:, 1:], summary[:, None]), 1)
        next_valid = torch.cat(
            (memory_valid[:, 1:], torch.ones(b, 1, dtype=torch.bool, device=images.device)), 1
        )
        actions = self.action_queries[None] + self.action_context(unified.mean(1))[:, None]
        for block in self.decoder:
            actions = block(actions, unified)
        output = {
            "F": vision.flatten(1, 2),
            "T": text,
            "S": semantic,
            "phrase_valid": phrase_valid,
            "Za": za,
            "Zs": zs,
            "Zg": zg,
            "Fg": spatial,
            "region_maps": regions,
            "geometry": geometry,
            "current_latents": current,
            "fusion_context": fused,
            "Z": unified,
            "memory": next_memory,
            "memory_valid": next_valid,
            "patch_valid": patch_valid,
            "alignment_attention": align_attention,
        }
        if debug:
            output.update(alignment_attention=align_attention, slot_attention=slot_attention)
        return self.action_out(actions), output


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
        vision_lr = 1e-5 if self.config.training_phase == "perception" else 5e-6
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
        cameras = []
        for key in self.config.image_keys[:2]:
            image = batch[key]
            if not isinstance(image, Tensor):
                raise TypeError(f"{key!r} must be a tensor")
            image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
            if image.ndim == 5:
                b, t = image.shape[:2]
                image = F.interpolate(image.flatten(0, 1), (224, 336), mode="bilinear", align_corners=False)
                image = image.view(b, t, 3, 224, 336)
            else:
                image = F.interpolate(image, (224, 336), mode="bilinear", align_corners=False)
            cameras.append(image)
        images = torch.stack(cameras, 2 if cameras[0].ndim == 5 else 1)
        language = batch.get(OBS_LANGUAGE, batch.get("task"))
        if language is None:
            raise ValueError("batch needs observation.language or task")
        language = [language] if isinstance(language, str) else list(language)
        state = batch[OBS_STATE]
        if not isinstance(state, Tensor) or len(language) != state.shape[0]:
            raise ValueError("state and language batch sizes must match")
        return images, state, language

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
        result = {
            key: self._tensor(batch, key, device)
            for key in ("camera_valid", "patch_valid", "phrase_spans", "phrase_valid", "state_valid")
        }
        if recurrent:
            result.update(memory=self._memory, memory_valid=self._memory_valid)
        return result

    def forward(self, batch: dict[str, object]) -> tuple[Tensor, dict[str, float]]:
        images, state, language = self._inputs(batch)
        if images.ndim == 6:
            memory = memory_valid = None
            steps = images.shape[1]
            for index in range(steps):
                kwargs = self._kwargs(batch, images.device)
                for key, value in tuple(kwargs.items()):
                    if isinstance(value, Tensor) and value.ndim > 1 and value.shape[1] == steps:
                        kwargs[key] = value[:, index]
                pred, output = self.model(
                    images[:, index],
                    state[:, index],
                    language,
                    memory=memory,
                    memory_valid=memory_valid,
                    **kwargs,
                )
                memory, memory_valid = output["memory"], output["memory_valid"]
        else:
            pred, output = self.model(images, state, language, **self._kwargs(batch, images.device))
        target = batch[ACTION].to(pred)
        pad = self._tensor(batch, "action_is_pad", pred.device)
        valid = (
            torch.ones(target.shape[:2], dtype=torch.bool, device=pred.device) if pad is None else ~pad.bool()
        )
        action, delta = action_losses(pred, target, valid)
        total = self.config.lambda_action * action + self.config.lambda_delta * delta
        losses = {"action_loss": action.item(), "delta_loss": delta.item()}
        tips, reliability = batch.get("tips_teacher_semantic"), batch.get("tips_teacher_reliability")
        if isinstance(tips, Tensor) and isinstance(reliability, Tensor):
            sem = semantic_kl(
                output["S"],
                tips.to(pred),
                output["patch_valid"].flatten(1),
                reliability.to(pred) * output["phrase_valid"],
            )
            total += self.config.lambda_semantic * sem
            losses["semantic_loss"] = sem.item()
        dino = batch.get("dino_teacher_relations")
        if isinstance(dino, Tensor):
            spatial = spatial_relation_loss(output["Fg"], dino.to(pred), output["patch_valid"])
            total += self.config.lambda_spatial * spatial
            losses["spatial_loss"] = spatial.item()
        region, region_valid = batch.get("region_target"), batch.get("region_valid")
        if isinstance(region, Tensor) and isinstance(region_valid, Tensor):
            predicted, target_region = output["region_maps"], region.to(pred)
            # Soft best matching is permutation invariant: no slot index is assigned
            # a fixed object/role. J may be smaller than eight.
            predicted = predicted[:, :, None]
            target_region = target_region[:, None]
            bce = F.binary_cross_entropy(
                predicted.expand(-1, -1, target_region.shape[2], -1, -1, -1),
                target_region.expand(-1, predicted.shape[1], -1, -1, -1, -1),
                reduction="none",
            ).mean((-1, -2, -3))
            dice = 1 - (2 * (predicted * target_region).sum((-1, -2, -3)) + EPS) / (
                (predicted + target_region).sum((-1, -2, -3)) + EPS
            )
            weight = region_valid.to(pred)
            pair_loss = 0.5 * bce + 0.5 * dice
            soft_match = (-pair_loss.detach() / 0.1).softmax(1)
            region_loss = ((pair_loss * soft_match).sum(1) * weight).sum() / weight.sum().clamp_min(1)
            total += self.config.lambda_region * region_loss
            losses["region_loss"] = region_loss.item()
        region_box = batch.get("region_box")
        region_camera_valid = batch.get("region_camera_valid")
        if (
            isinstance(region_box, Tensor)
            and isinstance(region_valid, Tensor)
            and isinstance(region_camera_valid, Tensor)
        ):
            # Grounding DINO is box-only: supervise mask-derived center/extent,
            # never turn the rectangle into a silhouette target.
            geometry = output["geometry"]
            center = (geometry[..., :2] + 1) / 2
            extent = (geometry[..., 2:4] * (12**0.5)).clamp(0, 1)
            predicted_box = torch.cat((center - extent / 2, center + extent / 2), -1).clamp(0, 1)
            target_box = region_box.to(pred)
            pair = F.smooth_l1_loss(
                predicted_box[:, :, None].expand(-1, -1, target_box.shape[1], -1, -1),
                target_box[:, None].expand(-1, predicted_box.shape[1], -1, -1, -1),
                beta=0.1,
                reduction="none",
            ).mean(-1)
            camera_weight = region_camera_valid.to(pair)[:, None]
            pair = (pair * camera_weight).sum(-1) / camera_weight.sum(-1).clamp_min(1)
            soft_match = (-pair.detach() / 0.1).softmax(1)
            confidence = region_valid.to(pair)
            box_loss = ((pair * soft_match).sum(1) * confidence).sum() / confidence.sum().clamp_min(1)
            total += self.config.lambda_region * box_loss
            losses["region_box_loss"] = box_loss.item()
        phrase_region = batch.get("phrase_region_target")
        phrase_region_valid = batch.get("phrase_region_valid")
        if isinstance(phrase_region, Tensor) and isinstance(phrase_region_valid, Tensor):
            attention = output["alignment_attention"].mean(1)
            target_mask = phrase_region.to(attention).flatten(2)
            mass = torch.einsum("bqn,bpn->bqp", attention, target_mask)
            best = mass.max(1).values.clamp_min(EPS)
            weight = phrase_region_valid.to(best) * output["phrase_valid"].to(best)
            query_loss = (-(best.log()) * weight).sum() / weight.sum().clamp_min(1)
            total += self.config.lambda_query_region * query_loss
            losses["query_region_loss"] = query_loss.item()
        losses["loss"] = total.item()
        return total, losses

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, object], **kwargs) -> Tensor:
        if (
            kwargs.get("inference_delay") is not None or kwargs.get("prev_chunk_left_over") is not None
        ) and not self._rtc_enabled():
            raise RuntimeError("RTC arguments require RTC inference")
        images, state, language = self._inputs(batch)
        if images.ndim == 6:
            images, state = images[:, -1], state[:, -1]
        actions, output = self.model(images, state, language, **self._kwargs(batch, images.device, True))
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
