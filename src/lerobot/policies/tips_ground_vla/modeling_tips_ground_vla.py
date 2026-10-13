"""TIPS-Ground-VLA student and its training-only distillation losses."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_tips_ground_vla import TIPSGroundVLAConfig


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def alignment_kl(student: Tensor, teacher: Tensor, valid: Tensor) -> Tensor:
    """KL(TIPS patch distribution || student patch distribution), over patches only."""
    if student.shape != teacher.shape:
        raise ValueError("TIPS target must match student alignment shape [B,C,P,196]")
    values = (teacher.clamp_min(1e-8) * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log())).sum(
        -1
    )
    return (values * valid).sum() / valid.sum().clamp_min(1)


def spatial_kl(student_features: Tensor, teacher: Tensor, valid_views: Tensor) -> Tensor:
    """KL of DINO/student patch-relation rows with their diagonal excluded."""
    b, c, n, _ = student_features.shape
    if teacher.shape != (b, c, n, n):
        raise ValueError("DINO target must have shape [B,C,196,196]")
    normalized = F.normalize(student_features, dim=-1)
    affinity = normalized @ normalized.transpose(-1, -2)
    diagonal = torch.eye(n, dtype=torch.bool, device=affinity.device)[None, None]
    student_log_probs = (affinity / 0.10).masked_fill(diagonal, -torch.inf).log_softmax(-1)
    teacher = teacher.masked_fill(diagonal, 0).clamp_min(0)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-8)
    # Do not form 0 * inf on the deliberately excluded diagonal.
    student_log_probs = student_log_probs.masked_fill(diagonal, 0)
    teacher_log_probs = torch.where(teacher > 0, teacher.log(), torch.zeros_like(teacher))
    values = (teacher * (teacher_log_probs - student_log_probs)).sum(-1)
    return (values * valid_views.unsqueeze(-1)).sum() / (valid_views.sum() * n).clamp_min(1)


class TeacherTargetGenerator:
    """Frozen TIPSv2/DINOv3 teachers used only while computing training losses."""

    def __init__(self, config: TIPSGroundVLAConfig) -> None:
        require_package("transformers", extra="tips-ground-vla")
        from transformers import AutoImageProcessor, AutoModel

        tips_source = config.tips_teacher_path or config.tips_teacher_id
        dino_source = config.dino_teacher_path or config.dino_teacher_id
        tips_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tips = AutoModel.from_pretrained(
            tips_source,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
            torch_dtype=tips_dtype,
        ).eval()
        self.dino = AutoModel.from_pretrained(
            # DINOv3 ViT-L produces NaNs in fp16 on RTX 3050; keep this frozen teacher in fp32.
            dino_source,
            local_files_only=config.local_files_only,
            torch_dtype=torch.float32,
        ).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_source, local_files_only=config.local_files_only
        )
        for module in (self.tips, self.dino):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, images: Tensor, phrases: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Return TIPS [B,C,1,196] and DINO [B,C,196,196] targets."""
        batch, cameras = images.shape[:2]
        flat_images = images.flatten(0, 1)
        device = flat_images.device
        self.tips.to(device)
        self.dino.to(device)
        tips_dtype = next(self.tips.parameters()).dtype
        dino_dtype = next(self.dino.parameters()).dtype
        with torch.autocast(device_type=device.type, enabled=False):
            tips_patches = self.tips.encode_image(flat_images.to(dtype=tips_dtype)).patch_tokens
            tips_text = self.tips.encode_text(list(phrases)).to(device=device, dtype=tips_patches.dtype)
            tips_logits = F.normalize(tips_patches, dim=-1) @ F.normalize(tips_text, dim=-1).unsqueeze(-1)
        tips_logits = tips_logits.squeeze(-1).view(batch, cameras, 16, 16)
        tips_logits = (
            F.interpolate(
                tips_logits.flatten(0, 1).unsqueeze(1), size=(14, 14), mode="bilinear", align_corners=False
            )
            .squeeze(1)
            .view(batch, cameras, 1, 196)
        )
        tips_target = (tips_logits / 0.07).softmax(-1)

        processed = self.dino_processor(images=flat_images, return_tensors="pt")["pixel_values"].to(
            device=device, dtype=dino_dtype
        )
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = self.dino(pixel_values=processed).last_hidden_state
            dino_patches = F.normalize(tokens[:, -196:], dim=-1)
            affinity = dino_patches @ dino_patches.transpose(-1, -2)
        diagonal = torch.eye(196, dtype=torch.bool, device=device).unsqueeze(0)
        dino_target = (affinity.masked_fill(diagonal, -torch.inf) / 0.10).softmax(-1)
        return tips_target, dino_target.view(batch, cameras, 196, 196)


class CrossBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_norm, self.kv_norm = nn.LayerNorm(256), nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.ff_norm = nn.LayerNorm(256)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(
        self, query: Tensor, kv: Tensor, *, key_mask: Tensor | None = None, bias: Tensor | None = None
    ) -> Tensor:
        # MultiheadAttention accepts a per-head additive mask as [B*h, Q, K].
        attn_mask = None
        if bias is not None:
            attn_mask = bias[:, None, :].expand(-1, query.shape[1], -1).repeat_interleave(4, 0)
        value, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(kv),
            self.kv_norm(kv),
            key_padding_mask=key_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        query = query + value
        return query + self.ff(self.ff_norm(query))


class SpatialBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.pointwise = nn.Conv2d(256, 256, 1)
        self.norm = nn.LayerNorm(256)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pointwise(self.depthwise(x))
        y = F.gelu(y).permute(0, 2, 3, 1)
        return x + self.norm(y).permute(0, 3, 1, 2)


class ActionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_norm, self.cross_norm, self.memory_norm, self.ff_norm = (
            nn.LayerNorm(256) for _ in range(4)
        )
        self.self_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        y, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)
        x = x + y
        y, _ = self.cross_attn(
            self.cross_norm(x), self.memory_norm(memory), self.memory_norm(memory), need_weights=False
        )
        x = x + y
        return x + self.ff(self.ff_norm(x))


class TIPSGroundVLA(nn.Module):
    """Student-only inference graph; teacher targets are passed to ``forward`` during training."""

    def __init__(
        self,
        config: TIPSGroundVLAConfig,
        *,
        vision: nn.Module | None = None,
        text: nn.Module | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.config = config
        if vision is None or text is None:
            require_package("timm", extra="tips-ground-vla")
            require_package("transformers", extra="tips-ground-vla")
            import timm
            from transformers import AutoModel, AutoTokenizer

            vision = vision or timm.create_model(config.vision_encoder_id, pretrained=True, num_classes=0)
            text = text or AutoModel.from_pretrained(
                config.text_encoder_id, local_files_only=config.local_files_only
            )
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                config.text_encoder_id, local_files_only=config.local_files_only
            )
        self.vision, self.text, self.tokenizer = vision, text, tokenizer
        self.visual_projection = nn.Sequential(nn.Linear(384, 256), nn.LayerNorm(256))
        self.text_projection = nn.Sequential(nn.Linear(384, 256), nn.LayerNorm(256))
        self.camera_embedding = nn.Embedding(config.max_cameras, 256)
        self.align_visual, self.align_phrase = (
            nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256)),
            nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256)),
        )
        self.align_queries, self.ground_queries, self.state_queries = (
            nn.Parameter(torch.randn(8, 256) * 0.02),
            nn.Parameter(torch.randn(8, 256) * 0.02),
            nn.Parameter(torch.randn(4, 256) * 0.02),
        )
        self.align_text, self.ground_text = nn.Linear(256, 256), nn.Linear(256, 256)
        self.align_block, self.ground_block, self.state_block = CrossBlock(), CrossBlock(), CrossBlock()
        self.spatial = nn.Sequential(SpatialBlock(), SpatialBlock())
        self.state_mlp = nn.Sequential(nn.Linear(config.state_dim, 128), nn.GELU(), nn.Linear(128, 4 * 256))
        self.state_slots = nn.Parameter(torch.randn(4, 256) * 0.02)
        self.fusion_norm, self.fusion_attn, self.fusion_ff_norm = (
            nn.LayerNorm(256),
            nn.MultiheadAttention(256, 4, batch_first=True),
            nn.LayerNorm(256),
        )
        self.fusion_ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
        self.head_types = nn.Parameter(torch.randn(3, 256) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(12, 256) * 0.02)
        self.decoder = nn.ModuleList([ActionBlock(), ActionBlock()])
        self.action_out = nn.Linear(256, config.action_dim)
        self._freeze_encoders()

    def _freeze_encoders(self) -> None:
        for parameter in self.text.parameters():
            parameter.requires_grad_(False)
        blocks = getattr(self.vision, "blocks", ())
        for index, block in enumerate(blocks):
            for parameter in block.parameters():
                parameter.requires_grad_(
                    index >= self.config.freeze_vision_blocks and self.config.training_phase == "joint"
                )

    def _encode_text(self, instructions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        ).to(device)
        hidden = self.text(**tokens).last_hidden_state
        return self.text_projection(hidden), tokens.attention_mask.bool()

    def _encode_vision(self, images: Tensor) -> Tensor:
        batch, cameras = images.shape[:2]
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, None, :, None, None]
        tokens = self.vision.forward_features(((images - mean) / std).flatten(0, 1))
        # timm ViTs return CLS followed by exactly 196 patch tokens for ViT-S/16.
        if tokens.ndim != 3 or tokens.shape[1] < 197:
            raise RuntimeError("student ViT must return CLS plus 196 patch tokens")
        tokens = tokens[:, -196:]
        if tokens.shape[-1] != 384:
            raise RuntimeError("TIPS-Ground-VLA v1 requires ViT-S/16 hidden_dim=384")
        patches = self.visual_projection(tokens).view(batch, cameras, 196, 256)
        return patches + self.camera_embedding(torch.arange(cameras, device=images.device))[None, :, None]

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        instructions: Sequence[str],
        *,
        camera_valid: Tensor | None = None,
        phrases: Sequence[Sequence[str]] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        b, c = images.shape[:2]
        if c > self.config.max_cameras:
            raise ValueError("too many cameras for configured camera embeddings")
        camera_valid = (
            torch.ones(b, c, dtype=torch.bool, device=images.device)
            if camera_valid is None
            else camera_valid.bool()
        )
        if not camera_valid.any(1).all():
            raise ValueError("each sample requires at least one valid camera")
        visual = self._encode_vision(images)
        text, text_mask = self._encode_text(instructions, images.device)
        pooled = _masked_mean(text, text_mask)
        phrase_text = (
            _masked_mean(text, text_mask).unsqueeze(1)
            if phrases is None
            else self._encode_phrases(phrases, images.device)
        )
        flat = visual.flatten(1, 2)
        token_invalid = ~camera_valid.unsqueeze(-1).expand(-1, -1, 196).flatten(1)
        semantic_v, semantic_t = (
            F.normalize(self.align_visual(visual), dim=-1),
            F.normalize(self.align_phrase(phrase_text), dim=-1),
        )
        logits = torch.einsum("bcnd,bpd->bcpn", semantic_v, semantic_t)
        logits = logits.masked_fill(~camera_valid[:, :, None, None], -1e4)
        align_map = (logits / 0.07).softmax(-1)
        prior = align_map.mean(2).detach()
        bias = (0.5 * torch.log((196 * prior).clamp_min(1e-6))).clamp(-2, 2).flatten(1)
        za = self.align_block(
            self.align_queries[None] + self.align_text(pooled)[:, None],
            flat,
            key_mask=token_invalid,
            bias=bias,
        )
        spatial = (
            self.spatial(visual.reshape(b * c, 14, 14, 256).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .reshape(b, c, 196, 256)
        )
        zg = self.ground_block(
            self.ground_queries[None] + self.ground_text(pooled)[:, None],
            spatial.flatten(1, 2),
            key_mask=token_invalid,
            bias=bias,
        )
        source = self.state_mlp(state).view(b, 4, 256) + self.state_slots
        zs = self.state_block(self.state_queries[None].expand(b, -1, -1), source)
        latent = torch.cat([za + self.head_types[0], zg + self.head_types[1], zs + self.head_types[2]], 1)
        y, _ = self.fusion_attn(
            self.fusion_norm(latent), self.fusion_norm(latent), self.fusion_norm(latent), need_weights=False
        )
        latent = latent + y
        latent = latent + self.fusion_ff(self.fusion_ff_norm(latent))
        actions = self.action_queries[None].expand(b, -1, -1)
        for block in self.decoder:
            actions = block(actions, latent)
        return self.action_out(actions), {
            "student_align_map": align_map,
            "dense_grounding": spatial,
            "Za": za,
            "Zg": zg,
            "Zs": zs,
            "Z": latent,
        }

    def _encode_phrases(self, phrases: Sequence[Sequence[str]], device: torch.device) -> Tensor:
        flattened = [phrase for sample in phrases for phrase in sample[: self.config.max_phrases]]
        sizes = [min(len(sample), self.config.max_phrases) for sample in phrases]
        if not flattened or len(set(sizes)) != 1:
            raise ValueError(
                "phrase batches must have a non-zero, uniform phrase count; pad in the data pipeline"
            )
        text, mask = self._encode_text(flattened, device)
        return _masked_mean(text, mask).view(len(phrases), sizes[0], 256)


class TIPSGroundVLAPolicy(PreTrainedPolicy):
    config_class, name = TIPSGroundVLAConfig, "tips_ground_vla"
    # ``lerobot-rollout`` drops queued actions when the task changes.  This
    # policy intentionally uses a small private FIFO instead of the common
    # ``_action_queue`` name, so declare it to the base-class queue contract.
    _action_queue_attrs = (*PreTrainedPolicy._action_queue_attrs, "_queue")
    # Direct action-space RTC must receive the real leftover length. Padding a
    # short prefix with zeros would blend invalid actions into the prediction.
    rtc_accepts_variable_prefix = True

    def supports_rtc(self) -> bool:
        return True

    def __init__(self, config: TIPSGroundVLAConfig, *, model: nn.Module | None = None, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = model or TIPSGroundVLA(config)
        # Keep teachers outside ``nn.Module``: training checkpoints must only contain the student.
        object.__setattr__(self, "_teachers", None)
        self.init_rtc_processor()
        self.reset()

    def init_rtc_processor(self) -> None:
        """Initialize action-space RTC after the rollout context injects its config."""
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_rtc_prefix(
        self,
        actions: Tensor,
        *,
        inference_delay: int | None,
        prev_chunk_left_over: Tensor | None,
    ) -> Tensor:
        """Blend the unexecuted previous chunk into a direct action prediction."""
        if prev_chunk_left_over is None:
            return actions
        if self.rtc_processor is None:
            raise RuntimeError("TIPS-Ground-VLA RTC is enabled but its processor is not initialized")

        previous = prev_chunk_left_over
        if previous.ndim == 2:
            previous = previous.unsqueeze(0)
        if previous.ndim != 3:
            raise ValueError("prev_chunk_left_over must have shape [T, A] or [B, T, A]")
        if previous.shape[0] == 1 and actions.shape[0] > 1:
            previous = previous.expand(actions.shape[0], -1, -1)
        if previous.shape[0] != actions.shape[0]:
            raise ValueError("previous action batch size must match the observation batch size")
        if previous.shape[2] != actions.shape[2]:
            raise ValueError("previous action dimension must match action_dim")

        delay = max(0, 0 if inference_delay is None else inference_delay)
        prefix_steps = min(previous.shape[1], actions.shape[1])
        if prefix_steps == 0:
            return actions
        weights = self.rtc_processor.get_prefix_weights(
            delay,
            self.config.rtc_config.execution_horizon,
            actions.shape[1],
        ).to(device=actions.device, dtype=actions.dtype)
        result = actions.clone()
        result[:, :prefix_steps] = torch.lerp(
            actions[:, :prefix_steps],
            previous[:, :prefix_steps].to(actions),
            weights[:prefix_steps].view(1, prefix_steps, 1),
        )
        return result

    def reset(self) -> None:
        self._queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _inputs(self, batch: dict[str, object]):
        cameras = []
        for key in self.config.image_keys:
            image = batch[key]
            if not isinstance(image, Tensor):
                raise TypeError(f"{key!r} must be a tensor")
            image = image[:, -1] if image.ndim == 5 else image
            image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
            cameras.append(F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False))
        images = torch.stack(cameras, 1)
        language = batch.get(OBS_LANGUAGE, batch.get("task"))
        if language is None:
            raise ValueError(f"batch must contain {OBS_LANGUAGE!r} or the dataset task string")
        language = [language] if isinstance(language, str) else list(language)
        state = batch[OBS_STATE]
        if not isinstance(state, Tensor):
            raise TypeError(f"{OBS_STATE!r} must be a tensor")
        if len(language) != state.shape[0]:
            raise ValueError(
                f"instruction batch size ({len(language)}) must match state batch size ({state.shape[0]})"
            )
        return images, state, language

    def _teacher_targets(self, images: Tensor, instructions: Sequence[str]) -> tuple[Tensor, Tensor]:
        teachers = self._teachers
        if teachers is None:
            teachers = TeacherTargetGenerator(self.config)
            object.__setattr__(self, "_teachers", teachers)
        return teachers(images, instructions)

    def forward(self, batch: dict[str, object]) -> tuple[Tensor, dict[str, float]]:
        images, state, instructions = self._inputs(batch)
        actions, outputs = self.model(images, state, instructions)
        target = batch[ACTION].to(actions)
        valid = ~batch.get(
            "action_is_pad", torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
        ).bool()
        action_loss = (F.l1_loss(actions, target, reduction="none") * valid.unsqueeze(-1)).sum() / (
            valid.sum() * target.shape[-1]
        ).clamp_min(1)
        loss = self.config.lambda_action * action_loss
        losses = {"action_loss": action_loss.item()}
        generated_targets: tuple[Tensor, Tensor] | None = None
        if self.config.lambda_align:
            teacher = batch.get("tips_teacher_align")
            if not isinstance(teacher, Tensor):
                generated_targets = self._teacher_targets(images, instructions)
                teacher = generated_targets[0]
            align_loss = alignment_kl(
                outputs["student_align_map"],
                teacher.to(actions),
                torch.ones(teacher.shape[:-1], dtype=torch.bool, device=actions.device),
            )
            loss = loss + self.config.lambda_align * align_loss
            losses["align_loss"] = align_loss.item()
        if self.config.lambda_spatial:
            teacher = batch.get("dino_teacher_relations")
            if not isinstance(teacher, Tensor):
                if generated_targets is None:
                    generated_targets = self._teacher_targets(images, instructions)
                teacher = generated_targets[1]
            grounding_loss = spatial_kl(
                outputs["dense_grounding"],
                teacher.to(actions),
                torch.ones(teacher.shape[:2], dtype=torch.bool, device=actions.device),
            )
            loss = loss + self.config.lambda_spatial * grounding_loss
            losses["spatial_loss"] = grounding_loss.item()
        losses["loss"] = loss.item()
        return loss, losses

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, object], **kwargs) -> Tensor:
        inference_delay = kwargs.get("inference_delay")
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
        if (inference_delay is not None or prev_chunk_left_over is not None) and not self._rtc_enabled():
            raise RuntimeError("RTC arguments require --inference.type=rtc")
        if inference_delay is not None and not isinstance(inference_delay, int):
            raise TypeError("inference_delay must be an int or None")
        if prev_chunk_left_over is not None and not isinstance(prev_chunk_left_over, Tensor):
            raise TypeError("prev_chunk_left_over must be a torch.Tensor or None")
        actions = self.model(*self._inputs(batch))[0]
        if self._rtc_enabled():
            actions = self._apply_rtc_prefix(
                actions,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, object], **kwargs) -> Tensor:
        assert not self._rtc_enabled(), "RTC uses predict_action_chunk, not select_action"
        if not self._queue:
            self._queue.extend(
                self.predict_action_chunk(batch)[:, : self.config.n_action_steps].transpose(0, 1)
            )
        return self._queue.popleft()
