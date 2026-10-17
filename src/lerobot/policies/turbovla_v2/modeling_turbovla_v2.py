"""Standalone TurboVLA-v2 implementation using both pretrained TIPSv2 towers."""

from collections import deque
from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_turbovla_v2 import TurboVLAV2Config


class TIPSv2Backbone(nn.Module):
    """One shared checkpoint exposing dense vision features and aligned text features."""

    def __init__(self, config: TurboVLAV2Config) -> None:
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.tips_model_id,
            revision=config.tips_model_revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        self.embedding_dim = int(self.model.config.embed_dim)
        self.patch_size = int(self.model.config.patch_size)
        self.num_patches = (config.image_size[0] // self.patch_size) * (
            config.image_size[1] // self.patch_size
        )
        self._tokenizer = self.model._load_tokenizer()
        if config.freeze_tips_vision:
            self.model.vision_encoder.requires_grad_(False)
        if config.freeze_tips_text:
            self.model.text_encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_tips_vision:
            self.model.vision_encoder.eval()
        if self.config.freeze_tips_text:
            self.model.text_encoder.eval()
        return self

    def encode_images(self, images: Tensor) -> Tensor:
        batch_size, views = images.shape[:2]
        flat = images.flatten(0, 1)
        context = torch.no_grad() if self.config.freeze_tips_vision else nullcontext()
        with context:
            _, _, patches = self.model.vision_encoder(flat)
        if patches.shape[1] != self.num_patches:
            raise RuntimeError(f"TIPSv2 returned {patches.shape[1]} patches, expected {self.num_patches}")
        return patches.view(batch_size, views, self.num_patches, self.embedding_dim)

    def encode_text(self, instructions: Sequence[str], device: torch.device) -> Tensor:
        ids, paddings = self._tokenizer.tokenize(list(instructions), max_len=self.config.max_text_length)
        ids = torch.from_numpy(ids).to(device)
        paddings = torch.from_numpy(paddings).to(device)
        context = torch.no_grad() if self.config.freeze_tips_text else nullcontext()
        with context:
            return self.model.text_encoder(ids, paddings)


class TurboVLAV2Model(nn.Module):
    def __init__(self, config: TurboVLAV2Config) -> None:
        super().__init__()
        self.config = config
        self.tips = TIPSv2Backbone(config)
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(self.tips.embedding_dim),
            nn.Linear(self.tips.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.text_projection = nn.Linear(self.tips.embedding_dim, config.hidden_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.view_embeddings = nn.Parameter(torch.zeros(1, len(config.image_keys), 1, config.hidden_dim))
        self.patch_embeddings = nn.Parameter(torch.zeros(1, 1, self.tips.num_patches, config.hidden_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)
        nn.init.trunc_normal_(self.patch_embeddings, std=0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, config.fusion_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.action_queries = nn.Parameter(torch.zeros(1, config.chunk_size, config.hidden_dim))
        self.action_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)

    def forward(self, instructions: Sequence[str], images: Tensor, state: Tensor) -> Tensor:
        vision = self.tips.encode_images(images)
        vision = self.vision_projection(vision)
        vision = vision + self.view_embeddings + self.patch_embeddings
        vision = vision.flatten(1, 2)
        text = self.text_projection(self.tips.encode_text(instructions, images.device)).unsqueeze(1)
        state_token = self.state_projection(state[:, -1] if state.ndim == 3 else state).unsqueeze(1)
        memory = self.fusion(torch.cat((vision, text, state_token), dim=1))
        queries = self.action_queries.expand(images.shape[0], -1, -1)
        return self.action_head(self.action_decoder(queries, memory))


def _build_model(config: TurboVLAV2Config) -> TurboVLAV2Model:
    require_package("transformers", extra="turbovla")
    require_package("sentencepiece", extra="turbovla")
    return TurboVLAV2Model(config)


class TurboVLAV2Policy(PreTrainedPolicy):
    config_class = TurboVLAV2Config
    name = "turbovla_v2"
    _fsdp_wrap_modules = ["TransformerEncoderLayer", "TransformerDecoderLayer"]

    def __init__(self, config: TurboVLAV2Config, *, model: nn.Module | None = None, **kwargs) -> None:
        del kwargs
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = model or _build_model(config)
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def _inputs(self, batch: dict[str, object]) -> tuple[list[str], Tensor, Tensor]:
        missing = [key for key in (*self.config.image_keys, OBS_STATE, OBS_LANGUAGE) if key not in batch]
        if missing:
            raise ValueError(f"TurboVLA-v2 batch is missing keys: {missing}")
        images = []
        for key in self.config.image_keys:
            image = batch[key]
            if not isinstance(image, Tensor):
                raise TypeError(f"{key!r} must be a tensor")
            images.append(image[:, -1] if image.ndim == 5 else image)
        pixels = torch.stack(images, dim=1)
        state = batch[OBS_STATE]
        if not isinstance(state, Tensor):
            raise TypeError(f"{OBS_STATE!r} must be a tensor")
        language = batch[OBS_LANGUAGE]
        instructions = [language] if isinstance(language, str) else [str(item) for item in language]
        if len(instructions) != pixels.shape[0]:
            raise ValueError("instruction and image batch sizes differ")
        return instructions, pixels, state

    def _predict(self, batch: dict[str, object]) -> Tensor:
        instructions, images, state = self._inputs(batch)
        actions = self.model(instructions, images, state)
        expected = (images.shape[0], self.config.chunk_size, self.config.action_dim)
        if tuple(actions.shape) != expected:
            raise RuntimeError(f"TurboVLA-v2 returned {tuple(actions.shape)}, expected {expected}")
        return actions

    def forward(self, batch: dict[str, object]) -> tuple[Tensor, dict[str, float]]:
        prediction = self._predict(batch)
        target = batch.get(ACTION)
        if not isinstance(target, Tensor) or target.shape != prediction.shape:
            raise ValueError(f"training action must have shape {tuple(prediction.shape)}")
        loss_values = F.l1_loss(prediction, target.to(prediction), reduction="none")
        padding = batch.get("action_is_pad")
        valid = torch.ones(prediction.shape[:2], device=prediction.device, dtype=torch.bool)
        if padding is not None:
            if not isinstance(padding, Tensor) or padding.shape != valid.shape:
                raise ValueError("action_is_pad must have shape [B,chunk_size]")
            valid = ~padding.bool()
        loss = (loss_values * valid.unsqueeze(-1)).sum() / (valid.sum().clamp_min(1) * self.config.action_dim)
        return loss, {"l1_loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, object], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        return self._predict(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, object], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        if not self._action_queue:
            actions = self._predict(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
