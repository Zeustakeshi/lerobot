"""Configuration for the TIPS-Ground-VLA v3 policy."""

import math
from dataclasses import dataclass, field
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


@LRSchedulerConfig.register_subclass("tips_ground_vla_cosine")
@dataclass
class TIPSGroundVLACosineConfig(LRSchedulerConfig):
    """Cosine decay with the v3-mandated 5% linear warmup."""

    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup = max(1, round(0.05 * num_training_steps))

        def schedule(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, num_training_steps - warmup)
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, schedule, -1)


@PreTrainedConfig.register_subclass("tips_ground_vla")
@dataclass
class TIPSGroundVLAConfig(PreTrainedConfig):
    """Independent ViT-S/16 VLA distilled from TIPSv2 and DINOv3 during training."""

    infer_from_dataset: bool = True
    image_keys: tuple[str, ...] = ()
    image_size: tuple[int, int] = (224, 336)
    state_dim: int = 0
    action_dim: int = 0
    chunk_size: int = 32
    n_action_steps: int = 4
    # Injected by ``lerobot-rollout --inference.type=rtc`` after checkpoint
    # loading. It is not needed during training or synchronous inference.
    rtc_config: RTCConfig | None = None
    max_text_length: int = 64
    max_phrases: int = 6
    max_cameras: int = 2
    hidden_dim: int = 384
    patch_grid: tuple[int, int] = (14, 21)
    vision_encoder_id: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    text_encoder_id: str = "microsoft/MiniLM-L12-H384-uncased"
    tips_teacher_id: str = "google/tipsv2-b14"
    dino_teacher_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    tips_teacher_path: str | None = None
    dino_teacher_path: str | None = None
    cache_teacher_targets: bool = False
    teacher_cache_dir: str | None = None
    teacher_cache_shard_size: int = 64
    teacher_prepare: Literal["auto", "require", "off"] = "auto"
    teacher_audit_frames: int = 100
    teacher_require_validated_regions: bool = True
    grounding_dino_id: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_box_threshold: float = 0.35
    grounding_dino_text_threshold: float = 0.25
    # ``None`` uses Hugging Face's default cache. With ``local_files_only=False``
    # (the default), all pretrained components are fetched automatically on first use.
    hub_cache_dir: str | None = None
    local_files_only: bool = False
    freeze_vision_blocks: int = 8
    training_phase: Literal["perception", "joint"] = "joint"
    lambda_action: float = 1.0
    lambda_delta: float = 0.10
    lambda_semantic: float = 0.20
    lambda_spatial: float = 0.05
    lambda_region: float = 0.50
    lambda_query_region: float = 0.05
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.05
    optimizer_grad_clip_norm: float = 1.0
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_size != 32:
            raise ValueError("TIPS-Ground-VLA v3 requires chunk_size=32")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.image_size != (224, 336) or self.patch_grid != (14, 21):
            raise ValueError("TIPS-Ground-VLA v3 requires 224x336 images and a 14x21 patch grid")
        if self.hidden_dim != 384 or self.max_cameras != 2:
            raise ValueError("TIPS-Ground-VLA v3 requires d=384 and exactly two camera slots")
        if self.max_phrases != 6 or self.max_text_length != 64:
            raise ValueError("TIPS-Ground-VLA v3 requires max_phrases=6 and max_text_length=64")
        if self.n_action_steps != 4:
            raise ValueError("TIPS-Ground-VLA v3 has a fixed execution horizon of 4")
        if self.training_phase == "perception":
            self.lambda_action, self.lambda_delta = 0.0, 0.0
            self.lambda_semantic, self.lambda_spatial = 0.30, 0.10
            self.lambda_region, self.lambda_query_region = 1.0, 0.10
        if self.teacher_audit_frames != 100:
            raise ValueError("TIPS-Ground-VLA v3 requires a 100-frame teacher audit set")
        if self.teacher_cache_shard_size < 1:
            raise ValueError("teacher_cache_shard_size must be positive")

    def adapt_to_dataset_features(self, features: dict[str, PolicyFeature]) -> None:
        if not self.infer_from_dataset:
            return
        images = {key: value for key, value in features.items() if value.type is FeatureType.VISUAL}
        state = features.get(OBS_STATE)
        action = features.get(ACTION)
        if not images or len(images) > 2 or state is None or action is None:
            raise ValueError("TIPS-Ground-VLA needs RGB cameras, observation.state, and action")
        if any(value.shape[0] != 3 for value in images.values()):
            raise ValueError("TIPS-Ground-VLA expects channel-first RGB images")
        if len(state.shape) != 1 or len(action.shape) != 1:
            raise ValueError("TIPS-Ground-VLA state and action must be vectors")
        self.image_keys = tuple(images)
        self.state_dim, self.action_dim = state.shape[0], action.shape[0]
        self.input_features = {
            **images,
            OBS_STATE: state,
            OBS_LANGUAGE: PolicyFeature(FeatureType.LANGUAGE, (1,)),
        }
        self.output_features = {ACTION: action}
        self.infer_from_dataset = False

    def validate_features(self) -> None:
        if not self.image_keys or self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("Create a fresh TIPS-Ground-VLA through lerobot-train with a dataset")
        if self.input_features is None or self.output_features is None:
            raise ValueError("TIPS-Ground-VLA requires explicit input/output features")
        if (
            OBS_LANGUAGE not in self.input_features
            or OBS_STATE not in self.input_features
            or ACTION not in self.output_features
        ):
            raise ValueError("TIPS-Ground-VLA requires language, state, and action features")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=(0.9, 0.95),
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return TIPSGroundVLACosineConfig()

    @property
    def observation_delta_indices(self):
        # Four causal observations from the same episode for recurrent-memory training.
        return [-3, -2, -1, 0]

    @property
    def action_delta_indices(self):
        return list(range(32))

    @property
    def reward_delta_indices(self):
        return None
