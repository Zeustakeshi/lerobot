"""Configuration for the TIPS-Ground-VLA v1 policy."""

from dataclasses import dataclass, field
from typing import Literal

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


@PreTrainedConfig.register_subclass("tips_ground_vla")
@dataclass
class TIPSGroundVLAConfig(PreTrainedConfig):
    """Independent ViT-S/16 VLA distilled from TIPSv2 and DINOv3 during training."""

    infer_from_dataset: bool = True
    image_keys: tuple[str, ...] = ()
    image_size: tuple[int, int] = (224, 224)
    state_dim: int = 0
    action_dim: int = 0
    chunk_size: int = 12
    n_action_steps: int = 4
    # Injected by ``lerobot-rollout --inference.type=rtc`` after checkpoint
    # loading. It is not needed during training or synchronous inference.
    rtc_config: RTCConfig | None = None
    max_text_length: int = 64
    max_phrases: int = 4
    max_cameras: int = 8
    vision_encoder_id: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    text_encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    tips_teacher_id: str = "google/tipsv2-b14"
    dino_teacher_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    tips_teacher_path: str | None = None
    dino_teacher_path: str | None = None
    cache_teacher_targets: bool = False
    # ``None`` uses Hugging Face's default cache. With ``local_files_only=False``
    # (the default), all pretrained components are fetched automatically on first use.
    hub_cache_dir: str | None = None
    local_files_only: bool = False
    freeze_vision_blocks: int = 10
    training_phase: Literal["warmup", "joint"] = "joint"
    lambda_action: float = 1.0
    lambda_align: float = 0.10
    lambda_spatial: float = 0.05
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
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
        if self.chunk_size != 12:
            raise ValueError("TIPS-Ground-VLA v1 requires chunk_size=12")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.image_size != (224, 224):
            raise ValueError("TIPS-Ground-VLA v1 requires 224x224 images")
        if not 1 <= self.max_phrases <= 4 or self.max_text_length > 64:
            raise ValueError("max_phrases must be 1..4 and max_text_length must be <= 64")
        if self.training_phase == "warmup":
            self.lambda_action, self.lambda_align, self.lambda_spatial = 0.0, 1.0, 0.5

    def adapt_to_dataset_features(self, features: dict[str, PolicyFeature]) -> None:
        if not self.infer_from_dataset:
            return
        images = {key: value for key, value in features.items() if value.type is FeatureType.VISUAL}
        state = features.get(OBS_STATE)
        action = features.get(ACTION)
        if not images or state is None or action is None:
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
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(12))

    @property
    def reward_delta_indices(self):
        return None
