"""Standalone configuration for TurboVLA-v2."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


@PreTrainedConfig.register_subclass("turbovla_v2")
@dataclass
class TurboVLAV2Config(PreTrainedConfig):
    tips_model_id: str = "google/tipsv2-b14"
    tips_model_revision: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    image_keys: tuple[str, ...] = ()
    image_size: tuple[int, int] = (252, 252)
    resize_images: bool = True
    state_dim: int = 0
    action_dim: int = 0
    chunk_size: int = 32
    n_action_steps: int = 4
    max_text_length: int = 64
    hidden_dim: int = 384
    num_heads: int = 8
    fusion_layers: int = 4
    decoder_layers: int = 3
    feedforward_dim: int = 1536
    dropout: float = 0.1
    freeze_tips_vision: bool = False
    freeze_tips_text: bool = True
    precision: str = "bfloat16"
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(size < 14 or size % 14 for size in self.image_size):
            raise ValueError("TurboVLA-v2 image_size must be divisible by TIPSv2 patch size 14")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be between 1 and chunk_size")
        if not 1 <= self.max_text_length <= 64:
            raise ValueError("TIPSv2 supports max_text_length between 1 and 64")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.precision not in {"bfloat16", "float32"}:
            raise ValueError("precision must be bfloat16 or float32")

    def adapt_to_dataset_features(self, features: dict[str, PolicyFeature]) -> None:
        images = {key: value for key, value in features.items() if value.type is FeatureType.VISUAL}
        state = features.get(OBS_STATE)
        action = features.get(ACTION)
        if not images or state is None or action is None:
            raise ValueError("TurboVLA-v2 requires RGB images, observation.state, and action")
        if any(len(value.shape) != 3 or value.shape[0] != 3 for value in images.values()):
            raise ValueError("TurboVLA-v2 requires channel-first RGB image features")
        if len(state.shape) != 1 or len(action.shape) != 1:
            raise ValueError("TurboVLA-v2 state and action must be vectors")
        self.image_keys = tuple(images)
        self.state_dim = state.shape[0]
        self.action_dim = action.shape[0]
        self.input_features = {
            **images,
            OBS_STATE: state,
            OBS_LANGUAGE: PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
        }
        self.output_features = {ACTION: action}

    def validate_features(self) -> None:
        if not self.image_keys or self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("Create TurboVLA-v2 through lerobot-train or load a complete checkpoint")
        if self.input_features is None or self.output_features is None:
            raise ValueError("TurboVLA-v2 requires explicit input and output features")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
