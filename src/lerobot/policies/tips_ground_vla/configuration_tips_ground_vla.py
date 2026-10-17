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
    """Lightweight Action-only ViT-S/16 VLA with grounding and robot-state heads."""

    infer_from_dataset: bool = True
    image_keys: tuple[str, ...] = ()
    image_size: tuple[int, int] = (224, 336)
    state_dim: int = 0
    action_dim: int = 0
    chunk_size: int = 32
    n_action_steps: int = 4
    memory_stride: int = 1
    # The representative multi-episode investigation found this memory path
    # slower and less accurate than the current-observation action-only path.
    use_temporal_memory: bool = False
    action_residual_from_state: bool = False
    # Predict causal action increments, accumulated from the observed state.
    # This prevents the absolute-action decoder from collapsing to a flat
    # conditional mean across heterogeneous demonstrations.
    action_delta_from_state: bool = True
    # ``direct`` preserves the validated checkpoint graph.  The two flow heads
    # are opt-in so loading an existing direct checkpoint remains strict and
    # byte-for-byte compatible.
    action_head: Literal["direct", "flow", "architecture_aware_flow"] = "direct"
    flow_num_steps: int = 4
    flow_num_layers: int = 4
    flow_time_alpha: float = 1.5
    flow_time_beta: float = 1.0
    flow_sigma_min: float = 0.0
    # Train-split increment standard deviations are 0.029--0.096 in the
    # normalized six-dimensional action space.  Scaling them near unit variance
    # prevents the flow prior from becoming an accumulated unit-noise random walk.
    flow_delta_scale: float = 0.075
    flow_reconstruction_weight: float = 0.1
    flow_motion_weight: float = 0.05
    # Injected by ``lerobot-rollout --inference.type=rtc`` after checkpoint
    # loading. It is not needed during training or synchronous inference.
    rtc_config: RTCConfig | None = None
    max_text_length: int = 64
    max_cameras: int = 2
    hidden_dim: int = 384
    patch_grid: tuple[int, int] = (14, 21)
    vision_encoder_id: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    text_encoder_id: str = "microsoft/MiniLM-L12-H384-uncased"
    # ``None`` uses Hugging Face's default cache. With ``local_files_only=False``
    # (the default), all pretrained components are fetched automatically on first use.
    hub_cache_dir: str | None = None
    local_files_only: bool = False
    freeze_vision_blocks: int = 8
    lambda_action: float = 1.0
    lambda_action_delta: float = 0.5
    # Training-only representation teachers.  Zero keeps the deployable,
    # action-only path byte-for-byte independent from teacher artifacts.
    lambda_tips: float = 0.0
    tips_semantic_target: Literal["projection", "alignment_attention"] = "projection"
    # Optional student-only semantic pooling into the action-consumed Alignment tokens.
    # Zero preserves checkpoint-compatible legacy inference.
    tips_semantic_action_scale: float = 0.0
    lambda_tips_text: float = 0.0
    tips_text_relational_weight: float = 1.0
    tips_text_shuffle_targets: bool = False
    lambda_dino: float = 0.0
    tips_teacher_id: str = "google/tipsv2-b14"
    dino_teacher_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    tips_teacher_path: str | None = None
    dino_teacher_path: str | None = None
    teacher_cache_dir: str | None = None
    teacher_prepare: Literal["auto", "require", "off"] = "off"
    tips_temperature: float = 0.005065968260169029
    student_semantic_temperature: float = 0.05
    first_four_loss_weight: float = 1.0
    grounding_conditioning: Literal["additive", "identity_preserving"] = "identity_preserving"
    grounding_context_scale: float = 0.25
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
        if self.max_text_length != 64:
            raise ValueError("TIPS-Ground-VLA v3 requires max_text_length=64")
        if self.n_action_steps != 4:
            raise ValueError("TIPS-Ground-VLA v3 has a fixed execution horizon of 4")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be positive")
        if self.action_residual_from_state and self.action_delta_from_state:
            raise ValueError("choose at most one state-relative action parameterization")
        if self.action_head not in {"direct", "flow", "architecture_aware_flow"}:
            raise ValueError(f"unsupported action_head={self.action_head!r}")
        if self.flow_num_steps < 1 or self.flow_num_layers < 1:
            raise ValueError("flow_num_steps and flow_num_layers must be positive")
        if self.flow_time_alpha <= 0 or self.flow_time_beta <= 0:
            raise ValueError("flow timestep beta parameters must be positive")
        if not 0 <= self.flow_sigma_min < 1:
            raise ValueError("flow_sigma_min must be in [0, 1)")
        if self.flow_delta_scale <= 0:
            raise ValueError("flow_delta_scale must be positive")
        if self.flow_reconstruction_weight < 0 or self.flow_motion_weight < 0:
            raise ValueError("flow auxiliary weights must be non-negative")
        if self.lambda_action_delta < 0:
            raise ValueError("lambda_action_delta must be non-negative")
        if self.lambda_tips < 0 or self.lambda_tips_text < 0 or self.lambda_dino < 0:
            raise ValueError("teacher loss weights must be non-negative")
        if self.tips_semantic_action_scale < 0:
            raise ValueError("tips_semantic_action_scale must be non-negative")
        if self.tips_text_relational_weight < 0:
            raise ValueError("tips_text_relational_weight must be non-negative")
        if self.tips_temperature <= 0 or self.student_semantic_temperature <= 0:
            raise ValueError("semantic temperatures must be positive")
        if self.first_four_loss_weight < 1:
            raise ValueError("first_four_loss_weight must be at least 1")
        if self.grounding_context_scale < 0:
            raise ValueError("grounding_context_scale must be non-negative")

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
        if not self.use_temporal_memory:
            return [0]
        # Four causal observations. A stride of four matches the normal
        # four-action closed-loop execution cadence; one preserves v3.1/v3.2.
        return [-3 * self.memory_stride, -2 * self.memory_stride, -self.memory_stride, 0]

    @property
    def action_delta_indices(self):
        return list(range(32))

    @property
    def reward_delta_indices(self):
        return None
