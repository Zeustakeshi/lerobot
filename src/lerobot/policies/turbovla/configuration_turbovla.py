# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the native TurboVLA policy integration."""

from dataclasses import dataclass, field
from typing import Literal

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE

TurboVLAVariant = Literal["custom", "libero", "robotwin"]

LIBERO_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
ROBOTWIN_IMAGE_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _libero_input_features() -> dict[str, PolicyFeature]:
    return {
        LIBERO_IMAGE_KEYS[0]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        LIBERO_IMAGE_KEYS[1]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        OBS_LANGUAGE: PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
    }


@PreTrainedConfig.register_subclass("turbovla")
@dataclass
class TurboVLAConfig(PreTrainedConfig):
    """Serializable model-construction contract for TurboVLA.

    Fresh training configurations infer their embodiment schema from the training
    dataset. ``libero`` and ``robotwin`` remain fixed presets for converted
    upstream checkpoints, whose tensor shapes and camera order are part of their
    weights.
    """

    variant: TurboVLAVariant = "libero"
    # The default preserves the released LIBERO construction contract for
    # backwards compatibility. `make_policy` replaces it for fresh training
    # when `infer_from_dataset` is enabled.
    infer_from_dataset: bool = True
    image_keys: tuple[str, ...] = LIBERO_IMAGE_KEYS
    image_size: tuple[int, int] = (256, 256)
    image_crop_size: tuple[int, int] | None = None
    resize_images: bool = False
    image_interpolation: str = "bilinear"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    state_dim: int = 8
    action_dim: int = 7
    chunk_size: int = 12
    n_action_steps: int = 12

    vision_encoder_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_encoder_revision: str | None = None
    language_encoder_id: str = "google-bert/bert-base-uncased"
    language_encoder_revision: str | None = None
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    attention_implementation: Literal["manual", "sdpa", "eager"] = "manual"

    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_interaction_layers: int = 6
    interaction_feedforward_dim: int = 2048
    interaction_inner_dim: int = 1024
    num_state_tokens: int = 2
    num_action_decoder_layers: int = 3
    action_mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    text_dropout: float = 0.0
    freeze_text_encoder: bool = True
    freeze_vision_encoder: bool = False
    local_files_only: bool = False
    max_text_length: int = 256
    text_padding_length: int | None = None
    text_padding_length_by_instruction: dict[str, int] = field(default_factory=dict)

    statistics_id: str | None = "turbovla/libero-four-suite-no-noops"
    statistics_revision: str | None = None
    config_schema_version: int = 1
    processor_schema_version: int = 1

    input_features: dict[str, PolicyFeature] | None = field(default_factory=_libero_input_features)
    output_features: dict[str, PolicyFeature] | None = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    )
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.variant not in ("custom", "libero", "robotwin"):
            raise ValueError(f"variant must be 'custom', 'libero', or 'robotwin', got {self.variant!r}")
        if self.chunk_size < 1 or self.n_action_steps < 1 or self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be between 1 and chunk_size")
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError("precision must be 'bfloat16' or 'float32'")
        if self.config_schema_version != 1 or self.processor_schema_version != 1:
            raise ValueError("Only TurboVLA config and processor schema version 1 are supported")
        if self.variant != "custom":
            expected = self._expected_variant_values()
            for name in ("image_keys", "state_dim", "action_dim", "chunk_size"):
                if getattr(self, name) != expected[name]:
                    raise ValueError(
                        f"TurboVLA {self.variant} requires {name}={expected[name]!r}, got {getattr(self, name)!r}"
                    )

    @classmethod
    def libero(cls, **kwargs: object) -> "TurboVLAConfig":
        """Build the released LIBERO preset."""
        defaults: dict[str, object] = {
            "variant": "libero",
            "infer_from_dataset": False,
            "image_keys": LIBERO_IMAGE_KEYS,
            "state_dim": 8,
            "action_dim": 7,
            "chunk_size": 12,
            "n_action_steps": 12,
            "statistics_id": "turbovla/libero-four-suite-no-noops",
            "input_features": _libero_input_features(),
            "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def robotwin(cls, **kwargs: object) -> "TurboVLAConfig":
        """Build the validated RoboTwin schema."""
        defaults: dict[str, object] = {
            "variant": "robotwin",
            "infer_from_dataset": False,
            "image_keys": ROBOTWIN_IMAGE_KEYS,
            "state_dim": 14,
            "action_dim": 14,
            "chunk_size": 50,
            "n_action_steps": 50,
            "vision_encoder_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "hidden_dim": 1024,
            "num_attention_heads": 16,
            "statistics_id": "turbovla/robotwin-released",
            "image_size": (224, 224),
            "input_features": {
                **{
                    key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                    for key in ROBOTWIN_IMAGE_KEYS
                },
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
                OBS_LANGUAGE: PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
            },
            "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def _expected_variant_values(self) -> dict[str, object]:
        if self.variant == "libero":
            return {"image_keys": LIBERO_IMAGE_KEYS, "state_dim": 8, "action_dim": 7, "chunk_size": 12}
        return {"image_keys": ROBOTWIN_IMAGE_KEYS, "state_dim": 14, "action_dim": 14, "chunk_size": 50}

    def adapt_to_dataset_features(self, features: dict[str, PolicyFeature]) -> None:
        """Derive a fresh custom policy's embodiment contract from dataset metadata.

        A trained checkpoint persists the resulting fields, so inference remains
        independent of the source dataset. Fixed upstream variants deliberately
        skip this method because changing their schema invalidates their weights.
        """
        if not self.infer_from_dataset:
            return

        image_features = {
            key: feature for key, feature in features.items() if feature.type is FeatureType.VISUAL
        }
        state_features = {
            key: feature for key, feature in features.items() if feature.type is FeatureType.STATE
        }
        action_features = {
            key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
        }
        if not image_features:
            raise ValueError("TurboVLA custom training requires at least one RGB camera in the dataset")
        if len(state_features) != 1 or OBS_STATE not in state_features:
            raise ValueError(
                f"TurboVLA custom training requires exactly one state feature named {OBS_STATE!r}; "
                f"got {sorted(state_features)}"
            )
        if len(action_features) != 1 or ACTION not in action_features:
            raise ValueError(
                f"TurboVLA custom training requires exactly one action feature named {ACTION!r}; "
                f"got {sorted(action_features)}"
            )

        invalid_images = {
            key: feature.shape
            for key, feature in image_features.items()
            if len(feature.shape) != 3 or feature.shape[0] != 3
        }
        if invalid_images:
            raise ValueError(
                "TurboVLA custom training requires channel-first RGB camera shapes (3, H, W); "
                f"got {invalid_images}"
            )
        state_shape = state_features[OBS_STATE].shape
        action_shape = action_features[ACTION].shape
        if len(state_shape) != 1 or len(action_shape) != 1:
            raise ValueError("TurboVLA custom state and action features must be one-dimensional vectors")

        self.variant = "custom"
        self.infer_from_dataset = False
        self.statistics_id = None
        self.image_keys = tuple(image_features)
        # DINOv3 and the native architecture use a square configured input.
        # Preserve the configured target (256 by default) and resize arbitrary
        # dataset cameras in the processor, just as the other VLA policies do.
        self.resize_images = True
        self.state_dim = state_shape[0]
        self.action_dim = action_shape[0]
        self.input_features = {
            **image_features,
            OBS_STATE: state_features[OBS_STATE],
            # Language is stored as complementary data in many LeRobot datasets,
            # rather than as a tensor feature. The processor reads it as `task`.
            OBS_LANGUAGE: PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
        }
        self.output_features = {ACTION: action_features[ACTION]}

    def validate_features(self) -> None:
        if self.input_features is None or self.output_features is None:
            raise ValueError("TurboVLA requires explicit input_features and output_features")

        image_features = tuple(
            key for key, feature in self.input_features.items() if feature.type is FeatureType.VISUAL
        )
        if not self.image_keys:
            raise ValueError(
                "TurboVLA custom policy has no inferred image keys. Create it through lerobot-train "
                "with a dataset, or provide explicit features in a saved checkpoint."
            )
        if image_features != self.image_keys:
            raise ValueError(
                f"TurboVLA {self.variant} requires cameras in order {self.image_keys!r}; got {image_features!r}"
            )
        for key in self.image_keys:
            if not self.resize_images and self.input_features[key].shape != (3, *self.image_size):
                raise ValueError(
                    f"TurboVLA camera {key!r} must have shape {(3, *self.image_size)!r}; "
                    f"got {self.input_features[key].shape!r}. Set resize_images=true to resize dataset cameras."
                )

        state = self.input_features.get(OBS_STATE)
        if state is None or state.type is not FeatureType.STATE or state.shape != (self.state_dim,):
            raise ValueError(f"TurboVLA {self.variant} requires {OBS_STATE!r} with shape ({self.state_dim},)")
        language = self.input_features.get(OBS_LANGUAGE)
        if language is None or language.type is not FeatureType.LANGUAGE:
            raise ValueError(f"TurboVLA requires a language feature at {OBS_LANGUAGE!r}")
        action = self.output_features.get(ACTION)
        if action is None or action.type is not FeatureType.ACTION or action.shape != (self.action_dim,):
            raise ValueError(f"TurboVLA {self.variant} requires {ACTION!r} with shape ({self.action_dim},)")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
