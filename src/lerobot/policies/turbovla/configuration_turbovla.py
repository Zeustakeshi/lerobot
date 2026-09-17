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

TurboVLAVariant = Literal["libero", "robotwin"]

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

    The configuration captures the released LIBERO and RoboTwin schemas, but no
    benchmark result is implied by constructing this configuration alone.
    """

    variant: TurboVLAVariant = "libero"
    image_keys: tuple[str, ...] = LIBERO_IMAGE_KEYS
    image_size: tuple[int, int] = (256, 256)
    image_crop_size: tuple[int, int] | None = None
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

    statistics_id: str = "turbovla/libero-four-suite-no-noops"
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
        if self.variant not in ("libero", "robotwin"):
            raise ValueError(f"variant must be 'libero' or 'robotwin', got {self.variant!r}")
        if self.n_action_steps < 1 or self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be between 1 and chunk_size")
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError("precision must be 'bfloat16' or 'float32'")
        if self.config_schema_version != 1 or self.processor_schema_version != 1:
            raise ValueError("Only TurboVLA config and processor schema version 1 are supported")
        expected = self._expected_variant_values()
        for name in ("image_keys", "state_dim", "action_dim", "chunk_size"):
            if getattr(self, name) != expected[name]:
                raise ValueError(
                    f"TurboVLA {self.variant} requires {name}={expected[name]!r}, got {getattr(self, name)!r}"
                )

    @classmethod
    def libero(cls, **kwargs: object) -> "TurboVLAConfig":
        """Build the released LIBERO preset."""
        return cls(**kwargs)

    @classmethod
    def robotwin(cls, **kwargs: object) -> "TurboVLAConfig":
        """Build the validated RoboTwin schema."""
        defaults: dict[str, object] = {
            "variant": "robotwin",
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

    def validate_features(self) -> None:
        if self.input_features is None or self.output_features is None:
            raise ValueError("TurboVLA requires explicit input_features and output_features")

        image_features = tuple(
            key for key, feature in self.input_features.items() if feature.type is FeatureType.VISUAL
        )
        if image_features != self.image_keys:
            raise ValueError(
                f"TurboVLA {self.variant} requires cameras in order {self.image_keys!r}; got {image_features!r}"
            )
        for key in self.image_keys:
            if self.input_features[key].shape != (3, *self.image_size):
                raise ValueError(
                    f"TurboVLA camera {key!r} must have shape {(3, *self.image_size)!r}; "
                    f"got {self.input_features[key].shape!r}"
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
