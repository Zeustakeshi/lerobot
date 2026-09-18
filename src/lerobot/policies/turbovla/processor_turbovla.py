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

"""Serializable TurboVLA preprocessing and action postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation, TransitionKey
from lerobot.processor import (
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE

from .configuration_turbovla import TurboVLAConfig


@dataclass
@ProcessorStepRegistry.register(name="turbovla_input_processor")
class TurboVLAInputProcessorStep(ObservationProcessorStep):
    """Canonicalize RGB cameras, state, and the exact runtime instruction.

    Images leave this step as channel-first float32 tensors in ``[0, 1]``.
    Text remains un-tokenized: the pinned BERT tokenizer in the model tokenizes
    the exact instruction during the forward pass.
    """

    image_keys: tuple[str, ...]
    image_size: tuple[int, int]
    state_dim: int
    image_crop_size: tuple[int, int] | None = None
    resize_images: bool = False
    interpolation: str = "bilinear"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        self.image_keys = tuple(self.image_keys)
        self.image_size = tuple(self.image_size)
        if self.image_crop_size is not None:
            self.image_crop_size = tuple(self.image_crop_size)
        if self.interpolation not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError("TurboVLA interpolation must be 'bilinear', 'bicubic', or 'nearest'")
        self.image_mean = tuple(self.image_mean)
        self.image_std = tuple(self.image_std)
        if len(self.image_mean) != 3 or len(self.image_std) != 3 or any(std <= 0 for std in self.image_std):
            raise ValueError("TurboVLA image_mean/image_std must contain 3 channels with positive std")

    def _prepare_image(self, key: str, value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"TurboVLA camera {key!r} must be a torch.Tensor")
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(
                f"TurboVLA camera {key!r} must use batched RGB channel-first layout [B,3,H,W]; "
                f"got {tuple(value.shape)}"
            )
        if value.dtype == torch.uint8:
            image = value.to(dtype=torch.float32) / 255.0
        elif value.is_floating_point():
            image = value.to(dtype=torch.float32)
        else:
            raise TypeError(f"TurboVLA camera {key!r} must be uint8 or floating point, got {value.dtype}")
        if not torch.isfinite(image).all():
            raise ValueError(f"TurboVLA camera {key!r} contains non-finite pixels")
        if image.numel() and (image.amin() < 0 or image.amax() > 1):
            raise ValueError(f"TurboVLA camera {key!r} floating pixels must be in [0, 1]")

        if self.image_crop_size is not None:
            crop_h, crop_w = self.image_crop_size
            height, width = image.shape[-2:]
            if crop_h > height or crop_w > width:
                raise ValueError(
                    f"TurboVLA crop {self.image_crop_size} exceeds camera {key!r} size {(height, width)}"
                )
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
            image = image[..., top : top + crop_h, left : left + crop_w]
        if self.resize_images and tuple(image.shape[-2:]) != self.image_size:
            interpolation_kwargs: dict[str, object] = {"size": self.image_size, "mode": self.interpolation}
            if self.interpolation != "nearest":
                interpolation_kwargs["align_corners"] = False
            image = F.interpolate(image, **interpolation_kwargs)
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"TurboVLA camera {key!r} must already be {self.image_size} after optional crop; "
                f"got {tuple(image.shape[-2:])}. Enable resize_images to resize dataset cameras."
            )
        mean = torch.tensor(self.image_mean, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
        return ((image - mean) / std).contiguous()

    def _runtime_tasks(self, observation: RobotObservation, batch_size: int) -> list[str]:
        task = observation.get("task")
        if task is None:
            task = observation.get(OBS_LANGUAGE)
        if task is None:
            task = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("task")
        if isinstance(task, str):
            tasks = [task]
        elif isinstance(task, (list, tuple)) and all(isinstance(item, str) for item in task):
            tasks = list(task)
        else:
            raise ValueError(
                "TurboVLA requires a runtime instruction under 'task' (or "
                f"{OBS_LANGUAGE!r}) as a string or sequence of strings"
            )
        if len(tasks) != batch_size:
            raise ValueError(
                f"TurboVLA received {len(tasks)} task instructions for image batch size {batch_size}"
            )
        if any(not item.strip() for item in tasks):
            raise ValueError("TurboVLA task instructions must be non-empty strings")
        return tasks

    def observation(self, observation: RobotObservation) -> RobotObservation:
        missing = [key for key in (*self.image_keys, OBS_STATE) if key not in observation]
        if missing:
            raise ValueError(f"TurboVLA observation is missing required keys: {missing}")
        new_observation = dict(observation)
        batch_size: int | None = None
        for key in self.image_keys:
            image = self._prepare_image(key, observation[key])
            if batch_size is None:
                batch_size = image.shape[0]
                if batch_size < 1:
                    raise ValueError("TurboVLA camera batches must contain at least one sample")
            elif image.shape[0] != batch_size:
                raise ValueError("TurboVLA camera batch sizes must match")
            new_observation[key] = image

        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"TurboVLA {OBS_STATE!r} must be a torch.Tensor")
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"TurboVLA {OBS_STATE!r} must have shape [B,{self.state_dim}], got {tuple(state.shape)}"
            )
        if state.shape[0] != batch_size:
            raise ValueError("TurboVLA state and camera batch sizes must match")
        if not state.is_floating_point():
            state = state.float()
        if not torch.isfinite(state).all():
            raise ValueError(f"TurboVLA {OBS_STATE!r} contains non-finite values")
        new_observation[OBS_STATE] = state
        new_observation[OBS_LANGUAGE] = self._runtime_tasks(observation, batch_size)
        new_observation.pop("task", None)
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "image_size": self.image_size,
            "state_dim": self.state_dim,
            "image_crop_size": self.image_crop_size,
            "resize_images": self.resize_images,
            "interpolation": self.interpolation,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _canonical_stats(
    config: TurboVLAConfig, stats: dict[str, dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    if not stats:
        raise ValueError(
            "TurboVLA requires dataset statistics for state normalization and action denormalization"
        )

    canonical = dict(stats)
    if ACTION not in canonical and len(canonical) == 1:
        only_value = next(iter(canonical.values()))
        if isinstance(only_value, dict) and ACTION in only_value:
            canonical = dict(only_value)
    if OBS_STATE not in canonical:
        if "state" in canonical:
            canonical[OBS_STATE] = canonical["state"]
        elif "proprio" in canonical:
            canonical[OBS_STATE] = canonical["proprio"]

    requirements = (
        (OBS_STATE, config.state_dim, ("mean", "std")),
        (ACTION, config.action_dim, ("min", "max")),
    )
    for key, expected_dim, names in requirements:
        if key not in canonical:
            raise ValueError(f"TurboVLA statistics are missing required feature {key!r}")
        missing = [name for name in names if name not in canonical[key]]
        if missing:
            raise ValueError(f"TurboVLA statistics for {key!r} are missing {missing}")
        tensors = {name: torch.as_tensor(canonical[key][name]) for name in names}
        if any(tuple(tensor.shape) != (expected_dim,) for tensor in tensors.values()):
            raise ValueError(f"TurboVLA statistics for {key!r} must have shape ({expected_dim},)")
        if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise ValueError(f"TurboVLA statistics for {key!r} must be finite")
        if "std" in tensors and (tensors["std"] <= 0).any():
            raise ValueError(f"TurboVLA statistics for {key!r} must have strictly positive std")
        if "min" in tensors and (tensors["max"] <= tensors["min"]).any():
            raise ValueError(f"TurboVLA statistics for {key!r} must have max greater than min")
    return canonical


def make_turbovla_pre_post_processors(
    config: TurboVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build TurboVLA's serializable, parity-oriented processor pipelines."""
    dataset_stats = _canonical_stats(config, dataset_stats)
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    # Language is intentionally kept as raw strings for the model-owned BERT tokenizer.
    # The generic normalizer tensorizes every configured observation unless explicitly scoped.
    steps.normalize.normalize_observation_keys = {*config.image_keys, OBS_STATE}
    turbovla_inputs = TurboVLAInputProcessorStep(
        image_keys=config.image_keys,
        image_size=config.image_size,
        state_dim=config.state_dim,
        image_crop_size=config.image_crop_size,
        resize_images=config.resize_images,
        interpolation=config.image_interpolation,
        image_mean=config.image_mean,
        image_std=config.image_std,
    )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            turbovla_inputs,
            steps.to_device,
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
