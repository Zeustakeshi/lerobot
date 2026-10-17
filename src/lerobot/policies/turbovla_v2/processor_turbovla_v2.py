"""Standalone preprocessing for TurboVLA-v2."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation, TransitionKey
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_LANGUAGE, OBS_STATE

from .configuration_turbovla_v2 import TurboVLAV2Config


@dataclass
@ProcessorStepRegistry.register(name="turbovla_v2_input_processor")
class TurboVLAV2InputProcessorStep(ObservationProcessorStep):
    image_keys: tuple[str, ...]
    image_size: tuple[int, int]
    state_dim: int
    resize_images: bool = True

    def observation(self, observation: RobotObservation) -> RobotObservation:
        result = dict(observation)
        batch_size = None
        for key in self.image_keys:
            if key not in observation or not isinstance(observation[key], torch.Tensor):
                raise ValueError(f"TurboVLA-v2 requires tensor camera {key!r}")
            image = observation[key]
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"{key!r} must have shape [B,3,H,W]")
            image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
            if image.numel() and (image.amin() < 0 or image.amax() > 1):
                raise ValueError("TIPSv2 image values must be in [0,1]")
            if tuple(image.shape[-2:]) != self.image_size:
                if not self.resize_images:
                    raise ValueError(f"{key!r} must have spatial size {self.image_size}")
                image = F.interpolate(image, self.image_size, mode="bilinear", align_corners=False)
            result[key] = image.contiguous()
            batch_size = image.shape[0] if batch_size is None else batch_size
            if image.shape[0] != batch_size:
                raise ValueError("TurboVLA-v2 camera batch sizes must match")
        state = observation.get(OBS_STATE)
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"{OBS_STATE!r} must have shape [B,{self.state_dim}]")
        result[OBS_STATE] = state.float()
        task = observation.get("task", observation.get(OBS_LANGUAGE))
        if task is None:
            task = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("task")
        tasks = [task] if isinstance(task, str) else list(task or [])
        if len(tasks) != batch_size or any(not isinstance(item, str) or not item.strip() for item in tasks):
            raise ValueError("TurboVLA-v2 requires one non-empty instruction per batch item")
        result[OBS_LANGUAGE] = tasks
        result.pop("task", None)
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "image_size": self.image_size,
            "state_dim": self.state_dim,
            "resize_images": self.resize_images,
        }


def make_turbovla_v2_pre_post_processors(config: TurboVLAV2Config, dataset_stats=None):
    if not dataset_stats:
        raise ValueError("TurboVLA-v2 requires dataset statistics")
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    steps.normalize.normalize_observation_keys = {*config.image_keys, OBS_STATE}
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            TurboVLAV2InputProcessorStep(
                image_keys=config.image_keys,
                image_size=config.image_size,
                state_dim=config.state_dim,
                resize_images=config.resize_images,
            ),
            steps.to_device,
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
