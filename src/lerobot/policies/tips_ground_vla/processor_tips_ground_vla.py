"""Processors for TIPS-Ground-VLA.

The model owns tokenization; the standard pipeline normalizes state/action and
leaves camera tensors in their dataset-native RGB range.
"""

from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_tips_ground_vla import TIPSGroundVLAConfig


def make_tips_ground_vla_pre_post_processors(
    config: TIPSGroundVLAConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None, **_: Any
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    steps = make_default_policy_processor_steps(config, dataset_stats or {}, normalizer_device=config.device)
    # Language must remain raw strings for the MiniLM tokenizer in the policy.
    steps.normalize.normalize_observation_keys = set(config.image_keys) | {"observation.state"}
    return make_policy_processor_pipelines(
        input_steps=[steps.rename_observations, steps.add_batch_dim, steps.to_device, steps.normalize],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
