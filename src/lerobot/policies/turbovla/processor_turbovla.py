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

"""Processor factory for the TurboVLA Phase 2 tracer."""

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_turbovla import TurboVLAConfig


def make_turbovla_pre_post_processors(
    config: TurboVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build serializable standard pipelines pending the Phase 4 parity transforms."""
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
