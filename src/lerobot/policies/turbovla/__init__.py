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

"""TurboVLA policy integration.

The modeling module is intentionally not imported here so importing LeRobot remains
lightweight and cannot initialize CUDA or download backbone checkpoints.
"""

from .configuration_turbovla import TurboVLAConfig
from .processor_turbovla import make_turbovla_pre_post_processors

__all__ = ["TurboVLAConfig", "make_turbovla_pre_post_processors"]
