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

"""Phase 2 model tracer for TurboVLA.

This module deliberately contains no DINOv3 or BERT imports. The faithful network
port is Phase 3; the scalar parameter below only exercises LeRobot's construction,
optimizer, device, save, and reload paths without making an inference claim.
"""

from typing import NoReturn

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_turbovla import TurboVLAConfig


class TurboVLAPolicy(PreTrainedPolicy):
    config_class = TurboVLAConfig
    name = "turbovla"

    def __init__(self, config: TurboVLAConfig, **kwargs: object) -> None:
        del kwargs
        super().__init__(config)
        config.validate_features()
        self._phase2_tracer = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _not_implemented() -> NoReturn:
        raise NotImplementedError(
            "TurboVLA inference and training require the Phase 3 model port; Phase 2 provides only "
            "the native registration/configuration/serialization tracer."
        )

    def reset(self) -> None:
        return None

    def get_optim_params(self) -> dict[str, object]:
        return {"params": self.parameters()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        del batch
        self._not_implemented()

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        del batch, kwargs
        self._not_implemented()

    def select_action(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        del batch, kwargs
        self._not_implemented()
