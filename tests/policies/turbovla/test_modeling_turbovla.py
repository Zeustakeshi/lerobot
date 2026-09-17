# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from lerobot.policies import get_policy_class
from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.modeling_turbovla import TurboVLAPolicy


def test_model_tracer_constructs_and_save_loads(tmp_path) -> None:
    config = TurboVLAConfig(device="cpu")
    policy = TurboVLAPolicy(config)
    policy._phase2_tracer.data.fill_(3.0)
    policy.save_pretrained(tmp_path)

    loaded = TurboVLAPolicy.from_pretrained(tmp_path)
    assert get_policy_class("turbovla") is TurboVLAPolicy
    assert torch.equal(loaded._phase2_tracer, torch.tensor(3.0))


def test_model_tracer_does_not_claim_inference() -> None:
    policy = TurboVLAPolicy(TurboVLAConfig(device="cpu"))
    with pytest.raises(NotImplementedError, match="Phase 3"):
        policy.select_action({})
