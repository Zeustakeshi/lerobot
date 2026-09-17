# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import subprocess
import sys

import draccus

from lerobot.configs import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import TurboVLAConfig, get_policy_class, make_policy_config


def test_native_registry_and_factories_need_no_plugin_registration() -> None:
    assert "turbovla" in PreTrainedConfig.get_known_choices()
    config = make_policy_config("turbovla", device="cpu")
    assert isinstance(config, TurboVLAConfig)
    assert get_policy_class("turbovla").__name__ == "TurboVLAPolicy"


def test_train_cli_accepts_turbovla_policy_type(tmp_path) -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=test/dummy",
            "--policy.type=turbovla",
            "--policy.device=cpu",
            f"--output_dir={tmp_path}",
            "--job_name=turbovla-smoke",
        ],
    )
    assert isinstance(config.policy, TurboVLAConfig)


def test_import_is_lightweight_and_does_not_initialize_cuda() -> None:
    code = """
import sys
import torch
before = torch.cuda.is_initialized()
import lerobot.policies.turbovla
assert torch.cuda.is_initialized() == before
assert 'lerobot.policies.turbovla.modeling_turbovla' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
