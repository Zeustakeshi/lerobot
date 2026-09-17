# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from lerobot.policies import make_pre_post_processors
from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def test_processor_factory_discovery_and_serialization(tmp_path) -> None:
    preprocessor, postprocessor = make_pre_post_processors(TurboVLAConfig(device="cpu"))
    assert preprocessor.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert postprocessor.name == POLICY_POSTPROCESSOR_DEFAULT_NAME

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    assert (tmp_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").is_file()
    assert (tmp_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json").is_file()
