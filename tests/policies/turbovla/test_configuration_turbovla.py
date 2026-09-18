# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import json

import pytest

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.turbovla.configuration_turbovla import (
    LIBERO_IMAGE_KEYS,
    ROBOTWIN_IMAGE_KEYS,
    TurboVLAConfig,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE, OBS_STATE


def test_libero_and_robotwin_presets_are_complete() -> None:
    libero = TurboVLAConfig.libero(device="cpu")
    robotwin = TurboVLAConfig.robotwin(device="cpu")

    libero.validate_features()
    robotwin.validate_features()
    assert libero.image_keys == LIBERO_IMAGE_KEYS
    assert not libero.local_files_only
    assert (libero.state_dim, libero.action_dim, libero.chunk_size) == (8, 7, 12)
    assert robotwin.image_keys == ROBOTWIN_IMAGE_KEYS
    assert (robotwin.state_dim, robotwin.action_dim, robotwin.chunk_size) == (14, 14, 50)


def test_variant_and_feature_mismatches_fail_early() -> None:
    with pytest.raises(ValueError, match="requires action_dim=14"):
        TurboVLAConfig(variant="robotwin", image_keys=ROBOTWIN_IMAGE_KEYS, state_dim=14, chunk_size=50)

    config = TurboVLAConfig(device="cpu")
    assert config.input_features is not None
    config.input_features.pop(OBS_LANGUAGE)
    with pytest.raises(ValueError, match="language feature"):
        config.validate_features()

    config = TurboVLAConfig(device="cpu")
    assert config.input_features is not None
    config.input_features[LIBERO_IMAGE_KEYS[0]] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    with pytest.raises(ValueError, match="must have shape"):
        config.validate_features()


def test_custom_config_infers_embodiment_from_dataset_features() -> None:
    config = TurboVLAConfig(device="cpu")
    config.adapt_to_dataset_features(
        {
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(9,)),
        }
    )

    config.validate_features()
    assert config.variant == "custom"
    assert config.image_keys == ("observation.images.wrist", "observation.images.top")
    assert config.image_size == (256, 256)
    assert config.resize_images
    assert (config.state_dim, config.action_dim) == (14, 9)


def test_custom_config_accepts_differently_sized_cameras_by_enabling_resize() -> None:
    config = TurboVLAConfig(device="cpu")
    config.adapt_to_dataset_features(
        {
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    config.validate_features()
    assert config.resize_images


def test_config_save_load_routes_through_base_class(tmp_path) -> None:
    config = TurboVLAConfig.robotwin(device="cpu")
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["type"] == "turbovla"
    assert saved["variant"] == "robotwin"
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, TurboVLAConfig)
    assert loaded == config


def test_inferred_custom_schema_is_saved_with_the_checkpoint(tmp_path) -> None:
    config = TurboVLAConfig(device="cpu")
    config.adapt_to_dataset_features(
        {
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }
    )
    config.save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, TurboVLAConfig)
    assert not loaded.infer_from_dataset
    assert loaded.variant == "custom"
    assert loaded.image_keys == ("observation.images.wrist",)
    assert (loaded.state_dim, loaded.action_dim) == (12, 6)
