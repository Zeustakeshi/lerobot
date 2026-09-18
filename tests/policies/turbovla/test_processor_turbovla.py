# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.processor_turbovla import TurboVLAInputProcessorStep
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_stats(config: TurboVLAConfig) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "mean": torch.arange(config.state_dim, dtype=torch.float32),
            "std": torch.full((config.state_dim,), 2.0),
        },
        ACTION: {
            "mean": torch.arange(config.action_dim, dtype=torch.float32) * 10,
            "std": torch.arange(1, config.action_dim + 1, dtype=torch.float32),
            "min": -torch.arange(1, config.action_dim + 1, dtype=torch.float32),
            "max": torch.arange(1, config.action_dim + 1, dtype=torch.float32),
        },
    }


def make_observation(config: TurboVLAConfig) -> dict[str, object]:
    observation: dict[str, object] = {
        key: torch.full((3, *config.image_size), 255, dtype=torch.uint8) for key in config.image_keys
    }
    observation[OBS_STATE] = torch.arange(config.state_dim, dtype=torch.float32) + 2
    observation["task"] = "pick up the red mug"
    return observation


def test_golden_preprocess_and_action_denormalization() -> None:
    config = TurboVLAConfig(device="cpu")
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=make_stats(config))

    processed = preprocessor(make_observation(config))
    assert processed[OBS_LANGUAGE] == ["pick up the red mug"]
    torch.testing.assert_close(processed[OBS_STATE], torch.ones(1, config.state_dim))
    for key in config.image_keys:
        assert processed[key].shape == (1, 3, *config.image_size)
        assert processed[key].dtype == torch.float32
        expected_image = (
            torch.ones_like(processed[key])
            - torch.tensor(config.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        ) / torch.tensor(config.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        torch.testing.assert_close(processed[key], expected_image)

    normalized_action = torch.ones(1, config.action_dim)
    expected = make_stats(config)[ACTION]["max"]
    torch.testing.assert_close(postprocessor(normalized_action), expected.unsqueeze(0))


def test_libero_camera_orientation_and_dino_normalization_match_upstream_recipe() -> None:
    config = TurboVLAConfig(device="cpu")
    raw = torch.arange(3 * 256 * 256, dtype=torch.int64).remainder(256).to(torch.uint8)
    raw = raw.reshape(1, 3, 256, 256).float() / 255.0
    env_step = LiberoProcessorStep()
    policy_step = TurboVLAInputProcessorStep(
        image_keys=config.image_keys,
        image_size=config.image_size,
        state_dim=config.state_dim,
        image_mean=config.image_mean,
        image_std=config.image_std,
    )

    rotated = env_step.observation({key: raw.clone() for key in config.image_keys})
    mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(config.image_std).view(1, 3, 1, 1)
    expected = (torch.flip(raw, dims=[2, 3]) - mean) / std
    for key in config.image_keys:
        torch.testing.assert_close(policy_step._prepare_image(key, rotated[key]), expected, rtol=0, atol=0)


def test_processor_save_load_is_golden_equivalent(tmp_path) -> None:
    config = TurboVLAConfig(device="cpu")
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=make_stats(config))
    expected_batch = preprocessor(make_observation(config))
    expected_action = postprocessor(torch.full((1, config.action_dim), 0.5))

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded_pre = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )
    loaded_post = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    actual_batch = loaded_pre(make_observation(config))
    actual_action = loaded_post(torch.full((1, config.action_dim), 0.5))

    assert actual_batch[OBS_LANGUAGE] == expected_batch[OBS_LANGUAGE]
    for key in (*config.image_keys, OBS_STATE):
        torch.testing.assert_close(actual_batch[key], expected_batch[key])
    torch.testing.assert_close(actual_action, expected_action)


def test_missing_or_invalid_stats_fail_loudly() -> None:
    config = TurboVLAConfig(device="cpu")
    with pytest.raises(ValueError, match="requires dataset statistics"):
        make_pre_post_processors(config)
    bad_stats = make_stats(config)
    bad_stats[ACTION]["max"][0] = bad_stats[ACTION]["min"][0]
    with pytest.raises(ValueError, match="max greater than min"):
        make_pre_post_processors(config, dataset_stats=bad_stats)


def test_upstream_libero_stats_shape_is_accepted() -> None:
    config = TurboVLAConfig(device="cpu")
    upstream_stats = {
        "libero_all4_no_noops": {
            "proprio": make_stats(config)[OBS_STATE],
            ACTION: make_stats(config)[ACTION],
        }
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=upstream_stats)
    processed = preprocessor(make_observation(config))
    torch.testing.assert_close(processed[OBS_STATE], torch.ones(1, config.state_dim))
    torch.testing.assert_close(
        postprocessor(torch.zeros(1, config.action_dim)), torch.zeros(1, config.action_dim)
    )


@pytest.mark.parametrize("missing", ["task", OBS_STATE, "observation.images.image2"])
def test_required_runtime_inputs_fail_loudly(missing: str) -> None:
    config = TurboVLAConfig(device="cpu")
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=make_stats(config))
    observation = make_observation(config)
    observation.pop(missing)
    with pytest.raises(ValueError, match="runtime instruction|missing required keys"):
        preprocessor(observation)


def test_camera_layout_and_float_range_fail_loudly() -> None:
    config = TurboVLAConfig(device="cpu")
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=make_stats(config))
    observation = make_observation(config)
    observation[config.image_keys[0]] = torch.zeros(128, 192, 3)
    with pytest.raises(ValueError, match="channel-first"):
        preprocessor(observation)

    observation = make_observation(config)
    observation[config.image_keys[0]] = torch.full((3, 128, 192), 1.01)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        preprocessor(observation)

    observation = make_observation(config)
    observation[config.image_keys[0]] = torch.zeros(3, 128, 192, dtype=torch.uint8)
    with pytest.raises(ValueError, match="must already be"):
        preprocessor(observation)


def test_processor_factory_discovery_and_serialization(tmp_path) -> None:
    config = TurboVLAConfig(device="cpu")
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=make_stats(config))
    assert preprocessor.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert postprocessor.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    assert (tmp_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").is_file()
    assert (tmp_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json").is_file()
