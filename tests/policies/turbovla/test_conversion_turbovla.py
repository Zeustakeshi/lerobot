# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from torch import nn

from lerobot.policies.turbovla import conversion_turbovla as conversion
from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.modeling_turbovla import TurboVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


class TinyTurboVLA(nn.Module):
    def __init__(self, config: TurboVLAConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.state_dim, config.action_dim)

    def forward(self, instructions, samples, state):
        del instructions, samples
        return self.projection(state)[:, None].expand(-1, 12, -1)


def _patch_policy(monkeypatch):
    class SyntheticPolicy(TurboVLAPolicy):
        def __init__(self, config: TurboVLAConfig, **kwargs) -> None:
            super().__init__(config, model=TinyTurboVLA(config), **kwargs)

    monkeypatch.setattr(conversion, "TurboVLAPolicy", SyntheticPolicy)
    monkeypatch.setattr(
        "lerobot.policies.turbovla.modeling_turbovla._build_model",
        lambda config: TinyTurboVLA(config),
    )
    return SyntheticPolicy


def test_strict_key_mapping_reports_unexpected_and_missing() -> None:
    config = TurboVLAConfig(device="cpu", precision="float32")
    target = TinyTurboVLA(config).state_dict()
    source = {
        "projection.weight": torch.zeros_like(target["projection.weight"]),
        "extra.weight": torch.zeros(1),
    }
    with pytest.raises(ValueError, match="unexpected source keys|missing target keys"):
        conversion.map_state_dict(source, target, strict=True)

    mapped, names, ignored = conversion.map_state_dict(
        source,
        target,
        ignore_patterns=("extra.*",),
        strict=False,
    )
    assert set(mapped) == {"projection.weight"}
    assert names == {"projection.weight": "projection.weight"}
    assert ignored == ["extra.weight"]


def test_robotwin_ema_safetensors_loader_prefers_ema_prefix(tmp_path) -> None:
    checkpoint = tmp_path / "robotwin.safetensors"
    save_safetensors(
        {
            "projection.weight": torch.ones(2, 2),
            "ema.projection.weight": torch.full((2, 2), 3.0),
        },
        checkpoint,
    )
    state = conversion.load_source_state(checkpoint, "robotwin-ema-safetensors")
    assert list(state) == ["projection.weight"]
    torch.testing.assert_close(state["projection.weight"], torch.full((2, 2), 3.0))


def test_libero_loader_prefers_released_ema_model_state_dict(tmp_path) -> None:
    checkpoint = tmp_path / "libero.pth"
    torch.save(
        {
            "model_state_dict": {"projection.weight": torch.ones(2, 2)},
            "ema_model_state_dict": {"projection.weight": torch.full((2, 2), 3.0)},
        },
        checkpoint,
    )

    state = conversion.load_source_state(checkpoint, "libero-pth")

    torch.testing.assert_close(state["projection.weight"], torch.full((2, 2), 3.0))


def test_libero_loader_supports_unified_model_state_dict(tmp_path) -> None:
    checkpoint = tmp_path / "libero-unified.pth"
    torch.save(
        {"model_state_dict": {"projection.weight": torch.full((2, 2), 7.0)}},
        checkpoint,
    )

    state = conversion.load_source_state(checkpoint, "libero-pth")

    torch.testing.assert_close(state["projection.weight"], torch.full((2, 2), 7.0))


def test_released_dinov3_layer_namespace_maps_to_current_transformers() -> None:
    source = {"vision_encoder.backbone.layer.0.attention.q_proj.weight": torch.ones(2, 2)}
    target = {"model.vision_encoder.backbone.model.layer.0.attention.q_proj.weight": torch.zeros(2, 2)}

    mapped, names, ignored = conversion.map_state_dict(source, target, strict=True)

    assert ignored == []
    assert names == {
        "vision_encoder.backbone.layer.0.attention.q_proj.weight": (
            "model.vision_encoder.backbone.model.layer.0.attention.q_proj.weight"
        )
    }
    torch.testing.assert_close(
        mapped["model.vision_encoder.backbone.model.layer.0.attention.q_proj.weight"], torch.ones(2, 2)
    )


def test_source_text_layout_is_preserved_in_converted_config() -> None:
    config = conversion.build_config(
        "libero",
        source_model_config={
            "text": {
                "model_name_or_path": "bert-base-uncased",
                "max_length": 256,
                "padding_length": 21,
                "padding_length_by_instruction": {"put the bowl on the plate": 11},
            },
            "interaction": {"attention_backend": "manual"},
        },
    )

    assert config.language_encoder_id == "bert-base-uncased"
    assert config.max_text_length == 256
    assert config.text_padding_length == 21
    assert config.text_padding_length_by_instruction == {"put the bowl on the plate": 11}
    assert config.attention_implementation == "manual"


def test_released_libero_checkpoint_strict_conversion(tmp_path) -> None:
    source_path = os.environ.get("TURBOVLA_LIBERO_SOURCE_PATH")
    if source_path is None:
        pytest.skip("Set TURBOVLA_LIBERO_SOURCE_PATH to run the released-checkpoint integration test.")
    source = Path(source_path)
    if not source.is_file():
        pytest.skip(f"TURBOVLA_LIBERO_SOURCE_PATH does not exist: {source}")

    output_dir = tmp_path / "converted"
    report = conversion.convert_checkpoint(
        source_path=source,
        output_dir=output_dir,
        variant="libero",
        source_kind="libero-pth",
        expected_sha256=os.environ.get("TURBOVLA_LIBERO_SOURCE_SHA256"),
        strict=True,
    )

    assert len(report.loaded_keys) == 672
    assert len(report.mapped_keys) == 672
    assert report.ignored_keys == []
    loaded = TurboVLAPolicy.from_pretrained(output_dir, strict=True)
    assert loaded.config.text_padding_length == 21
    assert loaded.config.text_padding_length_by_instruction["put the bowl on the plate"] == 11
    assert loaded.config.attention_implementation == "manual"
    actions = loaded.predict_action_chunk(
        {
            "observation.images.image": torch.zeros(1, 3, 256, 256),
            "observation.images.image2": torch.zeros(1, 3, 256, 256),
            OBS_STATE: torch.zeros(1, 8),
            OBS_LANGUAGE: ["put the bowl on the plate"],
        }
    )
    assert actions.shape == (1, 12, 7)
    assert torch.isfinite(actions).all()


def test_convert_checkpoint_writes_lerobot_artifacts_and_loads(monkeypatch, tmp_path) -> None:
    _patch_policy(monkeypatch)
    config = TurboVLAConfig(device="cpu", precision="float32")
    source_state = {
        "model.projection.weight": torch.full((config.action_dim, config.state_dim), 0.25),
        "model.projection.bias": torch.full((config.action_dim,), -0.5),
    }
    source = tmp_path / "libero.pth"
    torch.save({"state_dict": source_state}, source)
    output_dir = tmp_path / "converted"

    report = conversion.convert_checkpoint(
        source_path=source,
        output_dir=output_dir,
        variant="libero",
        source_kind="libero-pth",
        source_revision="cb53005",
        source_url="https://example.com/turbovla-libero.pth",
        checkpoint_license="other",
    )

    assert report.source_sha256 == conversion.sha256_file(source)
    assert report.source_revision == "cb53005"
    assert report.source_url == "https://example.com/turbovla-libero.pth"
    assert report.checkpoint_license == "other"
    expected_files = {
        "config.json",
        "model.safetensors",
        f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        "README.md",
        conversion.REPORT_FILENAME,
        conversion.PROVENANCE_FILENAME,
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    saved_state = load_safetensors(output_dir / "model.safetensors")
    torch.testing.assert_close(
        saved_state["model.projection.weight"], source_state["model.projection.weight"]
    )
    torch.testing.assert_close(saved_state["model.projection.bias"], source_state["model.projection.bias"])

    report_json = json.loads((output_dir / conversion.REPORT_FILENAME).read_text())
    assert report_json["mapped_keys"] == {
        "model.projection.bias": "model.projection.bias",
        "model.projection.weight": "model.projection.weight",
    }
    assert report_json["tensor_records"][0]["sha256"]
    provenance = json.loads((output_dir / conversion.PROVENANCE_FILENAME).read_text())
    assert provenance["variant"] == "libero"
    assert provenance["source_url"] == "https://example.com/turbovla-libero.pth"
    assert provenance["source_revision"] == "cb53005"
    assert provenance["benchmark_status"] == "pending"
    assert provenance["pending_benchmark_values"]["aggregate_success_rate"] is None
    assert provenance["config_schema_version"] == 1
    model_card = (output_dir / "README.md").read_text()
    assert "simulation-trained" in model_card
    assert "aggregate_success_rate:" in model_card

    loaded = TurboVLAPolicy.from_pretrained(output_dir, strict=True)
    torch.testing.assert_close(loaded.model.projection.weight, source_state["model.projection.weight"])


def test_expected_hash_mismatch_fails_before_writing(monkeypatch, tmp_path) -> None:
    _patch_policy(monkeypatch)
    source = tmp_path / "libero.pth"
    torch.save({"state_dict": {}}, source)
    output_dir = tmp_path / "converted"

    with pytest.raises(ValueError, match="hash mismatch"):
        conversion.convert_checkpoint(
            source_path=source,
            output_dir=output_dir,
            variant="libero",
            source_kind="libero-pth",
            expected_sha256="0" * 64,
        )
    assert not output_dir.exists()


def test_cli_help_is_import_lightweight() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.policies.turbovla.conversion_turbovla", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--source-kind" in result.stdout
    assert "--source-revision" in result.stdout
    assert "--source-url" in result.stdout
