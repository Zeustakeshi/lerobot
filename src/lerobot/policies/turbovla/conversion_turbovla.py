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

"""Checkpoint conversion utilities for TurboVLA.

The converter is deliberately strict: users must select the benchmark variant,
source hashes are checked when supplied, and every loaded source tensor must
either map to a LeRobot tensor or be explicitly ignored.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.libero_benchmark import TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES
from lerobot.policies.turbovla.modeling_turbovla import TurboVLAPolicy
from lerobot.policies.turbovla.processor_turbovla import make_turbovla_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

SourceKind = Literal["libero-pth", "robotwin-ema-safetensors"]
Variant = Literal["libero", "robotwin"]

CONVERSION_VERSION = 1
REPORT_FILENAME = "conversion_report.json"
PROVENANCE_FILENAME = "provenance.json"


@dataclass(frozen=True)
class TensorRecord:
    key: str
    shape: list[int]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class ConversionReport:
    conversion_version: int
    source_path: str
    source_revision: str | None
    source_url: str | None
    source_kind: SourceKind
    source_sha256: str
    checkpoint_license: str
    variant: Variant
    output_dir: str
    loaded_keys: list[str]
    ignored_keys: list[str]
    mapped_keys: dict[str, str]
    tensor_records: list[TensorRecord]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _extract_pickle_state(payload: object, path: Path) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in (
            "ema_model_state_dict",
            "ema_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
            "module",
            "ema",
        ):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and all(torch.is_tensor(value) for value in nested.values()):
                return dict(nested)
        if all(torch.is_tensor(value) for value in payload.values()):
            return dict(payload)
    raise ValueError(f"{path} does not contain a recognized PyTorch tensor state dict")


def _load_pickle_state(path: Path) -> dict[str, torch.Tensor]:
    return _extract_pickle_state(torch.load(path, map_location="cpu", weights_only=False), path)


def load_source_state(path: str | Path, source_kind: SourceKind) -> dict[str, torch.Tensor]:
    path = Path(path)
    if source_kind == "libero-pth":
        return _load_pickle_state(path)
    if source_kind == "robotwin-ema-safetensors":
        state = load_safetensors(path)
        ema_prefix = "ema."
        ema_state = {
            key.removeprefix(ema_prefix): value for key, value in state.items() if key.startswith(ema_prefix)
        }
        return ema_state or state
    raise ValueError(f"Unsupported TurboVLA source kind: {source_kind}")


def load_source_state_and_model_config(
    path: str | Path, source_kind: SourceKind
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any] | None]:
    path = Path(path)
    if source_kind != "libero-pth":
        return load_source_state(path, source_kind), None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_config = payload.get("model_config") if isinstance(payload, Mapping) else None
    if source_config is not None and not isinstance(source_config, Mapping):
        raise ValueError(f"{path} contains an invalid TurboVLA model_config")
    return _extract_pickle_state(payload, path), source_config


def _strip_known_wrappers(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "policy.",
        "net.",
        "network.",
        "student.",
        "ema.",
        "_orig_mod.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _load_key_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with Path(path).open() as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
    ):
        raise ValueError("TurboVLA key map must be a JSON object mapping source keys to target keys")
    return dict(mapping)


def map_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    key_map: Mapping[str, str] | None = None,
    ignore_patterns: tuple[str, ...] = (),
    strict: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, str], list[str]]:
    """Map upstream tensors into LeRobot names and validate strict coverage."""
    key_map = key_map or {}
    target_keys = set(target_state)
    mapped: dict[str, torch.Tensor] = {}
    mapped_names: dict[str, str] = {}
    ignored: list[str] = []
    unexpected: list[str] = []

    for source_key, tensor in source_state.items():
        if any(fnmatch.fnmatch(source_key, pattern) for pattern in ignore_patterns):
            ignored.append(source_key)
            continue
        target_key = key_map.get(source_key, key_map.get(_strip_known_wrappers(source_key)))
        if target_key is None:
            stripped = _strip_known_wrappers(source_key)
            if stripped in target_keys:
                target_key = stripped
            elif f"model.{stripped}" in target_keys:
                target_key = f"model.{stripped}"
            elif stripped.startswith("vision_encoder.backbone.layer."):
                dinov3_target = (
                    f"model.vision_encoder.backbone.model.{stripped.removeprefix('vision_encoder.backbone.')}"
                )
                if dinov3_target in target_keys:
                    target_key = dinov3_target
                else:
                    unexpected.append(source_key)
                    continue
            else:
                unexpected.append(source_key)
                continue
        if target_key not in target_state:
            raise ValueError(
                f"Mapped TurboVLA key {source_key!r} -> {target_key!r}, but target key is absent"
            )
        if tuple(tensor.shape) != tuple(target_state[target_key].shape):
            raise ValueError(
                f"TurboVLA key {source_key!r} maps to {target_key!r} with shape {tuple(tensor.shape)}, "
                f"expected {tuple(target_state[target_key].shape)}"
            )
        mapped[target_key] = tensor.detach().cpu().to(dtype=target_state[target_key].dtype).contiguous()
        mapped_names[source_key] = target_key

    missing = sorted(key for key in target_keys if key not in mapped)
    if strict and (unexpected or missing):
        details = []
        if unexpected:
            details.append(f"unexpected source keys: {unexpected[:20]}")
        if missing:
            details.append(f"missing target keys: {missing[:20]}")
        raise ValueError("Strict TurboVLA conversion failed with " + "; ".join(details))
    return mapped, mapped_names, sorted(ignored)


def _default_stats(config: TurboVLAConfig) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "mean": torch.zeros(config.state_dim),
            "std": torch.ones(config.state_dim),
        },
        ACTION: {
            "mean": torch.zeros(config.action_dim),
            "std": torch.ones(config.action_dim),
            "min": -torch.ones(config.action_dim),
            "max": torch.ones(config.action_dim),
        },
    }


def _load_stats(path: str | Path | None, config: TurboVLAConfig) -> dict[str, dict[str, torch.Tensor]]:
    if path is None:
        return _default_stats(config)
    with Path(path).open() as f:
        raw_stats = json.load(f)

    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.float32)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    # The file may contain a top-level dataset wrapper (e.g. {"libero_all4_no_noops": {...}, "metadata": {...}}).
    # Extract the first dataset entry and skip non-dataset keys like "metadata".
    if isinstance(raw_stats, dict):
        dataset_keys = [k for k in raw_stats if k != "metadata"]
        if len(dataset_keys) == 1:
            raw_stats = raw_stats[dataset_keys[0]]
        elif len(dataset_keys) > 1:
            raise ValueError(f"Multiple dataset entries in stats file: {dataset_keys}. "
                             "Pass the specific dataset stats file or update the loader.")

    return convert(raw_stats)


def _write_model_card(
    output_dir: Path,
    *,
    variant: Variant,
    source_sha256: str,
    source_kind: SourceKind,
    source_revision: str | None,
    source_url: str | None,
    checkpoint_license: str,
) -> None:
    pending_lines = "\n".join(
        f"- {key}: " for key in TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES
    )
    content = f"""---
library_name: lerobot
tags:
- lerobot
- turbovla
- robotics
license: {checkpoint_license}
---

# TurboVLA Converted Checkpoint

This checkpoint was converted for LeRobot's experimental TurboVLA policy.

- Variant: `{variant}`
- Source kind: `{source_kind}`
- Source URL: `{source_url or ""}`
- Source revision: `{source_revision or ""}`
- Source SHA256: `{source_sha256}`
- Checkpoint license: `{checkpoint_license}`

## Benchmark Status

Full LIBERO benchmark reproduction has not been run for this converted artifact yet.
The following values must remain blank until measured on the documented GPU setup:

{pending_lines}

The official TurboVLA weights are simulation-trained. This converted artifact does not imply real-robot
safety, robustness, or benchmark parity unless accompanied by an independently generated evaluation report.
Check the original checkpoint license before redistributing derived weights.
"""
    (output_dir / "README.md").write_text(content)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_config(
    variant: Variant,
    *,
    device: str = "cpu",
    precision: str = "float32",
    source_model_config: Mapping[str, Any] | None = None,
) -> TurboVLAConfig:
    text_config = source_model_config.get("text", {}) if source_model_config else {}
    interaction_config = source_model_config.get("interaction", {}) if source_model_config else {}
    if not isinstance(text_config, Mapping):
        raise ValueError("TurboVLA source model_config.text must be an object")
    if not isinstance(interaction_config, Mapping):
        raise ValueError("TurboVLA source model_config.interaction must be an object")
    source_text_kwargs: dict[str, Any] = {}
    for source_name, target_name in (
        ("model_name_or_path", "language_encoder_id"),
        ("max_length", "max_text_length"),
        ("padding_length", "text_padding_length"),
        ("padding_length_by_instruction", "text_padding_length_by_instruction"),
    ):
        if source_name in text_config:
            source_text_kwargs[target_name] = text_config[source_name]
    if "attention_backend" in interaction_config:
        source_text_kwargs["attention_implementation"] = interaction_config["attention_backend"]
    if variant == "libero":
        return TurboVLAConfig.libero(device=device, precision=precision, **source_text_kwargs)
    if variant == "robotwin":
        return TurboVLAConfig.robotwin(device=device, precision=precision, **source_text_kwargs)
    raise ValueError(f"Unsupported TurboVLA variant: {variant}")


def convert_checkpoint(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    variant: Variant,
    source_kind: SourceKind,
    source_revision: str | None = None,
    source_url: str | None = None,
    checkpoint_license: str = "other",
    expected_sha256: str | None = None,
    key_map_path: str | Path | None = None,
    ignore_patterns: tuple[str, ...] = (),
    stats_path: str | Path | None = None,
    strict: bool = True,
) -> ConversionReport:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    actual_sha256 = sha256_file(source_path)
    if expected_sha256 is not None and expected_sha256.lower() != actual_sha256:
        raise ValueError(
            f"TurboVLA source hash mismatch for {source_path}: expected {expected_sha256}, got {actual_sha256}"
        )

    source_state, source_model_config = load_source_state_and_model_config(source_path, source_kind)
    config = build_config(variant, source_model_config=source_model_config)
    policy = TurboVLAPolicy(config)
    target_state = policy.state_dict()
    converted_state, mapped_names, ignored_keys = map_state_dict(
        source_state,
        target_state,
        key_map=_load_key_map(key_map_path),
        ignore_patterns=ignore_patterns,
        strict=strict,
    )
    target_state.update(converted_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    save_safetensors(target_state, output_dir / SAFETENSORS_SINGLE_FILE)

    stats = _load_stats(stats_path, config)
    preprocessor, postprocessor = make_turbovla_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    _write_model_card(
        output_dir,
        variant=variant,
        source_sha256=actual_sha256,
        source_kind=source_kind,
        source_revision=source_revision,
        source_url=source_url,
        checkpoint_license=checkpoint_license,
    )

    tensor_records = [
        TensorRecord(
            key=key,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            sha256=sha256_tensor(tensor),
        )
        for key, tensor in sorted(target_state.items())
    ]
    report = ConversionReport(
        conversion_version=CONVERSION_VERSION,
        source_path=str(source_path),
        source_revision=source_revision,
        source_url=source_url,
        source_kind=source_kind,
        source_sha256=actual_sha256,
        checkpoint_license=checkpoint_license,
        variant=variant,
        output_dir=str(output_dir),
        loaded_keys=sorted(source_state),
        ignored_keys=ignored_keys,
        mapped_keys=dict(sorted(mapped_names.items())),
        tensor_records=tensor_records,
    )
    _write_json(output_dir / REPORT_FILENAME, asdict(report))
    _write_json(
        output_dir / PROVENANCE_FILENAME,
        {
            "conversion_version": CONVERSION_VERSION,
            "config_schema_version": config.config_schema_version,
            "processor_schema_version": config.processor_schema_version,
            "source_path": str(source_path),
            "source_url": source_url,
            "source_revision": source_revision,
            "source_kind": source_kind,
            "source_sha256": actual_sha256,
            "checkpoint_license": checkpoint_license,
            "variant": variant,
            "statistics_id": config.statistics_id,
            "statistics_revision": config.statistics_revision,
            "vision_encoder_id": config.vision_encoder_id,
            "vision_encoder_revision": config.vision_encoder_revision,
            "language_encoder_id": config.language_encoder_id,
            "language_encoder_revision": config.language_encoder_revision,
            "benchmark_status": "pending",
            "pending_benchmark_values": TURBOVLA_LIBERO_PENDING_BENCHMARK_VALUES,
        },
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert upstream TurboVLA checkpoints to LeRobot format.")
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("libero", "robotwin"))
    parser.add_argument("--source-kind", required=True, choices=("libero-pth", "robotwin-ema-safetensors"))
    parser.add_argument("--source-revision", help="Immutable upstream repository revision for provenance.")
    parser.add_argument("--source-url", help="Original checkpoint URL or release page for provenance.")
    parser.add_argument("--checkpoint-license", default="other", help="License identifier for converted weights.")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--key-map", type=Path, help="JSON object mapping upstream keys to LeRobot keys.")
    parser.add_argument(
        "--ignore-key",
        action="append",
        default=[],
        help="Glob pattern for source keys that are intentionally ignored. May be repeated.",
    )
    parser.add_argument(
        "--stats-json", type=Path, help="Dataset statistics JSON used to serialize processors."
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="Write only mapped tensors; reports gaps."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = convert_checkpoint(
        source_path=args.source_path,
        output_dir=args.output_dir,
        variant=args.variant,
        source_kind=args.source_kind,
        source_revision=args.source_revision,
        source_url=args.source_url,
        checkpoint_license=args.checkpoint_license,
        expected_sha256=args.expected_sha256,
        key_map_path=args.key_map,
        ignore_patterns=tuple(args.ignore_key),
        stats_path=args.stats_json,
        strict=not args.allow_partial,
    )
    print(f"Converted TurboVLA checkpoint to {report.output_dir}")
    print(f"Source SHA256: {report.source_sha256}")
    print(f"Mapped tensors: {len(report.mapped_keys)}")


if __name__ == "__main__":
    main()
