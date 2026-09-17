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
    source_kind: SourceKind
    source_sha256: str
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


def _load_pickle_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "module", "ema", "ema_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and all(torch.is_tensor(value) for value in nested.values()):
                return dict(nested)
        if all(torch.is_tensor(value) for value in payload.values()):
            return dict(payload)
    raise ValueError(f"{path} does not contain a recognized PyTorch tensor state dict")


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
            else:
                unexpected.append(source_key)
                continue
        if target_key not in target_state:
            raise ValueError(f"Mapped TurboVLA key {source_key!r} -> {target_key!r}, but target key is absent")
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

    return convert(raw_stats)


def _write_model_card(output_dir: Path, *, variant: Variant, source_sha256: str, source_kind: SourceKind) -> None:
    content = f"""---
library_name: lerobot
tags:
- lerobot
- turbovla
- robotics
license: other
---

# TurboVLA Converted Checkpoint

This checkpoint was converted for LeRobot's experimental TurboVLA policy.

- Variant: `{variant}`
- Source kind: `{source_kind}`
- Source SHA256: `{source_sha256}`

The official TurboVLA weights are simulation-trained. This converted artifact does not imply real-robot
safety, robustness, or benchmark parity unless accompanied by an independently generated evaluation report.
Check the original checkpoint license before redistributing derived weights.
"""
    (output_dir / "README.md").write_text(content)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_config(variant: Variant, *, device: str = "cpu", precision: str = "float32") -> TurboVLAConfig:
    if variant == "libero":
        return TurboVLAConfig.libero(device=device, precision=precision)
    if variant == "robotwin":
        return TurboVLAConfig.robotwin(device=device, precision=precision)
    raise ValueError(f"Unsupported TurboVLA variant: {variant}")


def convert_checkpoint(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    variant: Variant,
    source_kind: SourceKind,
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

    config = build_config(variant)
    policy = TurboVLAPolicy(config)
    target_state = policy.state_dict()
    source_state = load_source_state(source_path, source_kind)
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
    _write_model_card(output_dir, variant=variant, source_sha256=actual_sha256, source_kind=source_kind)

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
        source_kind=source_kind,
        source_sha256=actual_sha256,
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
            "source_kind": source_kind,
            "source_sha256": actual_sha256,
            "variant": variant,
            "statistics_id": config.statistics_id,
            "statistics_revision": config.statistics_revision,
            "vision_encoder_id": config.vision_encoder_id,
            "vision_encoder_revision": config.vision_encoder_revision,
            "language_encoder_id": config.language_encoder_id,
            "language_encoder_revision": config.language_encoder_revision,
            "checkpoint_license": "see-source-checkpoint",
        },
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert upstream TurboVLA checkpoints to LeRobot format.")
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("libero", "robotwin"))
    parser.add_argument("--source-kind", required=True, choices=("libero-pth", "robotwin-ema-safetensors"))
    parser.add_argument("--expected-sha256")
    parser.add_argument("--key-map", type=Path, help="JSON object mapping upstream keys to LeRobot keys.")
    parser.add_argument(
        "--ignore-key",
        action="append",
        default=[],
        help="Glob pattern for source keys that are intentionally ignored. May be repeated.",
    )
    parser.add_argument("--stats-json", type=Path, help="Dataset statistics JSON used to serialize processors.")
    parser.add_argument("--allow-partial", action="store_true", help="Write only mapped tensors; reports gaps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = convert_checkpoint(
        source_path=args.source_path,
        output_dir=args.output_dir,
        variant=args.variant,
        source_kind=args.source_kind,
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
