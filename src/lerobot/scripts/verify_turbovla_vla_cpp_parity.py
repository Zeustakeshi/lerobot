#!/usr/bin/env python
"""Compare a native TurboVLA checkpoint with a vla.cpp server request.

The inputs are deliberately explicit: one ``--image POLICY_KEY=PATH`` for
each checkpoint image key, raw robot state in the checkpoint's feature order,
and the instruction.  It can therefore verify a new embodiment without
hard-coding camera names, motor IDs, or a robot driver.

Example:
    python -m lerobot.scripts.verify_turbovla_vla_cpp_parity \
      --checkpoint /path/to/pretrained_model \
      --image observation.images.wrist=/tmp/wrist.png \
      --image observation.images.top=/tmp/top.png \
      --state '0,-45,40,60,-80,20' --task 'Pick up the object'
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from safetensors import safe_open

from lerobot.configs import PreTrainedConfig
from lerobot.policies.turbovla.configuration_turbovla import TurboVLAConfig
from lerobot.policies.turbovla.modeling_turbovla import TurboVLAPolicy
from lerobot.policies.turbovla.vla_cpp import VlaCppTurboVLAClient
from lerobot.utils.constants import OBS_LANGUAGE, OBS_STATE


def _parse_images(values: list[str], keys: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path:
            raise ValueError("Every --image must have the form POLICY_KEY=PATH")
        if key in result:
            raise ValueError(f"Image key {key!r} was given more than once")
        result[key] = Path(path)
    missing = [key for key in keys if key not in result]
    extra = sorted(set(result) - set(keys))
    if missing or extra:
        raise ValueError(f"Image keys must exactly match checkpoint keys; missing={missing}, extra={extra}")
    for key, path in result.items():
        if not path.is_file():
            raise FileNotFoundError(f"Image for {key!r} does not exist: {path}")
    return result


def _load_rgb(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0)


def _resize_and_crop(image: torch.Tensor, config: TurboVLAConfig) -> torch.Tensor:
    if config.image_crop_size is not None:
        crop_h, crop_w = config.image_crop_size
        height, width = image.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError(f"Crop {config.image_crop_size} exceeds input image {height}x{width}")
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        image = image[..., top : top + crop_h, left : left + crop_w]
    if config.resize_images and tuple(image.shape[-2:]) != tuple(config.image_size):
        kwargs: dict[str, object] = {"size": config.image_size, "mode": config.image_interpolation}
        if config.image_interpolation != "nearest":
            kwargs["align_corners"] = False
        image = F.interpolate(image, **kwargs)
    if tuple(image.shape[-2:]) != tuple(config.image_size):
        raise ValueError(f"Expected {config.image_size} image after preprocessing, got {tuple(image.shape[-2:])}")
    return image.contiguous()


def _read_stats(checkpoint: Path, config: TurboVLAConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    files = sorted(
        path
        for path in checkpoint.glob("*.safetensors")
        if path.name.endswith("_normalizer_processor.safetensors")
    )
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one normalizer safetensors file under {checkpoint}")
    with safe_open(str(files[0]), framework="pt", device="cpu") as stats:
        state_mean = stats.get_tensor(f"{OBS_STATE}.mean").float()
        state_std = stats.get_tensor(f"{OBS_STATE}.std").float()
        action_min = stats.get_tensor("action.min").float()
        action_max = stats.get_tensor("action.max").float()
    expected = (config.state_dim,)
    if tuple(state_mean.shape) != expected or tuple(state_std.shape) != expected:
        raise ValueError(f"State statistics must have shape {expected}")
    if tuple(action_min.shape) != (config.action_dim,) or tuple(action_max.shape) != (config.action_dim,):
        raise ValueError(f"Action statistics must have shape ({config.action_dim},)")
    return state_mean, state_std, action_min, action_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True, default=[], metavar="KEY=PATH")
    parser.add_argument("--state", required=True, help="Raw state values, comma-separated, in checkpoint order")
    parser.add_argument("--task", required=True)
    parser.add_argument("--vla-cpp-addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=0.05, help="Maximum allowed raw-action error")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    loaded = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(loaded, TurboVLAConfig):
        raise TypeError(f"Expected a TurboVLA checkpoint, got {type(loaded).__name__}")
    config = replace(loaded, inference_backend="pytorch", device=args.device)
    image_paths = _parse_images(args.image, config.image_keys)
    raw_state = torch.tensor([float(value) for value in args.state.split(",")], dtype=torch.float32).unsqueeze(0)
    if tuple(raw_state.shape) != (1, config.state_dim):
        raise ValueError(f"--state must contain {config.state_dim} values")

    state_mean, state_std, action_min, action_max = _read_stats(checkpoint, config)
    raw_images = {key: _resize_and_crop(_load_rgb(image_paths[key]), config) for key in config.image_keys}

    native = TurboVLAPolicy.from_pretrained(checkpoint, config=config).to(args.device).eval()
    image_mean = torch.tensor(config.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    image_std = torch.tensor(config.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    native_batch: dict[str, object] = {
        key: ((raw_images[key] - image_mean) / image_std).to(args.device) for key in config.image_keys
    }
    native_batch[OBS_STATE] = ((raw_state - state_mean) / (state_std + 1e-8)).to(args.device)
    native_batch[OBS_LANGUAGE] = [args.task]
    with torch.inference_mode():
        native_normalized = native._predict(native_batch).cpu()
    native_actions = (native_normalized.clamp(-1, 1) + 1) * 0.5 * (action_max - action_min) + action_min

    server_config = replace(
        loaded,
        inference_backend="vla_cpp",
        vla_cpp_addr=args.vla_cpp_addr,
        vla_cpp_timeout_ms=args.timeout_ms,
    )
    client = VlaCppTurboVLAClient(server_config)
    try:
        server_batch: dict[str, object] = {key: raw_images[key] for key in config.image_keys}
        server_batch[OBS_STATE] = raw_state
        server_batch[OBS_LANGUAGE] = [args.task]
        server_actions = client.predict(server_batch).cpu()
    finally:
        client.close()

    delta = (native_actions - server_actions).abs()
    print(f"native shape: {tuple(native_actions.shape)}")
    print(f"server shape: {tuple(server_actions.shape)}")
    print(f"max_abs_error: {delta.max().item():.6f}")
    print(f"mean_abs_error: {delta.mean().item():.6f}")
    print("native first action:", native_actions[0, 0].tolist())
    print("server first action:", server_actions[0, 0].tolist())
    if delta.max().item() > args.atol:
        raise SystemExit(f"PARITY FAILED: max error exceeds --atol={args.atol}")
    print("PARITY PASSED")


if __name__ == "__main__":
    main()
