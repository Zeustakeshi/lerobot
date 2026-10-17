"""Reproducible offline caches for TIPS-Ground-VLA representation teachers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset
from tqdm import tqdm

from .configuration_tips_ground_vla import TIPSGroundVLAConfig
from .teacher_supervision import NEIGHBOR_OFFSETS, local_relations, semantic_distribution

CACHE_VERSION = 3


def _current_image(value: torch.Tensor) -> torch.Tensor:
    return value[-1] if value.ndim == 4 else value


def _images(sample: dict[str, Any], keys: tuple[str, ...]) -> torch.Tensor:
    images = []
    for key in keys[:2]:
        image = _current_image(sample[key]).float()
        image = image.div(255) if image.max() > 1.5 else image
        images.append(F.interpolate(image[None], (224, 336), mode="bilinear", align_corners=False)[0])
    while len(images) < 2:
        images.append(torch.zeros(3, 224, 336))
    return torch.stack(images)


def _root(config: TIPSGroundVLAConfig, dataset: Dataset) -> Path:
    if config.teacher_cache_dir:
        return Path(config.teacher_cache_dir).expanduser().resolve()
    return Path(dataset.root) / "teacher_targets" / "tips_ground_vla_v4"


def _manifest(config: TIPSGroundVLAConfig, frames: int) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "frames": frames,
        "image_range": [0.0, 1.0],
        "image_size": [224, 336],
        "student_grid": [14, 21],
        "tips_teacher": config.tips_teacher_path or config.tips_teacher_id,
        "tips_temperature": config.tips_temperature,
        "tips_text_target": "encode_text_sentence_embedding_768",
        "dino_teacher": config.dino_teacher_path or config.dino_teacher_id,
        "dino_processor_rescale": False,
        "dino_dtype": "bfloat16",
        "dino_target": "eight_local_cosine_relations",
        "neighbor_offsets": [list(offset) for offset in NEIGHBOR_OFFSETS],
    }


def _is_complete(root: Path, expected: dict[str, Any], config: TIPSGroundVLAConfig) -> bool:
    path = root / "manifest.json"
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    required = []
    if config.lambda_tips:
        required.append("tips_teacher_semantic.npy")
    if config.lambda_tips_text:
        required.append("tips_teacher_text.npy")
    if config.lambda_dino:
        required.append("dino_teacher_local_relations.npy")
    return saved == {**expected, "complete": True} and all((root / name).exists() for name in required)


def _generate(dataset: Dataset, config: TIPSGroundVLAConfig, root: Path) -> None:
    from transformers import AutoImageProcessor, AutoModel

    root.mkdir(parents=True, exist_ok=True)
    frames, device = len(dataset), torch.device(config.device)
    tips_array = (
        np.lib.format.open_memmap(
            root / "tips_teacher_semantic.npy", mode="w+", dtype=np.float16, shape=(frames, 588)
        )
        if config.lambda_tips
        else None
    )
    text_array = (
        np.lib.format.open_memmap(
            root / "tips_teacher_text.npy", mode="w+", dtype=np.float16, shape=(frames, 768)
        )
        if config.lambda_tips_text
        else None
    )

    tips_source = config.tips_teacher_path or config.tips_teacher_id
    tips = (
        AutoModel.from_pretrained(
            tips_source,
            cache_dir=config.hub_cache_dir,
            local_files_only=config.local_files_only,
            trust_remote_code=True,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    with torch.inference_mode():
        for index in tqdm(range(frames), desc="TIPS semantic/text targets"):
            sample = dataset[index]
            instruction = str(sample.get("task", sample.get("observation.language", "")))
            text = tips.encode_text([instruction]).float()[0]
            if not torch.isfinite(text).all():
                raise RuntimeError("TIPS teacher emitted non-finite text features")
            if text_array is not None:
                text_array[index] = text.cpu().half().numpy()
            if tips_array is not None:
                images = _images(sample, config.image_keys).to(
                    device=device, dtype=next(tips.parameters()).dtype
                )
                patches = F.normalize(tips.encode_image(images).patch_tokens.float(), dim=-1)
                similarity = torch.einsum("cnd,d->cn", patches, F.normalize(text, dim=-1))
                similarity = F.interpolate(
                    similarity[:, None].view(2, 1, 16, 24), (14, 21), mode="bilinear", align_corners=False
                ).flatten()
                target = semantic_distribution(
                    similarity[None],
                    torch.ones_like(similarity[None], dtype=torch.bool),
                    config.tips_temperature,
                )
                tips_array[index] = target[0].cpu().half().numpy()
    del tips
    torch.cuda.empty_cache()

    if tips_array is not None:
        tips_array.flush()
    if text_array is not None:
        text_array.flush()
    if not config.lambda_dino:
        (root / "manifest.json").write_text(
            json.dumps({**_manifest(config, frames), "complete": True}, indent=2)
        )
        return

    dino_array = np.lib.format.open_memmap(
        root / "dino_teacher_local_relations.npy",
        mode="w+",
        dtype=np.float16,
        shape=(frames, 2, 294, len(NEIGHBOR_OFFSETS)),
    )
    dino_source = config.dino_teacher_path or config.dino_teacher_id
    processor = AutoImageProcessor.from_pretrained(
        dino_source, cache_dir=config.hub_cache_dir, local_files_only=config.local_files_only
    )
    dino = (
        AutoModel.from_pretrained(
            dino_source,
            cache_dir=config.hub_cache_dir,
            local_files_only=config.local_files_only,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    with torch.inference_mode():
        for index in tqdm(range(frames), desc="DINO finite local relations"):
            images = _images(dataset[index], config.image_keys)
            pixels = processor(
                images=list(images),
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_rescale=False,
            )["pixel_values"].to(device=device, dtype=next(dino.parameters()).dtype)
            tokens = dino(pixel_values=pixels, interpolate_pos_encoding=True).last_hidden_state[:, -294:]
            if not torch.isfinite(tokens).all():
                raise RuntimeError("DINO teacher emitted non-finite features; FP16 caches are invalid")
            relations, _ = local_relations(tokens[None].float())
            dino_array[index] = relations[0].cpu().half().numpy()
    dino_array.flush()
    (root / "manifest.json").write_text(json.dumps({**_manifest(config, frames), "complete": True}, indent=2))


class TeacherCacheDataset(Dataset):
    def __init__(self, dataset: Dataset, root: Path) -> None:
        self.dataset, self.root, self._arrays = dataset, root, {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def _open(self, name: str):
        if name not in self._arrays:
            path = self.root / f"{name}.npy"
            if path.exists():
                self._arrays[name] = np.load(path, mmap_mode="r")
        return self._arrays.get(name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        for name in ("tips_teacher_semantic", "tips_teacher_text", "dino_teacher_local_relations"):
            array = self._open(name)
            if array is not None:
                sample[name] = torch.from_numpy(np.array(array[index]))
        return sample

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self[index] for index in indices]


def prepare_teacher_cache(dataset: Dataset, config: TIPSGroundVLAConfig) -> TeacherCacheDataset:
    """Generate or validate a cache only when an auxiliary loss is enabled."""
    if not (config.lambda_tips or config.lambda_tips_text or config.lambda_dino):
        return TeacherCacheDataset(dataset, _root(config, dataset))
    root, expected = _root(config, dataset), _manifest(config, len(dataset))
    if not _is_complete(root, expected, config):
        if config.teacher_prepare in {"off", "require"}:
            raise RuntimeError(f"validated teacher cache required but missing or stale: {root}")
        _generate(dataset, config, root)
    return TeacherCacheDataset(dataset, root)
