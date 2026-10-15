"""Offline teacher preparation integrated with ``lerobot-train`` for TIPS-Ground-VLA v3.

The cache is a sidecar next to the LeRobot dataset.  It never mutates the source
dataset and no teacher object is attached to the deployable policy.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .configuration_tips_ground_vla import TIPSGroundVLAConfig

CACHE_VERSION = 1
PATCHES = 14 * 21
TEACHER_KEYS = (
    "tips_teacher_semantic",
    "tips_teacher_reliability",
    "dino_teacher_relations",
    "region_box",
    "region_valid",
    "region_camera_valid",
    "phrase_region_target",
    "phrase_region_valid",
    "phrase_spans",
    "phrase_valid",
)


def _stage(number: int, total: int, title: str, detail: str = "") -> None:
    line = f"  TIPS-Ground-VLA · Stage {number}/{total}  {title}"
    print(f"\n\033[1;36m{'━' * 78}\033[0m\n\033[1;37m{line}\033[0m")
    if detail:
        print(f"  \033[2m{detail}\033[0m")


def _phrases(instruction: str) -> list[str]:
    """Conservative, role-agnostic referring-phrase formatter.

    Phrase extraction is data preparation rather than a fourth student head.
    The full instruction remains the fallback when no clean concept is found.
    """
    text = re.sub(r"[^\w\s-]", " ", instruction.lower()).strip()
    parts = re.split(r"\b(?:into|inside|onto|towards?|to|in|on|from|with|using)\b", text)
    leading = re.compile(
        r"^(?:please\s+)?(?:pick\s+up|put|place|move|push|pull|open|close|press|rotate|wipe|pour|insert|stack|grasp|lift)\s+"
    )
    result = []
    for part in parts:
        part = leading.sub("", part).strip()
        part = re.sub(r"^(?:the|a|an)\s+", "", part).strip()
        if part and part not in result:
            result.append(part)
    return (result or [text])[:6]


def _current_image(value: Tensor) -> Tensor:
    return value[-1] if value.ndim == 4 else value


def _images(sample: dict[str, Any], keys: tuple[str, ...]) -> Tensor:
    images = []
    for key in keys[:2]:
        image = _current_image(sample[key]).float()
        image = image.div(255) if image.max() > 1.5 else image
        images.append(F.interpolate(image[None], (224, 336), mode="bilinear", align_corners=False)[0])
    while len(images) < 2:
        images.append(torch.zeros(3, 224, 336))
    return torch.stack(images)


def _cache_root(config: TIPSGroundVLAConfig, dataset: Dataset) -> Path:
    if config.teacher_cache_dir:
        return Path(config.teacher_cache_dir).expanduser().resolve()
    root = Path(dataset.root)
    return root / "teacher_targets" / "tips_ground_vla_v3"


def _manifest(config: TIPSGroundVLAConfig, length: int) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "frames": length,
        "tips_teacher": config.tips_teacher_id,
        "dino_teacher": config.dino_teacher_id,
        "grounding_teacher": config.grounding_dino_id,
        "student_grid": [14, 21],
        "image_size": [224, 336],
        "max_phrases": 6,
    }


def _complete(cache: Path, expected: dict[str, Any]) -> bool:
    path = cache / "manifest.json"
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    return all(saved.get(key) == value for key, value in expected.items()) and saved.get("complete") is True


def _targets_ready(cache: Path, expected: dict[str, Any]) -> bool:
    path = cache / "targets.json"
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    return all(saved.get(key) == value for key, value in expected.items()) and all(
        (cache / f"{key}.npy").exists() for key in TEACHER_KEYS
    )


def _download(config: TIPSGroundVLAConfig) -> None:
    import huggingface_hub.constants as hub_constants
    from huggingface_hub import snapshot_download

    models = (
        ("TIPSv2-B/14", config.tips_teacher_path or config.tips_teacher_id),
        ("DINOv3 ViT-L/16", config.dino_teacher_path or config.dino_teacher_id),
        ("Grounding DINO", config.grounding_dino_id),
        ("MiniLM tokenizer", config.text_encoder_id),
    )
    for label, source in tqdm(models, desc="Models", unit="model", colour="cyan"):
        if Path(source).exists():
            continue
        try:
            snapshot_download(source, local_files_only=config.local_files_only)
        except RuntimeError as error:
            # Some hf-xet builds leave an invalid resumable hash in the local
            # cache. Retry through the regular HTTP backend transparently so a
            # user does not have to know about Hugging Face transport details.
            if config.local_files_only or "hex hash" not in str(error):
                raise
            tqdm.write("  ! Xet resume metadata is invalid; retrying with the HTTP downloader")
            hub_constants.HF_HUB_DISABLE_XET = True
            snapshot_download(source, local_files_only=False)
        tqdm.write(f"  ✓ {label}: {source}")


def _token_spans(tokenizer: Any, instruction: str, phrases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(
        instruction,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    spans = np.zeros((6, 64), dtype=np.bool_)
    valid = np.zeros(6, dtype=np.bool_)
    lower = instruction.lower()
    cursor = 0
    for phrase_index, phrase in enumerate(phrases):
        start = lower.find(phrase.lower(), cursor)
        if start < 0:
            start = lower.find(phrase.lower())
        if start < 0:
            continue
        end, cursor = start + len(phrase), start + len(phrase)
        for token_index, (token_start, token_end) in enumerate(offsets[:64]):
            if token_end > start and token_start < end:
                spans[phrase_index, token_index] = True
        valid[phrase_index] = spans[phrase_index].any()
    if not valid.any():
        attention = np.asarray(encoded["attention_mask"], dtype=np.bool_)
        spans[0] = attention
        valid[0] = True
    return spans, valid


def _open_arrays(cache: Path, frames: int, mode: str) -> dict[str, np.memmap]:
    specs = {
        "tips_teacher_semantic": (np.float16, (frames, 6, 588)),
        "tips_teacher_reliability": (np.float16, (frames, 6)),
        "dino_teacher_relations": (np.float16, (frames, 2, PATCHES, PATCHES)),
        "region_box": (np.float16, (frames, 8, 2, 4)),
        "region_valid": (np.float16, (frames, 8)),
        "region_camera_valid": (np.bool_, (frames, 8, 2)),
        "phrase_region_target": (np.bool_, (frames, 6, 2, 14, 21)),
        "phrase_region_valid": (np.bool_, (frames, 6)),
        "phrase_spans": (np.bool_, (frames, 6, 64)),
        "phrase_valid": (np.bool_, (frames, 6)),
    }
    arrays = {}
    for key, (dtype, shape) in specs.items():
        path = cache / f"{key}.npy"
        arrays[key] = np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
    return arrays


def _tips_pass(dataset: Dataset, config: TIPSGroundVLAConfig, arrays: dict[str, np.memmap]) -> None:
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(config.device)
    model = (
        AutoModel.from_pretrained(
            config.tips_teacher_path or config.tips_teacher_id,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.text_encoder_id, local_files_only=config.local_files_only, use_fast=True
    )
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="TIPS semantic maps", unit="frame", colour="magenta"):
            sample = dataset[index]
            instruction = str(sample.get("task", ""))
            phrases = _phrases(instruction)
            spans, phrase_valid = _token_spans(tokenizer, instruction, phrases)
            arrays["phrase_spans"][index] = spans
            arrays["phrase_valid"][index] = phrase_valid
            images = _images(sample, config.image_keys).to(
                device=device, dtype=next(model.parameters()).dtype
            )
            patches = model.encode_image(images).patch_tokens
            if patches.shape[1] != 384:
                raise RuntimeError(f"TIPSv2 must produce 16x24=384 patches, got {patches.shape}")
            text = model.encode_text(phrases).to(patches)
            logits = torch.einsum("cnd,pd->pcn", F.normalize(patches, dim=-1), F.normalize(text, dim=-1))
            logits = F.interpolate(logits.view(-1, 1, 16, 24), (14, 21), mode="bilinear", align_corners=False)
            distribution = (logits.view(len(phrases), 2 * PATCHES) / 0.07).softmax(-1)
            arrays["tips_teacher_semantic"][index] = 0
            arrays["tips_teacher_semantic"][index, : len(phrases)] = distribution.cpu().half().numpy()
            # Similarity maps remain soft supervision. Reliability is finalized by audit.
            entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1) / np.log(2 * PATCHES)
            arrays["tips_teacher_reliability"][index] = 0
            arrays["tips_teacher_reliability"][index, : len(phrases)] = (1 - entropy).cpu().half().numpy()
    del model
    torch.cuda.empty_cache()


def _dino_pass(dataset: Dataset, config: TIPSGroundVLAConfig, arrays: dict[str, np.memmap]) -> None:
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device(config.device)
    source = config.dino_teacher_path or config.dino_teacher_id
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=config.local_files_only)
    model = AutoModel.from_pretrained(source, local_files_only=config.local_files_only).to(device).eval()
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="DINO spatial relations", unit="frame", colour="blue"):
            images = _images(dataset[index], config.image_keys)
            inputs = processor(
                images=list(images), return_tensors="pt", do_resize=False, do_center_crop=False
            )
            pixels = inputs["pixel_values"].to(device)
            output = model(pixel_values=pixels, interpolate_pos_encoding=True).last_hidden_state
            patches = output[:, -PATCHES:]
            if patches.shape[1] != PATCHES:
                raise RuntimeError(f"DINOv3 must produce 294 spatial patches, got {patches.shape}")
            normalized = F.normalize(patches.float(), dim=-1)
            arrays["dino_teacher_relations"][index] = (
                (normalized @ normalized.transpose(-1, -2)).cpu().half().numpy()
            )
    del model
    torch.cuda.empty_cache()


def _box_to_grid(box: Tensor) -> tuple[slice, slice]:
    x0, y0, x1, y1 = box.tolist()
    left, right = max(0, int(x0 / 336 * 21)), min(21, max(1, int(np.ceil(x1 / 336 * 21))))
    top, bottom = max(0, int(y0 / 224 * 14)), min(14, max(1, int(np.ceil(y1 / 224 * 14))))
    return slice(top, bottom), slice(left, right)


def _grounding_pass(dataset: Dataset, config: TIPSGroundVLAConfig, arrays: dict[str, np.memmap]) -> None:
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    device = torch.device(config.device)
    processor = AutoProcessor.from_pretrained(
        config.grounding_dino_id, local_files_only=config.local_files_only
    )
    model = (
        GroundingDinoForObjectDetection.from_pretrained(
            config.grounding_dino_id, local_files_only=config.local_files_only
        )
        .to(device)
        .eval()
    )
    with torch.inference_mode():
        for index in tqdm(
            range(len(dataset)), desc="Grounding DINO candidates", unit="frame", colour="green"
        ):
            sample = dataset[index]
            phrases = _phrases(str(sample.get("task", "")))
            images = _images(sample, config.image_keys)
            arrays["region_box"][index] = 0
            arrays["region_valid"][index] = 0
            arrays["region_camera_valid"][index] = False
            arrays["phrase_region_target"][index] = False
            arrays["phrase_region_valid"][index] = False
            candidates: list[tuple[float, int, int, Tensor]] = []
            for camera, image in enumerate(images):
                inputs = processor(images=image, text=[". ".join(phrases) + "."], return_tensors="pt").to(
                    device
                )
                outputs = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=config.grounding_dino_box_threshold,
                    text_threshold=config.grounding_dino_text_threshold,
                    target_sizes=[(224, 336)],
                )[0]
                per_phrase = dict.fromkeys(range(len(phrases)), 0)
                for score, label, box in zip(
                    result["scores"], result["text_labels"], result["boxes"], strict=True
                ):
                    phrase_index = max(
                        range(len(phrases)),
                        key=lambda i: len(set(label.lower().split()) & set(phrases[i].split())),
                    )
                    if per_phrase[phrase_index] >= 5:
                        continue
                    per_phrase[phrase_index] += 1
                    candidates.append((float(score), phrase_index, camera, box.detach().cpu()))
            for region_index, (score, phrase_index, camera, box) in enumerate(
                sorted(candidates, reverse=True)[:8]
            ):
                arrays["region_box"][index, region_index, camera] = box.numpy() / np.array(
                    [336, 224, 336, 224]
                )
                arrays["region_valid"][index, region_index] = score
                arrays["region_camera_valid"][index, region_index, camera] = True
                ys, xs = _box_to_grid(box)
                arrays["phrase_region_target"][index, phrase_index, camera, ys, xs] = True
                arrays["phrase_region_valid"][index, phrase_index] = True
    del model
    torch.cuda.empty_cache()


def _audit_pack(
    dataset: Dataset, config: TIPSGroundVLAConfig, cache: Path, arrays: dict[str, np.memmap]
) -> Path:
    audit = cache / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(dataset) - 1, config.teacher_audit_frames, dtype=int)
    cards = []
    for ordinal, index in enumerate(tqdm(indices, desc="Audit contact sheet", unit="frame", colour="yellow")):
        sample = dataset[int(index)]
        image = (_images(sample, config.image_keys)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for region in range(8):
            if (
                arrays["region_valid"][index, region] <= 0
                or not arrays["region_camera_valid"][index, region, 0]
            ):
                continue
            box = arrays["region_box"][index, region, 0].astype(np.float32) * np.array([336, 224, 336, 224])
            draw.rectangle(tuple(box.tolist()), outline="lime", width=2)
        name = f"frame_{ordinal:03d}_index_{index}.jpg"
        canvas.save(audit / name, quality=90)
        cards.append(
            f'<figure><img src="{html.escape(name)}"><figcaption>frame {index}</figcaption></figure>'
        )
    (audit / "index.html").write_text(
        "<html><style>body{background:#10131a;color:#eef;font-family:sans-serif}main{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}img{width:100%}figure{margin:0;background:#1b2030;padding:8px}</style><h1>TIPS-Ground-VLA v3 · Teacher audit</h1><p>Review every accepted box, then fill audit.json.</p><main>"
        + "".join(cards)
        + "</main></html>"
    )
    audit_json = audit / "audit.json"
    if not audit_json.exists():
        audit_json.write_text(
            json.dumps({"approved": False, "accepted_precision": None, "reviewed_frames": 0}, indent=2)
        )
    return audit_json


class TeacherCacheDataset(Dataset):
    """Attach memory-mapped teacher tensors to ordinary LeRobot samples."""

    def __init__(self, dataset: Dataset, cache: Path) -> None:
        self.dataset, self.cache = dataset, cache
        self._arrays: dict[str, np.memmap] | None = None
        manifest = json.loads((cache / "manifest.json").read_text())
        self._audit_precision = float(manifest.get("audit", {}).get("accepted_precision", 0))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def _ensure_arrays(self) -> dict[str, np.memmap]:
        if self._arrays is None:
            self._arrays = _open_arrays(self.cache, len(self), "r")
        return self._arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        arrays = self._ensure_arrays()
        for key in TEACHER_KEYS:
            sample[key] = torch.from_numpy(np.array(arrays[key][index]))
        # The manually measured accepted-label precision is the calibrated
        # reliability; raw Grounding-DINO scores are only proposal/ranking signals.
        sample["region_valid"] = (sample["region_valid"] > 0).to(torch.float32) * self._audit_precision
        sample["tips_teacher_reliability"] *= self._audit_precision
        return sample

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self[index] for index in indices]


def prepare_teacher_cache(dataset: Dataset, config: TIPSGroundVLAConfig) -> TeacherCacheDataset:
    """Prepare or reuse all three v3 teachers and return a cache-backed dataset."""
    cache = _cache_root(config, dataset)
    expected = _manifest(config, len(dataset))
    _stage(1, 5, "CHECK CACHE", str(cache))
    if _complete(cache, expected):
        print("  \033[1;32m✓ Validated teacher cache found — reusing it.\033[0m")
        return TeacherCacheDataset(dataset, cache)
    if config.teacher_prepare == "off":
        raise RuntimeError("teacher_prepare=off but the required v3 teacher cache does not exist")
    if config.teacher_prepare == "require":
        raise RuntimeError(f"Required teacher cache is incomplete: {cache}")

    cache.mkdir(parents=True, exist_ok=True)
    _stage(2, 5, "DOWNLOAD TEACHERS", "Frozen models; downloaded once and reused from Hugging Face cache")
    _download(config)
    if _targets_ready(cache, expected):
        print("  \033[1;32m✓ Teacher tensors already generated — skipping inference.\033[0m")
        arrays = _open_arrays(cache, len(dataset), "r")
    else:
        arrays = _open_arrays(cache, len(dataset), "w+")
        _stage(
            3, 5, "GENERATE TARGETS", "TIPS semantics → DINO spatial relations → Grounding DINO candidates"
        )
        _tips_pass(dataset, config, arrays)
        _dino_pass(dataset, config, arrays)
        _grounding_pass(dataset, config, arrays)
        for array in arrays.values():
            array.flush()
        (cache / "targets.json").write_text(json.dumps(expected, indent=2))

    _stage(
        4,
        5,
        "VALIDATE 100 FRAMES",
        "Hard region supervision is blocked until accepted-label precision is ≥90%",
    )
    audit_path = _audit_pack(dataset, config, cache, arrays)
    audit = json.loads(audit_path.read_text())
    if config.teacher_require_validated_regions and not (
        audit.get("approved") is True
        and audit.get("reviewed_frames", 0) >= 100
        and float(audit.get("accepted_precision") or 0) >= 0.90
    ):
        raise RuntimeError(
            "Teacher cache generated, but the mandatory audit gate is not approved. "
            f"Open {cache / 'audit/index.html'}, review 100 frames, then update {audit_path} "
            "with approved=true, reviewed_frames=100 and accepted_precision>=0.90; rerun the same lerobot-train command."
        )

    expected.update(complete=True, audit=audit)
    (cache / "manifest.json").write_text(json.dumps(expected, indent=2))
    _stage(5, 5, "TRAIN STUDENT", "All offline targets validated; handing control back to lerobot-train")
    return TeacherCacheDataset(dataset, cache)
