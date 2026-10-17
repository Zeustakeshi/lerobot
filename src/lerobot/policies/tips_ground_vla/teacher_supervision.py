"""Small, testable contracts for TIPSv2 and DINOv3 supervision.

The functions in this module contain no teacher models and are safe to import
at inference time.  Frozen teachers and cache generation live outside the
deployable policy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

EPS = 1e-8
NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def text_distillation_loss(
    student: Tensor, teacher: Tensor, relational_weight: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Match sentence features while preserving teacher semantic geometry."""
    if student.ndim != 2 or teacher.ndim != 2 or student.shape != teacher.shape:
        raise ValueError("student and teacher text features must have matching [B,D] shapes")
    if student.shape[0] < 1 or relational_weight < 0:
        raise ValueError("text distillation needs a non-empty batch and non-negative relational weight")
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    pointwise = (1 - (student * teacher).sum(-1)).mean()
    if student.shape[0] == 1:
        relational = pointwise.new_zeros(())
    else:
        off_diagonal = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
        relational = F.smooth_l1_loss(
            (student @ student.T)[off_diagonal],
            (teacher @ teacher.T)[off_diagonal],
            beta=0.05,
        )
    return pointwise + relational_weight * relational, pointwise, relational


def semantic_distribution(similarity: Tensor, valid: Tensor, temperature: float) -> Tensor:
    """Turn calibrated patch/text similarities into a masked distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if similarity.shape != valid.shape:
        raise ValueError("similarity and valid must have identical shapes")
    logits = similarity.float() / temperature
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    distribution = logits.softmax(-1).masked_fill(~valid, 0)
    return distribution / distribution.sum(-1, keepdim=True).clamp_min(EPS)


def semantic_kl(student: Tensor, teacher: Tensor, valid: Tensor) -> Tensor:
    """KL on valid patches, with no per-map min/max normalization."""
    if student.shape != teacher.shape or student.shape != valid.shape:
        raise ValueError("student, teacher and valid must have identical shapes")
    teacher = teacher.float().masked_fill(~valid, 0)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(EPS)
    student = student.float().masked_fill(~valid, 0)
    student = student / student.sum(-1, keepdim=True).clamp_min(EPS)
    return (teacher * (teacher.clamp_min(EPS).log() - student.clamp_min(EPS).log())).sum(-1).mean()


def local_relations(tokens: Tensor, height: int = 14, width: int = 21) -> tuple[Tensor, Tensor]:
    """Cosine relations to eight spatial neighbours without wrap-around.

    Returns ``[B,C,H*W,8]`` relations and a matching validity mask.  This
    preserves exact student/teacher coordinates while avoiding the large,
    mostly redundant all-pairs relation matrix.
    """
    if tokens.ndim != 4 or tokens.shape[2] != height * width:
        raise ValueError(f"tokens must be [B,C,{height * width},D]")
    grid = F.normalize(tokens.float(), dim=-1).view(*tokens.shape[:2], height, width, tokens.shape[-1])
    relations, masks = [], []
    base = torch.ones((*tokens.shape[:2], height, width), dtype=torch.bool, device=tokens.device)
    for dy, dx in NEIGHBOR_OFFSETS:
        shifted = torch.roll(grid, shifts=(dy, dx), dims=(2, 3))
        mask = base.clone()
        if dy < 0:
            mask[:, :, height + dy :] = False
        elif dy > 0:
            mask[:, :, :dy] = False
        if dx < 0:
            mask[:, :, :, width + dx :] = False
        elif dx > 0:
            mask[:, :, :, :dx] = False
        relations.append((grid * shifted).sum(-1))
        masks.append(mask)
    return torch.stack(relations, -1).flatten(2, 3), torch.stack(masks, -1).flatten(2, 3)


def spatial_relation_loss(student_tokens: Tensor, teacher_relations: Tensor, patch_valid: Tensor) -> Tensor:
    """Masked local-relation regression used by the DINOv3 auxiliary path."""
    student, neighbor_valid = local_relations(student_tokens)
    if teacher_relations.shape != student.shape:
        raise ValueError(f"teacher relations must have shape {tuple(student.shape)}")
    valid = neighbor_valid & patch_valid.bool().flatten(2, 3)[..., None]
    error = F.smooth_l1_loss(student, teacher_relations.float(), beta=0.05, reduction="none")
    return (error * valid).sum() / valid.sum().clamp_min(1)
