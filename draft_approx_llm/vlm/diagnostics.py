from __future__ import annotations

import math
from typing import Iterable

import torch


def percentile_rank_scores(
    scores: torch.Tensor, selected_indices: torch.Tensor
) -> torch.Tensor:
    """Return scale-free [0, 1] ranks on a selected token subset."""
    if scores.ndim != 1:
        raise ValueError(f"Expected one-dimensional scores, got {tuple(scores.shape)}")
    indices = selected_indices.to(scores.device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("selected_indices must be a non-empty one-dimensional tensor")
    values = scores[indices]
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float32, device=scores.device)
    if values.numel() == 1:
        ranks[0] = 1.0
    else:
        ranks[order] = torch.arange(
            values.numel(), dtype=torch.float32, device=scores.device
        ) / (values.numel() - 1)
    output = torch.zeros_like(scores, dtype=torch.float32)
    output[indices] = ranks
    return output


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.numel() != right.numel():
        raise ValueError("Correlation inputs must be one-dimensional and equal length")
    if left.numel() < 2:
        return math.nan
    left_centered = left.float() - left.float().mean()
    right_centered = right.float() - right.float().mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator) == 0.0:
        return math.nan
    return float(torch.dot(left_centered, right_centered) / denominator)


def set_overlap(left: Iterable[int], right: Iterable[int]) -> dict[str, float | int]:
    left_set, right_set = set(left), set(right)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection": intersection,
        "symmetric_difference": len(left_set ^ right_set),
        "jaccard": intersection / union if union else 1.0,
        "overlap_fraction": intersection / min(len(left_set), len(right_set))
        if left_set and right_set
        else 1.0,
    }
