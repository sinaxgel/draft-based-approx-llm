from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


def _centered_pool1d(values: torch.Tensor, kernel_size: int, mode: str) -> torch.Tensor:
    """Length-preserving centered pooling with the paper/repository padding rule."""
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got {tuple(values.shape)}")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if kernel_size == 1:
        return values.clone()
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    batched = values.float().view(1, 1, -1)
    if mode == "avg":
        padded = F.pad(batched, (left, right), mode="constant", value=0.0)
        pooled = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    elif mode == "max":
        padded = F.pad(
            batched,
            (left, right),
            mode="constant",
            value=torch.finfo(batched.dtype).min,
        )
        pooled = F.max_pool1d(padded, kernel_size=kernel_size, stride=1)
    else:
        raise ValueError(f"Unsupported pool mode: {mode}")
    return pooled.view(-1).to(values.dtype)


@dataclass
class MultimodalTokenLayout:
    """Token modality layout for a single Qwen2.5-VL sample."""

    input_ids: torch.Tensor
    visual_indices: torch.Tensor
    non_visual_indices: torch.Tensor
    image_visual_indices: list[torch.Tensor]
    image_grid_thw: torch.Tensor
    spatial_merge_size: int
    vision_start_indices: torch.Tensor
    vision_end_indices: torch.Tensor

    @classmethod
    def from_inputs(cls, inputs: dict[str, Any], config: Any) -> "MultimodalTokenLayout":
        input_ids = inputs["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {tuple(input_ids.shape)}")
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required")

        ids = input_ids[0]
        visual_indices = torch.where(ids == config.image_token_id)[0]
        non_visual_indices = torch.where(ids != config.image_token_id)[0]
        vision_start_indices = torch.where(ids == config.vision_start_token_id)[0]
        vision_end_indices = torch.where(ids == config.vision_end_token_id)[0]
        merge = int(config.vision_config.spatial_merge_size)
        per_image_counts = [int(row.prod().item()) // (merge * merge) for row in image_grid_thw]
        if sum(per_image_counts) != visual_indices.numel():
            raise ValueError(
                f"Visual token/grid mismatch: tokens={visual_indices.numel()}, expected={sum(per_image_counts)}"
            )
        image_visual_indices = list(torch.split(visual_indices, per_image_counts))
        if len(image_visual_indices) != len(vision_start_indices) or len(image_visual_indices) != len(vision_end_indices):
            raise ValueError(
                "Image count does not match vision boundary tokens: "
                f"images={len(image_visual_indices)}, starts={len(vision_start_indices)}, ends={len(vision_end_indices)}"
            )
        for image_index, indices in enumerate(image_visual_indices):
            if indices.numel() == 0 or not torch.all(indices[1:] == indices[:-1] + 1):
                raise ValueError(f"Image {image_index} visual tokens are not contiguous")
            if not (vision_start_indices[image_index] < indices[0] < indices[-1] < vision_end_indices[image_index]):
                raise ValueError(f"Image {image_index} visual tokens are outside their boundary tokens")

        return cls(
            input_ids=input_ids,
            visual_indices=visual_indices,
            non_visual_indices=non_visual_indices,
            image_visual_indices=image_visual_indices,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=merge,
            vision_start_indices=vision_start_indices,
            vision_end_indices=vision_end_indices,
        )

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def visual_token_count(self) -> int:
        return int(self.visual_indices.numel())

    @property
    def image_count(self) -> int:
        return len(self.image_visual_indices)

    def select_visual_tokens(self, scores: torch.Tensor, retention_ratio: float) -> torch.Tensor:
        if scores.ndim != 1 or scores.numel() != self.sequence_length:
            raise ValueError(
                f"Expected one score per input token ({self.sequence_length}), got {tuple(scores.shape)}"
            )
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}")
        keep_count = min(
            self.visual_token_count,
            max(1, math.ceil(self.visual_token_count * retention_ratio)),
        )
        return self.select_visual_tokens_by_count(scores, keep_count)

    def select_visual_tokens_by_count(
        self,
        scores: torch.Tensor,
        keep_count: int,
        always_keep_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Keep an exact number of visual tokens while preserving every text/control token."""
        if scores.ndim != 1 or scores.numel() != self.sequence_length:
            raise ValueError(
                f"Expected one score per input token ({self.sequence_length}), got {tuple(scores.shape)}"
            )
        if not 1 <= keep_count <= self.visual_token_count:
            raise ValueError(
                f"keep_count must be in [1, {self.visual_token_count}], got {keep_count}"
            )
        visual_indices = self.visual_indices.to(scores.device)
        mandatory_visual = torch.empty(0, dtype=torch.long, device=scores.device)
        if always_keep_indices is not None:
            mandatory = always_keep_indices.to(scores.device)
            mandatory_visual = mandatory[
                torch.isin(mandatory, visual_indices)
            ].unique(sorted=True)
        if mandatory_visual.numel() > keep_count:
            raise ValueError(
                f"keep_count={keep_count} is smaller than the number of mandatory "
                f"visual window tokens={mandatory_visual.numel()}"
            )
        candidates = visual_indices[~torch.isin(visual_indices, mandatory_visual)]
        remaining = keep_count - int(mandatory_visual.numel())
        if remaining:
            selected_offsets = torch.topk(
                scores[candidates], remaining, sorted=False
            ).indices
            selected_visual = torch.cat([mandatory_visual, candidates[selected_offsets]])
        else:
            selected_visual = mandatory_visual
        selected_visual = selected_visual.to(self.non_visual_indices.device)
        keep_indices = torch.cat([self.non_visual_indices, selected_visual]).sort().values
        return keep_indices

    def pool_visual_scores(
        self,
        scores: torch.Tensor,
        average_kernel: int,
        neighbor_kernel: int,
    ) -> torch.Tensor:
        """Apply Algorithm 2 AvgPool then neighborhood MaxPool per image.

        Pooling is reset at every image boundary so evidence cannot leak from one
        image into another through adjacent flattened token positions.
        """
        if scores.ndim != 1 or scores.numel() != self.sequence_length:
            raise ValueError(
                f"Expected one score per input token ({self.sequence_length}), "
                f"got {tuple(scores.shape)}"
            )
        pooled = scores.clone()
        for indices in self.image_visual_indices:
            device_indices = indices.to(scores.device)
            image_scores = scores[device_indices]
            image_scores = _centered_pool1d(image_scores, average_kernel, "avg")
            image_scores = _centered_pool1d(image_scores, neighbor_kernel, "max")
            pooled[device_indices] = image_scores
        return pooled

    def final_window_indices(self, window_size: int) -> torch.Tensor:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        start = max(0, self.sequence_length - window_size)
        return torch.arange(start, self.sequence_length, dtype=torch.long)

    def per_image_selected_counts(self, keep_indices: torch.Tensor) -> list[int]:
        keep = set(int(index) for index in keep_indices.cpu().tolist())
        return [sum(int(index) in keep for index in indices.cpu().tolist()) for indices in self.image_visual_indices]

    def image_spatial_shape(self, image_index: int) -> tuple[int, int]:
        grid = self.image_grid_thw[image_index]
        temporal, height, width = (int(value.item()) for value in grid)
        if temporal != 1:
            raise ValueError(f"Only still images are supported, got temporal grid={temporal}")
        return height // self.spatial_merge_size, width // self.spatial_merge_size

    def summary(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "visual_tokens": self.visual_token_count,
            "non_visual_tokens": int(self.non_visual_indices.numel()),
            "image_count": self.image_count,
            "visual_tokens_per_image": [int(indices.numel()) for indices in self.image_visual_indices],
            "image_grid_thw": self.image_grid_thw.cpu().tolist(),
            "vision_start_indices": self.vision_start_indices.cpu().tolist(),
            "vision_end_indices": self.vision_end_indices.cpu().tolist(),
        }
