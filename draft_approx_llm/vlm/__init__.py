"""Qwen2.5-VL research utilities for visual-token compression experiments."""

from .attention import (
    DraftEvidence,
    SpecPCScoringConfig,
    aggregate_specpc_attention,
    collect_draft_evidence,
)
from .generation import PreparedTargetInput, prepare_target_input, select_target_input, greedy_generate
from .layout import MultimodalTokenLayout

__all__ = [
    "DraftEvidence",
    "MultimodalTokenLayout",
    "PreparedTargetInput",
    "SpecPCScoringConfig",
    "aggregate_specpc_attention",
    "collect_draft_evidence",
    "greedy_generate",
    "prepare_target_input",
    "select_target_input",
]
