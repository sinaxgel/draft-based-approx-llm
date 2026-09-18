from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import Any

import torch


@dataclass
class DraftEvidence:
    lookahead_ids: torch.Tensor
    lookahead_text: str
    current_scores: torch.Tensor
    future_scores: torch.Tensor
    attention_steps: int
    future_query_steps: int
    elapsed_ms: float
    peak_memory_gib: float


def _reduce_prefill_attention(
    layers: tuple[torch.Tensor, ...],
    input_length: int,
    starting_layer: int,
    current_window: int,
) -> torch.Tensor:
    per_layer = []
    for attention in layers[starting_layer:]:
        if attention is None:
            continue
        query_start = max(0, attention.shape[-2] - current_window)
        values = attention[0, :, query_start:, :input_length].float()
        per_layer.append(values.mean(dim=(0, 1)).cpu())
    if not per_layer:
        raise RuntimeError("No prefill attention tensors were collected")
    return torch.stack(per_layer).amax(dim=0)


def _reduce_future_attention(
    steps: tuple[tuple[torch.Tensor, ...], ...],
    input_length: int,
    starting_layer: int,
) -> torch.Tensor:
    per_query = []
    for layers in steps:
        per_layer = []
        for attention in layers[starting_layer:]:
            if attention is None:
                continue
            values = attention[0, :, -1, :input_length].float()
            per_layer.append(values.mean(dim=0).cpu())
        if per_layer:
            per_query.append(torch.stack(per_layer).amax(dim=0))
    if not per_query:
        raise RuntimeError("No future-token attention tensors were collected")
    return torch.stack(per_query).amax(dim=0)


@torch.inference_mode()
def collect_draft_evidence(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    device_index: int,
    lookahead_tokens: int = 8,
    starting_layer: int = 8,
    current_window: int = 16,
) -> DraftEvidence:
    device = torch.device(f"cuda:{device_index}")
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    input_length = int(model_inputs["input_ids"].shape[1])
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.generate(
        **model_inputs,
        max_new_tokens=lookahead_tokens,
        min_new_tokens=lookahead_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
    elapsed_ms = (time.perf_counter() - started) * 1000
    attention_steps = output.attentions
    if attention_steps is None or len(attention_steps) < 2:
        raise RuntimeError("At least two attention steps are required for future-token scoring")

    current_scores = _reduce_prefill_attention(
        attention_steps[0], input_length, starting_layer, current_window
    )
    future_scores = _reduce_future_attention(
        attention_steps[1:], input_length, starting_layer
    )
    lookahead_ids = output.sequences[:, input_length:].detach().cpu()
    lookahead_text = processor.batch_decode(
        lookahead_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    evidence = DraftEvidence(
        lookahead_ids=lookahead_ids,
        lookahead_text=lookahead_text,
        current_scores=current_scores,
        future_scores=future_scores,
        attention_steps=len(attention_steps),
        future_query_steps=len(attention_steps) - 1,
        elapsed_ms=elapsed_ms,
        peak_memory_gib=peak_memory_gib,
    )
    del output, attention_steps, model_inputs
    gc.collect()
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
    return evidence
