from __future__ import annotations

import contextlib
from dataclasses import dataclass
import gc
import math
import time
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class SpecPCScoringConfig:
    """Paper-aligned attention aggregation settings for visual SpecPC."""

    query_window: int = 64
    starting_layer: int = 8
    weighted_query: bool = True

    def validate(self) -> None:
        if self.query_window < 1:
            raise ValueError("query_window must be positive")
        if self.starting_layer < 0:
            raise ValueError("starting_layer must be non-negative")


@dataclass
class DraftEvidence:
    lookahead_ids: torch.Tensor
    lookahead_text: str
    current_scores: torch.Tensor
    future_scores: torch.Tensor
    future_only_scores: torch.Tensor
    attention_steps: int
    future_query_steps: int
    query_window: int
    starting_layer: int
    always_keep_start: int
    elapsed_ms: float
    peak_memory_gib: float


def _query_weights(query_count: int, *, weighted: bool, device: torch.device) -> torch.Tensor:
    if query_count < 1:
        raise ValueError("query_count must be positive")
    if not weighted:
        return torch.ones(query_count, dtype=torch.float32, device=device)
    # Algorithm 2 assigns the j-th query weight j / n_window.
    return torch.arange(
        1, query_count + 1, dtype=torch.float32, device=device
    ) / query_count


def _aggregate_layers_and_queries(
    layers: Sequence[torch.Tensor | None],
    *,
    query_slice: slice,
    key_limit: int,
    starting_layer: int,
    weighted_query: bool,
) -> torch.Tensor:
    per_layer: list[torch.Tensor] = []
    for attention in layers[starting_layer:]:
        if attention is None:
            continue
        values = attention[0, :, query_slice, :key_limit].float()
        if values.shape[-2] == 0:
            continue
        weights = _query_weights(
            values.shape[-2], weighted=weighted_query, device=values.device
        )
        values = values * weights.view(1, -1, 1)
        per_layer.append(values.amax(dim=(0, 1)).cpu())
    if not per_layer:
        raise RuntimeError("No attention tensors were collected after layer skipping")
    return torch.stack(per_layer).amax(dim=0)


def aggregate_specpc_attention(
    attention_steps: Sequence[Sequence[torch.Tensor | None]],
    *,
    input_length: int,
    config: SpecPCScoringConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Aggregate draft attention according to Algorithm 2.

    ``current`` uses the final ``query_window`` prompt queries. ``future_only``
    uses attention queries produced after draft generation starts. ``future`` is
    the paper-style maximum over both sets of queries. Keys belonging to the
    final query window are omitted from scoring and returned through
    ``always_keep_start`` so callers can retain them unconditionally.
    """

    config.validate()
    if not attention_steps:
        raise RuntimeError("No generation attention steps were returned")
    window = min(config.query_window, input_length)
    always_keep_start = input_length - window
    if always_keep_start == 0:
        raise ValueError(
            f"query_window ({config.query_window}) leaves no scoreable keys for "
            f"input length {input_length}"
        )

    current_prefix = _aggregate_layers_and_queries(
        attention_steps[0],
        query_slice=slice(input_length - window, input_length),
        key_limit=always_keep_start,
        starting_layer=config.starting_layer,
        weighted_query=config.weighted_query,
    )

    future_parts: list[torch.Tensor] = []
    for layers in attention_steps[1:]:
        future_parts.append(
            _aggregate_layers_and_queries(
                layers,
                query_slice=slice(-1, None),
                key_limit=always_keep_start,
                starting_layer=config.starting_layer,
                weighted_query=False,
            )
        )
    if not future_parts:
        raise RuntimeError(
            "At least two generated tokens are required to obtain one future-token "
            "attention query"
        )
    future_only_prefix = torch.stack(future_parts).amax(dim=0)
    future_prefix = torch.maximum(current_prefix, future_only_prefix)

    def restore_full_length(prefix: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(input_length, dtype=prefix.dtype)
        output[:always_keep_start] = prefix
        return output

    return (
        restore_full_length(current_prefix),
        restore_full_length(future_prefix),
        restore_full_length(future_only_prefix),
        always_keep_start,
    )


class _QwenVLAttentionCollector:
    """Collect only window-to-prompt attention while FlashAttention runs normally."""

    def __init__(self, input_length: int, config: SpecPCScoringConfig):
        self.input_length = input_length
        self.config = config
        self.window = min(config.query_window, input_length)
        self.always_keep_start = input_length - self.window
        if self.always_keep_start == 0:
            raise ValueError("The query window leaves no keys available for scoring")
        self.per_layer: dict[int, list[torch.Tensor]] = {}

    def hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        layer_index = int(module.layer_idx)
        if layer_index < self.config.starting_layer:
            return
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get("position_embeddings")
        if hidden_states is None or position_embeddings is None:
            raise RuntimeError("Qwen attention hook did not receive hidden states/position embeddings")

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
            repeat_kv,
        )

        batch_size, query_count, _ = hidden_states.shape
        query_states = module.q_proj(hidden_states).view(
            batch_size, query_count, -1, module.head_dim
        ).transpose(1, 2)
        current_keys = module.k_proj(hidden_states).view(
            batch_size, query_count, -1, module.head_dim
        ).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, current_keys = apply_multimodal_rotary_pos_emb(
            query_states,
            current_keys,
            cos,
            sin,
            module.rope_scaling["mrope_section"],
        )

        cache = kwargs.get("past_key_value")
        key_states = current_keys
        if cache is not None:
            if hasattr(cache, "key_cache") and layer_index < len(cache.key_cache):
                key_states = cache.key_cache[layer_index]
            else:
                try:
                    key_states = cache[layer_index][0]
                except (IndexError, KeyError, TypeError):
                    key_states = current_keys
        key_states = repeat_kv(key_states, module.num_key_value_groups)

        if query_count > 1:
            query_states = query_states[:, :, -self.window :, :]
        logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            module.head_dim
        )
        if query_count > 1:
            key_positions = torch.arange(key_states.shape[-2], device=logits.device)
            query_positions = torch.arange(
                key_states.shape[-2] - query_states.shape[-2],
                key_states.shape[-2],
                device=logits.device,
            )
            causal_mask = key_positions.view(1, -1) > query_positions.view(-1, 1)
            logits = logits.masked_fill(causal_mask.view(1, 1, *causal_mask.shape), -torch.inf)
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scoreable = weights[..., : self.always_keep_start]
        if query_count > 1:
            query_weights = _query_weights(
                scoreable.shape[-2],
                weighted=self.config.weighted_query,
                device=scoreable.device,
            )
            scoreable = scoreable * query_weights.view(1, 1, -1, 1)
        reduced = scoreable.amax(dim=(1, 2)).cpu()
        self.per_layer.setdefault(layer_index, []).append(reduced)

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_steps = [self.per_layer[index] for index in sorted(self.per_layer)]
        if not layer_steps:
            raise RuntimeError("No language-model attention layers were collected")
        step_counts = {len(steps) for steps in layer_steps}
        if len(step_counts) != 1:
            raise RuntimeError(f"Inconsistent attention step counts across layers: {step_counts}")
        step_count = step_counts.pop()
        if step_count < 2:
            raise RuntimeError("No future-token attention query was collected")
        per_step = [
            torch.stack([steps[index] for steps in layer_steps]).amax(dim=0)[0]
            for index in range(step_count)
        ]
        current_prefix = per_step[0]
        future_only_prefix = torch.stack(per_step[1:]).amax(dim=0)
        future_prefix = torch.maximum(current_prefix, future_only_prefix)

        def restore(prefix: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(self.input_length, dtype=prefix.dtype)
            result[: self.always_keep_start] = prefix
            return result

        return restore(current_prefix), restore(future_prefix), restore(future_only_prefix)


@contextlib.contextmanager
def _collect_qwen_vl_attention(
    model: Any,
    *,
    input_length: int,
    config: SpecPCScoringConfig,
):
    collector = _QwenVLAttentionCollector(input_length, config)
    layers = model.model.layers
    handles = [
        layer.self_attn.register_forward_hook(collector.hook, with_kwargs=True)
        for layer in layers[config.starting_layer :]
    ]
    try:
        yield collector
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def collect_draft_evidence(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    device_index: int,
    lookahead_tokens: int = 2,
    starting_layer: int = 8,
    current_window: int = 64,
    weighted_query: bool = True,
) -> DraftEvidence:
    """Generate draft tokens and collect current/future SpecPC evidence."""

    if lookahead_tokens < 2:
        raise ValueError(
            "lookahead_tokens must be at least 2: Hugging Face generation exposes "
            "the first generated token as prompt logits, so the second token supplies "
            "the first true future attention query"
        )
    device = torch.device(f"cuda:{device_index}")
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    input_length = int(model_inputs["input_ids"].shape[1])
    scoring_config = SpecPCScoringConfig(
        query_window=current_window,
        starting_layer=starting_layer,
        weighted_query=weighted_query,
    )
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with _collect_qwen_vl_attention(
        model, input_length=input_length, config=scoring_config
    ) as collector:
        output = model.generate(
            **model_inputs,
            max_new_tokens=lookahead_tokens,
            min_new_tokens=lookahead_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_attentions=False,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
    elapsed_ms = (time.perf_counter() - started) * 1000
    current_scores, future_scores, future_only_scores = collector.aggregate()
    always_keep_start = collector.always_keep_start
    lookahead_ids = output.sequences[:, input_length:].detach().cpu()
    lookahead_text = processor.batch_decode(
        lookahead_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    evidence = DraftEvidence(
        lookahead_ids=lookahead_ids,
        lookahead_text=lookahead_text,
        current_scores=current_scores,
        future_scores=future_scores,
        future_only_scores=future_only_scores,
        attention_steps=lookahead_tokens,
        future_query_steps=lookahead_tokens - 1,
        query_window=min(current_window, input_length),
        starting_layer=starting_layer,
        always_keep_start=always_keep_start,
        elapsed_ms=elapsed_ms,
        peak_memory_gib=peak_memory_gib,
    )
    del output, collector, model_inputs
    gc.collect()
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
    return evidence
