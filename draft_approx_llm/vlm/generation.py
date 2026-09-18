from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch


@dataclass
class PreparedTargetInput:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    input_ids: torch.Tensor
    vision_encoder_ms: float
    position_strategy: str = "mrope"


@torch.inference_mode()
def prepare_target_input(
    model: Any,
    inputs: dict[str, Any],
    device_index: int,
) -> PreparedTargetInput:
    device = torch.device(f"cuda:{device_index}")
    values = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    input_ids = values["input_ids"]
    attention_mask = values["attention_mask"]
    inputs_embeds = model.model.embed_tokens(input_ids)

    with torch.cuda.device(device_index):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    image_embeds = model.visual(
        values["pixel_values"].to(model.visual.dtype),
        grid_thw=values["image_grid_thw"],
    )
    with torch.cuda.device(device_index):
        end_event.record()
        torch.cuda.synchronize()
        vision_encoder_ms = start_event.elapsed_time(end_event)

    image_mask = input_ids == model.config.image_token_id
    image_token_count = int(image_mask.sum().item())
    if image_token_count != image_embeds.shape[0]:
        raise ValueError(
            f"Image features and image tokens do not match: tokens={image_token_count}, "
            f"features={image_embeds.shape[0]}"
        )
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
    )
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=values["image_grid_thw"],
        attention_mask=attention_mask,
    )
    return PreparedTargetInput(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        input_ids=input_ids,
        vision_encoder_ms=vision_encoder_ms,
        position_strategy="mrope",
    )


def select_target_input(
    prepared: PreparedTargetInput,
    keep_indices: torch.Tensor,
    position_strategy: str = "mrope",
) -> PreparedTargetInput:
    """Select compressed embeddings and assign their Qwen2.5-VL positions.

    ``mrope`` preserves the original three-axis visual positions, retaining the
    image grid geometry after sparse token selection. ``contiguous`` implements
    the paper's text-model policy by reassigning 0..N-1 on all three axes.  The
    latter is exposed as an explicit ablation because it discards 2D visual
    coordinates.
    """
    keep = keep_indices.to(prepared.inputs_embeds.device)
    if position_strategy == "mrope":
        position_ids = prepared.position_ids.index_select(2, keep)
    elif position_strategy == "contiguous":
        length = int(keep.numel())
        position_ids = torch.arange(
            length,
            dtype=prepared.position_ids.dtype,
            device=prepared.position_ids.device,
        ).view(1, 1, length).expand(3, prepared.position_ids.shape[1], length)
    else:
        raise ValueError(
            f"Unknown position_strategy={position_strategy!r}; expected 'mrope' or 'contiguous'"
        )
    return PreparedTargetInput(
        inputs_embeds=prepared.inputs_embeds.index_select(1, keep),
        attention_mask=prepared.attention_mask.index_select(1, keep),
        position_ids=position_ids,
        input_ids=prepared.input_ids.index_select(1, keep),
        vision_encoder_ms=prepared.vision_encoder_ms,
        position_strategy=position_strategy,
    )


def _eos_ids(model: Any) -> set[int]:
    value = model.generation_config.eos_token_id
    if value is None:
        value = model.config.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


@torch.inference_mode()
def greedy_generate(
    model: Any,
    prepared: PreparedTargetInput,
    device_index: int,
    max_new_tokens: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = torch.device(f"cuda:{device_index}")
    eos_ids = _eos_ids(model)

    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
    wall_started = time.perf_counter()
    with torch.cuda.device(device_index):
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()
    output = model(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        position_ids=prepared.position_ids,
        use_cache=True,
        return_dict=True,
    )
    with torch.cuda.device(device_index):
        prefill_end.record()
        torch.cuda.synchronize()
    prefill_ms = prefill_start.elapsed_time(prefill_end)

    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past_key_values = output.past_key_values
    del output
    next_position = int(prepared.position_ids.max().item()) + 1

    with torch.cuda.device(device_index):
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
    for offset in range(max_new_tokens - 1):
        if int(next_token.item()) in eos_ids:
            break
        token_embeds = model.model.embed_tokens(next_token.to(device))
        full_attention_mask = torch.ones(
            (1, prepared.inputs_embeds.shape[1] + len(generated)),
            dtype=prepared.attention_mask.dtype,
            device=device,
        )
        token_position = torch.full(
            (3, 1, 1),
            next_position + offset,
            dtype=prepared.position_ids.dtype,
            device=device,
        )
        output = model(
            inputs_embeds=token_embeds,
            attention_mask=full_attention_mask,
            position_ids=token_position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        past_key_values = output.past_key_values
        del output
    with torch.cuda.device(device_index):
        decode_end.record()
        torch.cuda.synchronize()
    decode_ms = decode_start.elapsed_time(decode_end)
    total_ms = (time.perf_counter() - wall_started) * 1000
    return torch.cat(generated, dim=1).cpu(), {
        "prefill_ms": round(prefill_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "total_ms": round(total_ms, 3),
    }
