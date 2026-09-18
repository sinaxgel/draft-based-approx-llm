#!/usr/bin/env python3
"""Visual-only Future-SpecPC smoke test for Qwen2.5-VL 3B -> 7B."""

from __future__ import annotations

import gc
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from draft_approx_llm.vlm import (
    MultimodalTokenLayout,
    collect_draft_evidence,
    greedy_generate,
    prepare_target_input,
    select_target_input,
)


DRAFT_PATH = "/home/ubuntu/public/model/qwen2.5-vl-3b-instruct"
TARGET_PATH = "/home/ubuntu/public/model/qwen2.5-vl-7b-instruct"
ASSET_DIR = Path("/home/ubuntu/zhangpengcheng/vlm-work/assets")
OUTPUT_DIR = Path("/home/ubuntu/zhangpengcheng/outputs/vlm-specpc-smoke")
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
RESULT_PATH = OUTPUT_DIR / "results.json"
LOOKAHEAD_TOKENS = 8
MAX_NEW_TOKENS = 24
RETENTION_RATIOS = (0.50, 0.25)


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def build_cases() -> list[dict[str, Any]]:
    red = ASSET_DIR / "red_square.png"
    blue = ASSET_DIR / "blue_circle.png"
    for path in (red, blue):
        if not path.exists():
            raise FileNotFoundError(path)
    return [
        {
            "name": "single_red_square",
            "image_paths": [str(red)],
            "question": "What color and shape is shown? Answer with only the color and shape.",
            "expected_terms": ["red", "square"],
        },
        {
            "name": "second_image_blue_circle",
            "image_paths": [str(red), str(blue)],
            "question": "What color and shape is shown in the second image? Answer with only the color and shape.",
            "expected_terms": ["blue", "circle"],
        },
    ]


def build_inputs(processor: Any, case: dict[str, Any]) -> dict[str, torch.Tensor]:
    content: list[dict[str, str]] = [{"type": "image"} for _ in case["image_paths"]]
    content.append({"type": "text", "text": case["question"]})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.open(path).convert("RGB") for path in case["image_paths"]]
    return dict(processor(text=[text], images=images, padding=True, return_tensors="pt"))


def load_model(path: str, device_index: int, attention: str) -> Any:
    started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    ).eval()
    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 2**30
    log(
        f"Loaded {path} on cuda:{device_index} attention={attention} "
        f"elapsed={time.perf_counter() - started:.2f}s allocated={allocated:.2f}GiB"
    )
    return model


def score_mass_per_image(layout: MultimodalTokenLayout, scores: torch.Tensor) -> list[float]:
    visual_total = float(scores[layout.visual_indices].clamp_min(0).sum().item())
    if visual_total == 0:
        return [0.0] * layout.image_count
    return [
        round(float(scores[indices].clamp_min(0).sum().item()) / visual_total, 6)
        for indices in layout.image_visual_indices
    ]


def save_heatmaps(
    layout: MultimodalTokenLayout,
    scores: torch.Tensor,
    case_name: str,
    image_paths: list[str],
    score_name: str,
) -> list[str]:
    output_paths = []
    for image_index, (indices, image_path) in enumerate(zip(layout.image_visual_indices, image_paths)):
        height, width = layout.image_spatial_shape(image_index)
        values = scores[indices].float().cpu().numpy().reshape(height, width)
        low, high = float(values.min()), float(values.max())
        if high > low:
            values = (values - low) / (high - low)
        else:
            values = np.zeros_like(values)
        heat = Image.fromarray(np.uint8(values * 255), mode="L")
        base = Image.open(image_path).convert("RGB")
        heat = heat.resize(base.size, Image.Resampling.BILINEAR)
        red = Image.new("RGB", base.size, (255, 0, 0))
        alpha = heat.point(lambda value: int(value * 0.58))
        overlay = Image.composite(red, base, alpha)
        output_path = HEATMAP_DIR / f"{case_name}_{score_name}_image{image_index + 1}.png"
        overlay.save(output_path)
        output_paths.append(str(output_path))
    return output_paths


def random_scores(sequence_length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(sequence_length, generator=generator)


def semantic_check(text: str, expected_terms: list[str]) -> tuple[bool, list[str]]:
    lowered = text.lower()
    missing = [term for term in expected_terms if term.lower() not in lowered]
    return not missing, missing


def main() -> None:
    log("Starting visual-only Future-SpecPC smoke test")
    log(
        f"Python={platform.python_version()} torch={torch.__version__} "
        f"transformers={transformers.__version__} CUDA={torch.version.cuda}"
    )
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Need two visible GPUs, found {torch.cuda.device_count()}")
    for index in range(2):
        log(f"Visible cuda:{index}={torch.cuda.get_device_name(index)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(DRAFT_PATH, local_files_only=True)
    draft = load_model(DRAFT_PATH, 0, "eager")
    target = load_model(TARGET_PATH, 1, "flash_attention_2")

    all_results: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case_index, case in enumerate(build_cases()):
        log(f"CASE_START {case['name']}")
        inputs = build_inputs(processor, case)
        layout = MultimodalTokenLayout.from_inputs(inputs, draft.config)
        log(f"TOKEN_LAYOUT case={case['name']} {layout.summary()}")

        evidence = collect_draft_evidence(
            draft,
            processor,
            inputs,
            device_index=0,
            lookahead_tokens=LOOKAHEAD_TOKENS,
            starting_layer=8,
            current_window=16,
        )
        log(
            f"DRAFT_EVIDENCE case={case['name']} lookahead={evidence.lookahead_text!r} "
            f"steps={evidence.attention_steps} future_queries={evidence.future_query_steps} "
            f"elapsed_ms={evidence.elapsed_ms:.2f} peak={evidence.peak_memory_gib:.2f}GiB"
        )
        heatmaps = {
            "current": save_heatmaps(
                layout, evidence.current_scores, case["name"], case["image_paths"], "current"
            ),
            "future": save_heatmaps(
                layout, evidence.future_scores, case["name"], case["image_paths"], "future"
            ),
        }
        current_mass = score_mass_per_image(layout, evidence.current_scores)
        future_mass = score_mass_per_image(layout, evidence.future_scores)

        with torch.cuda.device(1):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        prepared = prepare_target_input(target, inputs, device_index=1)
        log(
            f"TARGET_VISUAL case={case['name']} vision_encoder_ms={prepared.vision_encoder_ms:.3f}"
        )

        methods: list[tuple[str, float, torch.Tensor]] = [
            ("dense", 1.0, torch.arange(layout.sequence_length)),
        ]
        for ratio in RETENTION_RATIOS:
            methods.extend(
                [
                    (
                        "random",
                        ratio,
                        layout.select_visual_tokens(
                            random_scores(layout.sequence_length, 1000 + case_index), ratio
                        ),
                    ),
                    (
                        "current",
                        ratio,
                        layout.select_visual_tokens(evidence.current_scores, ratio),
                    ),
                    (
                        "future",
                        ratio,
                        layout.select_visual_tokens(evidence.future_scores, ratio),
                    ),
                ]
            )

        dense_pass = False
        for method, ratio, keep_indices in methods:
            selected = select_target_input(prepared, keep_indices)
            with torch.cuda.device(1):
                torch.cuda.reset_peak_memory_stats()
            output_ids, timings = greedy_generate(
                target,
                selected,
                device_index=1,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            semantic_pass, missing_terms = semantic_check(output_text, case["expected_terms"])
            if method == "dense":
                dense_pass = semantic_pass
            with torch.cuda.device(1):
                peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
            selected_counts = layout.per_image_selected_counts(keep_indices)
            retained_visual = sum(selected_counts)
            draft_cost = evidence.elapsed_ms if method in {"current", "future"} else 0.0
            pipeline_total_ms = draft_cost + prepared.vision_encoder_ms + timings["total_ms"]
            result = {
                "case": case["name"],
                "method": method,
                "retention_ratio_requested": ratio,
                "original_tokens": layout.sequence_length,
                "original_visual_tokens": layout.visual_token_count,
                "compressed_tokens": int(keep_indices.numel()),
                "retained_visual_tokens": retained_visual,
                "visual_retention_ratio": retained_visual / layout.visual_token_count,
                "retained_visual_tokens_per_image": selected_counts,
                "generated_tokens": int(output_ids.shape[1]),
                "output": output_text,
                "semantic_pass": semantic_pass,
                "missing_terms": missing_terms,
                "draft_evidence_ms": round(draft_cost, 3),
                "target_vision_ms": round(prepared.vision_encoder_ms, 3),
                "target_prefill_ms": timings["prefill_ms"],
                "target_decode_ms": timings["decode_ms"],
                "target_generation_ms": timings["total_ms"],
                "pipeline_total_ms": round(pipeline_total_ms, 3),
                "target_peak_memory_gib": round(peak_memory_gib, 3),
            }
            all_results.append(result)
            log(
                f"METHOD_OK case={case['name']} method={method} ratio={ratio:.2f} "
                f"visual={retained_visual}/{layout.visual_token_count} per_image={selected_counts} "
                f"prefill_ms={timings['prefill_ms']:.3f} semantic_pass={semantic_pass} "
                f"output={output_text!r}"
            )
            del selected, output_ids
            gc.collect()
        if not dense_pass:
            raise RuntimeError(f"Dense manual-generation baseline failed for {case['name']}")

        case_summaries.append(
            {
                "case": case["name"],
                "question": case["question"],
                "expected_terms": case["expected_terms"],
                "layout": layout.summary(),
                "lookahead_text": evidence.lookahead_text,
                "attention_steps": evidence.attention_steps,
                "future_query_steps": evidence.future_query_steps,
                "draft_elapsed_ms": round(evidence.elapsed_ms, 3),
                "draft_peak_memory_gib": round(evidence.peak_memory_gib, 3),
                "current_score_mass_per_image": current_mass,
                "future_score_mass_per_image": future_mass,
                "heatmaps": heatmaps,
            }
        )
        del prepared, evidence, inputs
        gc.collect()
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
        log(f"CASE_DONE {case['name']}")

    payload = {
        "status": "passed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Mechanism smoke test only. Draft uses eager attention and timings include warm-up; "
            "do not use these numbers as final speed claims."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "draft_model": DRAFT_PATH,
            "target_model": TARGET_PATH,
        },
        "lookahead_tokens": LOOKAHEAD_TOKENS,
        "retention_ratios": list(RETENTION_RATIOS),
        "cases": case_summaries,
        "results": all_results,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    nonempty = all(bool(result["output"]) for result in all_results)
    if not nonempty:
        raise RuntimeError("At least one compressed generation returned empty output")
    log(f"Wrote results to {RESULT_PATH}")
    log("ALL_VISUAL_SPECPC_PIPELINE_TESTS_PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("VLM_SPECPC_SMOKE_FAILED")
        raise
