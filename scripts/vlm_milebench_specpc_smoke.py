#!/usr/bin/env python3
"""Run visual-only Future-SpecPC on short real MileBench samples."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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


IMAGE_MARKER = re.compile(r"\{(?:image|table)#(\d+)\}")
DEFAULT_TASKS = ("MovingDirection", "TQA", "WebQA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/data/manifests/milebench/smoke.jsonl"),
    )
    parser.add_argument(
        "--draft-model", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct"
    )
    parser.add_argument(
        "--target-model", default="/home/ubuntu/public/model/qwen2.5-vl-7b-instruct"
    )
    parser.add_argument(
        "--processor", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct"
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/outputs/milebench-specpc-smoke/results.json"),
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_records(path: Path, tasks: list[str]) -> list[dict[str, Any]]:
    wanted = set(tasks)
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = [record for record in records if record["task"] in wanted]
    found = {record["task"] for record in selected}
    missing = wanted - found
    if missing:
        raise ValueError(f"Tasks missing from manifest: {sorted(missing)}")
    return selected


def build_inputs(processor: Any, record: dict[str, Any]) -> dict[str, torch.Tensor]:
    context = record["context"]
    content: list[dict[str, str]] = [
        {"type": "text", "text": record["task_instruction"] + "\n"}
    ]
    position = 0
    seen = []
    for match in IMAGE_MARKER.finditer(context):
        if match.start() > position:
            content.append({"type": "text", "text": context[position : match.start()]})
        image_index = int(match.group(1)) - 1
        if not 0 <= image_index < len(record["image_paths"]):
            raise ValueError(f"Invalid marker {match.group(0)}")
        seen.append(image_index)
        content.append({"type": "image"})
        position = match.end()
    if position < len(context):
        content.append({"type": "text", "text": context[position:]})
    if seen != list(range(len(record["image_paths"]))):
        raise ValueError(
            f"Image markers mismatch for {record['task']}/{record['sample_id']}: {seen}"
        )
    if record["choices"]:
        choice_text = "\nChoice list:\n" + "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(record["choices"])
        )
        content.append({"type": "text", "text": choice_text + "\nYour answer is: "})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = []
    for path in record["image_paths"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    try:
        return dict(processor(text=[prompt], images=images, padding=True, return_tensors="pt"))
    finally:
        for image in images:
            image.close()


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


def random_scores(sequence_length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(sequence_length, generator=generator)


def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def answer_matches(output: str, answer: str, choices: list[str]) -> bool:
    output_norm = normalized(output)
    answer_norm = normalized(answer)
    if answer_norm and answer_norm in output_norm:
        return True
    if answer in choices:
        letter = chr(65 + choices.index(answer)).lower()
        return bool(re.match(rf"^\s*(?:option\s*)?{letter}(?:\b|[.)])", output.lower()))
    return False


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(0)
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Need two visible GPUs, found {torch.cuda.device_count()}")
    records = load_records(args.manifest, args.tasks)
    log(f"Starting real MileBench SpecPC smoke tasks={args.tasks} budgets={args.budgets}")
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    draft = load_model(args.draft_model, 0, "eager")
    target = load_model(args.target_model, 1, "flash_attention_2")
    payload: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Real-data mechanism smoke test. Two generated draft tokens provide one future-token "
            "attention query. Do not use timings as final speed claims."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "processor": args.processor,
        },
        "tasks": args.tasks,
        "budgets": args.budgets,
        "cases": [],
        "results": [],
    }
    save(args.output, payload)

    for case_index, record in enumerate(records):
        case = f"{record['task']}/{record['sample_id']}"
        log(f"CASE_START {case} images={record['image_count']}")
        inputs = build_inputs(processor, record)
        layout = MultimodalTokenLayout.from_inputs(inputs, draft.config)
        evidence = collect_draft_evidence(
            draft,
            processor,
            inputs,
            device_index=0,
            lookahead_tokens=2,
            starting_layer=8,
            current_window=16,
        )
        log(
            f"DRAFT_EVIDENCE {case} layout={layout.summary()} "
            f"lookahead={evidence.lookahead_text!r} future_queries={evidence.future_query_steps} "
            f"elapsed_ms={evidence.elapsed_ms:.2f} peak={evidence.peak_memory_gib:.2f}GiB"
        )
        prepared = prepare_target_input(target, inputs, device_index=1)
        non_visual = int(layout.non_visual_indices.numel())
        methods: list[tuple[str, int, torch.Tensor]] = [
            ("dense", layout.sequence_length, torch.arange(layout.sequence_length))
        ]
        for budget in sorted(set(args.budgets)):
            visual_budget = min(layout.visual_token_count, max(1, budget - non_visual))
            methods.extend(
                [
                    (
                        "random",
                        budget,
                        layout.select_visual_tokens_by_count(
                            random_scores(layout.sequence_length, 1000 + case_index + budget),
                            visual_budget,
                        ),
                    ),
                    (
                        "current",
                        budget,
                        layout.select_visual_tokens_by_count(
                            evidence.current_scores, visual_budget
                        ),
                    ),
                    (
                        "future",
                        budget,
                        layout.select_visual_tokens_by_count(evidence.future_scores, visual_budget),
                    ),
                ]
            )

        for method, budget, keep_indices in methods:
            selected = select_target_input(prepared, keep_indices)
            with torch.cuda.device(1):
                torch.cuda.reset_peak_memory_stats()
            output_ids, timings = greedy_generate(
                target,
                selected,
                device_index=1,
                max_new_tokens=args.max_new_tokens,
            )
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            with torch.cuda.device(1):
                peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
            selected_counts = layout.per_image_selected_counts(keep_indices)
            result = {
                "task": record["task"],
                "sample_id": record["sample_id"],
                "method": method,
                "requested_cmax": budget,
                "original_tokens": layout.sequence_length,
                "original_visual_tokens": layout.visual_token_count,
                "compressed_tokens": int(keep_indices.numel()),
                "retained_visual_tokens": sum(selected_counts),
                "retained_visual_tokens_per_image": selected_counts,
                "output": output_text,
                "answer": record["answer"],
                "answer_match": answer_matches(output_text, record["answer"], record["choices"]),
                "target_vision_ms": round(prepared.vision_encoder_ms, 3),
                "target_prefill_ms": timings["prefill_ms"],
                "target_decode_ms": timings["decode_ms"],
                "target_generation_ms": timings["total_ms"],
                "target_peak_memory_gib": round(peak_memory_gib, 3),
            }
            payload["results"].append(result)
            save(args.output, payload)
            log(
                f"METHOD_OK {case} method={method} cmax={budget} "
                f"tokens={result['compressed_tokens']} visual={result['retained_visual_tokens']} "
                f"prefill_ms={result['target_prefill_ms']:.2f} match={result['answer_match']} "
                f"output={output_text!r}"
            )
            del selected, output_ids
            gc.collect()

        payload["cases"].append(
            {
                "task": record["task"],
                "sample_id": record["sample_id"],
                "answer": record["answer"],
                "layout": layout.summary(),
                "lookahead_text": evidence.lookahead_text,
                "future_query_steps": evidence.future_query_steps,
                "draft_elapsed_ms": round(evidence.elapsed_ms, 3),
                "draft_peak_memory_gib": round(evidence.peak_memory_gib, 3),
            }
        )
        save(args.output, payload)
        del prepared, evidence, inputs
        gc.collect()
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
        log(f"CASE_DONE {case}")

    payload["status"] = "passed"
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["summary"] = {
        "cases": len(payload["cases"]),
        "runs": len(payload["results"]),
        "answer_matches": sum(bool(item["answer_match"]) for item in payload["results"]),
    }
    save(args.output, payload)
    log(f"ALL_MILEBENCH_SPECPC_SMOKE_TESTS_PASSED output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("MILEBENCH_SPECPC_SMOKE_FAILED")
        raise
