#!/usr/bin/env python3
"""Run a resumable six-task MileBench pilot for visual Future-SpecPC."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import platform
import random
import re
import statistics
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
DEFAULT_TASKS = (
    "EgocentricNavigation",
    "MovingDirection",
    "SceneTransition",
    "SlideVQA",
    "TQA",
    "WebQA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/data/manifests/milebench/all.jsonl"),
    )
    parser.add_argument("--draft-model", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct")
    parser.add_argument("--target-model", default="/home/ubuntu/public/model/qwen2.5-vl-7b-instruct")
    parser.add_argument("--processor", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260918)
    parser.add_argument("--budgets", nargs="+", type=int, default=[768])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--lookahead-tokens", type=int, default=2)
    parser.add_argument("--query-window", type=int, default=64)
    parser.add_argument("--starting-layer", type=int, default=8)
    parser.add_argument("--average-kernel", type=int, default=32)
    parser.add_argument("--neighbor-kernel", type=int, default=32)
    parser.add_argument("--position-strategy", choices=("mrope", "contiguous"), default="mrope")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/outputs/milebench-specpc-pilot/results.json"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_records(path: Path, tasks: list[str], count: int, seed: int) -> list[dict[str, Any]]:
    wanted = set(tasks)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["task"] in wanted:
                    grouped[record["task"]].append(record)
    missing = wanted - set(grouped)
    if missing:
        raise ValueError(f"Tasks missing from manifest: {sorted(missing)}")
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        candidates = sorted(grouped[task], key=lambda item: int(item["sample_id"]))
        if count < 1 or count > len(candidates):
            raise ValueError(f"Invalid samples-per-task={count} for {task} ({len(candidates)} available)")
        picker = random.Random(seed + task_index)
        selected.extend(sorted(picker.sample(candidates, count), key=lambda item: int(item["sample_id"])))
    return selected


def build_inputs(processor: Any, record: dict[str, Any]) -> dict[str, torch.Tensor]:
    context = record["context"]
    content: list[dict[str, str]] = [{"type": "text", "text": record["task_instruction"] + "\n"}]
    position = 0
    seen: list[int] = []
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
        raise ValueError(f"Image markers mismatch for {record['task']}/{record['sample_id']}: {seen}")
    if record["choices"]:
        choices = "\nChoice list:\n" + "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(record["choices"])
        )
        content.append({"type": "text", "text": choices + "\nYour answer is: "})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    images = []
    for path in record["image_paths"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    try:
        return dict(processor(text=[prompt], images=images, padding=True, return_tensors="pt"))
    finally:
        for image in images:
            image.close()


def load_model(path: str, device_index: int) -> Any:
    started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    ).eval()
    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 2**30
    log(f"Loaded {path} on cuda:{device_index} elapsed={time.perf_counter() - started:.2f}s allocated={allocated:.2f}GiB")
    return model


def random_scores(sequence_length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(sequence_length, generator=generator)


def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def answer_matches(output: str, answer: str, choices: list[str]) -> bool:
    if normalized(answer) and normalized(answer) in normalized(output):
        return True
    if answer in choices:
        letter = chr(65 + choices.index(answer)).lower()
        return bool(re.match(rf"^\s*(?:option\s*)?{letter}(?:\b|[.)])", output.lower()))
    return False


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def result_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["task"],
        int(item["sample_id"]),
        item["method"],
        int(item["requested_cmax"]),
        item.get("random_seed"),
    )


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[(item["task"], item["method"], item["requested_cmax"])].append(item)
    output = []
    for (task, method, budget), items in sorted(groups.items()):
        output.append(
            {
                "task": task,
                "method": method,
                "requested_cmax": budget,
                "runs": len(items),
                "accuracy": sum(bool(item["answer_match"]) for item in items) / len(items),
                "mean_compressed_tokens": round(statistics.fmean(item["compressed_tokens"] for item in items), 2),
                "mean_prefill_ms": round(statistics.fmean(item["target_prefill_ms"] for item in items), 2),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Need two visible GPUs, found {torch.cuda.device_count()}")
    records = load_records(args.manifest, args.tasks, args.samples_per_task, args.sample_seed)
    experiment = {
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "sample_seed": args.sample_seed,
        "sample_ids": {task: [int(r["sample_id"]) for r in records if r["task"] == task] for task in args.tasks},
        "budgets": sorted(set(args.budgets)),
        "random_seeds": args.random_seeds,
        "lookahead_tokens": args.lookahead_tokens,
        "query_window": args.query_window,
        "starting_layer": args.starting_layer,
        "weighted_query": True,
        "average_kernel": args.average_kernel,
        "neighbor_kernel": args.neighbor_kernel,
        "position_strategy": args.position_strategy,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("experiment") != experiment:
            raise ValueError("Existing output configuration does not match this run")
        payload["status"] = "running"
        payload["resumed_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        payload = {
            "status": "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "Visual-only SpecPC pilot. Future=max(weighted prompt-window queries, generated future queries).",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "draft_model": args.draft_model,
                "target_model": args.target_model,
                "processor": args.processor,
            },
            "experiment": experiment,
            "cases": [],
            "results": [],
        }
    save(args.output, payload)
    completed = {result_key(item) for item in payload["results"]}
    log(f"Starting/resuming pilot cases={len(records)} completed_runs={len(completed)} config={experiment}")

    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    draft = load_model(args.draft_model, 0)
    target = load_model(args.target_model, 1)

    for case_index, record in enumerate(records):
        case = f"{record['task']}/{record['sample_id']}"
        expected = [(record["task"], int(record["sample_id"]), "dense", 0, None)]
        for budget in experiment["budgets"]:
            expected.extend((record["task"], int(record["sample_id"]), method, budget, None) for method in ("current", "future"))
            expected.extend((record["task"], int(record["sample_id"]), "random", budget, seed) for seed in args.random_seeds)
        if all(key in completed for key in expected):
            log(f"CASE_SKIP_COMPLETED {case}")
            continue

        log(f"CASE_START {case} index={case_index + 1}/{len(records)} images={record['image_count']}")
        inputs = build_inputs(processor, record)
        layout = MultimodalTokenLayout.from_inputs(inputs, draft.config)
        evidence = collect_draft_evidence(
            draft,
            processor,
            inputs,
            device_index=0,
            lookahead_tokens=args.lookahead_tokens,
            starting_layer=args.starting_layer,
            current_window=args.query_window,
            weighted_query=True,
        )
        current_scores = layout.pool_visual_scores(
            evidence.current_scores, args.average_kernel, args.neighbor_kernel
        )
        future_scores = layout.pool_visual_scores(
            evidence.future_scores, args.average_kernel, args.neighbor_kernel
        )
        mandatory = layout.final_window_indices(evidence.query_window)
        prepared = prepare_target_input(target, inputs, device_index=1)
        non_visual = int(layout.non_visual_indices.numel())
        methods: list[tuple[str, int, int | None, torch.Tensor]] = [
            ("dense", 0, None, torch.arange(layout.sequence_length))
        ]
        for budget in experiment["budgets"]:
            visual_budget = min(layout.visual_token_count, max(1, budget - non_visual))
            methods.extend(
                [
                    ("current", budget, None, layout.select_visual_tokens_by_count(current_scores, visual_budget, mandatory)),
                    ("future", budget, None, layout.select_visual_tokens_by_count(future_scores, visual_budget, mandatory)),
                ]
            )
            for seed in args.random_seeds:
                methods.append(
                    (
                        "random",
                        budget,
                        seed,
                        layout.select_visual_tokens_by_count(
                            random_scores(layout.sequence_length, seed + case_index * 1009),
                            visual_budget,
                            mandatory,
                        ),
                    )
                )

        for method, budget, random_seed, keep_indices in methods:
            key = (record["task"], int(record["sample_id"]), method, budget, random_seed)
            if key in completed:
                continue
            strategy = "mrope" if method == "dense" else args.position_strategy
            selected = select_target_input(prepared, keep_indices, strategy)
            with torch.cuda.device(1):
                torch.cuda.reset_peak_memory_stats()
            output_ids, timings = greedy_generate(target, selected, 1, args.max_new_tokens)
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            with torch.cuda.device(1):
                peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
            selected_counts = layout.per_image_selected_counts(keep_indices)
            result = {
                "task": record["task"],
                "sample_id": int(record["sample_id"]),
                "method": method,
                "random_seed": random_seed,
                "requested_cmax": budget,
                "position_strategy": strategy,
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
            completed.add(key)
            save(args.output, payload)
            log(
                f"METHOD_OK {case} method={method} seed={random_seed} cmax={budget} "
                f"tokens={result['compressed_tokens']} match={result['answer_match']} "
                f"prefill_ms={result['target_prefill_ms']:.2f} output={output_text!r}"
            )
            del selected, output_ids
            gc.collect()

        payload["cases"] = [
            item for item in payload["cases"]
            if not (item["task"] == record["task"] and int(item["sample_id"]) == int(record["sample_id"]))
        ]
        payload["cases"].append(
            {
                "task": record["task"],
                "sample_id": int(record["sample_id"]),
                "answer": record["answer"],
                "layout": layout.summary(),
                "lookahead_text": evidence.lookahead_text,
                "future_query_steps": evidence.future_query_steps,
                "draft_elapsed_ms": round(evidence.elapsed_ms, 3),
                "draft_peak_memory_gib": round(evidence.peak_memory_gib, 3),
            }
        )
        payload["summary"] = summarize(payload["results"])
        save(args.output, payload)
        del prepared, evidence, inputs, current_scores, future_scores
        gc.collect()
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
        log(f"CASE_DONE {case}")

    payload["status"] = "passed"
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["summary"] = summarize(payload["results"])
    save(args.output, payload)
    log(f"ALL_MILEBENCH_SPECPC_PILOT_TESTS_PASSED output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("MILEBENCH_SPECPC_PILOT_FAILED")
        raise
