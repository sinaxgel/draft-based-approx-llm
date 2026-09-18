#!/usr/bin/env python3
"""Evaluate diagnostic visual-token selections with the target VLM."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import statistics
from typing import Any

import torch
from transformers import AutoProcessor

from draft_approx_llm.vlm import (
    MultimodalTokenLayout,
    greedy_generate,
    prepare_target_input,
    select_target_input,
)
from vlm_milebench_specpc_pilot import (
    answer_matches,
    build_inputs,
    load_model,
    log,
    save,
)


METHODS = ("dense", "current", "future_only", "max_raw", "rank_mean", "rank_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-results",
        type=Path,
        default=Path(
            "/home/ubuntu/zhangpengcheng/outputs/vlm-specpc-signal-diagnostic/results.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/data/manifests/milebench/all.jsonl"),
    )
    parser.add_argument("--target-model", default="/home/ubuntu/public/model/qwen2.5-vl-7b-instruct")
    parser.add_argument("--processor", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct")
    parser.add_argument("--budget", type=int, default=768)
    parser.add_argument("--position-strategy", choices=("mrope", "contiguous"), default="mrope")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/outputs/vlm-specpc-fusion-pilot/results.json"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[(record["task"], int(record["sample_id"]))] = record
    return records


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[item["method"]].append(item)
        task_groups[(item["task"], item["method"])].append(item)

    def row(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "runs": len(items),
            "accuracy": sum(bool(item["answer_match"]) for item in items) / len(items),
            "mean_tokens": round(statistics.fmean(item["compressed_tokens"] for item in items), 2),
            "mean_prefill_ms": round(statistics.fmean(item["target_prefill_ms"] for item in items), 2),
        }

    return {
        "all": {method: row(groups[method]) for method in METHODS if groups[method]},
        "by_task": {
            task: {
                method: row(task_groups[(task, method)])
                for method in METHODS
                if task_groups[(task, method)]
            }
            for task in sorted({item["task"] for item in results})
        },
    }


def main() -> None:
    args = parse_args()
    diagnostic = json.loads(args.diagnostic_results.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "passed":
        raise ValueError("Diagnostic selection file is not complete")
    budget_key = str(args.budget)
    if args.budget not in diagnostic["configuration"]["budgets"]:
        raise ValueError(f"Budget {args.budget} is absent from diagnostic results")
    manifest = load_manifest(args.manifest)
    experiment = {
        "diagnostic_results": str(args.diagnostic_results),
        "budget": args.budget,
        "position_strategy": args.position_strategy,
        "max_new_tokens": args.max_new_tokens,
        "methods": list(METHODS),
        "case_ids": [
            [case["task"], int(case["sample_id"])] for case in diagnostic["cases"]
        ],
    }
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("experiment") != experiment:
            raise ValueError("Existing fusion output does not match this experiment")
        payload["status"] = "running"
    else:
        payload = {
            "status": "running",
            "experiment": experiment,
            "results": [],
        }
    save(args.output, payload)
    completed = {
        (item["task"], int(item["sample_id"]), item["method"])
        for item in payload["results"]
    }

    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    target = load_model(args.target_model, 0)
    for case_index, diagnostic_case in enumerate(diagnostic["cases"]):
        key = (diagnostic_case["task"], int(diagnostic_case["sample_id"]))
        record = manifest[key]
        log(f"FUSION_CASE_START {key[0]}/{key[1]} {case_index + 1}/{len(diagnostic['cases'])}")
        inputs = build_inputs(processor, record)
        layout = MultimodalTokenLayout.from_inputs(inputs, target.config)
        prepared = prepare_target_input(target, inputs, device_index=0)
        method_indices: dict[str, torch.Tensor] = {
            "dense": torch.arange(layout.sequence_length)
        }
        stored_methods = diagnostic_case["budgets"][budget_key]["methods"]
        for method in METHODS:
            if method == "dense":
                continue
            method_indices[method] = torch.tensor(
                stored_methods[method]["selected_visual_indices"]
                + layout.non_visual_indices.tolist(),
                dtype=torch.long,
            ).unique(sorted=True)

        for method in METHODS:
            result_key = (key[0], key[1], method)
            if result_key in completed:
                continue
            keep_indices = method_indices[method]
            strategy = "mrope" if method == "dense" else args.position_strategy
            selected = select_target_input(prepared, keep_indices, strategy)
            output_ids, timings = greedy_generate(
                target, selected, device_index=0, max_new_tokens=args.max_new_tokens
            )
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            result = {
                "task": key[0],
                "sample_id": key[1],
                "method": method,
                "compressed_tokens": int(keep_indices.numel()),
                "output": output_text,
                "answer": record["answer"],
                "answer_match": answer_matches(
                    output_text, record["answer"], record["choices"]
                ),
                "target_prefill_ms": timings["prefill_ms"],
                "target_decode_ms": timings["decode_ms"],
                "target_generation_ms": timings["total_ms"],
            }
            payload["results"].append(result)
            completed.add(result_key)
            save(args.output, payload)
            log(
                f"FUSION_METHOD_OK {key[0]}/{key[1]} method={method} "
                f"match={result['answer_match']} output={output_text!r}"
            )
            del selected, output_ids
            gc.collect()
        del prepared, inputs
        gc.collect()
        with torch.cuda.device(0):
            torch.cuda.empty_cache()

    payload["status"] = "passed"
    payload["summary"] = summarize(payload["results"])
    save(args.output, payload)
    log(f"ALL_FUSION_PILOT_TESTS_PASSED output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("VLM_SPECPC_FUSION_PILOT_FAILED")
        raise
