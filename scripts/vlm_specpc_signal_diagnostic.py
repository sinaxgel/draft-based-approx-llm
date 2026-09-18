#!/usr/bin/env python3
"""Measure how much future-token evidence changes visual SpecPC selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch
from transformers import AutoProcessor

from draft_approx_llm.vlm import MultimodalTokenLayout, collect_draft_evidence
from draft_approx_llm.vlm.diagnostics import (
    pearson_correlation,
    percentile_rank_scores,
    set_overlap,
)
from vlm_milebench_specpc_pilot import (
    DEFAULT_TASKS,
    build_inputs,
    load_model,
    load_records,
    log,
)


METHODS = ("current", "future_only", "max_raw", "rank_mean", "rank_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/data/manifests/milebench/all.jsonl"),
    )
    parser.add_argument("--draft-model", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct")
    parser.add_argument("--processor", default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20260918)
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--lookahead-tokens", type=int, default=2)
    parser.add_argument("--query-window", type=int, default=64)
    parser.add_argument("--starting-layer", type=int, default=8)
    parser.add_argument("--average-kernel", type=int, default=32)
    parser.add_argument("--neighbor-kernel", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/ubuntu/zhangpengcheng/outputs/vlm-specpc-signal-diagnostic/results.json"
        ),
    )
    return parser.parse_args()


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return round(statistics.fmean(finite), 6) if finite else None


def selected_visual(
    layout: MultimodalTokenLayout, keep_indices: torch.Tensor
) -> list[int]:
    visual = set(int(index) for index in layout.visual_indices.tolist())
    return [int(index) for index in keep_indices.tolist() if int(index) in visual]


def selection_details(
    layout: MultimodalTokenLayout, keep_indices: torch.Tensor
) -> dict[str, Any]:
    selected = selected_visual(layout, keep_indices)
    counts = layout.per_image_selected_counts(keep_indices)
    return {
        "selected_visual_indices": selected,
        "selected_visual_count": len(selected),
        "selected_per_image": counts,
        "zero_image_count": sum(count == 0 for count in counts),
        "zero_image_fraction": sum(count == 0 for count in counts) / len(counts),
    }


def summarize(cases: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(cases),
        "score_statistics": {},
        "budgets": {},
    }
    score_keys = sorted(cases[0]["score_statistics"])
    for key in score_keys:
        summary["score_statistics"][key] = finite_mean(
            [float(case["score_statistics"][key]) for case in cases]
        )
    for budget in budgets:
        budget_key = str(budget)
        budget_summary: dict[str, Any] = {}
        for method in METHODS:
            budget_summary[method] = {
                "mean_selected_visual_tokens": finite_mean(
                    [
                        float(case["budgets"][budget_key]["methods"][method]["selected_visual_count"])
                        for case in cases
                    ]
                ),
                "mean_zero_image_fraction": finite_mean(
                    [
                        float(case["budgets"][budget_key]["methods"][method]["zero_image_fraction"])
                        for case in cases
                    ]
                ),
            }
            if method != "current":
                budget_summary[method]["vs_current"] = {
                    key: finite_mean(
                        [
                            float(case["budgets"][budget_key]["vs_current"][method][key])
                            for case in cases
                        ]
                    )
                    for key in ("jaccard", "overlap_fraction", "symmetric_difference")
                }
        summary["budgets"][budget_key] = budget_summary
    return summary


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() < 1:
        raise RuntimeError("At least one visible GPU is required")
    records = load_records(
        args.manifest, args.tasks, args.samples_per_task, args.sample_seed
    )
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    draft = load_model(args.draft_model, 0)
    payload: dict[str, Any] = {
        "status": "running",
        "configuration": {
            "tasks": args.tasks,
            "samples_per_task": args.samples_per_task,
            "sample_seed": args.sample_seed,
            "budgets": sorted(set(args.budgets)),
            "lookahead_tokens": args.lookahead_tokens,
            "query_window": args.query_window,
            "starting_layer": args.starting_layer,
            "average_kernel": args.average_kernel,
            "neighbor_kernel": args.neighbor_kernel,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for case_index, record in enumerate(records):
        case_name = f"{record['task']}/{record['sample_id']}"
        log(f"DIAGNOSTIC_CASE_START {case_name} {case_index + 1}/{len(records)}")
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
        current = layout.pool_visual_scores(
            evidence.current_scores, args.average_kernel, args.neighbor_kernel
        )
        future_only = layout.pool_visual_scores(
            evidence.future_only_scores, args.average_kernel, args.neighbor_kernel
        )
        max_raw = layout.pool_visual_scores(
            evidence.future_scores, args.average_kernel, args.neighbor_kernel
        )
        current_rank = percentile_rank_scores(current, layout.visual_indices)
        future_rank = percentile_rank_scores(future_only, layout.visual_indices)
        rank_mean = (current_rank + future_rank) / 2
        rank_max = torch.maximum(current_rank, future_rank)
        method_scores = {
            "current": current,
            "future_only": future_only,
            "max_raw": max_raw,
            "rank_mean": rank_mean,
            "rank_max": rank_max,
        }

        scoreable_visual = layout.visual_indices[
            layout.visual_indices < evidence.always_keep_start
        ]
        raw_current = evidence.current_scores[scoreable_visual]
        raw_future = evidence.future_only_scores[scoreable_visual]
        pooled_current = current[scoreable_visual]
        pooled_future = future_only[scoreable_visual]
        score_statistics = {
            "raw_future_gt_current_fraction": float((raw_future > raw_current).float().mean()),
            "pooled_future_gt_current_fraction": float(
                (pooled_future > pooled_current).float().mean()
            ),
            "raw_pearson": pearson_correlation(raw_current, raw_future),
            "pooled_pearson": pearson_correlation(pooled_current, pooled_future),
            "rank_pearson": pearson_correlation(
                current_rank[scoreable_visual], future_rank[scoreable_visual]
            ),
            "raw_current_mean": float(raw_current.mean()),
            "raw_future_mean": float(raw_future.mean()),
            "raw_current_max": float(raw_current.max()),
            "raw_future_max": float(raw_future.max()),
        }
        mandatory = layout.final_window_indices(evidence.query_window)
        budget_payload: dict[str, Any] = {}
        non_visual = int(layout.non_visual_indices.numel())
        for budget in sorted(set(args.budgets)):
            visual_budget = min(
                layout.visual_token_count, max(1, budget - non_visual)
            )
            selections = {
                method: layout.select_visual_tokens_by_count(
                    scores, visual_budget, mandatory
                )
                for method, scores in method_scores.items()
            }
            details = {
                method: selection_details(layout, keep_indices)
                for method, keep_indices in selections.items()
            }
            current_set = details["current"]["selected_visual_indices"]
            comparisons = {
                method: set_overlap(
                    current_set, details[method]["selected_visual_indices"]
                )
                for method in METHODS
                if method != "current"
            }
            max_selected = torch.tensor(
                details["max_raw"]["selected_visual_indices"], dtype=torch.long
            )
            future_dominant = (
                evidence.future_only_scores[max_selected]
                > evidence.current_scores[max_selected]
            )
            budget_payload[str(budget)] = {
                "visual_budget": visual_budget,
                "compressed_tokens": visual_budget + non_visual,
                "methods": details,
                "vs_current": comparisons,
                "max_raw_selected_future_dominant_fraction": float(
                    future_dominant.float().mean()
                ),
            }
        case_payload = {
            "task": record["task"],
            "sample_id": int(record["sample_id"]),
            "lookahead_text": evidence.lookahead_text,
            "layout": layout.summary(),
            "score_statistics": score_statistics,
            "budgets": budget_payload,
        }
        payload["cases"].append(case_payload)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log(
            f"DIAGNOSTIC_CASE_DONE {case_name} "
            f"future_gt={score_statistics['raw_future_gt_current_fraction']:.4f} "
            f"raw_corr={score_statistics['raw_pearson']:.4f}"
        )

    payload["status"] = "passed"
    payload["summary"] = {
        "all": summarize(payload["cases"], sorted(set(args.budgets))),
        "by_task": {
            task: summarize(
                [case for case in payload["cases"] if case["task"] == task],
                sorted(set(args.budgets)),
            )
            for task in args.tasks
        },
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"ALL_SIGNAL_DIAGNOSTICS_PASSED output={args.output}")


if __name__ == "__main__":
    main()
