#!/usr/bin/env python3
"""Create paired statistical summaries for a completed VLM SpecPC pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    ]
    return [round(percentile(means, 0.025), 4), round(percentile(means, 0.975), 4)]


def analyze_group(
    cases: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    future = [float(case["future"]["answer_match"]) for case in cases]
    current = [float(case["current"]["answer_match"]) for case in cases]
    dense = [float(case["dense"]["answer_match"]) for case in cases]
    random_means = [
        statistics.fmean(float(item["answer_match"]) for item in case["random"])
        for case in cases
    ]
    future_current = [a - b for a, b in zip(future, current)]
    future_random = [a - b for a, b in zip(future, random_means)]
    return {
        "cases": len(cases),
        "accuracy": {
            "dense": round(statistics.fmean(dense), 4),
            "current": round(statistics.fmean(current), 4),
            "future": round(statistics.fmean(future), 4),
            "random_seed_mean": round(statistics.fmean(random_means), 4),
        },
        "paired_future_minus_current": {
            "mean": round(statistics.fmean(future_current), 4),
            "bootstrap_95_ci": bootstrap_ci(future_current, bootstrap_samples, seed),
            "wins": sum(value > 0 for value in future_current),
            "losses": sum(value < 0 for value in future_current),
            "ties": sum(value == 0 for value in future_current),
        },
        "paired_future_minus_random": {
            "mean": round(statistics.fmean(future_random), 4),
            "bootstrap_95_ci": bootstrap_ci(future_random, bootstrap_samples, seed + 1),
        },
        "mean_prefill_ms": {
            method: round(
                statistics.fmean(
                    case[method]["target_prefill_ms"]
                    if method != "random"
                    else statistics.fmean(item["target_prefill_ms"] for item in case["random"])
                    for case in cases
                ),
                2,
            )
            for method in ("dense", "current", "future", "random")
        },
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    by_case: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for item in payload["results"]:
        case = by_case[(item["task"], int(item["sample_id"]))]
        if item["method"] == "random":
            case.setdefault("random", []).append(item)
        else:
            case[item["method"]] = item
    complete = {
        key: value
        for key, value in by_case.items()
        if all(method in value for method in ("dense", "current", "future", "random"))
        and len(value["random"]) > 0
    }
    tasks = sorted({task for task, _ in complete})
    analysis: dict[str, Any] = {
        "source": str(args.results),
        "source_status": payload.get("status"),
        "complete_cases": len(complete),
        "groups": {},
    }
    for task_index, task in enumerate(tasks):
        cases = [value for (case_task, _), value in complete.items() if case_task == task]
        analysis["groups"][task] = analyze_group(
            cases, args.bootstrap_samples, args.seed + task_index * 10
        )
    analysis["groups"]["ALL"] = analyze_group(
        list(complete.values()), args.bootstrap_samples, args.seed + 1000
    )

    json_output = args.json_output or args.results.with_name("analysis.json")
    markdown_output = args.markdown_output or args.results.with_name("summary.md")
    json_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        "# MileBench visual SpecPC pilot",
        "",
        "| Task | N | Dense | Current | Future | Random | F-C (95% CI) | W/L/T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [*tasks, "ALL"]:
        group = analysis["groups"][name]
        accuracy = group["accuracy"]
        comparison = group["paired_future_minus_current"]
        low, high = comparison["bootstrap_95_ci"]
        rows.append(
            f"| {name} | {group['cases']} | {accuracy['dense']:.3f} | "
            f"{accuracy['current']:.3f} | {accuracy['future']:.3f} | "
            f"{accuracy['random_seed_mean']:.3f} | {comparison['mean']:+.3f} "
            f"([{low:+.3f}, {high:+.3f}]) | "
            f"{comparison['wins']}/{comparison['losses']}/{comparison['ties']} |"
        )
    rows.extend(
        [
            "",
            "Random is averaged within each sample over all configured random seeds. "
            "Intervals are paired non-parametric bootstrap intervals over samples.",
            "",
        ]
    )
    markdown_output.write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"json": str(json_output), "markdown": str(markdown_output)}, indent=2))


if __name__ == "__main__":
    main()
