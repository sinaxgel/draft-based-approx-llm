#!/usr/bin/env python3
"""Validate the six MileBench tasks used by DRAFT and build stable manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


TASKS = (
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
        "--dataset-root",
        type=Path,
        default=Path("/home/ubuntu/TS8T/zhangpengcheng/datasets/MileBench/selected"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/data/manifests/milebench"),
    )
    return parser.parse_args()


def normalize_record(task: str, task_root: Path, annotation: dict[str, Any]) -> dict[str, Any]:
    instance = annotation["task_instance"]
    image_paths = [task_root / "images" / relative for relative in instance["images_path"]]
    return {
        "dataset": "MileBench",
        "task": task,
        "sample_id": int(annotation["sample_id"]),
        "task_instruction_id": int(annotation["task_instruction_id"]),
        "task_instruction": None,
        "context": instance["context"],
        "choices": [str(choice) for choice in instance.get("choice_list", [])],
        "answer": str(annotation["response"]),
        "image_paths": [str(path.resolve()) for path in image_paths],
        "image_count": len(image_paths),
        "image_quantity_level": annotation.get("image_quantity_level"),
    }


def inspect_smoke_images(record: dict[str, Any]) -> list[dict[str, Any]]:
    details = []
    for raw_path in record["image_paths"]:
        path = Path(raw_path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            details.append(
                {
                    "path": raw_path,
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                }
            )
    return details


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    missing_paths: list[str] = []
    for task in TASKS:
        task_root = args.dataset_root / task
        annotation_path = task_root / f"{task}.json"
        with annotation_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        instructions = payload["meta_data"]["task_instruction"]
        records = []
        for annotation in payload["data"]:
            record = normalize_record(task, task_root, annotation)
            instruction_id = record["task_instruction_id"]
            if not 0 <= instruction_id < len(instructions):
                raise ValueError(f"Invalid instruction id {instruction_id} for {task}")
            record["task_instruction"] = instructions[instruction_id]
            declared_images = annotation.get("images_number")
            if declared_images is not None and int(declared_images) != record["image_count"]:
                raise ValueError(
                    f"Image count mismatch for {task}/{record['sample_id']}: "
                    f"declared={declared_images}, paths={record['image_count']}"
                )
            missing_paths.extend(path for path in record["image_paths"] if not Path(path).is_file())
            records.append(record)

        if len(records) != int(payload["meta_data"]["num_sample"]):
            raise ValueError(f"Sample count mismatch for {task}")
        counts = sorted(record["image_count"] for record in records)
        task_summaries[task] = {
            "samples": len(records),
            "image_references": sum(counts),
            "unique_image_references": len(
                {path for record in records for path in record["image_paths"]}
            ),
            "min_images": counts[0],
            "median_images": (counts[99] + counts[100]) / 2,
            "max_images": counts[-1],
            "image_count_histogram": dict(sorted(Counter(counts).items())),
        }
        all_records.extend(records)

    if missing_paths:
        preview = "\n".join(missing_paths[:20])
        raise FileNotFoundError(f"Missing {len(missing_paths)} referenced images:\n{preview}")

    smoke_records = []
    smoke_image_details: dict[str, Any] = {}
    for task in TASKS:
        candidates = [record for record in all_records if record["task"] == task]
        selected = min(candidates, key=lambda record: (record["image_count"], record["sample_id"]))
        smoke_records.append(selected)
        smoke_image_details[f"{task}/{selected['sample_id']}"] = inspect_smoke_images(selected)

    write_jsonl(args.output_dir / "all.jsonl", all_records)
    write_jsonl(args.output_dir / "smoke.jsonl", smoke_records)
    summary = {
        "status": "passed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(args.dataset_root.resolve()),
        "tasks": task_summaries,
        "total_samples": len(all_records),
        "total_image_references": sum(record["image_count"] for record in all_records),
        "smoke_selection_rule": "minimum image count, then minimum sample id",
        "smoke_samples": [
            {
                "task": record["task"],
                "sample_id": record["sample_id"],
                "image_count": record["image_count"],
            }
            for record in smoke_records
        ],
        "smoke_image_details": smoke_image_details,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
