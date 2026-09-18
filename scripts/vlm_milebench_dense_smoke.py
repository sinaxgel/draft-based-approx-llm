#!/usr/bin/env python3
"""Run a dense Qwen2.5-VL baseline on one real sample from each DRAFT task."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


IMAGE_MARKER = re.compile(r"\{(?:image|table)#(\d+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/data/manifests/milebench/smoke.jsonl"),
    )
    parser.add_argument(
        "--model",
        default="/home/ubuntu/public/model/qwen2.5-vl-7b-instruct",
    )
    parser.add_argument(
        "--processor",
        default="/home/ubuntu/public/model/qwen2.5-vl-3b-instruct",
        help="Processor/tokenizer path. The local 7B snapshot does not contain tokenizer files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/zhangpengcheng/outputs/milebench-dense-smoke/results.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def build_content(record: dict[str, Any]) -> list[dict[str, str]]:
    context = record["context"]
    image_paths = record["image_paths"]
    content: list[dict[str, str]] = [
        {"type": "text", "text": record["task_instruction"] + "\n"}
    ]
    position = 0
    seen: list[int] = []
    for match in IMAGE_MARKER.finditer(context):
        if match.start() > position:
            content.append({"type": "text", "text": context[position : match.start()]})
        image_index = int(match.group(1)) - 1
        if not 0 <= image_index < len(image_paths):
            raise ValueError(
                f"Invalid image marker {match.group(0)} for {record['task']}/{record['sample_id']}"
            )
        content.append({"type": "image"})
        seen.append(image_index)
        position = match.end()
    if position < len(context):
        content.append({"type": "text", "text": context[position:]})
    if seen != list(range(len(image_paths))):
        raise ValueError(
            f"Image markers do not match image paths for {record['task']}/{record['sample_id']}: "
            f"markers={seen}, paths={len(image_paths)}"
        )
    if record["choices"]:
        choice_text = "\nChoice list:\n" + "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(record["choices"])
        )
        content.append({"type": "text", "text": choice_text + "\nYour answer is: "})
    return content


def open_images(paths: list[str]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


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


def save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, found {torch.cuda.device_count()}")
    log(
        f"Starting MileBench dense smoke: samples={len(records)} model={args.model} "
        f"processor={args.processor} "
        f"gpu={torch.cuda.get_device_name(0)}"
    )
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    model_started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    log(f"Model loaded in {time.perf_counter() - model_started:.2f}s")

    payload: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "model": args.model,
            "processor": args.processor,
        },
        "results": [],
    }
    save_results(args.output, payload)

    for record in records:
        case = f"{record['task']}/{record['sample_id']}"
        log(f"CASE_START {case} images={record['image_count']}")
        images = open_images(record["image_paths"])
        try:
            content = build_content(record)
            messages = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            preprocess_started = time.perf_counter()
            inputs = dict(
                processor(text=[prompt], images=images, padding=True, return_tensors="pt")
            )
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        finally:
            for image in images:
                image.close()

        input_tokens = int(inputs["input_ids"].shape[1])
        image_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
        image_grid = inputs["image_grid_thw"].tolist()
        model_inputs = {
            key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - generation_started) * 1000
        generated = output_ids[:, input_tokens:]
        output_text = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        result = {
            "task": record["task"],
            "sample_id": record["sample_id"],
            "image_count": record["image_count"],
            "input_tokens": input_tokens,
            "visual_tokens": image_tokens,
            "image_grid_thw": image_grid,
            "preprocess_ms": round(preprocess_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "generated_tokens": int(generated.shape[1]),
            "answer": record["answer"],
            "output": output_text,
            "answer_match": answer_matches(output_text, record["answer"], record["choices"]),
        }
        payload["results"].append(result)
        save_results(args.output, payload)
        log(
            f"CASE_DONE {case} tokens={input_tokens} visual={image_tokens} "
            f"generation_ms={generation_ms:.2f} peak={result['peak_memory_gib']:.2f}GiB "
            f"match={result['answer_match']} output={output_text!r}"
        )
        del inputs, model_inputs, output_ids, generated
        gc.collect()
        torch.cuda.empty_cache()

    payload["status"] = "passed"
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["summary"] = {
        "samples": len(payload["results"]),
        "nonempty_outputs": sum(bool(item["output"]) for item in payload["results"]),
        "answer_matches": sum(bool(item["answer_match"]) for item in payload["results"]),
    }
    save_results(args.output, payload)
    if payload["summary"]["nonempty_outputs"] != len(records):
        raise RuntimeError("At least one sample produced an empty output")
    log(f"ALL_MILEBENCH_DENSE_SMOKE_TESTS_PASSED output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("MILEBENCH_DENSE_SMOKE_FAILED")
        raise
