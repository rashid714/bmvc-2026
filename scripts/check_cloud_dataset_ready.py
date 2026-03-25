#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cloud_datasets import UnifiedCloudDatasetBuilder


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def format_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def summarize_samples(samples: list) -> dict:
    by_source = Counter([s.source_dataset for s in samples])
    modality_counts = {
        "text": int(sum(1 for s in samples if s.modality_available.get("text", False))),
        "image": int(sum(1 for s in samples if s.modality_available.get("image", False))),
        "audio": int(sum(1 for s in samples if s.modality_available.get("audio", False))),
        "video": int(sum(1 for s in samples if s.modality_available.get("video", False))),
    }
    return {
        "total_samples": len(samples),
        "by_source": dict(by_source),
        "modality_counts": modality_counts,
    }


def run_once(
    sources: list[str],
    train_rows: int,
    val_rows: int,
    cache_dir: Path,
    report_path: Path,
) -> dict:
    t0 = time.time()
    samples = UnifiedCloudDatasetBuilder.build_multimodal_dataset(
        sources=sources,
        splits={"train": train_rows, "validation": val_rows},
        cache_dir=str(cache_dir),
        report_path=str(report_path),
    )
    elapsed = round(time.time() - t0, 2)
    summary = summarize_samples(samples)
    summary["elapsed_seconds"] = elapsed
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify cloud dataset download, cache reuse, and multimodal coverage"
    )
    parser.add_argument("--sources", type=str, default="goemotions,dailydialog,tweet_eval,mine,mine_gdrive,emoticon,raza,coco")
    parser.add_argument("--train-rows", type=int, default=200)
    parser.add_argument("--val-rows", type=int, default=40)
    parser.add_argument("--cache-dir", type=str, default="data/hf_datasets")
    parser.add_argument("--report-path", type=str, default="data/source_availability_report.json")
    parser.add_argument("--run-twice", action="store_true", help="Run twice to verify no double download behavior")
    parser.add_argument("--output-json", type=str, default="data/cloud_dataset_check.json")
    args = parser.parse_args()

    repo_root = Path.cwd()
    cache_dir = (repo_root / args.cache_dir).resolve()
    report_path = (repo_root / args.report_path).resolve()
    output_json = (repo_root / args.output_json).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Force all HF artifacts to remain inside the project folder.
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HOME", str((repo_root / ".hf_home").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((repo_root / "models" / "hf_models").resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((repo_root / "models" / "hf_hub").resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    before_size = dir_size_bytes(cache_dir)
    run1 = run_once(
        sources=sources,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        cache_dir=cache_dir,
        report_path=report_path,
    )
    after_run1 = dir_size_bytes(cache_dir)

    result = {
        "sources": sources,
        "cache_dir": str(cache_dir),
        "report_path": str(report_path),
        "cache_size_mb_before": format_mb(before_size),
        "run1": run1,
        "cache_size_mb_after_run1": format_mb(after_run1),
        "cache_growth_mb_run1": format_mb(after_run1 - before_size),
    }

    if args.run_twice:
        run2 = run_once(
            sources=sources,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            cache_dir=cache_dir,
            report_path=report_path,
        )
        after_run2 = dir_size_bytes(cache_dir)
        result["run2"] = run2
        result["cache_size_mb_after_run2"] = format_mb(after_run2)
        result["cache_growth_mb_run2"] = format_mb(after_run2 - after_run1)

    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved dataset readiness report to {output_json}")


if __name__ == "__main__":
    main()
