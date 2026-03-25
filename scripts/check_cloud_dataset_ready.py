#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
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
    parser.add_argument("--require-min-samples", type=int, default=1)
    parser.add_argument(
        "--require-modalities",
        type=str,
        default="text",
        help="Comma-separated required modalities from: text,image,audio,video",
    )
    parser.add_argument(
        "--strict-mine-gdrive",
        action="store_true",
        help="Fail if mine_gdrive requested but MINE_GDRIVE_ROOT/manifest is missing",
    )
    parser.add_argument(
        "--max-cache-growth-mb-run2",
        type=float,
        default=-1.0,
        help="If set >=0 and --run-twice, fail when cache growth on run2 exceeds this threshold",
    )
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

    if args.strict_mine_gdrive and any(s.lower() in {"mine_gdrive", "mine_drive", "mine_google_drive"} for s in sources):
        root = os.environ.get("MINE_GDRIVE_ROOT", "").strip()
        if not root:
            raise SystemExit("STRICT CHECK FAILED: mine_gdrive requested but MINE_GDRIVE_ROOT is not set")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise SystemExit(f"STRICT CHECK FAILED: MINE_GDRIVE_ROOT does not exist: {root_path}")
        manifest_candidates = [
            "manifest.jsonl",
            "manifest.json",
            "metadata.jsonl",
            "metadata.json",
            "annotations.jsonl",
            "annotations.json",
            "data.jsonl",
            "data.json",
        ]
        if not any((root_path / m).exists() for m in manifest_candidates):
            raise SystemExit(
                f"STRICT CHECK FAILED: no metadata manifest found under {root_path} "
                "(expected one of manifest/metadata/annotations/data json or jsonl files)"
            )

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

    failures: list[str] = []
    required_modalities = [m.strip().lower() for m in args.require_modalities.split(",") if m.strip()]
    allowed = {"text", "image", "audio", "video"}
    invalid_modalities = [m for m in required_modalities if m not in allowed]
    if invalid_modalities:
        failures.append(f"Invalid require-modalities values: {invalid_modalities}")

    min_samples = max(0, int(args.require_min_samples))
    run1 = result["run1"]
    if int(run1["total_samples"]) < min_samples:
        failures.append(
            f"Run1 loaded only {run1['total_samples']} samples, required >= {min_samples}"
        )

    for modality in required_modalities:
        if int(run1["modality_counts"].get(modality, 0)) <= 0:
            failures.append(f"Required modality missing in run1: {modality}")

    if args.run_twice and args.max_cache_growth_mb_run2 >= 0:
        growth2 = float(result.get("cache_growth_mb_run2", 0.0))
        if growth2 > float(args.max_cache_growth_mb_run2):
            failures.append(
                f"Run2 cache growth {growth2} MB exceeded threshold {args.max_cache_growth_mb_run2} MB"
            )

    result["status"] = "ok" if not failures else "failed"
    result["failures"] = failures

    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved dataset readiness report to {output_json}")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
