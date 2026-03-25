#!/usr/bin/env python3
"""
Pre-download models and datasets only (no training).

Purpose:
- Warm Hugging Face model cache.
- Warm Hugging Face dataset cache.
- Verify large cache (20-30GB target profile) before running training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cloud_datasets import (
    MINEDatasetLoader,
    MINEGoogleDriveDatasetLoader,
    EmoticonDatasetLoader,
    RazaIntentDatasetLoader,
    MSCOCOCaptionsLoader,
    VoxCelebDatasetLoader,
    GoEmotionsDatasetLoader,
    DailyDialogDatasetLoader,
    TweetEvalEmotionDatasetLoader,
)

LOGGER = logging.getLogger("predownload_assets")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def resolve_dataset_plan(profile: str, max_rows_per_source: int | None, sources: list[str] | None) -> tuple[list[str], int, int]:
    if sources is None:
        if profile == "ultra_30gb":
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "mine_gdrive", "emoticon", "raza", "coco", "voxceleb"]
        elif profile == "large_20gb":
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "mine_gdrive", "emoticon", "raza", "coco"]
        else:
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "emoticon", "raza"]

    if max_rows_per_source is None:
        if profile == "ultra_30gb":
            train_rows = 40000
        elif profile == "large_20gb":
            train_rows = 25000
        else:
            train_rows = 5000
    else:
        train_rows = int(max_rows_per_source)

    val_rows = max(1, train_rows // 5)
    return sources, train_rows, val_rows


def warm_models(models: list[str]) -> None:
    LOGGER.info("Model download plan: %s", ", ".join(models))
    for model_name in models:
        LOGGER.info("Downloading tokenizer: %s", model_name)
        AutoTokenizer.from_pretrained(model_name)
        LOGGER.info("Downloading model weights: %s", model_name)
        AutoModel.from_pretrained(model_name)
        LOGGER.info("Model ready in cache: %s", model_name)


def _load_split_with_fallback(loader: Callable[[str, int | None], list], preferred_split: str, limit: int) -> int:
    try:
        data = loader(split=preferred_split, limit=limit)
        return len(data)
    except Exception:
        # Some datasets only expose train. Fall back without failing the run.
        data = loader(split="train", limit=limit)
        return len(data)


def warm_datasets(sources: list[str], train_rows: int, val_rows: int) -> None:
    loaders: dict[str, Callable[[str, int | None], list]] = {
        "goemotions": GoEmotionsDatasetLoader.load_split,
        "dailydialog": DailyDialogDatasetLoader.load_split,
        "tweet_eval": TweetEvalEmotionDatasetLoader.load_split,
        "mine": MINEDatasetLoader.load_mine_split,
        "mine_gdrive": MINEGoogleDriveDatasetLoader.load_split,
        "emoticon": EmoticonDatasetLoader.load_emoticon_split,
        "raza": RazaIntentDatasetLoader.load_intent_split,
        "coco": MSCOCOCaptionsLoader.load_coco_split,
        "voxceleb": VoxCelebDatasetLoader.load_voxceleb_split,
    }

    LOGGER.info("Dataset download plan: sources=%s", ", ".join(sources))
    LOGGER.info("Rows per source: train=%d, validation=%d", train_rows, val_rows)

    for source in sources:
        loader = loaders.get(source)
        if loader is None:
            LOGGER.warning("Skipping unknown dataset source: %s", source)
            continue

        LOGGER.info("Downloading dataset source: %s", source)
        train_count = _load_split_with_fallback(loader, "train", train_rows)
        val_count = _load_split_with_fallback(loader, "validation", val_rows)
        LOGGER.info("Cached source=%s train=%d validation=%d", source, train_count, val_count)


def print_cache_paths() -> None:
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    hf_datasets = os.environ.get("HF_DATASETS_CACHE", "<default>")
    hf_models = os.environ.get("TRANSFORMERS_CACHE", "<default>")
    LOGGER.info("HF_HOME=%s", hf_home)
    LOGGER.info("HF_DATASETS_CACHE=%s", hf_datasets)
    LOGGER.info("TRANSFORMERS_CACHE=%s", hf_models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download models and datasets only (no training)")
    parser.add_argument("--config", type=str, default="configs/multimodal_ultra_30gb.json")
    parser.add_argument("--dataset-profile", type=str, default=None, choices=["balanced", "large_20gb", "ultra_30gb"])
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    args = parser.parse_args()

    setup_logging()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Default all caches inside repo data/ and models/ folders unless explicitly overridden.
    repo_root = Path.cwd()
    datasets_cache = Path(config.get("hf_cache_dir", str(repo_root / "data" / "hf_datasets"))).resolve()
    transformers_cache = Path(config.get("model_cache_dir", str(repo_root / "models" / "hf_models"))).resolve()
    hub_cache = Path(config.get("hf_hub_cache_dir", str(repo_root / "models" / "hf_hub"))).resolve()
    hf_home = (repo_root / ".hf_home").resolve()
    datasets_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    mine_gdrive_root = config.get("mine_gdrive_root")
    if mine_gdrive_root:
        os.environ["MINE_GDRIVE_ROOT"] = str(Path(mine_gdrive_root).expanduser().resolve())

    profile = args.dataset_profile or config.get("dataset_profile", "balanced")
    models = config.get("llm_downloads", ["roberta-large", "distilroberta-base"])

    sources, train_rows, val_rows = resolve_dataset_plan(
        profile=profile,
        max_rows_per_source=args.max_rows_per_source or config.get("max_rows_per_source"),
        sources=config.get("cloud_sources"),
    )

    LOGGER.info("Pre-download started")
    LOGGER.info("Profile=%s", profile)
    print_cache_paths()

    warm_models(models)
    warm_datasets(sources, train_rows, val_rows)

    LOGGER.info("Pre-download complete")
    LOGGER.info("You can now run training without initial model/dataset download delay")


if __name__ == "__main__":
    main()
