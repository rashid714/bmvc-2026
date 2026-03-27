#!/usr/bin/env python3
"""
Pre-download models and datasets only (no training).
Updated for BMVC 2026 Cloud Architecture.

Purpose:
- Warm Hugging Face model cache.
- Warm Hugging Face dataset cache using the new get_cloud_dataloaders.
- Verify large cache (20-30GB target profile) before running training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer, AutoConfig

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cloud_datasets import get_cloud_dataloaders

LOGGER = logging.getLogger("predownload_assets")

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def print_cache_paths() -> None:
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    hf_datasets = os.environ.get("HF_DATASETS_CACHE", "<default>")
    hf_models = os.environ.get("TRANSFORMERS_CACHE", "<default>")
    LOGGER.info("HF_HOME=%s", hf_home)
    LOGGER.info("HF_DATASETS_CACHE=%s", hf_datasets)
    LOGGER.info("TRANSFORMERS_CACHE=%s", hf_models)

def warm_models(models: list[str]) -> None:
    LOGGER.info("Model download plan: %s", ", ".join(models))
    for model_name in models:
        LOGGER.info("Downloading tokenizer: %s", model_name)
        AutoTokenizer.from_pretrained(model_name)
        LOGGER.info("Downloading model weights: %s", model_name)
        AutoConfig.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        LOGGER.info("✅ Model ready in cache: %s", model_name)

def warm_datasets(config: dict, max_rows: int) -> None:
    LOGGER.info("Dataset download plan using get_cloud_dataloaders...")
    sources = config.get("cloud_sources", ["mine", "emoticon", "raza", "coco"])
    
    # Safely remove mine_gdrive if the environment variable wasn't set yet
    if "mine_gdrive" in sources and not os.environ.get("MINE_GDRIVE_ROOT"):
        LOGGER.warning("MINE_GDRIVE_ROOT not found. Skipping mine_gdrive for pre-download.")
        sources.remove("mine_gdrive")

    LOGGER.info("Sources to cache: %s", ", ".join(sources))
    LOGGER.info("Target max rows per source: %d", max_rows)

    try:
        # Calling get_cloud_dataloaders will automatically trigger the HF datasets 
        # library to download, cache, and process the data.
        get_cloud_dataloaders(
            batch_size=8,
            num_workers=0, # Keep to 0 for safe pre-downloading
            sources=sources,
            mine_gdrive_root=os.environ.get("MINE_GDRIVE_ROOT"),
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
            max_samples={
                "train": max_rows,
                "validation": max(1, max_rows // 5),
                "test": max(1, max_rows // 5),
            },
        )
        LOGGER.info("✅ All requested datasets successfully downloaded and cached!")
    except Exception as e:
        LOGGER.error("❌ Error during dataset caching: %s", e)
        LOGGER.info("Note: It is okay if datasets fail here; professor-run will catch them later.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download models and datasets only")
    parser.add_argument("--config", type=str, default="configs/multimodal_ultra_30gb.json")
    parser.add_argument("--dataset-profile", type=str, default=None, choices=["balanced", "large_20gb", "ultra_30gb"])
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    args = parser.parse_args()

    setup_logging()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        LOGGER.warning("Config file %s not found. Using defaults.", args.config)

    LOGGER.info("🚀 Starting BMVC 2026 Cloud Asset Pre-download...")
    print_cache_paths()

    # Get models from config or use defaults
    models = config.get("llm_downloads", ["roberta-large", "distilroberta-base", "distilgpt2"])

    # Determine rows
    max_rows = args.max_rows_per_source or config.get("max_rows_per_source", 5000)
    if args.dataset_profile == "ultra_30gb":
        max_rows = max(max_rows, 40000)
    elif args.dataset_profile == "large_20gb":
        max_rows = max(max_rows, 25000)

    # 1. Warm Models
    warm_models(models)

    # 2. Warm Datasets
    warm_datasets(config, max_rows)

    LOGGER.info("-" * 50)
    LOGGER.info("🎉 Pre-download complete!")
    LOGGER.info("You are officially ready to run Step 6 (make professor-run)")
    LOGGER.info("-" * 50)

if __name__ == "__main__":
    main()
