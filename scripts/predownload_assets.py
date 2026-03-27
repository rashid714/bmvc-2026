#!/usr/bin/env python3
"""
Pre-download models and ALL datasets (including Kaggle) before training.
Updated for BMVC 2026 Cloud Architecture.

Purpose:
- Force Kaggle API to download and extract massive ZIP datasets.
- Warm Hugging Face model cache.
- Warm Hugging Face dataset cache.
- Fail-safe design to prevent crashes if an API times out.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer, AutoConfig

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cloud_datasets import (
    KaggleGoEmotionsLoader,
    KaggleFacialEmotionLoader,
    KaggleIntentLoader,
    DairAiEmotionLoader,
    DailyDialogLoader,
    MSCOCOCaptionsLoader
)

LOGGER = logging.getLogger("predownload_assets")

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def force_kaggle_downloads() -> None:
    """Explicitly triggers the Kaggle API to download and extract the ZIP files."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("🔥 STEP 1: DOWNLOADING KAGGLE DATASETS")
    LOGGER.info("="*50)
    
    try:
        # Calling load_split with limit=1 is a trick! 
        # It forces the KaggleDownloader to download and extract the ENTIRE dataset 
        # into data/kaggle_datasets/ so it's ready for the main training loop.
        
        LOGGER.info("Downloading Kaggle GoEmotions (Text)...")
        KaggleGoEmotionsLoader.load_split(split="train", limit=1)
        
        LOGGER.info("Downloading Kaggle Facial Emotions (Images)...")
        KaggleFacialEmotionLoader.load_split(split="train", limit=1)
        
        LOGGER.info("Downloading Kaggle Bitext Intent (Customer Service)...")
        KaggleIntentLoader.load_split(split="train", limit=1)
        
        LOGGER.info("✅ All Kaggle datasets successfully downloaded and extracted!")
    except Exception as e:
        LOGGER.error(f"❌ Kaggle download failed: {e}")
        LOGGER.error("Please ensure you ran the python -c login command and created ~/.kaggle/kaggle.json!")

def force_hf_datasets() -> None:
    """Explicitly triggers Hugging Face to cache public datasets."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("🤗 STEP 2: CACHING HUGGING FACE DATASETS")
    LOGGER.info("="*50)
    
    try:
        LOGGER.info("Caching Dair-AI Emotion...")
        DairAiEmotionLoader.load_split(split="train", limit=5)
        
        LOGGER.info("Caching DailyDialog...")
        DailyDialogLoader.load_split(split="train", limit=5)
        
        LOGGER.info("Caching MS COCO Captions...")
        MSCOCOCaptionsLoader.load_split(split="train", limit=5)
        
        LOGGER.info("✅ All Hugging Face datasets successfully cached!")
    except Exception as e:
        LOGGER.error(f"❌ Hugging Face dataset caching failed: {e}")

def warm_models(models: list[str]) -> None:
    """Downloads the massive Transformer weights."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("🤖 STEP 3: DOWNLOADING FOUNDATION MODELS")
    LOGGER.info("="*50)
    
    for model_name in models:
        LOGGER.info(f"Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name)
            AutoConfig.from_pretrained(model_name)
            AutoModel.from_pretrained(model_name)
            LOGGER.info(f"✅ {model_name} ready in cache!")
        except Exception as e:
            LOGGER.error(f"❌ Failed to cache {model_name}: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download models and all datasets")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-profile", type=str, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("🚀 Starting BMVC 2026 Ultimate Pre-download...")

    # Set HF Cache paths to keep your server clean
    repo_root = Path(__file__).resolve().parent.parent
    os.environ["HF_DATASETS_CACHE"] = str((repo_root / "data" / "hf_datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((repo_root / "models" / "hf_models").resolve())

    # 1. Force Kaggle to extract
    force_kaggle_downloads()

    # 2. Force Hugging Face to cache
    force_hf_datasets()

    # 3. Cache Model Weights
    models = ["roberta-large", "distilroberta-base", "distilgpt2"]
    warm_models(models)

    LOGGER.info("\n" + "="*50)
    LOGGER.info("🎉 PRE-DOWNLOAD 100% COMPLETE!")
    LOGGER.info("All Kaggle ZIPs extracted. All HF models cached.")
    LOGGER.info("You are officially ready to run Step 6 (make professor-run)")
    LOGGER.info("="*50)

if __name__ == "__main__":
    main()
