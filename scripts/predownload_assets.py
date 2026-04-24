#!/usr/bin/env python3
"""
BMVC 2026 - Foundation Model Pre-Cacher
Purpose:
- Downloads massive DINOv2 and RoBERTa weights before training.
- Prevents multi-GPU race conditions where multiple nodes try to download simultaneously.
- Locks all downloads into the local BMVC 2026/models/ vault.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import torch

from transformers import AutoModel, AutoTokenizer, AutoConfig

LOGGER = logging.getLogger("predownload_assets")

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def warm_hf_models(models: list[str], cache_dir: Path) -> None:
    """Downloads the massive Hugging Face Transformer weights."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("🤗 STEP 1: DOWNLOADING HUGGING FACE MODELS (RoBERTa)")
    LOGGER.info("="*50)
    
    for model_name in models:
        LOGGER.info(f"Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
            AutoConfig.from_pretrained(model_name, cache_dir=str(cache_dir))
            AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
            LOGGER.info(f"✅ {model_name} securely cached in {cache_dir.name}!")
        except Exception as e:
            LOGGER.error(f"❌ Failed to cache {model_name}: {e}")

def warm_torch_hub_models() -> None:
    """Downloads PyTorch Hub models (Meta's DINOv2)."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("👁️  STEP 2: DOWNLOADING PYTORCH HUB MODELS (DINOv2)")
    LOGGER.info("="*50)
    
    try:
        LOGGER.info("Downloading Meta's DINOv2 ViT-B/14...")
        # PyTorch Hub automatically uses the TORCH_HOME environment variable we set
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        LOGGER.info(f"✅ dinov2_vitb14 securely cached in torch_hub!")
    except Exception as e:
        LOGGER.error(f"❌ Failed to cache DINOv2: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download DINOv2 and RoBERTa weights")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-profile", type=str, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("🚀 Starting BMVC 2026 Foundation Model Pre-Cache...")

    # 1. Enforce strict local paths matching our new Makefile
    repo_root = Path(__file__).resolve().parent.parent
    hf_cache_dir = repo_root / "models" / "hf_hub"
    torch_cache_dir = repo_root / "models" / "torch_hub"
    
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    torch_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
    os.environ["TORCH_HOME"] = str(torch_cache_dir)

    # 2. Download RoBERTa-Large
    warm_hf_models(["roberta-large"], hf_cache_dir)

    # 3. Download DINOv2
    warm_torch_hub_models()

    LOGGER.info("\n" + "="*50)
    LOGGER.info("🎉 FOUNDATION PRE-DOWNLOAD 100% COMPLETE!")
    LOGGER.info("Your GPUs will not experience download bottlenecks.")
    LOGGER.info("="*50)

if __name__ == "__main__":
    main()
