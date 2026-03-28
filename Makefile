# Save this file as: Makefile
SHELL := /bin/bash
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Configurable variables
# -----------------------------------------------------------------------------
PYTHON ?= python
TORCHRUN ?= torchrun

CONFIG ?= configs/multimodal_cloud.json
ULTRA_CONFIG ?= configs/multimodal_ultra_30gb.json

OUTPUT_DIR ?= checkpoints/advanced-results-final
SMOKE_OUTPUT_DIR ?= checkpoints/advanced-smoke
PROFESSOR_OUTPUT_DIR ?= checkpoints/professor-run
BEST_OUTPUT_DIR ?= checkpoints/advanced-best-ultra

BATCH_SIZE ?= 8
EPOCHS ?= 4
NUM_WORKERS ?= 2
MAX_ROWS ?= 2000
SEEDS ?= 41 42 43

HF_DATASETS_CACHE ?= $(PWD)/data/hf_datasets
TRANSFORMERS_CACHE ?= $(PWD)/models/hf_models
HF_HOME ?= $(PWD)/models/hf_hub
MINE_GDRIVE_ROOT ?= $(PWD)/data/mine_gdrive

GPU_COUNT := $(shell $(PYTHON) -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
define ensure_cache_dirs
    mkdir -p data/hf_datasets models/hf_models models/hf_hub checkpoints
endef

define export_cache_env
    export HF_DATASETS_CACHE="$(HF_DATASETS_CACHE)"; \
    export TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)"; \
    export HF_HOME="$(HF_HOME)"; \
    export MINE_GDRIVE_ROOT="$(MINE_GDRIVE_ROOT)";
endef

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
.PHONY: help
help:
    @echo "BEAR BMVC 2026 - Updated Makefile"
    @echo "================================="
    @echo ""
    @echo "Main targets:"
    @echo "  make install             - Install dependencies"
    @echo "  make preflight           - Create cache dirs and run lightweight readiness check"
    @echo "  make advanced-smoke      - Quick 1-epoch advanced multimodal smoke test"
    @echo "  make advanced-single-gpu - Advanced training on a single GPU"
    @echo "  make advanced-multi-gpu  - Advanced training using all detected GPUs"
    @echo "  make advanced-cloud      - Auto-picks single/multi-GPU advanced training"
    @echo "  make predownload-ultra   - Pre-download datasets and models only"
    @echo "  make advanced-best       - Bigger advanced run using ultra config"
    @echo "  make professor-run       - One-command reproducible advanced run"
    @echo "  make organize-paper      - Organize paper-ready artifacts"
    @echo "  make clean               - Remove Python/artifact junk"
    @echo ""
    @echo "Configurable variables (examples):"
    @echo "  make advanced-single-gpu BATCH_SIZE=4 EPOCHS=2 MAX_ROWS=500"
    @echo "  make professor-run CONFIG=configs/multimodal_cloud.json"
    @echo ""

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
.PHONY: install
install:
    pip install -r requirements.txt
    @if [ -f requirements-inference.txt ]; then pip install -r requirements-inference.txt; fi
    @echo "Dependencies installed."

# -----------------------------------------------------------------------------
# Preflight / readiness
# -----------------------------------------------------------------------------
.PHONY: preflight
preflight:
    @$(ensure_cache_dirs)
    @mkdir -p data
    @echo "Running lightweight dataset/cache readiness check..."
    @$(export_cache_env) \
    $(PYTHON) scripts/check_cloud_dataset_ready.py \
        --run-twice \
        --sources "mine,kaggle_goemotions,kaggle_facial,kaggle_intent,hf_emotion,hf_dailydialog,hf_coco" \
        --train-rows 2 \
        --val-rows 1 \
        --cache-dir data/hf_datasets \
        --report-path data/source_availability_report.json \
        --output-json data/cloud_dataset_check.json
    @echo "Preflight complete."

# -----------------------------------------------------------------------------
# Advanced multimodal training
# -----------------------------------------------------------------------------
.PHONY: advanced-smoke
advanced-smoke:
    @$(ensure_cache_dirs)
    @echo "Running advanced smoke test..."
    @$(export_cache_env) \
    $(PYTHON) scripts/train_advanced_multimodal.py \
        --config $(CONFIG) \
        --output-dir $(SMOKE_OUTPUT_DIR) \
        --epochs 1 \
        --batch-size 4 \
        --max-rows-per-source 20 \
        --seeds 41 \
        --num-workers 0
    @echo "Smoke test complete. Check: $(SMOKE_OUTPUT_DIR)"

.PHONY: advanced-single-gpu
advanced-single-gpu:
    @$(ensure_cache_dirs)
    @echo "Running advanced single-GPU training..."
    @$(export_cache_env) \
    $(PYTHON) scripts/train_advanced_multimodal.py \
        --config $(CONFIG) \
        --output-dir $(OUTPUT_DIR) \
        --epochs $(EPOCHS) \
        --batch-size $(BATCH_SIZE) \
        --max-rows-per-source $(MAX_ROWS) \
        --num-workers $(NUM_WORKERS) \
        --seeds $(SEEDS)

.PHONY: advanced-multi-gpu
advanced-multi-gpu:
    @$(ensure_cache_dirs)
    @if [ "$(GPU_COUNT)" -lt 2 ]; then \
        echo "ERROR: advanced-multi-gpu requires at least 2 GPUs, found $(GPU_COUNT)."; \
        exit 1; \
    fi
    @echo "Running advanced multi-GPU training on $(GPU_COUNT) GPUs..."
    @$(export_cache_env) \
    $(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_advanced_multimodal.py \
        --config $(CONFIG) \
        --output-dir $(OUTPUT_DIR) \
        --epochs $(EPOCHS) \
        --batch-size $(BATCH_SIZE) \
        --max-rows-per-source $(MAX_ROWS) \
        --num-workers 4 \
        --seeds $(SEEDS)

.PHONY: advanced-cloud
advanced-cloud:
    @$(ensure_cache_dirs)
    @echo "Detected $(GPU_COUNT) GPU(s)."
    @if [ "$(GPU_COUNT)" -gt 1 ]; then \
        $(MAKE) advanced-multi-gpu CONFIG="$(CONFIG)" OUTPUT_DIR="$(OUTPUT_DIR)" EPOCHS="$(EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" MAX_ROWS="$(MAX_ROWS)" SEEDS="$(SEEDS)"; \
    else \
        $(MAKE) advanced-single-gpu CONFIG="$(CONFIG)" OUTPUT_DIR="$(OUTPUT_DIR)" EPOCHS="$(EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" MAX_ROWS="$(MAX_ROWS)" NUM_WORKERS="$(NUM_WORKERS)" SEEDS="$(SEEDS)"; \
    fi

# -----------------------------------------------------------------------------
# Pre-download only
# -----------------------------------------------------------------------------
.PHONY: predownload-ultra
predownload-ultra:
    @$(ensure_cache_dirs)
    @echo "Pre-downloading datasets and models only..."
    @$(export_cache_env) \
    $(PYTHON) scripts/predownload_assets.py \
        --config $(ULTRA_CONFIG) \
        --dataset-profile ultra_30gb \
        --max-rows-per-source 40000
    @echo "Pre-download complete."

# -----------------------------------------------------------------------------
# Bigger / best run
# -----------------------------------------------------------------------------
.PHONY: advanced-best
advanced-best:
    @$(ensure_cache_dirs)
    @echo "Starting advanced best run..."
    @if [ "$(GPU_COUNT)" -gt 1 ]; then \
        $(export_cache_env) \
        $(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_advanced_multimodal.py \
            --config $(ULTRA_CONFIG) \
            --dataset-profile ultra_30gb \
            --output-dir $(BEST_OUTPUT_DIR) \
            --epochs 5 \
            --batch-size 8 \
            --max-rows-per-source 40000 \
            --num-workers 4 \
            --seeds $(SEEDS); \
    else \
        $(export_cache_env) \
        $(PYTHON) scripts/train_advanced_multimodal.py \
            --config $(ULTRA_CONFIG) \
            --dataset-profile ultra_30gb \
            --output-dir $(BEST_OUTPUT_DIR) \
            --epochs 5 \
            --batch-size 8 \
            --max-rows-per-source 30000 \
            --num-workers 2 \
            --seeds $(SEEDS); \
    fi
    @echo "Advanced best run complete."

# -----------------------------------------------------------------------------
# Professor one-command run
# -----------------------------------------------------------------------------
.PHONY: professor-run
professor-run:
    @$(ensure_cache_dirs)
    @mkdir -p $(PROFESSOR_OUTPUT_DIR)
    @$(PYTHON) -c "import torch,sys; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || (echo "ERROR: CUDA is not available. Install CUDA-enabled PyTorch and try again." && exit 1)
    @$(export_cache_env) \
    if [ ! -d "$$MINE_GDRIVE_ROOT" ]; then \
        echo "WARNING: MINE dataset not found at $$MINE_GDRIVE_ROOT"; \
        echo "The run will continue using other sources if available."; \
    else \
        echo "Using MINE dataset from: $$MINE_GDRIVE_ROOT"; \
    fi; \
    $(PYTHON) scripts/check_cloud_dataset_ready.py \
        --run-twice \
        --sources "mine,kaggle_goemotions,kaggle_facial,kaggle_intent,hf_emotion,hf_dailydialog,hf_coco" \
        --train-rows 2 \
        --val-rows 1 \
        --cache-dir data/hf_datasets \
        --report-path data/source_availability_report.json \
        --output-json data/cloud_dataset_check.json; \
    if [ "$(GPU_COUNT)" -gt 1 ]; then \
        $(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_advanced_multimodal.py \
            --config $(CONFIG) \
            --batch-size 8 \
            --epochs 10 \
            --output-dir $(PROFESSOR_OUTPUT_DIR); \
    else \
        $(PYTHON) scripts/train_advanced_multimodal.py \
            --config $(CONFIG) \
            --batch-size 8 \
            --epochs 10 \
            --output-dir $(PROFESSOR_OUTPUT_DIR); \
    fi
    @echo "Professor run complete: $(PROFESSOR_OUTPUT_DIR)"

# -----------------------------------------------------------------------------
# Research organization
# -----------------------------------------------------------------------------
.PHONY: organize-paper
organize-paper:
    @echo "Organizing research paper data..."
    @$(PYTHON) scripts/organize_paper_data.py $(PROFESSOR_OUTPUT_DIR) research_paper_data
    @echo "Paper folder organized."

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + || true
    find . -type f -name "*.pyc" -delete || true
    rm -rf .pytest_cache .mypy_cache
    @echo "Clean complete."
