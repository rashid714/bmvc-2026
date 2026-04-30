# Save this file exactly as: Makefile
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

# VRAM Optimized Defaults (DINOv2 and RoBERTa base layers are frozen)
BATCH_SIZE ?= 8
EPOCHS ?= 6
NUM_WORKERS ?= 4
# Set massively high so we never truncate the Silver Standard dataset
MAX_ROWS ?= 100000 
# Run all 3 seeds. The VRAM flusher in the Python script will protect the GPU.
SEEDS ?= 41 42 43

# Strict Local Vault Architecture
MODELS_DIR ?= $(PWD)/models
HF_HOME ?= $(MODELS_DIR)/hf_hub
TRANSFORMERS_CACHE ?= $(MODELS_DIR)/hf_hub
HF_DATASETS_CACHE ?= $(MODELS_DIR)/hf_hub
TORCH_HOME ?= $(MODELS_DIR)/torch_hub
MINE_CURATED_ROOT ?= $(PWD)/data/mine_curated

GPU_COUNT := $(shell $(PYTHON) -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
define ensure_cache_dirs
	mkdir -p data/mine_curated/images data/fane models/hf_hub models/torch_hub checkpoints
endef

define export_cache_env
	export HF_HOME="$(HF_HOME)"; \
	export TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)"; \
	export HF_DATASETS_CACHE="$(HF_DATASETS_CACHE)"; \
	export TORCH_HOME="$(TORCH_HOME)"; \
	export MINE_CURATED_ROOT="$(MINE_CURATED_ROOT)";
endef

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "🎓 BMVC 2026 - SILVER STANDARD TRAINING PIPELINE"
	@echo "================================================="
	@echo ""
	@echo "Main targets:"
	@echo "  make install             - Install dependencies"
	@echo "  make preflight           - Create vaults and verify Curated MINE / FANE data"
	@echo "  make predownload         - Cache DINOv2 and RoBERTa foundation models"
	@echo "  make advanced-smoke      - Quick 1-epoch architecture compilation test"
	@echo "  make advanced-single-gpu - Full training on a single GPU"
	@echo "  make advanced-multi-gpu  - Full training using all detected GPUs"
	@echo "  make advanced-cloud      - Auto-picks single/multi-GPU training"
	@echo "  make professor-run       - Balances data and runs reproducible defense run"
	@echo "  make organize-paper      - Compile BMVC-ready artifacts"
	@echo "  make clean               - Remove Python cache junk"
	@echo ""

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
.PHONY: install
install:
	pip install -r requirements.txt
	@if [ -f requirements-inference.txt ]; then pip install -r requirements-inference.txt; fi
	@echo "✅ Dependencies installed."

# -----------------------------------------------------------------------------
# Preflight / readiness
# -----------------------------------------------------------------------------
.PHONY: preflight
preflight:
	@$(ensure_cache_dirs)
	@echo "🔍 Running Silver Standard dataset readiness check..."
	@$(export_cache_env) \
	$(PYTHON) scripts/check_cloud_dataset_ready.py \
		--sources "mine_curated,fane" \
		--train-rows 2 \
		--val-rows 1 \
		--report-path data/source_availability_report.json \
		--output-json data/cloud_dataset_check.json
	@echo "✅ Preflight complete. Vaults secured."

# -----------------------------------------------------------------------------
# Foundation Pre-Cache
# -----------------------------------------------------------------------------
.PHONY: predownload
predownload:
	@$(ensure_cache_dirs)
	@echo "🤖 Downloading DINOv2 and RoBERTa weights to local vaults..."
	@$(export_cache_env) \
	$(PYTHON) scripts/predownload_assets.py
	@echo "✅ Foundation models cached."

# -----------------------------------------------------------------------------
# Advanced multimodal training
# -----------------------------------------------------------------------------
.PHONY: advanced-smoke
advanced-smoke:
	@$(ensure_cache_dirs)
	@echo "💨 Running advanced smoke test (1 Epoch)..."
	@$(export_cache_env) \
	$(PYTHON) scripts/train_advanced_multimodal.py \
		--config $(CONFIG) \
		--output-dir $(SMOKE_OUTPUT_DIR) \
		--epochs 1 \
		--batch-size 4 \
		--max-rows-per-source 20 \
		--seeds 41 \
		--num-workers 0
	@echo "✅ Smoke test complete. Architecture compiled successfully: $(SMOKE_OUTPUT_DIR)"

.PHONY: advanced-single-gpu
advanced-single-gpu:
	@$(ensure_cache_dirs)
	@echo "🚀 Running Silver Standard training (Single GPU)..."
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
		echo "❌ ERROR: advanced-multi-gpu requires at least 2 GPUs, found $(GPU_COUNT)."; \
		exit 1; \
	fi
	@echo "🚀 Running Silver Standard training on $(GPU_COUNT) GPUs..."
	@$(export_cache_env) \
	$(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_advanced_multimodal.py \
		--config $(CONFIG) \
		--output-dir $(OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--max-rows-per-source $(MAX_ROWS) \
		--num-workers $(NUM_WORKERS) \
		--seeds $(SEEDS)

.PHONY: advanced-cloud
advanced-cloud:
	@$(ensure_cache_dirs)
	@echo "🖥️  Detected $(GPU_COUNT) GPU(s)."
	@if [ "$(GPU_COUNT)" -gt 1 ]; then \
		$(MAKE) advanced-multi-gpu CONFIG="$(CONFIG)" OUTPUT_DIR="$(OUTPUT_DIR)" EPOCHS="$(EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" MAX_ROWS="$(MAX_ROWS)" SEEDS="$(SEEDS)"; \
	else \
		$(MAKE) advanced-single-gpu CONFIG="$(CONFIG)" OUTPUT_DIR="$(OUTPUT_DIR)" EPOCHS="$(EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" MAX_ROWS="$(MAX_ROWS)" NUM_WORKERS="$(NUM_WORKERS)" SEEDS="$(SEEDS)"; \
	fi

# -----------------------------------------------------------------------------
# Professor one-command run
# -----------------------------------------------------------------------------
.PHONY: balance-fane
balance-fane:
	@echo "⚖️  Balancing FANE dataset splits..."
	@$(PYTHON) scripts/balance_fane.py

.PHONY: professor-run
professor-run: balance-fane
	@$(ensure_cache_dirs)
	@mkdir -p $(PROFESSOR_OUTPUT_DIR)
	@$(PYTHON) -c "import torch,sys; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 1)" \
	|| (echo "❌ ERROR: CUDA is not available. Install CUDA-enabled PyTorch and try again." && exit 1)
	@$(export_cache_env) \
	if [ ! -d "$$MINE_CURATED_ROOT" ]; then \
		echo "⚠️ WARNING: Curated dataset not found at $$MINE_CURATED_ROOT"; \
	else \
		echo "✅ Using Pure MINE dataset from: $$MINE_CURATED_ROOT"; \
	fi; \
	if [ "$(GPU_COUNT)" -gt 1 ]; then \
		$(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_advanced_multimodal.py \
			--config $(CONFIG) \
			--batch-size $(BATCH_SIZE) \
			--epochs $(EPOCHS) \
			--output-dir $(PROFESSOR_OUTPUT_DIR); \
	else \
		$(PYTHON) scripts/train_advanced_multimodal.py \
			--config $(CONFIG) \
			--batch-size $(BATCH_SIZE) \
			--epochs $(EPOCHS) \
			--output-dir $(PROFESSOR_OUTPUT_DIR); \
	fi
	@echo "🏆 Professor run complete: $(PROFESSOR_OUTPUT_DIR)"

# -----------------------------------------------------------------------------
# Research organization
# -----------------------------------------------------------------------------
.PHONY: organize-paper
organize-paper:
	@echo "📄 Organizing BMVC research paper data..."
	@$(PYTHON) scripts/organize_paper_data.py $(PROFESSOR_OUTPUT_DIR) research_paper_data
	@echo "✅ Paper folder organized."

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true
	rm -rf .pytest_cache .mypy_cache
	@echo "🧹 Clean complete."
