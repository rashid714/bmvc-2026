# ============================================================================
# BEAR BMVC 2026 Research System - Corrected Makefile
# Save this file as: Makefile
# IMPORTANT: commands under each target must start with a TAB, not spaces.
# ============================================================================

SHELL := /bin/bash

# Change this if you want a different official PyTorch CUDA wheel.
# Example alternatives from the official PyTorch selector include cu126 / cu128 etc.
PYTORCH_CUDA_INDEX ?= https://download.pytorch.org/whl/cu126

.PHONY: help install install-gpu install-cpu doctor-gpu demo train \
        real-data real-train real-train-bear real-train-tri \
        cloud-train cloud-test \
        multimodal-smoke multimodal-single-gpu multimodal-multi-gpu multimodal-cloud \
        multimodal-ablation-no-reliability multimodal-ablation-text-only \
        advanced-smoke advanced-single-gpu advanced-multi-gpu advanced-cloud advanced-best \
        predownload-ultra professor-run \
        aws-help aws-setup aws-verify aws-smoke aws-train aws-download-results aws-stop aws-status \
        real-train-2 real-train-ml real-dashboard benchmark evaluate serve test-api \
        ablation baselines paper research clean all organize-paper

help:
    @echo "BEAR BMVC 2026 Research System - Makefile"
    @echo "========================================="
    @echo ""
    @echo "Recommended first-time GPU setup:"
    @echo "  make install"
    @echo "  make doctor-gpu"
    @echo "  make professor-run"
    @echo ""
    @echo "Available targets:"
    @echo "  make install              - Install dependencies with CUDA-enabled PyTorch"
    @echo "  make install-gpu          - Same as install"
    @echo "  make install-cpu          - Install CPU-only PyTorch"
    @echo "  make doctor-gpu           - Check PyTorch/CUDA/GPU import health"
    @echo "  make demo                 - Run Streamlit demo app"
    @echo "  make train                - Train world-class model"
    @echo "  make real-data            - Download real public datasets"
    @echo "  make real-train           - Real fine-tuning on GoEmotions"
    @echo "  make real-train-bear      - Train BEAR variant on GoEmotions"
    @echo "  make real-train-tri       - Train BEAR tri-task (emotion+intention+action)"
    @echo "  make cloud-train          - Cloud-ready distributed BEAR tri-task training (text-only)"
    @echo "  make cloud-test           - Test cloud checkpoint on held-out data"
    @echo ""
    @echo "MULTIMODAL CLOUD TRAINING:"
    @echo "  make multimodal-smoke"
    @echo "  make multimodal-single-gpu"
    @echo "  make multimodal-multi-gpu"
    @echo "  make multimodal-cloud"
    @echo ""
    @echo "ADVANCED MULTIMODAL:"
    @echo "  make advanced-smoke"
    @echo "  make advanced-single-gpu"
    @echo "  make advanced-multi-gpu"
    @echo "  make advanced-cloud"
    @echo "  make advanced-best"
    @echo "  make predownload-ultra"
    @echo "  make professor-run"
    @echo ""
    @echo "AWS EC2:"
    @echo "  make aws-help"
    @echo "  make aws-setup"
    @echo "  make aws-verify"
    @echo "  make aws-smoke"
    @echo "  make aws-train"
    @echo "  make aws-download-results"
    @echo "  make aws-status"
    @echo ""
    @echo "RESEARCH & PAPER:"
    @echo "  make ablation"
    @echo "  make baselines"
    @echo "  make paper"
    @echo "  make research"
    @echo "  make organize-paper"
    @echo ""
    @echo "UTILITY:"
    @echo "  make clean"

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------

install: install-gpu

install-gpu:
    @echo "Installing CUDA-enabled PyTorch from official wheel index: $(PYTORCH_CUDA_INDEX)"
    python -m pip install --upgrade pip setuptools wheel
    pip install torch torchvision torchaudio --index-url $(PYTORCH_CUDA_INDEX)
    pip install -r requirements.txt
    @if [ -f requirements-inference.txt ]; then pip install -r requirements-inference.txt; fi
    @echo "Dependencies installed with CUDA-enabled PyTorch."

install-cpu:
    @echo "Installing CPU-only PyTorch"
    python -m pip install --upgrade pip setuptools wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    @if [ -f requirements-inference.txt ]; then pip install -r requirements-inference.txt; fi
    @echo "Dependencies installed with CPU-only PyTorch."

doctor-gpu:
    @echo "=== GPU / PyTorch Doctor ==="
    @echo "Python: $$(python --version)"
    @echo "Pip: $$(pip --version)"
    @echo ""
    @echo "nvidia-smi:"
    @which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
    @echo ""
    @echo "Torch import test:"
    @unset LD_LIBRARY_PATH; \
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); print('GPU 0:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 0: none'); print('torch path:', torch.__file__)"
    @echo ""
    @echo "If the import above fails with an NCCL symbol error, your runtime is still mismatched."

# ----------------------------------------------------------------------------
# Basic targets
# ----------------------------------------------------------------------------

demo:
    streamlit run app.py

train:
    python scripts/train.py --config world-class

real-data:
    python scripts/download_real_datasets.py --output-dir data/real

real-train:
    python scripts/train_real_emotion.py --output-dir checkpoints/goemotions-real

real-train-bear:
    python scripts/train_bear_goemotions.py --output-dir checkpoints/bear-goemotions-real

real-train-tri:
    python scripts/train_bear_intention_real.py --output-dir checkpoints/bear-tritask-real --seeds 41,42,43

cloud-train:
    bash scripts/run_cloud_supercomputer.sh configs/cloud_supercomputer.json 1

cloud-test:
    python scripts/test_bear_intention_cloud.py --checkpoint checkpoints/bear-tritask-cloud/seed_41/best_model.pt --output-json checkpoints/bear-tritask-cloud/test_metrics.json

# ----------------------------------------------------------------------------
# Multimodal cloud training
# ----------------------------------------------------------------------------

multimodal-smoke:
    python scripts/train_multimodal_cloud.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/multimodal-smoke \
        --epochs 1 --batch-size 8 --max-rows-per-source 100 --seeds 41 --num-workers 0

multimodal-single-gpu:
    python scripts/train_multimodal_cloud.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/multimodal-single-gpu \
        --epochs 4 --batch-size 16 --max-rows-per-source 2000 --num-workers 2

multimodal-multi-gpu:
    torchrun --nproc_per_node=4 \
        scripts/train_multimodal_cloud.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/multimodal-multi-gpu \
        --epochs 4 --batch-size 32 --max-rows-per-source 5000 --num-workers 4

multimodal-cloud:
    bash scripts/run_multimodal_supercomputer.sh configs/multimodal_cloud.json 4 checkpoints/multimodal-full

multimodal-ablation-no-reliability:
    python scripts/train_multimodal_cloud.py \
        --output-dir checkpoints/ablations/multimodal-no-reliability \
        --epochs 2 --batch-size 16 --seeds 41 42

multimodal-ablation-text-only:
    python scripts/train_bear_intention_cloud.py \
        --output-dir checkpoints/ablations/text-only \
        --epochs 2 --batch-size 16 --seeds 41 42

# ----------------------------------------------------------------------------
# Advanced multimodal targets
# ----------------------------------------------------------------------------

advanced-smoke:
    @echo "Running ADVANCED smoke test (dual-layer LLM + PDF)..."
    python scripts/train_advanced_multimodal.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/advanced-smoke \
        --epochs 1 --batch-size 4 --max-rows-per-source 20 \
        --seeds 41 --num-workers 0
    @echo "Smoke test complete."

advanced-single-gpu:
    @echo "Advanced training (single GPU, dual-layer LLM)..."
    python scripts/train_advanced_multimodal.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/advanced-single-gpu \
        --epochs 4 --batch-size 16 --max-rows-per-source 2000 --num-workers 2 \
        --seeds 41 42 43
    @echo "Training complete."

advanced-multi-gpu:
    @echo "Advanced training (4x GPUs, dual-layer LLM)..."
    torchrun --nproc_per_node=4 \
        scripts/train_advanced_multimodal.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/advanced-multi-gpu \
        --epochs 4 --batch-size 32 --max-rows-per-source 5000 --num-workers 4 \
        --seeds 41 42 43
    @echo "Training complete."

advanced-cloud:
    @echo "ADVANCED CLOUD TRAINING (Dual-Layer LLM + Auto-PDF Reports)"
    @echo "Architecture: RoBERTa-large + DistilGPT2 dual-layer fusion"
    @GPU_COUNT=$$(nvidia-smi -L | wc -l); \
    if [ $$GPU_COUNT -gt 1 ]; then \
        echo "Detected $$GPU_COUNT GPUs - using distributed training..."; \
        unset LD_LIBRARY_PATH; \
        torchrun --nproc_per_node=$$GPU_COUNT scripts/train_advanced_multimodal.py \
            --config configs/multimodal_cloud.json \
            --output-dir checkpoints/advanced-results-final \
            --epochs 4 --batch-size 32 --max-rows-per-source 5000 --num-workers 4 \
            --seeds 41 42 43; \
    else \
        echo "Using single GPU..."; \
        unset LD_LIBRARY_PATH; \
        python scripts/train_advanced_multimodal.py \
            --config configs/multimodal_cloud.json \
            --output-dir checkpoints/advanced-results-final \
            --epochs 4 --batch-size 16 --max-rows-per-source 2500 --num-workers 2 \
            --seeds 41 42 43; \
    fi
    @echo ""
    @echo "ADVANCED TRAINING COMPLETE"
    @echo "Results: checkpoints/advanced-results-final"

advanced-best:
    @echo "Best profile: ultra_30gb (target cache 20-30GB)"
    @echo "LLMs: roberta-large + distilroberta-base"
    @mkdir -p .hf_cache
    @GPU_COUNT=$$(nvidia-smi -L | wc -l); \
    if [ $$GPU_COUNT -gt 1 ]; then \
        echo "Detected $$GPU_COUNT GPUs, using distributed mode"; \
        unset LD_LIBRARY_PATH; \
        HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets TRANSFORMERS_CACHE=.hf_cache/transformers \
        torchrun --nproc_per_node=$$GPU_COUNT scripts/train_advanced_multimodal.py \
            --config configs/multimodal_ultra_30gb.json \
            --dataset-profile ultra_30gb \
            --output-dir checkpoints/advanced-best-ultra \
            --epochs 5 --batch-size 32 --max-rows-per-source 40000 --num-workers 4 \
            --seeds 41 42 43; \
    else \
        echo "Using single GPU mode"; \
        unset LD_LIBRARY_PATH; \
        HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets TRANSFORMERS_CACHE=.hf_cache/transformers \
        python scripts/train_advanced_multimodal.py \
            --config configs/multimodal_ultra_30gb.json \
            --dataset-profile ultra_30gb \
            --output-dir checkpoints/advanced-best-ultra \
            --epochs 5 --batch-size 16 --max-rows-per-source 30000 --num-workers 2 \
            --seeds 41 42 43; \
    fi
    @echo "Run complete"
    @echo "Results: checkpoints/advanced-best-ultra"
    @echo "Paper folder: checkpoints/research_paper_data"

predownload-ultra:
    @echo "Pre-download mode: models + datasets only (no training)"
    @mkdir -p .hf_cache
    HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets TRANSFORMERS_CACHE=.hf_cache/transformers \
    python scripts/predownload_assets.py \
        --config configs/multimodal_ultra_30gb.json \
        --dataset-profile ultra_30gb \
        --max-rows-per-source 40000
    @echo "Cache usage:"
    @du -sh .hf_cache || true

# ----------------------------------------------------------------------------
# Professor one-command run (corrected)
# ----------------------------------------------------------------------------

professor-run:
    @echo "Professor one-command run (strict + reproducible)"
    @mkdir -p data/hf_datasets models/hf_models models/hf_hub checkpoints/professor-run
    @unset LD_LIBRARY_PATH; \
    python -c "import torch,sys; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || (echo 'ERROR: PyTorch/CUDA import failed or CUDA is not available. Run: make doctor-gpu' && exit 1)
    @CONFIG_PATH="configs/multimodal_cloud.json"; \
    SOURCE_LIST="mine_gdrive,mine,emoticon,raza,coco"; \
    if [ -z "$$MINE_GDRIVE_ROOT" ] || [ ! -d "$$MINE_GDRIVE_ROOT" ]; then \
        echo "MINE_GDRIVE_ROOT not set/found -> running without mine_gdrive source"; \
        python -c "import json, pathlib; cfg=json.load(open('configs/multimodal_cloud.json','r',encoding='utf-8')); cfg['cloud_sources']=[s for s in cfg.get('cloud_sources',[]) if s!='mine_gdrive']; pathlib.Path('/tmp/multimodal_cloud_nominegdrive.json').write_text(json.dumps(cfg), encoding='utf-8'); print('Wrote /tmp/multimodal_cloud_nominegdrive.json')"; \
        CONFIG_PATH="/tmp/multimodal_cloud_nominegdrive.json"; \
        SOURCE_LIST="mine,emoticon,raza,coco"; \
    else \
        echo "Using mine_gdrive source from MINE_GDRIVE_ROOT=$$MINE_GDRIVE_ROOT"; \
    fi; \
    export HF_DATASETS_CACHE="$$PWD/data/hf_datasets"; \
    export TRANSFORMERS_CACHE="$$PWD/models/hf_models"; \
    export HF_HOME="$$PWD/models/hf_hub"; \
    python scripts/check_cloud_dataset_ready.py \
        --run-twice \
        --sources "$$SOURCE_LIST" \
        --train-rows 2 --val-rows 1 \
        --cache-dir data/hf_datasets \
        --report-path data/source_availability_report.json \
        --output-json data/cloud_dataset_check.json; \
    GPU_COUNT=$$(python -c "import torch; print(torch.cuda.device_count())"); \
    echo "Using $$GPU_COUNT CUDA GPU(s)"; \
    if [ "$$GPU_COUNT" -gt 1 ]; then \
        unset LD_LIBRARY_PATH; \
        torchrun --nproc_per_node=$$GPU_COUNT scripts/train_multimodal_cloud.py \
            --config "$$CONFIG_PATH" \
            --strict-preflight \
            --output-dir checkpoints/professor-run; \
    else \
        unset LD_LIBRARY_PATH; \
        python scripts/train_multimodal_cloud.py \
            --config "$$CONFIG_PATH" \
            --strict-preflight \
            --output-dir checkpoints/professor-run; \
    fi
    @echo "Complete: checkpoints/professor-run"

# ----------------------------------------------------------------------------
# AWS EC2 deployment
# ----------------------------------------------------------------------------

aws-setup:
    @echo "Setting up AWS environment..."
    wget -q https://raw.githubusercontent.com/YOUR-USERNAME/bmvc-2026/main/setup_aws.py -O setup_aws.py
    python3 setup_aws.py
    @echo "AWS environment ready."

aws-verify:
    @echo "Verifying AWS setup..."
    nvidia-smi -L | wc -l | xargs -I {} echo "{} GPUs detected"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

aws-smoke:
    @echo "Running AWS smoke test (1 epoch)..."
    python scripts/train_multimodal_cloud.py \
        --config configs/multimodal_cloud.json \
        --output-dir checkpoints/aws-smoke \
        --epochs 1 --batch-size 4 --max-rows-per-source 20 \
        --seeds 41 --num-workers 0
    @echo "Smoke test complete."

aws-train:
    @echo "Starting AWS full training..."
    @GPU_COUNT=$$(nvidia-smi -L | wc -l); \
    if [ $$GPU_COUNT -gt 1 ]; then \
        unset LD_LIBRARY_PATH; \
        torchrun --nproc_per_node=$$GPU_COUNT scripts/train_multimodal_cloud.py \
            --config configs/multimodal_cloud.json \
            --output-dir checkpoints/aws-results \
            --epochs 4 --batch-size 32 --max-rows-per-source 5000 --num-workers 4; \
    else \
        unset LD_LIBRARY_PATH; \
        python scripts/train_multimodal_cloud.py \
            --config configs/multimodal_cloud.json \
            --output-dir checkpoints/aws-results \
            --epochs 4 --batch-size 16 --max-rows-per-source 2500 --num-workers 2; \
    fi
    @echo "Training complete."

aws-download-results:
    @echo "To download results from AWS, run on LOCAL machine:"
    @echo "scp -r -i YOUR-KEY.pem ec2-user@AWS-IP:~/bmvc-2026/checkpoints/aws-results ./"

aws-stop:
    @echo "AWS instance stop reminder:"
    @echo "aws ec2 stop-instances --instance-ids i-XXXXXXXXX"

aws-status:
    @echo "AWS Setup Status:"
    @echo "GPUs: $$(nvidia-smi -L | wc -l)"
    @echo "Python: $$(python3 --version)"
    @echo "PyTorch: $$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    @echo "Storage: $$(df -h / | tail -1 | awk '{print $$4}')"
    @echo "CUDA Check:"
    @nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

aws-help:
    @echo "==============================================================="
    @echo "AWS EC2 QUICK START"
    @echo "==============================================================="
    @echo "1) make aws-setup"
    @echo "2) make aws-verify"
    @echo "3) make aws-smoke"
    @echo "4) make aws-train"
    @echo "5) make aws-download-results"
    @echo "Additional: make aws-status / make aws-stop"

# ----------------------------------------------------------------------------
# Other research utilities
# ----------------------------------------------------------------------------

real-train-2:
    python scripts/train_real_tweet_eval.py --output-dir checkpoints/tweeteval-emotion-real

real-train-ml:
    python scripts/train_real_goemotions_multilabel.py --output-dir checkpoints/goemotions-multilabel-real --seeds 41,42,43

real-dashboard:
    streamlit run results_dashboard.py

benchmark:
    python scripts/benchmark_suite.py --run-all

evaluate:
    python scripts/evaluate.py --config world-class \
        --model-path checkpoints/world-class/best_model.pt

serve:
    cd inference && uvicorn api:app --host 0.0.0.0 --port 8000

test-api:
    curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text":"Amazing work! Love it!","image_caption":"Celebration"}'

ablation:
    python scripts/ablation_study.py --num-samples 500 --num-epochs 15

baselines:
    python scripts/baseline_comparison.py

paper:
    python scripts/generate_paper.py

research: ablation baselines paper
    @echo ""
    @echo "Complete research package generated!"
    @echo " - Ablation study: results/ablation/"
    @echo " - Baseline comparison: results/baselines/"
    @echo " - BMVC paper: results/paper/"

organize-paper:
    @echo "Organizing research paper data"
    python scripts/organize_paper_data.py checkpoints/advanced-results-final research_paper_data
    @echo "Paper folder organized"

clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    rm -rf checkpoints results .wandb .streamlit

all: install demo
