.PHONY: install test demo train benchmark evaluate serve clean docs help

help:
	@echo "BEAR BMVC 2026 Research System - Makefile"
	@echo "========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install all dependencies (Standard/T4 Optimized)"
	@echo "  make demo            - Run Streamlit demo app"
	@echo "  make train           - Train world-class model"
	@echo "  make real-data       - Download real public datasets"
	@echo "  make real-train      - Real fine-tuning on GoEmotions"
	@echo "  make real-train-bear - Train BEAR variant on GoEmotions"
	@echo "  make real-train-tri  - Train BEAR tri-task (emotion+intention+action)"
	@echo "  make cloud-train     - Cloud-ready distributed BEAR tri-task training (text-only)"
	@echo "  make cloud-test      - Test cloud checkpoint on held-out data"
	@echo ""
	@echo "MULTIMODAL CLOUD TRAINING (NEW - BMVC 2026 MAIN):"
	@echo "  make multimodal-smoke        - Quick test (1 epoch, 100 samples)"
	@echo "  make multimodal-single-gpu   - Single GPU training (debug mode)"
	@echo "  make multimodal-multi-gpu    - Multi-GPU training (4 GPUs)"
	@echo "  make multimodal-cloud        - Full production run with launcher"
	@echo "  make multimodal-ablation-* - Ablation studies"
	@echo ""
	@echo "ADVANCED MULTIMODAL (Top-Tier - BMVC 2026 PRODUCTION 🚀):"
	@echo "  make advanced-smoke        - Quick test (1 epoch, dual-layer LLM, auto-PDF)"
	@echo "  make advanced-single-gpu   - Single GPU (advanced model, PDF output)"
	@echo "  make advanced-multi-gpu    - Multi-GPU (4 GPUs, advanced model)"
	@echo "  make advanced-cloud        - Full production (dual-layer LLM + auto-PDF)"
	@echo "  make advanced-best         - Maximum quality run (ultra 20-30GB dataset)"
	@echo "  make predownload-ultra     - Download models/datasets only (no training)"
	@echo "  make professor-run         - Download + full training + organize results"
	@echo ""
	@echo "AWS EC2 DEPLOYMENT (🚀 Recommended for Professor):"
	@echo "  make aws-help                - Print AWS quick start guide"
	@echo "  make aws-setup               - Auto-setup AWS environment"
	@echo "  make aws-verify              - Verify GPUs and PyTorch"
	@echo "  make aws-smoke               - Quick test on AWS (1 min)"
	@echo "  make aws-train               - Full training on AWS (12+ hours)"
	@echo "  make aws-download-results    - Download results locally"
	@echo "  make aws-status              - Check AWS environment"
	@echo ""
	@echo "  make real-train-2    - Real fine-tuning on TweetEval Emotion"
	@echo "  make real-train-ml   - Real multi-label GoEmotions (3 seeds)"
	@echo "  make real-dashboard  - Launch real-metrics dashboard"
	@echo "  make benchmark       - Run full benchmark suite (all 4 configs)"
	@echo "  make evaluate        - Evaluate best model"
	@echo "  make serve           - Start FastAPI production server"
	@echo ""
	@echo "RESEARCH & PAPER GENERATION:"
	@echo "  make ablation         - Run ablation study"
	@echo "  make baselines        - Compare against SOTA baselines"
	@echo "  make paper           - Generate BMVC paper draft"
	@echo "  make research        - Run ablation + baselines + paper"
	@echo ""
	@echo "UTILITY:"
	@echo "  make clean           - Remove artifacts"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-inference.txt
	@echo "Dependencies installed for standard/T4 GPU architecture!"

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

multimodal-smoke:
	python scripts/train_multimodal_cloud.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/multimodal-smoke \
		--epochs 1 --batch-size 8 --max-rows-per-source 100 --seeds 41 --num-workers 0

multimodal-single-gpu:
	python scripts/train_multimodal_cloud.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/multimodal-single-gpu \
		--epochs 4 --batch-size 8 --max-rows-per-source 2000 --num-workers 2

multimodal-multi-gpu:
	torchrun --nproc_per_node=4 \
		scripts/train_multimodal_cloud.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/multimodal-multi-gpu \
		--epochs 4 --batch-size 8 --max-rows-per-source 5000 --num-workers 4

multimodal-cloud:
	bash scripts/run_multimodal_supercomputer.sh configs/multimodal_cloud.json 4 checkpoints/multimodal-full

multimodal-ablation-no-reliability:
	python scripts/train_multimodal_cloud.py \
		--output-dir checkpoints/ablations/multimodal-no-reliability \
		--epochs 2 --batch-size 8 --seeds 41 42

multimodal-ablation-text-only:
	python scripts/train_bear_intention_cloud.py \
		--output-dir checkpoints/ablations/text-only \
		--epochs 2 --batch-size 8 --seeds 41 42

# ═══════════════════════════════════════════════════════════════════
# ADVANCED MULTIMODAL TARGETS (Dual-Layer LLM + Auto-PDF)
# ═══════════════════════════════════════════════════════════════════

advanced-smoke:
	@echo "🚀 Running ADVANCED smoke test (dual-layer LLM + PDF)..."
	python scripts/train_advanced_multimodal.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/advanced-smoke \
		--epochs 1 --batch-size 4 --max-rows-per-source 20 \
		--seeds 41 --num-workers 0
	@echo "✅ Smoke test complete! PDF at: checkpoints/advanced-smoke/RESEARCH_RESULTS_REPORT.pdf"

advanced-single-gpu:
	@echo "🎯 Advanced training (single GPU, dual-layer LLM)..."
	python scripts/train_advanced_multimodal.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/advanced-single-gpu \
		--epochs 4 --batch-size 8 --max-rows-per-source 2000 --num-workers 2 \
		--seeds 41 42 43
	@echo "✅ Training complete! PDF at: checkpoints/advanced-single-gpu/RESEARCH_RESULTS_REPORT.pdf"

advanced-multi-gpu:
	@echo "⚡ Advanced training (4× GPUs, dual-layer LLM)..."
	torchrun --nproc_per_node=4 \
		scripts/train_advanced_multimodal.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/advanced-multi-gpu \
		--epochs 4 --batch-size 8 --max-rows-per-source 5000 --num-workers 4 \
		--seeds 41 42 43
	@echo "✅ Training complete! PDF at: checkpoints/advanced-multi-gpu/RESEARCH_RESULTS_REPORT.pdf"

advanced-cloud:
	@echo "🌩️  ADVANCED CLOUD TRAINING (Dual-Layer LLM + Auto-PDF Reports)"
	@echo "Architecture: RoBERTa-large + DistilGPT2 dual-layer fusion"
	@echo "This will automatically generate publication-ready PDFs after training..."
	@echo ""
	@GPU_COUNT=$$(nvidia-smi -L | wc -l); \
	if [ $$GPU_COUNT -gt 1 ]; then \
		echo "🔥 Detected $$GPU_COUNT GPUs - using distributed training..."; \
		torchrun --nproc_per_node=$$GPU_COUNT scripts/train_advanced_multimodal.py \
			--config configs/multimodal_cloud.json \
			--output-dir checkpoints/advanced-results-final \
			--epochs 4 --batch-size 8 --max-rows-per-source 5000 --num-workers 4 \
			--seeds 41 42 43; \
	else \
		echo "Using single GPU..."; \
		python scripts/train_advanced_multimodal.py \
			--config configs/multimodal_cloud.json \
			--output-dir checkpoints/advanced-results-final \
			--epochs 4 --batch-size 8 --max-rows-per-source 2500 --num-workers 2 \
			--seeds 41 42 43; \
	fi
	@echo ""
	@echo "✅ ADVANCED TRAINING COMPLETE!"
	@echo "📂 Results:"
	@echo "   - PDF: checkpoints/advanced-results-final/RESEARCH_RESULTS_REPORT.pdf"
	@echo "   - CSV: checkpoints/advanced-results-final/RESULTS_TABLE.csv"
	@echo "   - LaTeX: checkpoints/advanced-results-final/RESULTS_LATEX_TABLE.txt"
	@echo "   - JSON: checkpoints/advanced-results-final/summary.json"
	@echo "Ready for research paper writing! 📝"

advanced-best:
	@echo "Best profile: ultra_30gb (target cache 20-30GB)"
	@echo "LLMs: roberta-large + distilroberta-base"
	@mkdir -p .hf_cache
	@GPU_COUNT=$$(nvidia-smi -L | wc -l); \
	if [ $$GPU_COUNT -gt 1 ]; then \
		echo "Detected $$GPU_COUNT GPUs, using distributed mode"; \
		HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets TRANSFORMERS_CACHE=.hf_cache/transformers torchrun --nproc_per_node=$$GPU_COUNT scripts/train_advanced_multimodal.py \
			--config configs/multimodal_ultra_30gb.json \
			--dataset-profile ultra_30gb \
			--output-dir checkpoints/advanced-best-ultra \
			--epochs 5 --batch-size 8 --max-rows-per-source 40000 --num-workers 4 \
			--seeds 41 42 43; \
	else \
		echo "Using single GPU mode"; \
		HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets TRANSFORMERS_CACHE=.hf_cache/transformers python scripts/train_advanced_multimodal.py \
			--config configs/multimodal_ultra_30gb.json \
			--dataset-profile ultra_30gb \
			--output-dir checkpoints/advanced-best-ultra \
			--epochs 5 --batch-size 8 --max-rows-per-source 30000 --num-workers 2 \
			--seeds 41 42 43; \
	fi
	@echo "Run complete"
	@echo "   - Results: checkpoints/advanced-best-ultra/"
	@echo "   - Paper folder: checkpoints/research_paper_data/"

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

professor-run:
	@echo "Professor one-command run (strict + reproducible, 16GB VRAM limits)"
	@mkdir -p data/hf_datasets models/hf_models models/hf_hub checkpoints/professor-run
	@python -c "import torch,sys; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 1)" || (echo "ERROR: CUDA is not available. Install a CUDA-enabled PyTorch build and run again." && exit 1)
	@CONFIG_PATH="configs/multimodal_cloud.json"; \
	SOURCE_LIST="mine_gdrive,mine,emoticon,raza,coco"; \
	if [ -z "$$MINE_GDRIVE_ROOT" ] || [ ! -d "$$MINE_GDRIVE_ROOT" ]; then \
		echo "MINE_GDRIVE_ROOT not set/found -> running without mine_gdrive source"; \
		python -c "import json, pathlib; cfg=json.load(open('configs/multimodal_cloud.json','r',encoding='utf-8')); cfg['cloud_sources']=[s for s in cfg.get('cloud_sources',[]) if s!='mine_gdrive']; pathlib.Path('/tmp/multimodal_cloud_nominegdrive.json').write_text(json.dumps(cfg), encoding='utf-8')"; \
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
	if [ $$GPU_COUNT -gt 1 ]; then \
		torchrun --nproc_per_node=$$GPU_COUNT scripts/train_multimodal_cloud.py \
			--config "$$CONFIG_PATH" \
			--strict-preflight \
			--batch-size 8 \
			--output-dir checkpoints/professor-run; \
	else \
		python scripts/train_multimodal_cloud.py \
			--config "$$CONFIG_PATH" \
			--strict-preflight \
			--batch-size 8 \
			--output-dir checkpoints/professor-run; \
	fi
	@echo "Complete: checkpoints/professor-run"

# ═══════════════════════════════════════════════════════════════════
# AWS EC2 DEPLOYMENT TARGETS (NEW)
# ═══════════════════════════════════════════════════════════════════

aws-setup:
	@echo "Setting up AWS environment..."
	wget -q https://raw.githubusercontent.com/YOUR-USERNAME/bmvc-2026/main/setup_aws.py -O setup_aws.py
	python3 setup_aws.py
	@echo "✅ AWS environment ready!"

aws-verify:
	@echo "Verifying AWS setup..."
	nvidia-smi -L | wc -l | xargs -I {} echo "✅ {} GPUs detected"
	python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA Available: {torch.cuda.is_available()}')"

aws-smoke:
	@echo "🔥 Running AWS smoke test (1 epoch)..."
	python scripts/train_multimodal_cloud.py \
		--config configs/multimodal_cloud.json \
		--output-dir checkpoints/aws-smoke \
		--epochs 1 --batch-size 4 --max-rows-per-source 20 \
		--seeds 41 --num-workers 0
	@echo "✅ Smoke test complete! Check: checkpoints/aws-smoke/summary.json"

aws-train:
	@echo "🚀 Starting AWS full training (12-24 hours on T4)..."
	@GPU_COUNT=$$(nvidia-smi -L | wc -l); \
	if [ $$GPU_COUNT -gt 1 ]; then \
		torchrun --nproc_per_node=$$GPU_COUNT scripts/train_multimodal_cloud.py \
			--config configs/multimodal_cloud.json \
			--output-dir checkpoints/aws-results \
			--epochs 4 --batch-size 8 --max-rows-per-source 5000 --num-workers 4; \
	else \
		python scripts/train_multimodal_cloud.py \
			--config configs/multimodal_cloud.json \
			--output-dir checkpoints/aws-results \
			--epochs 4 --batch-size 8 --max-rows-per-source 2500 --num-workers 2; \
	fi
	@echo "✅ Training complete! Results at: checkpoints/aws-results/"

aws-download-results:
	@echo "📊 To download results from AWS, run on LOCAL machine:"
	@echo "   scp -r -i YOUR-KEY.pem ec2-user@AWS-IP:~/bmvc-2026/checkpoints/aws-results ./"
	@echo ""
	@echo "   Replace: YOUR-KEY.pem with your AWS key file"
	@echo "   Replace: AWS-IP with your EC2 public IP"

aws-stop:
	@echo "⏹️  AWS instance will be stopped (costs will pause)"
	@echo "   Run this on AWS instance, then on local machine:"
	@echo "   aws ec2 stop-instances --instance-ids i-XXXXXXXXX"

aws-status:
	@echo "AWS Setup Status:"
	@echo "  GPUs: $$(nvidia-smi -L | wc -l)"
	@echo "  Python: $$(python3 --version)"
	@echo "  PyTorch: $$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "  Storage: $$(df -h / | tail -1 | awk '{print $$4}')"
	@echo "  CUDA Check:"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

aws-help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "AWS EC2 QUICK START"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "1️⃣  Setup Environment:"
	@echo "   $$ make aws-setup"
	@echo ""
	@echo "2️⃣  Verify Setup:"
	@echo "   $$ make aws-verify"
	@echo ""
	@echo "3️⃣  Run Smoke Test (1 min):"
	@echo "   $$ make aws-smoke"
	@echo ""
	@echo "4️⃣  Run Full Training:"
	@echo "   $$ make aws-train"
	@echo ""
	@echo "5️⃣  Download Results:"
	@echo "   $$ make aws-download-results"
	@echo ""
	@echo "Additional:"
	@echo "   $$ make aws-status        - Check current setup"
	@echo "   $$ make aws-stop          - Stop instance"
	@echo ""
	@echo "Full Guide: AWS_DEPLOYMENT_GUIDE.md"
	@echo "Quick Ref:  AWS_QUICK_REFERENCE.md"
	@echo ""

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
	@echo "✅ Complete research package generated!"
	@echo "   - Ablation study: results/ablation/"
	@echo "   - Baseline comparison: results/baselines/"
	@echo "   - BMVC paper: results/paper/"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf checkpoints results .wandb .streamlit

all: install demo

.PHONY: organize-paper
organize-paper:
	@echo "Organizing research paper data"
	python scripts/organize_paper_data.py checkpoints/advanced-results-final research_paper_data
	@echo "Paper folder organized"
