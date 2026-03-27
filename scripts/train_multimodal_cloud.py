#!/usr/bin/env python3
"""
Cloud-Scale Multimodal Emotion-Intention-Action Training.

This is the main training script for BMVC 2026 research.
Production-grade distributed training with multimodal fusion.

Usage:
    # Single GPU
    python scripts/train_multimodal_cloud.py \
        --output-dir checkpoints/multimodal-cloud \
        --epochs 10 --batch-size 32 --seeds 41 42 43

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node 4 \
        scripts/train_multimodal_cloud.py \
        --output-dir checkpoints/multimodal-cloud \
        --epochs 10 --batch-size 32

Author: Research Team
Date: 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cloud_datasets import get_cloud_dataloaders
from models.mine_model import MINEModel
from training.losses import MultiTaskLoss
from training.eval import evaluate_tritask

logger = logging.getLogger(__name__)


def ensure_repo_cache_paths(config: dict, repo_root: Path) -> None:
    """Force all HF caches into the repository for reproducible and controlled storage."""
    hf_cache_dir = (repo_root / "data" / "hf_datasets").resolve()
    model_cache_dir = (repo_root / "models" / "hf_models").resolve()
    hub_cache_dir = (repo_root / "models" / "hf_hub").resolve()
    hf_home = (repo_root / ".hf_home").resolve()

    for p in (hf_cache_dir, model_cache_dir, hub_cache_dir, hf_home):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(model_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache_dir)

    config["hf_cache_dir"] = str(hf_cache_dir)
    config["model_cache_dir"] = str(model_cache_dir)
    config["hf_hub_cache_dir"] = str(hub_cache_dir)


def preflight_validate_config(config: dict, repo_root: Path) -> None:
    """Fail fast on common setup problems to avoid long failing jobs."""
    sources = [str(s).lower() for s in config.get("cloud_sources", [])]

    if not sources:
        raise ValueError("cloud_sources is empty. Add at least one dataset source in config.")

    if any(s in {"mine_gdrive", "mine_drive", "mine_google_drive"} for s in sources):
        mine_root = config.get("mine_gdrive_root") or os.environ.get("MINE_GDRIVE_ROOT", "")
        if not mine_root:
            raise ValueError(
                "mine_gdrive requested but mine_gdrive_root/MINE_GDRIVE_ROOT is missing. "
                "Set it to the extracted Google Drive dataset folder."
            )

        mine_root_path = Path(mine_root).expanduser()
        if not mine_root_path.is_absolute():
            mine_root_path = (repo_root / mine_root_path).resolve()
        else:
            mine_root_path = mine_root_path.resolve()

        if not mine_root_path.exists() or not mine_root_path.is_dir():
            raise ValueError(f"mine_gdrive_root path not found: {mine_root_path}")

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
        if not any((mine_root_path / name).exists() for name in manifest_candidates):
            raise ValueError(
                f"No metadata manifest found under {mine_root_path}. "
                "Expected one of manifest/metadata/annotations/data .json or .jsonl files."
            )

        # Keep resolved path in config/env for consistency across all loaders.
        config["mine_gdrive_root"] = str(mine_root_path)
        os.environ["MINE_GDRIVE_ROOT"] = str(mine_root_path)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" not in os.environ:
        # Single GPU mode
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Multi-GPU mode
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, world_size, device


def setup_logging(rank: int, output_dir: Path):
    """Configure logging for distributed training."""
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    fp16: bool = False,
    grad_accum_steps: int = 1,
) -> float:
    """Train for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    scaler = torch.amp.GradScaler('cuda') if fp16 else None
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["emotion_label"].to(device)
        intention_labels = batch["intention_labels"].to(device)
        action_labels = batch["action_labels"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        
        # Optional multimodal features
        image_features = batch.get("image_features")
        if image_features is not None:
            image_features = image_features.to(device)
        
        audio_features = batch.get("audio_features")
        if audio_features is not None:
            audio_features = audio_features.to(device)
        
        video_features = batch.get("video_features")
        if video_features is not None:
            video_features = video_features.to(device)
        
        # Forward pass
        if fp16:
            with torch.amp.autocast('cuda'):
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_features=image_features,
                    audio_features=audio_features,
                    video_features=video_features,
                    modality_mask=modality_mask,
                )
                
                loss_dict = criterion(
                    emotion_logits=model_output["emotion_logits"],
                    intention_logits=model_output["intention_logits"],
                    action_logits=model_output["action_logits"],
                    emotion_labels=emotion_labels,
                    intention_labels=intention_labels,
                    action_labels=action_labels,
                )
                
                loss = loss_dict["total_loss"]
        else:
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_features=image_features,
                audio_features=audio_features,
                video_features=video_features,
                modality_mask=modality_mask,
            )
            
            loss_dict = criterion(
                emotion_logits=model_output["emotion_logits"],
                intention_logits=model_output["intention_logits"],
                action_logits=model_output["action_logits"],
                emotion_labels=emotion_labels,
                intention_labels=intention_labels,
                action_labels=action_labels,
            )
            
            loss = loss_dict["total_loss"]
        
        # Backward pass
        loss = loss / grad_accum_steps
        
        if fp16:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item() * grad_accum_steps:.4f}"
            )
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_emotion_preds = []
    all_intention_preds = []
    all_action_preds = []
    all_emotion_labels = []
    all_intention_labels = []
    all_action_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            intention_labels = batch["intention_labels"].to(device)
            action_labels = batch["action_labels"].to(device)
            modality_mask = batch["modality_mask"].to(device)
            
            image_features = batch.get("image_features")
            if image_features is not None:
                image_features = image_features.to(device)
            
            audio_features = batch.get("audio_features")
            if audio_features is not None:
                audio_features = audio_features.to(device)
            
            video_features = batch.get("video_features")
            if video_features is not None:
                video_features = video_features.to(device)
            
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_features=image_features,
                audio_features=audio_features,
                video_features=video_features,
                modality_mask=modality_mask,
            )
            
            loss_dict = criterion(
                emotion_logits=model_output["emotion_logits"],
                intention_logits=model_output["intention_logits"],
                action_logits=model_output["action_logits"],
                emotion_labels=emotion_labels,
                intention_labels=intention_labels,
                action_labels=action_labels,
            )
            
            loss = loss_dict["total_loss"]
            
            total_loss += loss.item()
            num_batches += 1
            
            # Predictions
            preds = model.get_predictions(model_output)
            
            all_emotion_preds.append(preds["emotion_preds"].cpu())
            all_intention_preds.append(preds["intention_preds"].cpu())
            all_action_preds.append(preds["action_preds"].cpu())
            
            all_emotion_labels.append(emotion_labels.cpu())
            all_intention_labels.append(intention_labels.cpu())
            all_action_labels.append(action_labels.cpu())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute metrics
    metrics = evaluate_tritask(
        emotion_preds=torch.cat(all_emotion_preds),
        intention_preds=torch.cat(all_intention_preds),
        action_preds=torch.cat(all_action_preds),
        emotion_labels=torch.cat(all_emotion_labels),
        intention_labels=torch.cat(all_intention_labels),
        action_labels=torch.cat(all_action_labels),
    )
    
    logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    logger.info(f"Emotion Accuracy: {metrics['emotion_accuracy']:.4f}")
    logger.info(f"Intention F1: {metrics['intention_micro_f1']:.4f}")
    logger.info(f"Action F1: {metrics['action_micro_f1']:.4f}")
    
    return avg_loss, metrics


def run_seed(
    seed: int,
    config: dict,
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict:
    """Run training for a single seed."""
    logger.info(f"\n{'='*80}")
    logger.info(f"[seed={seed}] Starting training")
    logger.info(f"{'='*80}")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory for seed
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataloaders
    logger.info("Loading multimodal datasets from cloud sources...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        sources=config.get("cloud_sources", ["mine", "emoticon", "raza"]),
        mine_gdrive_root=config.get("mine_gdrive_root"),
        cache_dir=config.get("hf_cache_dir"),
        max_samples={
            "train": config.get("max_rows_per_source", 5000),
            "validation": max(1, config.get("max_rows_per_source", 5000) // 5),
            "test": max(1, config.get("max_rows_per_source", 5000) // 5),
        },
    )
    
    # Model
    logger.info("Initializing multimodal BEAR model...")
    model = MINEModel(
        text_backbone=config.get("text_backbone", "distilroberta-base"),
        hidden_dim=config.get("hidden_dim", 768),
        use_multimodal=True,
    )
    model = model.to(device)
    
    # Distributed data parallel
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )
    
    total_steps = len(train_loader) * config.get("epochs", 10)
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=total_steps,
    )
    
    # Loss
    criterion = MultiTaskLoss(
        emotion_weight=1.0,
        intention_weight=1.2,
        action_weight=1.0,
    )
    
    # Early Stopping tracking
    best_val_loss = float("inf")
    best_epoch = -1
    patience = config.get("early_stopping_patience", 2)
    patience_counter = 0
    
    seed_metrics = {
        "seed": seed,
        "train_losses": [],
        "val_losses": [],
        "emotion_accuracies": [],
        "intention_f1s": [],
        "action_f1s": [],
    }
    
    for epoch in range(1, config.get("epochs", 10) + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            fp16=config.get("fp16", True),
            grad_accum_steps=config.get("grad_accum_steps", 1),
        )
        
        val_loss, metrics = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        
        seed_metrics["train_losses"].append(train_loss)
        seed_metrics["val_losses"].append(val_loss)
        seed_metrics["emotion_accuracies"].append(metrics["emotion_accuracy"])
        seed_metrics["intention_f1s"].append(metrics["intention_micro_f1"])
        seed_metrics["action_f1s"].append(metrics["action_micro_f1"])
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset patience if we improved
            
            # Save best model
            if rank == 0:
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(
                    model_to_save.state_dict(),
                    seed_dir / "best_model.pt",
                )
                logger.info(f"Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
            if rank == 0:
                logger.info(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                if rank == 0:
                    logger.info("🛑 Early stopping triggered! Model has stopped improving.")
                break
        
        if rank == 0:
            logger.info(f"\n[seed={seed}] epoch={epoch}/{config['epochs']} "
                       f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}\n")
    
    # -----------------------------------------------------------------
    # BUG FIX: Load the Best Weights before Final Test Evaluation
    # -----------------------------------------------------------------
    best_model_path = seed_dir / "best_model.pt"
    if best_model_path.exists():
        if rank == 0:
            logger.info(f"\nLoading best model weights (from epoch {best_epoch}) for final test evaluation...")
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(torch.load(best_model_path, map_location=device))
    # -----------------------------------------------------------------

    # Test set evaluation
    logger.info(f"\nEvaluating on test set...")
    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
        device=device,
        epoch=best_epoch if best_epoch != -1 else config.get("epochs", 10),
    )
    
    seed_metrics["test_loss"] = test_loss
    seed_metrics["test_emotion_accuracy"] = test_metrics["emotion_accuracy"]
    seed_metrics["test_intention_f1"] = test_metrics["intention_micro_f1"]
    seed_metrics["test_action_f1"] = test_metrics["action_micro_f1"]
    seed_metrics["best_epoch"] = best_epoch
    seed_metrics["best_val_loss"] = best_val_loss
    
    if rank == 0:
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(seed_metrics, f, indent=2)
        
        logger.info(f"\n[seed={seed}] FINAL TEST RESULTS (From best epoch {best_epoch}):")
        logger.info(f"  Test Emotion Accuracy: {test_metrics['emotion_accuracy']:.4f}")
        logger.info(f"  Test Intention F1: {test_metrics['intention_micro_f1']:.4f}")
        logger.info(f"  Test Action F1: {test_metrics['action_micro_f1']:.4f}")
        logger.info(f"{'='*80}\n")
    
    return seed_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Cloud multimodal training for BMVC 2026"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cloud_supercomputer.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/multimodal-cloud",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to run (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Max rows per dataset source (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers (overrides config)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mixed precision",
    )
    parser.add_argument(
        "--strict-preflight",
        action="store_true",
        help="Fail fast on missing mine_gdrive path/manifest and other preflight issues",
    )
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    # Load config
    output_dir = Path(args.output_dir)
    
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    
    # Override config with CLI args
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seeds:
        config["seeds"] = args.seeds
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.max_rows_per_source:
        config["max_rows_per_source"] = args.max_rows_per_source
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.fp16:
        config["fp16"] = True
    
    # Set defaults
    config.setdefault("epochs", 10)
    config.setdefault("batch_size", 16)
    config.setdefault("seeds", [41, 42, 43])
    config.setdefault("learning_rate", 2e-5)
    config.setdefault("max_rows_per_source", 5000)
    config.setdefault("num_workers", 4)
    config.setdefault("fp16", True)
    config.setdefault("grad_accum_steps", 1)
    config.setdefault("text_backbone", "distilroberta-base")
    config.setdefault("hidden_dim", 768)
    config.setdefault("cloud_sources", ["mine", "emoticon", "raza"])
    config.setdefault("strict_preflight", True)
    config.setdefault("early_stopping_patience", 2)

    if args.strict_preflight:
        config["strict_preflight"] = True

    repo_root = Path.cwd().resolve()
    ensure_repo_cache_paths(config, repo_root)

    if config.get("strict_preflight", True):
        preflight_validate_config(config, repo_root)
    
    # Setup logging
    setup_logging(rank, output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info("MULTIMODAL BEAR TRAINING - BMVC 2026")
    logger.info(f"{'='*80}")
    logger.info(f"Rank: {rank}/{world_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    logger.info(f"{'='*80}\n")
    
    # Run multiple seeds
    all_seed_metrics = {}
    
    for seed in config["seeds"]:
        seed_metrics = run_seed(
            seed=seed,
            config=config,
            output_dir=output_dir,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        all_seed_metrics[str(seed)] = seed_metrics
    
    # Summarize results
    if rank == 0:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "seeds": all_seed_metrics,
        }
        
        # Compute mean ± std across seeds
        metrics_names = ["emotion_accuracies", "intention_f1s", "action_f1s"]
        for metric_name in metrics_names:
            all_runs = [
                m[metric_name][-1] for m in all_seed_metrics.values() if metric_name in m
            ]
            if all_runs:
                summary[f"{metric_name}_mean"] = float(np.mean(all_runs))
                summary[f"{metric_name}_std"] = float(np.std(all_runs))
        
        # Compute test results mean ± std
        test_emotion = [m.get("test_emotion_accuracy", 0) for m in all_seed_metrics.values()]
        test_intention = [m.get("test_intention_f1", 0) for m in all_seed_metrics.values()]
        test_action = [m.get("test_action_f1", 0) for m in all_seed_metrics.values()]
        
        summary["test_emotion_accuracy_mean"] = float(np.mean(test_emotion))
        summary["test_emotion_accuracy_std"] = float(np.std(test_emotion))
        summary["test_intention_f1_mean"] = float(np.mean(test_intention))
        summary["test_intention_f1_std"] = float(np.std(test_intention))
        summary["test_action_f1_mean"] = float(np.mean(test_action))
        summary["test_action_f1_std"] = float(np.std(test_action))
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        with open(output_dir / "seed_metrics.json", "w") as f:
            json.dump(all_seed_metrics, f, indent=2)
        
        with open(output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE - FINAL RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Test Emotion Accuracy: {summary['test_emotion_accuracy_mean']:.4f} ± {summary['test_emotion_accuracy_std']:.4f}")
        logger.info(f"Test Intention F1: {summary['test_intention_f1_mean']:.4f} ± {summary['test_intention_f1_std']:.4f}")
        logger.info(f"Test Action F1: {summary['test_action_f1_mean']:.4f} ± {summary['test_action_f1_std']:.4f}")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"Artifacts saved to: {output_dir}")
        logger.info(f"  - summary.json: overall results")
        logger.info(f"  - seed_metrics.json: per-seed detailed results")
        logger.info(f"  - run_config.json: hyperparameters used")
        logger.info(f"  - seed_*/best_model.pt: trained checkpoints")
        logger.info(f"  - training.log: full training log\n")


if __name__ == "__main__":
    main()
