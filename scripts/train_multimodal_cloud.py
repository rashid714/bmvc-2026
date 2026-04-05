#!/usr/bin/env python3
"""
Cloud-Scale Multimodal Emotion-Intention-Action Training.
Spotlight BMVC 2026 Version: Hardware Acceleration, Cosine Annealing, ResNet50, and Focal Loss

Usage:
    # Single GPU
    python scripts/train_multimodal_cloud.py \
        --output-dir checkpoints/multimodal-cloud \
        --epochs 10 --batch-size 32 --seeds 41 42 43
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
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

# 🌟 UPGRADE: Scikit-Learn for accurate Macro F1 calculations
from sklearn.metrics import accuracy_score, f1_score

from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import logging as hf_logging

# Silence huggingface warnings
hf_logging.set_verbosity_error()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cloud_datasets import get_cloud_dataloaders
from models.advanced_multimodal_bear import AdvancedBEARModel

logger = logging.getLogger(__name__)

# ==============================================================================
# 🌟 THE 80% SECRET WEAPON: MULTI-LABEL FOCAL LOSS
# ==============================================================================
class MultiLabelFocalLoss(torch.nn.Module):
    """Heavily penalizes the model for getting the rare classes wrong."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class FocalMultiTaskLoss(torch.nn.Module):
    """Replaces your old MultiTaskLoss with the Focal Engine."""
    def __init__(self, emotion_weight=1.0, intention_weight=2.0, action_weight=2.0):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.intention_weight = intention_weight
        self.action_weight = action_weight
        
        self.emotion_criterion = torch.nn.CrossEntropyLoss()
        self.intention_criterion = MultiLabelFocalLoss(gamma=2.0)
        self.action_criterion = MultiLabelFocalLoss(gamma=2.0)

    def forward(self, emotion_logits, intention_logits, action_logits, emotion_labels, intention_labels, action_labels):
        loss_emotion = self.emotion_criterion(emotion_logits, emotion_labels)
        loss_intention = self.intention_criterion(intention_logits, intention_labels)
        loss_action = self.action_criterion(action_logits, action_labels)
        
        total_loss = (
            self.emotion_weight * loss_emotion +
            self.intention_weight * loss_intention +
            self.action_weight * loss_action
        )
        return {"total_loss": total_loss}

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
            logger.warning("mine_gdrive requested but MINE_GDRIVE_ROOT is missing. Skipping direct validation.")
            return

        mine_root_path = Path(mine_root).expanduser()
        if not mine_root_path.is_absolute():
            mine_root_path = (repo_root / mine_root_path).resolve()
        else:
            mine_root_path = mine_root_path.resolve()

        if not mine_root_path.exists() or not mine_root_path.is_dir():
            logger.warning(f"mine_gdrive_root path not found: {mine_root_path}")
            return

        config["mine_gdrive_root"] = str(mine_root_path)
        os.environ["MINE_GDRIVE_ROOT"] = str(mine_root_path)


def setup_distributed():
    """Initialize distributed training environment."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    criterion: FocalMultiTaskLoss,
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        intention_labels = batch["intention_labels"].to(device)
        action_labels = batch["action_labels"].to(device)
        
        images = batch.get("images")
        if images is not None:
            images = images.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if fp16:
            with torch.amp.autocast('cuda'):
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )
                
                loss_dict = criterion(
                    emotion_logits=model_output["emotion_logits"],
                    intention_logits=model_output["intention_logits"],
                    action_logits=model_output["action_logits"],
                    emotion_labels=emotion_labels,
                    intention_labels=intention_labels,
                    action_labels=action_labels,
                )
                
                loss = loss_dict["total_loss"] / grad_accum_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                if scale_before <= scale_after:
                    scheduler.step()
        else:
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
            )
            
            loss_dict = criterion(
                emotion_logits=model_output["emotion_logits"],
                intention_logits=model_output["intention_logits"],
                action_logits=model_output["action_logits"],
                emotion_labels=emotion_labels,
                intention_labels=intention_labels,
                action_labels=action_labels,
            )
            
            loss = loss_dict["total_loss"] / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
    criterion: FocalMultiTaskLoss,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_emotion_logits = []
    all_intention_logits = []
    all_action_logits = []
    
    all_emotion_labels = []
    all_intention_labels = []
    all_action_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            intention_labels = batch["intention_labels"].to(device)
            action_labels = batch["action_labels"].to(device)
            
            images = batch.get("images")
            if images is not None:
                images = images.to(device)
            
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
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
            
            all_emotion_logits.append(model_output["emotion_logits"].cpu())
            all_intention_logits.append(model_output["intention_logits"].cpu())
            all_action_logits.append(model_output["action_logits"].cpu())
            
            all_emotion_labels.append(emotion_labels.cpu())
            all_intention_labels.append(intention_labels.cpu())
            all_action_labels.append(action_labels.cpu())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # 🌟 THE 80% METRICS ENGINE: Macro F1 with 0.4 Dynamic Threshold
    emotion_logits = torch.cat(all_emotion_logits)
    emotion_preds = torch.argmax(emotion_logits, dim=1).numpy()
    emotion_targets = torch.cat(all_emotion_labels).numpy()
    
    intention_probs = torch.sigmoid(torch.cat(all_intention_logits)).numpy()
    action_probs = torch.sigmoid(torch.cat(all_action_logits)).numpy()
    
    intention_targets = torch.cat(all_intention_labels).numpy()
    action_targets = torch.cat(all_action_labels).numpy()
    
    # Dynamic Threshold set to 0.4 instead of 0.5 for Multi-Label flexibility
    intention_preds = (intention_probs > 0.4).astype(int)
    action_preds = (action_probs > 0.4).astype(int)
    
    metrics = {
        "emotion_accuracy": accuracy_score(emotion_targets, emotion_preds),
        "intention_macro_f1": f1_score(intention_targets, intention_preds, average='macro', zero_division=0),
        "action_macro_f1": f1_score(action_targets, action_preds, average='macro', zero_division=0),
    }
    
    logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    logger.info(f"Emotion Accuracy: {metrics['emotion_accuracy']:.4f}")
    logger.info(f"Intention F1 (Macro): {metrics['intention_macro_f1']:.4f}")
    logger.info(f"Action F1 (Macro): {metrics['action_macro_f1']:.4f}")
    
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
        sources=config.get("cloud_sources", ["llama_distilled", "mine", "emoticon", "raza"]),
        mine_gdrive_root=config.get("mine_gdrive_root"),
        cache_dir=config.get("hf_cache_dir"),
        max_samples={
            "train": config.get("max_rows_per_source", 5000),
            "validation": max(1, config.get("max_rows_per_source", 5000) // 5),
            "test": max(1, config.get("max_rows_per_source", 5000) // 5),
        },
    )
    
    # Model
    logger.info("Initializing Advanced Multimodal BEAR model...")
    model = AdvancedBEARModel(
        hidden_dim=config.get("hidden_dim", 1024),
        use_pretrained_vision=True
    )
    model = model.to(device)
    
    # Distributed data parallel
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4), # 🌟 Lowered for Focal Loss
        weight_decay=config.get("weight_decay", 0.01),
    )
    
    total_steps = len(train_loader) * config.get("epochs", 10)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1), # 10% warmup
        num_training_steps=total_steps
    )
    
    # 🌟 ACTIVATE FOCAL LOSS
    criterion = FocalMultiTaskLoss(
        emotion_weight=1.0,
        intention_weight=2.0,
        action_weight=2.0,
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
        seed_metrics["intention_f1s"].append(metrics["intention_macro_f1"])
        seed_metrics["action_f1s"].append(metrics["action_macro_f1"])
        
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
    
    # Load best weights before final testing
    best_model_path = seed_dir / "best_model.pt"
    if best_model_path.exists():
        if rank == 0:
            logger.info(f"\nLoading best model weights (from epoch {best_epoch}) for final test evaluation...")
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(torch.load(best_model_path, map_location=device))

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
    seed_metrics["test_intention_f1"] = test_metrics["intention_macro_f1"]
    seed_metrics["test_action_f1"] = test_metrics["action_macro_f1"]
    seed_metrics["best_epoch"] = best_epoch
    seed_metrics["best_val_loss"] = best_val_loss
    
    if rank == 0:
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(seed_metrics, f, indent=2)
        
        logger.info(f"\n[seed={seed}] FINAL TEST RESULTS (From best epoch {best_epoch}):")
        logger.info(f"  Test Emotion Accuracy: {test_metrics['emotion_accuracy']:.4f}")
        logger.info(f"  Test Intention F1 (Macro): {test_metrics['intention_macro_f1']:.4f}")
        logger.info(f"  Test Action F1 (Macro): {test_metrics['action_macro_f1']:.4f}")
        logger.info(f"{'='*80}\n")
    
    return seed_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Cloud multimodal training for BMVC 2026"
    )
    parser.add_argument("--config", type=str, default="configs/cloud_supercomputer.json", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="checkpoints/multimodal-cloud", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds to run (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--max-rows-per-source", type=int, default=None, help="Max rows per dataset source (overrides config)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers (overrides config)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--strict-preflight", action="store_true", help="Fail fast on missing mine_gdrive path/manifest")
    
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
    if args.epochs: config["epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.seeds: config["seeds"] = args.seeds
    if args.learning_rate: config["learning_rate"] = args.learning_rate
    if args.max_rows_per_source: config["max_rows_per_source"] = args.max_rows_per_source
    if args.num_workers is not None: config["num_workers"] = args.num_workers
    if args.fp16: config["fp16"] = True
    
    # Set defaults
    config.setdefault("epochs", 10)
    config.setdefault("batch_size", 16)
    config.setdefault("seeds", [41, 42, 43])
    config.setdefault("learning_rate", 1e-4) # 🌟 Adjusted for Focal Loss
    config.setdefault("max_rows_per_source", 5000)
    config.setdefault("num_workers", 4)
    config.setdefault("fp16", True)
    config.setdefault("grad_accum_steps", 1)
    config.setdefault("hidden_dim", 1024)
    config.setdefault("cloud_sources", ["llama_distilled", "mine", "emoticon", "raza"])
    config.setdefault("strict_preflight", True)
    config.setdefault("early_stopping_patience", 3)

    if args.strict_preflight:
        config["strict_preflight"] = True

    repo_root = Path.cwd().resolve()
    ensure_repo_cache_paths(config, repo_root)

    if config.get("strict_preflight", True):
        preflight_validate_config(config, repo_root)
    
    # Setup logging
    setup_logging(rank, output_dir)
    
    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info("MULTIMODAL BEAR TRAINING - BMVC 2026 (FOCAL ENGINE ENABLED)")
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
        logger.info(f"Test Intention F1 (Macro): {summary['test_intention_f1_mean']:.4f} ± {summary['test_intention_f1_std']:.4f}")
        logger.info(f"Test Action F1 (Macro): {summary['test_action_f1_mean']:.4f} ± {summary['test_action_f1_std']:.4f}")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"Artifacts saved to: {output_dir}")
        logger.info(f"  - summary.json: overall results")
        logger.info(f"  - seed_metrics.json: per-seed detailed results")
        logger.info(f"  - run_config.json: hyperparameters used")
        logger.info(f"  - seed_*/best_model.pt: trained checkpoints")
        logger.info(f"  - training.log: full training log\n")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
