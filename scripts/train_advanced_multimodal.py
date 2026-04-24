#!/usr/bin/env python3
"""
BMVC 2026 - Silver Standard Training Loop
Spotlight Version: Dynamic Pos-Weights, Gradient Accumulation, DINOv2 + RoBERTa
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from models.advanced_multimodal_bear import AdvancedBEARModel
from training.eval import evaluate_tritask
from data.cloud_datasets import get_cloud_dataloaders
from training.pdf_report_generator import (
    generate_research_report_pdf,
    generate_raw_data_export,
)

# =============================================================================
# 🌟 UPGRADE 1: Silver Standard Loss Engine (Dynamic Inverse Weighting)
# =============================================================================
class SilverStandardLoss(nn.Module):
    def __init__(self, pos_weight_intent: torch.Tensor, pos_weight_action: torch.Tensor, 
                 emo_w: float = 1.0, int_w: float = 2.0, act_w: float = 2.0):
        super().__init__()
        # Label smoothing prevents overconfidence on the single-label emotion head
        self.emo_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        # BCE with dynamic positive weights forces the model to respect rare classes
        self.int_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_intent)
        self.act_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_action)
        self.weights = (emo_w, int_w, act_w)

    def forward(self, emotion_logits, intention_logits, action_logits, 
                emotion_labels, intention_labels, action_labels):
        l_emo = self.emo_loss(emotion_logits, emotion_labels)
        l_int = self.int_loss(intention_logits, intention_labels)
        l_act = self.act_loss(action_logits, action_labels)
        
        total = (self.weights[0] * l_emo) + (self.weights[1] * l_int) + (self.weights[2] * l_act)
        return {"total_loss": total, "emo_loss": l_emo, "int_loss": l_int, "act_loss": l_act}

def compute_dynamic_pos_weights(loader, device, num_intent=12, num_action=15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scans the dataset to calculate perfect inverse class weights for BCE loss."""
    int_pos = torch.zeros(num_intent, device=device)
    act_pos = torch.zeros(num_action, device=device)
    total_samples = 0
    
    for batch in loader:
        int_pos += batch["intention_labels"].to(device).sum(dim=0)
        act_pos += batch["action_labels"].to(device).sum(dim=0)
        total_samples += batch["intention_labels"].size(0)
        
    int_neg = total_samples - int_pos
    act_neg = total_samples - act_pos
    
    # Cap the multiplier at 50x so ultra-rare classes don't explode the gradients
    pw_int = torch.clamp(int_neg / (int_pos + 1e-5), min=1.0, max=50.0)
    pw_act = torch.clamp(act_neg / (act_pos + 1e-5), min=1.0, max=50.0)
    return pw_int, pw_act

# =============================================================================
# Setup Helpers
# =============================================================================
def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    log_path = Path(output_dir) / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_advanced_multimodal")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.propagate = False
    if logger.handlers: logger.handlers.clear()
    
    if rank == 0:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh, ch = logging.FileHandler(log_path), logging.StreamHandler()
        fh.setFormatter(formatter); ch.setFormatter(formatter)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

# =============================================================================
# Training / Evaluation Loops
# =============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, rank, logger, epoch, fp16=True):
    model.train()
    total_loss = 0.0
    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # 🌟 UPGRADE 2: Gradient Accumulation (Effective Batch Size = Batch * 4)
    accumulation_steps = 4 

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        emotion_labels = batch["emotion_labels"].to(device, non_blocking=True)
        intention_labels = batch["intention_labels"].to(device, non_blocking=True)
        action_labels = batch["action_labels"].to(device, non_blocking=True)
        images = batch.get("images")
        if images is not None: images = images.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                model_output = model(input_ids, attention_mask, images=images)
                loss_dict = criterion(model_output["emotion_logits"], model_output["intention_logits"], model_output["action_logits"],
                                      emotion_labels, intention_labels, action_labels)
                loss = loss_dict["total_loss"] / accumulation_steps

            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        else:
            # Fallback for CPU
            model_output = model(input_ids, attention_mask, images=images)
            loss_dict = criterion(model_output["emotion_logits"], model_output["intention_logits"], model_output["action_logits"],
                                  emotion_labels, intention_labels, action_labels)
            loss = loss_dict["total_loss"] / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        total_loss += float(loss.item() * accumulation_steps)

        if rank == 0 and batch_idx % 20 == 0:
            logger.info("Epoch %d | Batch %d/%d | Loss: %.4f", epoch, batch_idx, len(train_loader), float(loss.item() * accumulation_steps))

    return total_loss / max(len(train_loader), 1)

@torch.no_grad()
def evaluate_one_epoch(model, data_loader, criterion, device, rank, logger, split_name="Validation"):
    model.eval()
    total_loss, num_batches = 0.0, 0
    all_emo_preds, all_emo_labels = [], []
    all_int_preds, all_int_labels = [], []
    all_act_preds, all_act_labels = [], []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        emotion_labels = batch["emotion_labels"].to(device, non_blocking=True)
        intention_labels = batch["intention_labels"].to(device, non_blocking=True)
        action_labels = batch["action_labels"].to(device, non_blocking=True)
        images = batch.get("images")
        if images is not None: images = images.to(device, non_blocking=True)

        model_output = model(input_ids, attention_mask, images=images)
        loss_dict = criterion(model_output["emotion_logits"], model_output["intention_logits"], model_output["action_logits"],
                              emotion_labels, intention_labels, action_labels)
        
        total_loss += float(loss_dict["total_loss"].item())
        num_batches += 1

        all_emo_preds.append(model_output["emotion_logits"].cpu())
        all_emo_labels.append(emotion_labels.cpu())
        all_int_preds.append(model_output["intention_logits"].cpu())
        all_int_labels.append(intention_labels.cpu())
        all_act_preds.append(model_output["action_logits"].cpu())
        all_act_labels.append(action_labels.cpu())

    metrics = evaluate_tritask(
        emotion_preds=torch.cat(all_emo_preds, dim=0), intention_preds=torch.cat(all_int_preds, dim=0), action_preds=torch.cat(all_act_preds, dim=0),
        emotion_labels=torch.cat(all_emo_labels, dim=0), intention_labels=torch.cat(all_int_labels, dim=0), action_labels=torch.cat(all_act_labels, dim=0),
    )
    
    avg_loss = total_loss / max(num_batches, 1)
    if rank == 0:
        logger.info("%s Loss: %.4f | Emo Acc: %.4f | Int F1: %.4f | Act F1: %.4f", 
                    split_name, avg_loss, metrics["emotion_accuracy"], metrics["intention_macro_f1"], metrics["action_macro_f1"])
    return avg_loss, metrics

# =============================================================================
# Main Execution
# =============================================================================
def run_seed(seed, config, output_dir, rank, world_size, local_rank, device, logger):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[Seed %s] Booting Silver Standard Pipeline...", seed)

    # Hardcoded perfectly to the Curated Datasets
    sources = ["mine_curated", "fane"]

    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=config.get("batch_size", 16),
        eval_batch_size=config.get("eval_batch_size", 32),
        num_workers=config.get("num_workers", 4),
        distributed=(world_size > 1),
        sources=sources,
    )

    if rank == 0: logger.info("📊 Calculating Inverse Class Weights for Focal/BCE Loss...")
    pw_int, pw_act = compute_dynamic_pos_weights(train_loader, device)
    
    criterion = SilverStandardLoss(
        pos_weight_intent=pw_int, pos_weight_action=pw_act,
        emo_w=config.get("emotion_weight", 1.0), int_w=config.get("intention_weight", 2.0), act_w=config.get("action_weight", 2.0)
    )

    model = AdvancedBEARModel(hidden_dim=config.get("hidden_dim", 1024)).to(device)
    if world_size > 1: model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 3e-5), weight_decay=config.get("weight_decay", 0.05))
    total_steps = max(1, len(train_loader) * config.get("epochs", 6))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.15), num_training_steps=total_steps)

    best_val_loss, best_epoch, seed_metrics = float("inf"), -1, []

    for epoch in range(1, config.get("epochs", 6) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, rank, logger, epoch, config.get("fp16", True))
        val_loss, val_metrics = evaluate_one_epoch(model, val_loader, criterion, device, rank, logger, "Validation")

        if rank == 0:
            seed_metrics.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss), **{k: float(v) for k, v in val_metrics.items()}})
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                torch.save((model.module if isinstance(model, DDP) else model).state_dict(), seed_dir / "best_model.pt")
                logger.info("🎯 [Seed %s] New Best Model! Epoch %d (val_loss=%.4f)", seed, epoch, val_loss)

    if (seed_dir / "best_model.pt").exists():
        (model.module if isinstance(model, DDP) else model).load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=device))
    
    test_loss, test_metrics = evaluate_one_epoch(model, test_loader, criterion, device, rank, logger, "Test")
    if rank == 0:
        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"epochs": seed_metrics, "best_epoch": best_epoch, "test": {k: float(v) for k, v in test_metrics.items()}}, f, indent=2)
    return seed_metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal_cloud.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/results-final")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    with open(args.config, "r", encoding="utf-8") as f: config = json.load(f)
    
    config.update({"epochs": args.epochs, "batch_size": args.batch_size, "seeds": args.seeds, "num_workers": args.num_workers})
    logger = setup_logging(rank, args.output_dir)

    if rank == 0:
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║ BMVC 2026 - SILVER STANDARD PIPELINE                               ║")
        logger.info("║ Core: DINOv2 + RoBERTa-Large + Inverse Weighted Loss               ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")

    for seed in config.get("seeds", [41, 42, 43]):
        run_seed(seed, config, args.output_dir, rank, world_size, local_rank, device, logger)

    if rank == 0:
        logger.info("✅ TRAINING COMPLETE. Aggregating PDF Data...")
        try:
            generate_research_report_pdf(args.output_dir, str(Path(args.output_dir) / "summary.json"), args.config)
        except Exception: pass

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
