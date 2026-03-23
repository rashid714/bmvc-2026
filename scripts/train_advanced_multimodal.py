#!/usr/bin/env python3
"""
BMVC 2026 - Advanced Multimodal Training
Fully Automated: No Code Needed
Uses Dual-Layer LLM + Automatic PDF Report Generation
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Import advanced model
from models.advanced_multimodal_bear import AdvancedBEARModel
from training.losses import MultiTaskLoss
from training.eval import evaluate_tritask
from data.cloud_datasets import get_cloud_dataloaders
from training.pdf_report_generator import generate_research_report_pdf, generate_raw_data_export


# ═══════════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return rank, world_size, device


def setup_logging(rank, output_dir):
    """Setup logging."""
    log_path = Path(output_dir) / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if rank == 0:
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger


def log_download_plan(logger, config):
    """Log which LLMs and datasets will be downloaded on first run."""
    llms = config.get("llm_downloads", ["roberta-large", "distilroberta-base"])
    dataset_sources = config.get("cloud_sources", ["mine", "emoticon", "raza"])
    profile = config.get("dataset_profile", "balanced")

    logger.info("LLMs to auto-download (first run only, then cached): %s", ", ".join(llms))
    logger.info("Dataset sources to auto-download: %s", ", ".join(dataset_sources))
    logger.info("Dataset profile: %s", profile)

    if profile in {"large_20gb", "ultra_30gb"}:
        logger.info("Large dataset mode enabled: expected cache usage ~20-30GB on first run.")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, rank, logger, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        intention_labels = batch["intention_labels"].to(device)
        action_labels = batch["action_labels"].to(device)
        
        # Optional multimodal features
        image_features = batch.get("image_features", None)
        audio_features = batch.get("audio_features", None)
        video_features = batch.get("video_features", None)
        
        if image_features is not None:
            image_features = image_features.to(device)
        if audio_features is not None:
            audio_features = audio_features.to(device)
        if video_features is not None:
            video_features = video_features.to(device)
        
        # Forward pass
        model_output = model(
            input_ids, attention_mask,
            image_features=image_features,
            audio_features=audio_features,
            video_features=video_features
        )
        
        # Compute loss
        loss_dict = criterion(
            emotion_logits=model_output["emotion_logits"],
            intention_logits=model_output["intention_logits"],
            action_logits=model_output["action_logits"],
            emotion_labels=emotion_labels,
            intention_labels=intention_labels,
            action_labels=action_labels,
        )
        loss = loss_dict["total_loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if rank == 0 and batch_idx % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
    
    return total_loss / len(train_loader)


def evaluate_one_epoch(model, val_loader, device, rank, logger):
    """Evaluate one epoch."""
    model.eval()
    all_emotion_preds = []
    all_emotion_labels = []
    all_intention_preds = []
    all_intention_labels = []
    all_action_preds = []
    all_action_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            intention_labels = batch["intention_labels"].to(device)
            action_labels = batch["action_labels"].to(device)
            
            image_features = batch.get("image_features", None)
            audio_features = batch.get("audio_features", None)
            video_features = batch.get("video_features", None)
            
            if image_features is not None:
                image_features = image_features.to(device)
            if audio_features is not None:
                audio_features = audio_features.to(device)
            if video_features is not None:
                video_features = video_features.to(device)
            
            # Forward pass
            model_output = model(
                input_ids, attention_mask,
                image_features=image_features,
                audio_features=audio_features,
                video_features=video_features
            )
            
            # Get predictions
            emotion_preds = torch.argmax(model_output["emotion_logits"], dim=1)
            intention_preds = (torch.sigmoid(model_output["intention_logits"]) > 0.5).long()
            action_preds = (torch.sigmoid(model_output["action_logits"]) > 0.5).long()
            
            all_emotion_preds.append(emotion_preds.cpu().numpy())
            all_emotion_labels.append(emotion_labels.cpu().numpy())
            all_intention_preds.append(intention_preds.cpu().numpy())
            all_intention_labels.append(intention_labels.cpu().numpy())
            all_action_preds.append(action_preds.cpu().numpy())
            all_action_labels.append(action_labels.cpu().numpy())
    
    # Compute metrics
    emotion_preds = np.concatenate(all_emotion_preds)
    emotion_labels = np.concatenate(all_emotion_labels)
    intention_preds = np.concatenate(all_intention_preds)
    intention_labels = np.concatenate(all_intention_labels)
    action_preds = np.concatenate(all_action_preds)
    action_labels = np.concatenate(all_action_labels)
    
    metrics = evaluate_tritask(
        emotion_preds, emotion_labels,
        intention_preds, intention_labels,
        action_preds, action_labels
    )
    
    if rank == 0:
        logger.info(f"Validation - Emotion Acc: {metrics['emotion_accuracy']:.4f}")
        logger.info(f"Validation - Intention F1: {metrics['intention_micro_f1']:.4f}")
        logger.info(f"Validation - Action F1: {metrics['action_micro_f1']:.4f}")
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_seed(seed, config, output_dir, rank, world_size, device, logger):
    """Run training for one seed."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Seed {seed}] Starting training...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=config.get("batch_size", 32),
        eval_batch_size=config.get("eval_batch_size", 64),
        num_workers=config.get("num_workers", 4),
        max_rows_per_source=config.get("max_rows_per_source", 5000),
        distributed=(world_size > 1),
        sources=config.get("cloud_sources", ["mine", "emoticon", "raza"]),
        dataset_profile=config.get("dataset_profile", "balanced"),
        cache_dir=config.get("hf_cache_dir")
    )
    
    # Create model (USING ADVANCED MODEL!)
    model = AdvancedBEARModel(hidden_dim=1024)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        emotion_weight=config.get("emotion_weight", 1.0),
        intention_weight=config.get("intention_weight", 1.2),
        action_weight=config.get("action_weight", 1.0),
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    total_steps = len(train_loader) * config.get("epochs", 4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.get("warmup_fraction", 0.1)),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_loss = float('inf')
    seed_metrics = []
    
    for epoch in range(config.get("epochs", 4)):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, rank, logger, epoch+1
        )
        
        # Validate
        val_metrics = evaluate_one_epoch(model, val_loader, device, rank, logger)
        
        if rank == 0:
            seed_metrics.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics
            })
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                checkpoint_path = seed_dir / "best_model.pt"
                torch.save(model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(), checkpoint_path)
                logger.info(f"[Seed {seed}] Saved best model at epoch {epoch+1}")
    
    # Test evaluation
    if rank == 0:
        model_state = model.state_dict() if not isinstance(model, DDP) else model.module.state_dict()
        test_metrics = evaluate_one_epoch(model, test_loader, device, rank, logger)
        
        logger.info(f"[Seed {seed}] FINAL TEST RESULTS:")
        logger.info(f"  Test Emotion Accuracy: {test_metrics['emotion_accuracy']:.4f}")
        logger.info(f"  Test Intention F1: {test_metrics['intention_micro_f1']:.4f}")
        logger.info(f"  Test Action F1: {test_metrics['action_micro_f1']:.4f}")
        
        # Save metrics
        metrics_path = seed_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "epochs": seed_metrics,
                "test": test_metrics
            }, f, indent=2)
    
    return seed_metrics


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="BMVC 2026 Advanced Multimodal Training")
    parser.add_argument("--config", type=str, default="configs/multimodal_cloud.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/results-final")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument("--dataset-profile", type=str, default=None, choices=["balanced", "large_20gb", "ultra_30gb"])
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override with CLI args
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.max_rows_per_source is not None:
        config["max_rows_per_source"] = args.max_rows_per_source
    if args.dataset_profile is not None:
        config["dataset_profile"] = args.dataset_profile
    
    # Setup logging
    logger = setup_logging(rank, args.output_dir)
    
    if rank == 0:
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║   BMVC 2026 - ADVANCED MULTIMODAL TRAINING                        ║")
        logger.info("║   Dual-Layer LLM + Automatic PDF Generation                      ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        log_download_plan(logger, config)
    
    # Multi-seed training
    all_seed_metrics = {}
    
    for seed in config.get("seeds", [41, 42, 43]):
        seed_metrics = run_seed(seed, config, args.output_dir, rank, world_size, device, logger)
        all_seed_metrics[seed] = seed_metrics
    
    # Aggregate results
    if rank == 0:
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║   TRAINING COMPLETE - AGGREGATING RESULTS                         ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")
        
        # Collect test metrics from all seeds
        emotion_accs = []
        intention_f1s = []
        action_f1s = []
        
        for seed in config.get("seeds", [41, 42, 43]):
            metrics_path = Path(args.output_dir) / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    emotion_accs.append(metrics["test"]["emotion_accuracy"])
                    intention_f1s.append(metrics["test"]["intention_micro_f1"])
                    action_f1s.append(metrics["test"]["action_micro_f1"])
        
        # Compute mean and std
        summary = {
            "test_emotion_accuracy_mean": float(np.mean(emotion_accs)),
            "test_emotion_accuracy_std": float(np.std(emotion_accs)),
            "test_intention_f1_mean": float(np.mean(intention_f1s)),
            "test_intention_f1_std": float(np.std(intention_f1s)),
            "test_action_f1_mean": float(np.mean(action_f1s)),
            "test_action_f1_std": float(np.std(action_f1s)),
            "config": config,
        }
        
        # Save summary
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ FINAL RESULTS (Mean ± Std across {len(config.get('seeds', []))} seeds):")
        logger.info(f"   Test Emotion Accuracy: {summary['test_emotion_accuracy_mean']:.4f} ± {summary['test_emotion_accuracy_std']:.4f}")
        logger.info(f"   Test Intention F1: {summary['test_intention_f1_mean']:.4f} ± {summary['test_intention_f1_std']:.4f}")
        logger.info(f"   Test Action F1: {summary['test_action_f1_mean']:.4f} ± {summary['test_action_f1_std']:.4f}")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # AUTO-GENERATE PDF REPORTS
        # ═══════════════════════════════════════════════════════════════════════════
        
        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║   GENERATING AUTOMATED REPORTS (PDF + CSV + LaTeX)                ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")
        
        try:
            # PDF Report
            pdf_path = generate_research_report_pdf(
                args.output_dir,
                str(summary_path),
                args.config
            )
            logger.info(f"✅ PDF Report: {pdf_path}")
            
            # CSV + LaTeX exports
            csv_path, latex_path = generate_raw_data_export(args.output_dir, str(summary_path))
            logger.info(f"✅ CSV Export: {csv_path}")
            logger.info(f"✅ LaTeX Export: {latex_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not generate reports: {e}")
            logger.info("   (Install reportlab: pip install reportlab pillow)")
        
        # Save config
        config_path = Path(args.output_dir) / "run_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("")
        logger.info(f"📂 Artifacts saved to: {args.output_dir}")
        logger.info(f"   - summary.json (publication-ready metrics)")
        logger.info(f"   - RESEARCH_RESULTS_REPORT.pdf (for paper)")
        logger.info(f"   - RESULTS_TABLE.csv (for tables)")
        logger.info(f"   - RESULTS_LATEX_TABLE.txt (for LaTeX)")
        logger.info(f"   - seed_**/best_model.pt (trained weights)")
        logger.info(f"   - seed_**/metrics.json (per-seed results)")
        logger.info(f"   - run_config.json (exact configuration)")

        # ═══════════════════════════════════════════════════════════════════════════
        # ORGANIZE RESEARCH PAPER FOLDER (FOR IMMEDIATE PAPER WRITING)
        # ═══════════════════════════════════════════════════════════════════════════
        
        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║   ORGANIZING RESEARCH PAPER FOLDER                                ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")
        
        try:
            # Import and run the paper organizer
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            
            from organize_paper_data import create_research_paper_folder
            
            paper_output = Path(args.output_dir).parent / "research_paper_data"
            create_research_paper_folder(args.output_dir, str(paper_output))
            
            logger.info("")
            logger.info(f"✅ Research paper folder ready: {paper_output}")
            logger.info(f"   📖 See: {paper_output}/README_FOR_PAPER_WRITING.md")
            logger.info(f"   📊 Tables: {paper_output}/1_RESULTS_TABLES/")
            logger.info(f"   📝 Templates: {paper_output}/5_PAPER_TEMPLATE/")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not organize paper folder: {e}")
            logger.info("   (Manually run: python scripts/organize_paper_data.py)")

if __name__ == "__main__":
    main()
