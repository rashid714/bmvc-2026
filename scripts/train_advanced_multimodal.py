#!/usr/bin/env python3
"""
BMVC 2026 - Advanced Multimodal Training
Spotlight Version: Dynamic Loading, Safe Focal Engine, Macro F1, Cosine Annealing
Uses Dual-Layer LLM + ResNet50 Vision + Automatic PDF Report Generation
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

# 🌟 UPGRADE 1: Cosine Annealing for deep convergence
from transformers import get_cosine_schedule_with_warmup
from transformers import logging as hf_logging

# Silence noisy HF warnings
hf_logging.set_verbosity_error()

from models.advanced_multimodal_bear import AdvancedBEARModel
# 🌟 UPGRADE 2: Swapped to Focal MultiTaskLoss
from training.losses import MultiTaskLoss as FocalMultiTaskLoss
from training.eval import evaluate_tritask
from data.cloud_datasets import get_cloud_dataloaders
from training.pdf_report_generator import (
    generate_research_report_pdf,
    generate_raw_data_export,
)

# =============================================================================
# Setup helpers
# =============================================================================

def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """
    Initialize distributed training if launched with torchrun.
    Returns: rank, world_size, local_rank, device
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        return rank, world_size, local_rank, device

    rank, world_size, local_rank = 0, 1, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    """
    Setup logging and avoid duplicate handlers.
    """
    log_path = Path(output_dir) / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_advanced_multimodal")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    if rank == 0:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def apply_runtime_env_from_config(config: Dict[str, Any], logger: logging.Logger | None = None) -> None:
    """
    Push important config paths into environment so loaders can use them.
    """
    if config.get("hf_cache_dir"):
        os.environ["HF_DATASETS_CACHE"] = str(config["hf_cache_dir"])
    if config.get("model_cache_dir"):
        os.environ["TRANSFORMERS_CACHE"] = str(config["model_cache_dir"])
    if config.get("mine_gdrive_root"):
        os.environ["MINE_GDRIVE_ROOT"] = str(config["mine_gdrive_root"])
    if config.get("hf_home"):
        os.environ["HF_HOME"] = str(config["hf_home"])

    if logger is not None:
        logger.info("HF_DATASETS_CACHE=%s", os.environ.get("HF_DATASETS_CACHE", ""))
        logger.info("TRANSFORMERS_CACHE=%s", os.environ.get("TRANSFORMERS_CACHE", ""))
        logger.info("MINE_GDRIVE_ROOT=%s", os.environ.get("MINE_GDRIVE_ROOT", ""))
        logger.info("HF_HOME=%s", os.environ.get("HF_HOME", ""))


def log_download_plan(logger: logging.Logger, config: Dict[str, Any]) -> None:
    llms = config.get("llm_downloads", ["roberta-large", "distilroberta-base"])
    dataset_sources = config.get(
        "cloud_sources",
        ["llama_distilled", "mine", "emoticon", "raza", "kaggle_goemotions", "kaggle_facial", "kaggle_intent"],
    )
    profile = config.get("dataset_profile", "balanced")

    logger.info("LLMs to auto-download/cache: %s", ", ".join(llms))
    logger.info("Dataset sources: %s", ", ".join(dataset_sources))
    logger.info("Dataset profile: %s", profile)
    if profile in {"large_20gb", "ultra_30gb"}:
        logger.info("Large dataset mode enabled: expected cache usage ~20-30GB on first run.")


# =============================================================================
# Training / evaluation
# =============================================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    rank,
    logger,
    epoch,
    fp16=True,
):
    model.train()
    total_loss = 0.0

    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        emotion_labels = batch["emotion_labels"].to(device, non_blocking=True)
        intention_labels = batch["intention_labels"].to(device, non_blocking=True)
        action_labels = batch["action_labels"].to(device, non_blocking=True)

        images = batch.get("images")
        if images is not None:
            images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                model_output = model(input_ids, attention_mask, images=images)
                loss_dict = criterion(
                    emotion_logits=model_output["emotion_logits"],
                    intention_logits=model_output["intention_logits"],
                    action_logits=model_output["action_logits"],
                    emotion_labels=emotion_labels,
                    intention_labels=intention_labels,
                    action_labels=action_labels,
                )
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            if scale_before <= scale_after:
                scheduler.step()
        else:
            model_output = model(input_ids, attention_mask, images=images)
            loss_dict = criterion(
                emotion_logits=model_output["emotion_logits"],
                intention_logits=model_output["intention_logits"],
                action_logits=model_output["action_logits"],
                emotion_labels=emotion_labels,
                intention_labels=intention_labels,
                action_labels=action_labels,
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += float(loss.item())

        if rank == 0 and batch_idx % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(
                "Epoch %d | Batch %d/%d | Loss: %.4f | Avg: %.4f",
                epoch,
                batch_idx,
                len(train_loader),
                float(loss.item()),
                avg_loss,
            )

    return total_loss / max(len(train_loader), 1)


@torch.no_grad()
def evaluate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    rank,
    logger,
    split_name="Validation",
):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_emotion_preds = []
    all_emotion_labels = []
    all_intention_preds = []
    all_intention_labels = []
    all_action_preds = []
    all_action_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        emotion_labels = batch["emotion_labels"].to(device, non_blocking=True)
        intention_labels = batch["intention_labels"].to(device, non_blocking=True)
        action_labels = batch["action_labels"].to(device, non_blocking=True)

        images = batch.get("images")
        if images is not None:
            images = images.to(device, non_blocking=True)

        model_output = model(input_ids, attention_mask, images=images)

        loss_dict = criterion(
            emotion_logits=model_output["emotion_logits"],
            intention_logits=model_output["intention_logits"],
            action_logits=model_output["action_logits"],
            emotion_labels=emotion_labels,
            intention_labels=intention_labels,
            action_labels=action_labels,
        )
        loss = loss_dict["total_loss"]
        total_loss += float(loss.item())
        num_batches += 1

        all_emotion_preds.append(model_output["emotion_logits"].cpu())
        all_emotion_labels.append(emotion_labels.cpu())
        
        all_intention_preds.append(model_output["intention_logits"].cpu())
        all_intention_labels.append(intention_labels.cpu())
        
        all_action_preds.append(model_output["action_logits"].cpu())
        all_action_labels.append(action_labels.cpu())

    emotion_preds = torch.cat(all_emotion_preds, dim=0)
    emotion_labels = torch.cat(all_emotion_labels, dim=0)
    
    intention_preds = torch.cat(all_intention_preds, dim=0)
    intention_labels = torch.cat(all_intention_labels, dim=0)
    
    action_preds = torch.cat(all_action_preds, dim=0)
    action_labels = torch.cat(all_action_labels, dim=0)

    metrics = evaluate_tritask(
        emotion_preds=emotion_preds,
        intention_preds=intention_preds,
        action_preds=action_preds,
        emotion_labels=emotion_labels,
        intention_labels=intention_labels,
        action_labels=action_labels,
    )
    avg_loss = total_loss / max(num_batches, 1)

    if rank == 0:
        logger.info("%s Loss: %.4f", split_name, avg_loss)
        logger.info("%s - Emotion Acc: %.4f", split_name, metrics["emotion_accuracy"])
        # 🌟 UPGRADE 4: Log Macro F1 directly
        logger.info("%s - Intention F1 (Macro): %.4f", split_name, metrics["intention_macro_f1"])
        logger.info("%s - Action F1 (Macro): %.4f", split_name, metrics["action_macro_f1"])

    return avg_loss, metrics


# =============================================================================
# Main training loop
# =============================================================================

def run_seed(seed, config, output_dir, rank, world_size, local_rank, device, logger):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[Seed %s] Starting training...", seed)

    apply_runtime_env_from_config(config, logger)

    # 🌟 CRITICAL FIX: Force Llama Distilled to be included in the dataset sources
    sources = config.get(
        "cloud_sources",
        ["llama_distilled", "mine", "emoticon", "raza", "kaggle_goemotions", "kaggle_facial", "kaggle_intent"],
    )
    if "llama_distilled" not in [s.lower() for s in sources]:
        sources.insert(0, "llama_distilled")
        logger.info("🔥 OVERRIDE: 'llama_distilled' explicitly added to data sources.")

    # 🌟 UNCAPPED DATA LOADER: Reads all available Llama (~16k) naturally
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=config.get("batch_size", 32),
        eval_batch_size=config.get("eval_batch_size", 64),
        num_workers=config.get("num_workers", 4),
        max_rows_per_source=config.get("max_rows_per_source", 40000), 
        distributed=(world_size > 1),
        sources=sources,
        data_dir=config.get("data_dir"),
    )

    model = AdvancedBEARModel(hidden_dim=config.get("hidden_dim", 1024), use_pretrained_vision=True)
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # 🌟 THE GOLDEN RATIO WEIGHTS: Strong enough to learn Llama annotations, gentle enough for Kaggle noise.
    criterion = FocalMultiTaskLoss(
        emotion_weight=config.get("emotion_weight", 1.0),    
        intention_weight=config.get("intention_weight", 2.0),  
        action_weight=config.get("action_weight", 2.0),     
    )

    # 🌟 GOLDEN LEARNING RATE
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 3e-5), 
        weight_decay=config.get("weight_decay", 0.05),     
    )

    total_steps = max(1, len(train_loader) * config.get("epochs", 4))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.get("warmup_fraction", 0.15)), 
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    seed_metrics = []

    for epoch in range(1, config.get("epochs", 4) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            rank,
            logger,
            epoch,
            config.get("fp16", True),
        )

        val_loss, val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            rank,
            logger,
            "Validation",
        )

        if rank == 0:
            seed_metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    **{k: float(v) for k, v in val_metrics.items()},
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint_path = seed_dir / "best_model.pt"
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), checkpoint_path)
                logger.info(
                    "[Seed %s] Saved best model at epoch %d (val_loss=%.4f)",
                    seed,
                    epoch,
                    val_loss,
                )

    checkpoint_path = seed_dir / "best_model.pt"
    if checkpoint_path.exists():
        model_to_load = model.module if isinstance(model, DDP) else model
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_to_load.load_state_dict(state_dict)
        if rank == 0:
            logger.info("[Seed %s] Loaded best checkpoint from epoch %d for final test.", seed, best_epoch)

    test_loss, test_metrics = evaluate_one_epoch(
        model,
        test_loader,
        criterion,
        device,
        rank,
        logger,
        "Test",
    )

    if rank == 0:
        logger.info("[Seed %s] FINAL TEST RESULTS:", seed)
        logger.info(" Test Loss: %.4f", test_loss)
        logger.info(" Test Emotion Accuracy: %.4f", test_metrics["emotion_accuracy"])
        logger.info(" Test Intention F1 (Macro): %.4f", test_metrics["intention_macro_f1"])
        logger.info(" Test Action F1 (Macro): %.4f", test_metrics["action_macro_f1"])

        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epochs": seed_metrics,
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss),
                    "test_loss": float(test_loss),
                    "test": {k: float(v) for k, v in test_metrics.items()},
                },
                f,
                indent=2,
            )

    return seed_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal_cloud.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/results-final")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default=None,
        choices=["balanced", "large_20gb", "ultra_30gb"],
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

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

    logger = setup_logging(rank, args.output_dir)

    config.setdefault("epochs", 4)
    config.setdefault("batch_size", 32)
    config.setdefault("eval_batch_size", 64)
    config.setdefault("num_workers", 4)
    config.setdefault("max_rows_per_source", 40000)  # 🌟 Dynamic Uncapped default
    config.setdefault("seeds", [41, 42, 43])
    config.setdefault("fp16", True)
    config.setdefault("warmup_fraction", 0.15)
    config.setdefault("learning_rate", 3e-5) # 🌟 Balanced LR
    config.setdefault("weight_decay", 0.05)  
    config.setdefault("hidden_dim", 1024)
    config.setdefault("emotion_weight", 1.0)
    config.setdefault("intention_weight", 2.0)
    config.setdefault("action_weight", 2.0)

    if rank == 0:
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║ BMVC 2026 - ADVANCED MULTIMODAL TRAINING (DYNAMIC LOADING)         ║")
        logger.info("║ Dual-Layer LLM + ResNet50 Vision + Auto-PDF Generation             ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")
        
        sources = config.get("cloud_sources", ["llama_distilled", "mine", "emoticon", "raza", "kaggle_goemotions", "kaggle_facial", "kaggle_intent"])
        if "llama_distilled" not in [s.lower() for s in sources]:
             sources.insert(0, "llama_distilled")
             config["cloud_sources"] = sources
        
        log_download_plan(logger, config)

    for seed in config.get("seeds", [41, 42, 43]):
        run_seed(seed, config, args.output_dir, rank, world_size, local_rank, device, logger)

    if rank == 0:
        logger.info("╔════════════════════════════════════════════════════════════════════╗")
        logger.info("║ TRAINING COMPLETE - AGGREGATING RESULTS                            ║")
        logger.info("╚════════════════════════════════════════════════════════════════════╝")

        emotion_accs, intention_f1s, action_f1s = [], [], []
        for seed in config.get("seeds", [41, 42, 43]):
            metrics_path = Path(args.output_dir) / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                emotion_accs.append(m["test"]["emotion_accuracy"])
                intention_f1s.append(m["test"]["intention_macro_f1"])
                action_f1s.append(m["test"]["action_macro_f1"])

        summary = {
            "test_emotion_accuracy_mean": float(np.mean(emotion_accs)) if emotion_accs else 0.0,
            "test_emotion_accuracy_std": float(np.std(emotion_accs)) if emotion_accs else 0.0,
            "test_intention_f1_mean": float(np.mean(intention_f1s)) if intention_f1s else 0.0,
            "test_intention_f1_std": float(np.std(intention_f1s)) if intention_f1s else 0.0,
            "test_action_f1_mean": float(np.mean(action_f1s)) if action_f1s else 0.0,
            "test_action_f1_std": float(np.std(action_f1s)) if action_f1s else 0.0,
            "config": config,
        }

        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("FINAL RESULTS:")
        logger.info(
            " Emotion Acc: %.4f ± %.4f",
            summary["test_emotion_accuracy_mean"],
            summary["test_emotion_accuracy_std"],
        )
        logger.info(
            " Intention F1 (Macro): %.4f ± %.4f",
            summary["test_intention_f1_mean"],
            summary["test_intention_f1_std"],
        )
        logger.info(
            " Action F1 (Macro): %.4f ± %.4f",
            summary["test_action_f1_mean"],
            summary["test_action_f1_std"],
        )

        try:
            generate_research_report_pdf(args.output_dir, str(summary_path), args.config)
            generate_raw_data_export(args.output_dir, str(summary_path))
        except Exception:
            pass

        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from organize_paper_data import create_research_paper_folder

            paper_output = Path(args.output_dir).parent / "research_paper_data"
            create_research_paper_folder(args.output_dir, str(paper_output))
        except Exception:
            pass

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
