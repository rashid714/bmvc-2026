#!/usr/bin/env python3

import sys
import torch
import warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

# Silence sklearn zero-division warnings 
warnings.filterwarnings("ignore")

# 1. Map to your actual project files perfectly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.advanced_multimodal_bear import AdvancedBEARModel
from data.cloud_datasets import get_cloud_dataloaders

def generate_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    print("📦 Loading Silver Standard Test Dataset...")
    _, _, test_loader = get_cloud_dataloaders(
        batch_size=8, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    print("🧠 Loading Advanced BEAR Model weights (Seed 41)...")
    model = AdvancedBEARModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "professor-run" / "seed_41" / "best_model.pt"

    if not model_path.exists():
        print(f"❌ ERROR: Could not find {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("⏳ Extracting Per-Class Metrics... (This will take ~60 seconds)")
    all_int_preds, all_int_labels = [], []
    all_act_preds, all_act_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            images = batch.get("images")
            if images is not None: images = images.to(device, non_blocking=True)

            out = model(input_ids, attention_mask, images=images)

            # Using your eval.py's exact 0.4 threshold
            int_preds = (torch.sigmoid(out["intention_logits"]).cpu().numpy() > 0.4).astype(int)
            act_preds = (torch.sigmoid(out["action_logits"]).cpu().numpy() > 0.4).astype(int)

            all_int_preds.extend(int_preds)
            all_int_labels.extend(batch["intention_labels"].cpu().numpy())
            all_act_preds.extend(act_preds)
            all_act_labels.extend(batch["action_labels"].cpu().numpy())

    # Convert lists to numpy arrays to check their exact dimensions
    all_int_labels_np = np.array(all_int_labels)
    all_int_preds_np = np.array(all_int_preds)
    all_act_labels_np = np.array(all_act_labels)
    all_act_preds_np = np.array(all_act_preds)

    # 🌟 SOTA FIX: Dynamically detect the exact number of classes from your dataset!
    actual_num_intentions = all_int_labels_np.shape[1]
    actual_num_actions = all_act_labels_np.shape[1]

    # Generate the exact correct list of names so sklearn never crashes
    INTENTION_NAMES = [f"Intention_{i}" for i in range(actual_num_intentions)]
    ACTION_NAMES = [f"Action_{i}" for i in range(actual_num_actions)]

    print("\n" + "="*50)
    print("🎯 INTENTION DETECTION (The Good & The Bad)")
    print("="*50)
    int_report = classification_report(all_int_labels_np, all_int_preds_np, target_names=INTENTION_NAMES, output_dict=True, zero_division=0)
    _print_extreme_cases(int_report, INTENTION_NAMES)

    print("\n" + "="*50)
    print("🏃 ACTION PREDICTION (The Good & The Bad)")
    print("="*50)
    act_report = classification_report(all_act_labels_np, all_act_preds_np, target_names=ACTION_NAMES, output_dict=True, zero_division=0)
    _print_extreme_cases(act_report, ACTION_NAMES)

def _print_extreme_cases(report, class_names):
    scores = []
    for cls in class_names:
        if cls in report:
            scores.append((cls, report[cls]['f1-score']))
            
    scores.sort(key=lambda x: x[1])
    
    print("\n🚨 HARDEST CASES (Lowest F1):")
    for name, f1 in scores[:5]:
        print(f"  ❌ {name:<15}: {f1:.4f}")
        
    print("\n🌟 BEST CASES (Highest F1):")
    for name, f1 in reversed(scores[-5:]):
        print(f"  ✅ {name:<15}: {f1:.4f}")

if __name__ == "__main__":
    generate_report()
