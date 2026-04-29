#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

def generate_bmvc_plots():
    # 1. SMART PATH FINDER: Guarantees it stays inside bmvc-2026
    current_dir = Path.cwd()
    if current_dir.name == "scripts":
        project_root = current_dir.parent
    else:
        project_root = current_dir

    training_path = project_root / "checkpoints" / "professor-run"
    output_dir = project_root / "research_paper_data" / "6_VISUAL_GUIDES"
    
    print("\n🚀 STARTING GRAPH GENERATOR...")
    print(f"🔍 Looking for training data in: {training_path}")
    
    # 2. AGGRESSIVE FOLDER CREATION
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving graphs directly to: {output_dir}\n")
    
    all_metrics = []
    # 3. Checks for all possible seed names (1, 2, 3 or 41, 42, 43)
    for seed in [1, 2, 3, 41, 42, 43]:
        metric_file = training_path / f"seed_{seed}" / "metrics.json"
        if metric_file.exists():
            print(f"   ✅ Found data for Seed {seed}")
            with open(metric_file, "r", encoding="utf-8") as f:
                all_metrics.append(json.load(f))
                
    if not all_metrics:
        print(f"\n❌ ERROR: Could not find any metrics.json files inside {training_path}")
        print("Please check that your training finished successfully!")
        return

    epochs = [ep["epoch"] for ep in all_metrics[0]["epochs"]]
    last_epoch = epochs[-1]
    
    # Calculate the mean losses
    train_loss = np.mean([[ep["train_loss"] for ep in run["epochs"]] for run in all_metrics], axis=0)
    val_loss = np.mean([[ep["val_loss"] for ep in run["epochs"]] for run in all_metrics], axis=0)

    # Plot 1: The Loss Curve (Train vs Val)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', linewidth=2.5, color='#1f77b4')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2.5, color='#d62728')
    plt.title('BMVC 2026: Training vs Validation Loss\n(Averaged across 3 seeds)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Tri-Task Focal Loss', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    loss_path = output_dir / 'bmvc_loss_curve.png'
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"\n✅ SUCCESS: Saved Loss curve to {loss_path}")
    
    # Calculate Validation Means
    val_emo = np.mean([[ep["emotion_accuracy"] for ep in run["epochs"]] for run in all_metrics], axis=0)
    val_int = np.mean([[ep["intention_macro_f1"] for ep in run["epochs"]] for run in all_metrics], axis=0)
    val_act = np.mean([[ep["action_macro_f1"] for ep in run["epochs"]] for run in all_metrics], axis=0)

    # Calculate Final Test Means
    test_emo = np.mean([run.get("test_emotion_accuracy", 0) for run in all_metrics])
    test_int = np.mean([run.get("test_intention_f1", 0) for run in all_metrics])
    test_act = np.mean([run.get("test_action_f1", 0) for run in all_metrics])

    # Plot 2: The Unified Validation + Test Curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, val_emo, label='Validation Emotion (Acc)', marker='o', color='#2ca02c', linewidth=2.5, alpha=0.7)
    plt.plot(epochs, val_int, label='Validation Intention (F1)', marker='s', color='#ff7f0e', linewidth=2.5, alpha=0.7)
    plt.plot(epochs, val_act, label='Validation Action (F1)', marker='^', color='#9467bd', linewidth=2.5, alpha=0.7)
    
    plt.plot(last_epoch, test_emo, marker='*', markersize=18, color='darkgreen', label='TEST Emotion (Acc)', linestyle='None', zorder=5)
    plt.plot(last_epoch, test_int, marker='*', markersize=18, color='darkorange', label='TEST Intention (F1)', linestyle='None', zorder=5)
    plt.plot(last_epoch, test_act, marker='*', markersize=18, color='indigo', label='TEST Action (F1)', linestyle='None', zorder=5)

    plt.title('BMVC 2026: Validation vs TEST Performance\n(Averaged across 3 seeds)', fontsize=15, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Score', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    metric_path = output_dir / 'bmvc_train_val_test_curve.png'
    plt.savefig(metric_path, dpi=300)
    plt.close()
    print(f"✅ SUCCESS: Saved perfectly unified Train/Val/Test graph to {metric_path}\n")

if __name__ == "__main__":
    generate_bmvc_plots()
