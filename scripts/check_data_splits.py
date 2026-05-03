#!/usr/bin/env python3

import sys
import torch
import numpy as np
from pathlib import Path

# Fix paths to load your project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

def count_split_distribution(dataloader, split_name):
    print(f"⏳ Scanning {split_name} Split...")
    
    emo_counts = {}
    int_counts = None
    act_counts = None

    for batch in dataloader:
        # 1. EMOTION (Single-Label): Count occurrences of each integer
        emos = batch["emotion_labels"].cpu().numpy()
        for e in emos:
            emo_counts[e] = emo_counts.get(e, 0) + 1

        # 2. INTENTION (Multi-Label): Sum the columns of the 1s and 0s
        ints = batch["intention_labels"].cpu().numpy()
        if int_counts is None:
            int_counts = np.zeros(ints.shape[1])
        int_counts += ints.sum(axis=0)

        # 3. ACTION (Multi-Label): Sum the columns of the 1s and 0s
        acts = batch["action_labels"].cpu().numpy()
        if act_counts is None:
            act_counts = np.zeros(acts.shape[1])
        act_counts += acts.sum(axis=0)

    return emo_counts, int_counts, act_counts

def run_distribution_check():
    print("📦 Loading Datasets (This might take a minute depending on data size)...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=32, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    splits = {
        "TRAINING": train_loader,
        "VALIDATION": val_loader,
        "TESTING": test_loader
    }

    # Store results to print nicely at the end
    results = {}

    for name, loader in splits.items():
        if loader is not None:
            e_count, i_count, a_count = count_split_distribution(loader, name)
            results[name] = {"Emotion": e_count, "Intention": i_count, "Action": a_count}

    # Print the Final Report
    print("\n" + "="*60)
    print("📊 DATASET SPLIT DISTRIBUTION REPORT")
    print("="*60)

    for split_name, data in results.items():
        print(f"\n--- {split_name} SET ---")
        
        print("\nEmotion (Single-Label):")
        # Sort by class ID for clean reading
        for class_id in sorted(data["Emotion"].keys()):
            print(f"  Class {class_id}: {data['Emotion'][class_id]} samples")

        print("\nIntention (Multi-Label):")
        for class_id, count in enumerate(data["Intention"]):
            print(f"  Class {class_id}: {int(count)} samples")

        print("\nAction (Multi-Label):")
        for class_id, count in enumerate(data["Action"]):
            print(f"  Class {class_id}: {int(count)} samples")

if __name__ == "__main__":
    run_distribution_check()
