#!/usr/bin/env python3

import sys
import torch
import numpy as np
from pathlib import Path

# Fix paths to load your project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

# Define the exact names from your paper's Table 3
EMOTION_CLASSES = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", 
    "Sad", "Surprise", "Confused", "Shy"
]

INTENTION_CLASSES = [
    "Informing/Stating", "Seeking Information", "Requesting Help", "Complaining", 
    "Agreeing", "Disagreeing", "Warning", "Greeting", 
    "Apologizing", "Suggesting", "Expressing Gratitude", "Expressing Confusion"
]

ACTION_CLASSES = [
    "No Action/Still", "Standing", "Sitting", "Walking", "Running", 
    "Pointing", "Typing/Texting", "Shouting/Yelling", "Crying", "Smiling/Laughing", 
    "Holding an Object", "Looking Away", "Gesturing", "Waving", "Reading/Examining"
]

def get_dataloader_totals(dataloader):
    # Initialize empty arrays/dictionaries for counting
    emo_counts = np.zeros(len(EMOTION_CLASSES), dtype=int)
    int_counts = np.zeros(len(INTENTION_CLASSES), dtype=int)
    act_counts = np.zeros(len(ACTION_CLASSES), dtype=int)

    for batch in dataloader:
        # 1. EMOTION (Single-Label)
        emos = batch["emotion_labels"].cpu().numpy()
        for e in emos:
            if 0 <= e < len(EMOTION_CLASSES):
                emo_counts[e] += 1

        # 2. INTENTION (Multi-Label)
        ints = batch["intention_labels"].cpu().numpy()
        int_counts += ints.sum(axis=0).astype(int)

        # 3. ACTION (Multi-Label)
        acts = batch["action_labels"].cpu().numpy()
        act_counts += acts.sum(axis=0).astype(int)

    return emo_counts, int_counts, act_counts

def generate_perfect_table_3():
    print("📦 Loading Datasets... (This will take a minute to scan all images)")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=32, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    print("⏳ Scanning Training Data...")
    tr_emo, tr_int, tr_act = get_dataloader_totals(train_loader)
    
    print("⏳ Scanning Validation Data...")
    val_emo, val_int, val_act = get_dataloader_totals(val_loader)
    
    print("⏳ Scanning Testing Data...")
    te_emo, te_int, te_act = get_dataloader_totals(test_loader)

    # Aggregate Grand Totals
    total_emo = tr_emo + val_emo + te_emo
    total_int = tr_int + val_int + te_int
    total_act = tr_act + val_act + te_act

    # Print the Final Publication-Ready Table
    print("\n" + "="*60)
    print("📊 TABLE 3: CLASS-WISE DISTRIBUTION (GRAND TOTALS)")
    print("="*60)
    print(f"{'Task':<15} {'ID':<5} {'Class':<25} {'Samples':>10}")
    print("-" * 60)

    # Print Emotion
    print("Emotion")
    for i, name in enumerate(EMOTION_CLASSES):
        # We use i+1 because your table starts IDs at 1, not 0
        print(f"{'':<15} {i+1:<5} {name:<25} {total_emo[i]:>10,}")
    print("-" * 60)

    # Print Intention
    print("Intention")
    for i, name in enumerate(INTENTION_CLASSES):
        print(f"{'':<15} {i+1:<5} {name:<25} {total_int[i]:>10,}")
    print("-" * 60)

    # Print Action
    print("Action")
    for i, name in enumerate(ACTION_CLASSES):
        print(f"{'':<15} {i+1:<5} {name:<25} {total_act[i]:>10,}")
    print("="*60)
    
    print("\n💡 Tip: The numbers generated above are the ABSOLUTE TRUE COUNTS of your dataset.")
    print("You can copy these numbers directly into your BMVC paper to update the old Table 3!")

if __name__ == "__main__":
    generate_perfect_table_3()
