#!/usr/bin/env python3

import sys
import torch
from pathlib import Path

# 🌟 CRITICAL FIX: Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the actual dataloader function we know works!
from data.cloud_datasets import get_cloud_dataloaders

# Your Taxonomies
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy", "Frustrated", "Excited"]
INTENTIONS = ["Informing/Stating", "Seeking Information", "Requesting Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Expressing Gratitude", "Expressing Confusion", "Denying", "Confirming", "Instructing/Commanding", "Inquiring", "Threatening", "Consoling", "Persuading", "Promising"]
ACTIONS = ["No Action/Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing/Texting", "Shouting/Yelling", "Crying", "Smiling/Laughing", "Holding an Object", "Looking Away", "Gesturing", "Waving", "Reading/Examining"]

def main():
    print("🚀 Booting up the Dataloaders to calculate exact distribution...")
    
    try:
        train_loader, val_loader, test_loader = get_cloud_dataloaders(
            batch_size=32, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
        )
    except Exception as e:
        print(f"❌ Error loading dataloaders: {e}")
        return

    emo_counts = None
    int_counts = None
    act_counts = None

    print("⏳ Scanning all dataset splits... (This will take a minute or two)")

    # Scan through all three datasets to get the absolute total
    for loader, name in [(train_loader, "Train"), (val_loader, "Validation"), (test_loader, "Test")]:
        print(f"  👉 Counting {name} Set...")
        for batch in loader:
            emo_lbls = batch["emotion_labels"]
            int_lbls = batch["intention_labels"]
            act_lbls = batch["action_labels"]

            # Initialize the counters on the first batch dynamically
            if emo_counts is None:
                # Emotion is a 1D tensor of class indices
                emo_counts = torch.zeros(len(EMOTIONS), dtype=torch.long)
                # Intentions and Actions are 2D multi-label tensors
                int_counts = torch.zeros(int_lbls.size(1), dtype=torch.long)
                act_counts = torch.zeros(act_lbls.size(1), dtype=torch.long)

            # 1. Count Emotions (Single Label)
            batch_emo_counts = torch.bincount(emo_lbls.cpu().long(), minlength=len(emo_counts))
            # Only add up to the size of our tracking tensor
            emo_counts[:len(batch_emo_counts)] += batch_emo_counts[:len(emo_counts)]

            # 2. Count Intentions & Actions (Multi Label)
            int_counts += int_lbls.sum(dim=0).cpu().long()
            act_counts += act_lbls.sum(dim=0).cpu().long()

    # =========================================================
    # PRINTING THE FINAL RESULTS
    # =========================================================
    print("\n" + "="*60)
    print(" 🎭 EMOTION DISTRIBUTION (Single-Label)")
    print("="*60)
    for i in range(len(emo_counts)):
        name = EMOTIONS[i] if i < len(EMOTIONS) else f"Emotion_{i}"
        print(f" - [{i:02d}] {name:<25}: {emo_counts[i].item():>9,} samples")

    print("\n" + "="*60)
    print(" 🎯 INTENTION DISTRIBUTION (Multi-Label)")
    print("="*60)
    for i in range(len(int_counts)):
        name = INTENTIONS[i] if i < len(INTENTIONS) else f"Intention_{i}"
        print(f" - [{i:02d}] {name:<25}: {int_counts[i].item():>9,} samples")

    print("\n" + "="*60)
    print(" 🏃 ACTION DISTRIBUTION (Multi-Label)")
    print("="*60)
    for i in range(len(act_counts)):
        name = ACTIONS[i] if i < len(ACTIONS) else f"Action_{i}"
        print(f" - [{i:02d}] {name:<25}: {act_counts[i].item():>9,} samples")

    print("\n" + "="*60)
    print("✅ Analysis Complete!")

if __name__ == "__main__":
    main()
