#!/usr/bin/env python3

import sys
import torch
from pathlib import Path

# 🌟 CRITICAL FIX: Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.cloud_datasets import get_cloud_dataloaders

# These are just translation dictionaries. 
# The script will automatically figure out how many to actually use!
EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTION_NAMES = ["Informing/Stating", "Seeking Information", "Requesting Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Expressing Gratitude", "Expressing Confusion"]
ACTION_NAMES = ["No Action/Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing/Texting", "Shouting/Yelling", "Crying", "Smiling/Laughing", "Holding an Object", "Looking Away", "Gesturing", "Waving", "Reading/Examining"]

def main():
    print("🚀 Booting up the Dataloaders for AUTO-DETECTION...")
    
    try:
        train_loader, val_loader, test_loader = get_cloud_dataloaders(
            batch_size=32, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
        )
    except Exception as e:
        print(f"❌ Error loading dataloaders: {e}")
        return

    # We will let the dataset tell us exactly how big these should be
    emo_counts = torch.zeros(50, dtype=torch.long)  # Temp buffer
    int_counts = None
    act_counts = None
    max_emo_idx = 0

    print("⏳ Scanning datasets to auto-detect dimensions... (This takes a minute)")

    for loader, name in [(train_loader, "Train"), (val_loader, "Validation"), (test_loader, "Test")]:
        print(f"  👉 Scanning {name} Set...")
        for batch in loader:
            emo_lbls = batch["emotion_labels"]
            int_lbls = batch["intention_labels"]
            act_lbls = batch["action_labels"]

            # 🌟 AUTO-DETECT SHAPES ON THE FIRST BATCH
            if int_counts is None:
                detected_intentions = int_lbls.size(1)
                detected_actions = act_lbls.size(1)
                print(f"     ✅ Auto-Detected {detected_intentions} Intentions & {detected_actions} Actions from tensor shapes!")
                int_counts = torch.zeros(detected_intentions, dtype=torch.long)
                act_counts = torch.zeros(detected_actions, dtype=torch.long)

            # 1. Count Emotions (Single Label)
            current_max = int(torch.max(emo_lbls).item())
            if current_max > max_emo_idx:
                max_emo_idx = current_max
            
            batch_emo_counts = torch.bincount(emo_lbls.cpu().long(), minlength=50)
            emo_counts += batch_emo_counts

            # 2. Count Intentions & Actions (Multi Label)
            int_counts += int_lbls.sum(dim=0).cpu().long()
            act_counts += act_lbls.sum(dim=0).cpu().long()

    # Trim the emotion counts to the exact maximum class detected in the data
    emo_counts = emo_counts[:max_emo_idx + 1]
    print(f"     ✅ Auto-Detected {len(emo_counts)} Emotions from the data!")

    # =========================================================
    # PRINTING THE FINAL RESULTS
    # =========================================================
    print("\n" + "="*60)
    print(" 🎭 EMOTION DISTRIBUTION (Single-Label)")
    print("="*60)
    for i in range(len(emo_counts)):
        name = EMOTION_NAMES[i] if i < len(EMOTION_NAMES) else f"Emotion_{i}"
        print(f" - [{i:02d}] {name:<25}: {emo_counts[i].item():>9,} samples")

    print("\n" + "="*60)
    print(" 🎯 INTENTION DISTRIBUTION (Multi-Label)")
    print("="*60)
    for i in range(len(int_counts)):
        name = INTENTION_NAMES[i] if i < len(INTENTION_NAMES) else f"Intention_{i}"
        print(f" - [{i:02d}] {name:<25}: {int_counts[i].item():>9,} samples")

    print("\n" + "="*60)
    print(" 🏃 ACTION DISTRIBUTION (Multi-Label)")
    print("="*60)
    for i in range(len(act_counts)):
        name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"Action_{i}"
        print(f" - [{i:02d}] {name:<25}: {act_counts[i].item():>9,} samples")

    print("\n" + "="*60)
    print("✅ Analysis Complete!")

if __name__ == "__main__":
    main()
