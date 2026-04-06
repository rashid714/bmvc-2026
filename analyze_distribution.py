#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path
from collections import Counter

# 🌟 CRITICAL FIX: Ensure the project root is in the Python path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.cloud_datasets import UnifiedCloudDatasetBuilder

# The exact taxonomies we defined
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy", "Frustrated", "Excited"]
INTENTIONS = ["Informing/Stating", "Seeking Information", "Requesting Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Expressing Gratitude", "Expressing Confusion", "Denying", "Confirming", "Instructing/Commanding", "Inquiring (Status)", "Threatening", "Consoling/Comforting", "Persuading", "Promising"]
ACTIONS = ["No Action/Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing/Texting", "Shouting/Yelling", "Crying", "Smiling/Laughing", "Holding an Object", "Looking Away", "Gesturing", "Waving", "Reading/Examining"]

def main():
    print("🔍 Scanning local datasets... This will take a minute...")
    
    config_path = "configs/multimodal_cloud.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {config_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {config_path}")
        return

    sources = config.get("cloud_sources", ["llama_distilled", "mine", "goemotions", "dailydialog", "tweet_eval", "emoticon", "raza"])
    
    # Ensure llama_distilled is prioritized and not accidentally removed
    if "llama_distilled" not in [s.lower() for s in sources]:
        sources.insert(0, "llama_distilled")

    print(f"📂 Loading sources: {', '.join(sources)}")

    # Load exactly how the training script loads it
    samples = UnifiedCloudDatasetBuilder.build_multimodal_dataset(
        sources=sources, 
        splits={"train": 40000, "validation": 5000, "test": 5000}
    )

    if not samples:
        print("❌ Error: No samples loaded. Check your data directories.")
        return

    print(f"\n✅ Successfully loaded {len(samples):,} total samples. Calculating distributions...\n")

    emo_counts = Counter()
    int_counts = Counter()
    act_counts = Counter()

    # Safely aggregate the labels
    for s in samples:
        emo_counts[s.emotion_label] += 1
        
        for idx in (s.intention_labels or []):
            if isinstance(idx, int) and 0 <= idx < 20:
                int_counts[idx] += 1
                
        for idx in (s.action_labels or []):
            if isinstance(idx, int) and 0 <= idx < 15:
                act_counts[idx] += 1

    # Print beautifully aligned tables
    print("="*60)
    print(" 🧠 EMOTION DISTRIBUTION (Single-Label)")
    print("="*60)
    for i, name in enumerate(EMOTIONS):
        print(f" - [{i:02d}] {name:<25}: {emo_counts.get(i, 0):>9,} samples")

    print("\n" + "="*60)
    print(" 🎯 INTENTION DISTRIBUTION (Multi-Label)")
    print("="*60)
    for i, name in enumerate(INTENTIONS):
        print(f" - [{i:02d}] {name:<25}: {int_counts.get(i, 0):>9,} samples")

    print("\n" + "="*60)
    print(" 🏃 ACTION DISTRIBUTION (Multi-Label)")
    print("="*60)
    for i, name in enumerate(ACTIONS):
        print(f" - [{i:02d}] {name:<25}: {act_counts.get(i, 0):>9,} samples")
        
    print("\n" + "="*60)
    print("✅ Analysis Complete!")

if __name__ == "__main__":
    main()
