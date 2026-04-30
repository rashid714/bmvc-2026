#!/usr/bin/env python3

import json
import random
from pathlib import Path

# Setup paths based on your repo structure
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "fane"
target_json = data_dir / "distilled_annotations.json"

def balance_splits():
    if not target_json.exists():
        print(f"❌ Error: Could not find {target_json}")
        return

    print(f"📦 Loading FANE dataset from {target_json.name}...")
    
    with open(target_json, "r", encoding="utf-8") as f:
        fane_data = json.load(f)

    # 1. Shuffle the data using a set seed so it is mathematically reproducible
    random.seed(42) 
    random.shuffle(fane_data)

    total_samples = len(fane_data)
    print(f"📊 Total FANE Samples found: {total_samples}")

    # 2. Calculate the exact 80/10/10 cutoffs
    train_cutoff = int(total_samples * 0.80)
    val_cutoff = train_cutoff + int(total_samples * 0.10)

    train_count, val_count, test_count = 0, 0, 0
    
    # 3. Re-assign the "split" tag for every single image
    for i, sample in enumerate(fane_data):
        if i < train_cutoff:
            sample["split"] = "train"
            train_count += 1
        elif i < val_cutoff:
            sample["split"] = "val"
            val_count += 1
        else:
            sample["split"] = "test"
            test_count += 1

    # 4. OVERWRITE the exact same file so cloud_datasets.py finds it instantly
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(fane_data, f, indent=2)

    print("\n✅ FANE Data Splitting Complete!")
    print("New Balanced Distribution:")
    print(f"   ├─ Train: {train_count} samples")
    print(f"   ├─ Val:   {val_count} samples")
    print(f"   └─ Test:  {test_count} samples")
    print(f"\nSaved perfectly back into {target_json.name}!")
    print("You are ready to run your training script!")

if __name__ == "__main__":
    balance_splits()
