#!/usr/bin/env python3

import sys
import re
from pathlib import Path
from collections import defaultdict

# Fix paths to load your project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

def extract_image_paths(dataloader):
    paths = []
    dataset = dataloader.dataset
    for sample in dataset.samples:
        if sample.image_path:
            paths.append(str(sample.image_path))
    return paths

def get_prefix(filename):
    name = Path(filename).stem
    # Remove trailing numbers (e.g., 'shy145' -> 'shy')
    prefix = re.sub(r'[0-9]+$', '', name)
    return prefix.rstrip('_')

def run_visual_leak_check():
    print("📦 Loading Datasets...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    train_paths = extract_image_paths(train_loader)
    test_paths = extract_image_paths(test_loader)

    # Group the exact paths by their prefix
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for p in train_paths:
        train_dict[get_prefix(p)].append(p)
        
    for p in test_paths:
        test_dict[get_prefix(p)].append(p)

    train_prefixes = set(train_dict.keys())
    test_prefixes = set(test_dict.keys())
    
    prefix_overlap = train_prefixes.intersection(test_prefixes)
    
    print("\n" + "="*70)
    print("📸 EXACT LEAKED IMAGE PATHS 📸")
    print("="*70)

    if not prefix_overlap:
        print("✅ No overlaps found. The data is clean!")
        return

    print(f"⚠️ Found {len(prefix_overlap)} overlapping subjects/sequences.\n")
    
    for prefix in list(prefix_overlap):
        print(f"--- OVERLAP: '{prefix}' ---")
        
        # Grab up to 3 examples from the training set
        print("   In TRAINING SET:")
        for path in train_dict[prefix][:3]:
            print(f"      {path}")
            
        # Grab up to 3 examples from the testing set
        print("   In TESTING SET (The Leak):")
        for path in test_dict[prefix][:3]:
            print(f"      {path}")
            
        print("-" * 40 + "\n")

    print("💡 TO DO: Open a 'TRAINING SET' path and a 'TESTING SET' path from the")
    print("same overlap group on your computer. You will see they are almost identical")
    print("frames from the exact same video!")

if __name__ == "__main__":
    run_visual_leak_check()
