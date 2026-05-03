#!/usr/bin/env python3

import sys
from pathlib import Path

# Fix paths to load your project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

def extract_image_paths(dataloader):
    """Extracts all image paths directly from the MultimodalSample objects."""
    paths = []
    
    # Access the underlying CloudMultimodalDataset
    dataset = dataloader.dataset
    
    # Loop directly through the dataclass objects in your SilverDataset architecture
    for sample in dataset.samples:
        if sample.image_path:
            paths.append(str(sample.image_path))
            
    return paths

def run_leakage_check():
    print("📦 Loading Datasets...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    print("🔍 Extracting image filenames from Training Set...")
    train_paths = extract_image_paths(train_loader)
    
    print("🔍 Extracting image filenames from Testing Set...")
    test_paths = extract_image_paths(test_loader)

    if not train_paths or not test_paths:
        print("\n❌ Could not find image paths. Please make sure dataset.samples is accessible.")
        return

    # Convert to sets for fast comparison
    train_set = set(train_paths)
    test_set = set(test_paths)

    # 1. Check for EXACT duplicates
    exact_duplicates = train_set.intersection(test_set)
    
    print("\n" + "="*50)
    print("🚨 DATA LEAKAGE REPORT 🚨")
    print("="*50)
    print(f"Total Unique Train Images: {len(train_set)}")
    print(f"Total Unique Test Images:  {len(test_set)}")
    
    if len(exact_duplicates) > 0:
        print(f"\n❌ FATAL LEAK: Found {len(exact_duplicates)} EXACT identical images in both Train and Test!")
        print("Here are the first 5 duplicates:")
        for dup in list(exact_duplicates)[:5]:
            print(f"  - {dup}")
    else:
        print("\n✅ Good News: No exact filename duplicates found between Train and Test.")

    # 2. Check for "Flipbook" Sequence Leaks (Same prefix/Subject)
    # Assuming files are named like 'subject01_frame001.jpg' -> we check 'subject01'
    print("\n🔍 Checking for 'Flipbook' (Subject-Level) Leaks...")
    
    def get_prefix(filename):
        # Grabs the first part of a filename before the last underscore or number
        import re
        name = Path(filename).stem
        # Remove trailing numbers (e.g., 'actor1_045' -> 'actor1_')
        prefix = re.sub(r'[0-9]+$', '', name)
        # Also remove trailing underscores just in case (e.g. 'actor1_' -> 'actor1')
        return prefix.rstrip('_')

    train_prefixes = set(get_prefix(p) for p in train_set)
    test_prefixes = set(get_prefix(p) for p in test_set)
    
    prefix_overlap = train_prefixes.intersection(test_prefixes)
    
    if len(prefix_overlap) > 0:
        print(f"\n⚠️ WARNING: Found {len(prefix_overlap)} shared Subject/Video sequences across splits!")
        print("This means different frames of the SAME person are in both Train and Test.")
        print("Here are the first 5 overlapping subjects:")
        for prefix in list(prefix_overlap)[:5]:
            print(f"  - {prefix}")
    else:
        print("\n✅ Clean Split: No subjects appear to be mixed between Train and Test.")

if __name__ == "__main__":
    run_leakage_check()
