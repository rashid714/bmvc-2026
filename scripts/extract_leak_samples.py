#!/usr/bin/env python3

import sys
import re
import shutil
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
    # Remove trailing numbers to get the "person" identifier (e.g., 'shy145' -> 'shy')
    prefix = re.sub(r'[0-9]+$', '', name)
    return prefix.rstrip('_')

def run_visual_extraction():
    print("📦 Loading Datasets...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    train_paths = extract_image_paths(train_loader)
    test_paths = extract_image_paths(test_loader)

    # Group the exact paths by their person/prefix
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for p in train_paths:
        train_dict[get_prefix(p)].append(p)
        
    for p in test_paths:
        test_dict[get_prefix(p)].append(p)

    train_prefixes = set(train_dict.keys())
    test_prefixes = set(test_dict.keys())
    
    # Find the people that exist in BOTH training and testing
    prefix_overlap = train_prefixes.intersection(test_prefixes)

    if not prefix_overlap:
        print("✅ No overlapping people found. The data is clean!")
        return

    # Create the Output Directory Structure
    output_dir = project_root / "research_paper_data" / "7_LEAK_INSPECTION"
    train_out_dir = output_dir / "TRAINING_SAME_PERSON"
    test_out_dir = output_dir / "TESTING_SAME_PERSON"

    # Clear old folders if they exist so we get a fresh start
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"📸 COPYING LEAKED IMAGES TO: {output_dir}")
    print("="*70)

    print(f"⚠️ Found {len(prefix_overlap)} overlapping people/identities.\n")
    
    for person in list(prefix_overlap):
        print(f"📁 Copying images for person: '{person}'")
        
        # Create a specific folder for this person in both Train and Test directories
        person_train_dir = train_out_dir / person
        person_test_dir = test_out_dir / person
        
        person_train_dir.mkdir(exist_ok=True)
        person_test_dir.mkdir(exist_ok=True)
        
        # Copy up to 10 examples from the training set
        for path_str in train_dict[person][:10]:
            src_path = Path(path_str)
            if src_path.exists():
                shutil.copy2(src_path, person_train_dir / src_path.name)
            
        # Copy up to 10 examples from the testing set
        for path_str in test_dict[person][:10]:
            src_path = Path(path_str)
            if src_path.exists():
                shutil.copy2(src_path, person_test_dir / src_path.name)

    print("\n✅ DONE!")
    print("Go open your file explorer and look inside this folder:")
    print(str(output_dir))
    print("Open the 'TRAINING_SAME_PERSON' folder and 'TESTING_SAME_PERSON' folder side-by-side!")

if __name__ == "__main__":
    run_visual_extraction()
