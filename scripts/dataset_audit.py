#!/usr/bin/env python3

import sys
import torch
from pathlib import Path

# Ensure the project root is accessible
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

def generate_detailed_audit():
    print("\n" + "═"*80)
    print(" 🏛️  BMVC 2026: OFFICIAL MULTIMODAL DATASET AUDIT REPORT")
    print("═"*80)
    print("Initializing Dataloaders... (Reading dataset files and checking dimensions)\n")

    try:
        train_loader, val_loader, test_loader = get_cloud_dataloaders(
            batch_size=32, eval_batch_size=32, num_workers=4, distributed=False
        )
    except Exception as e:
        print(f"❌ Error loading dataloaders: {e}")
        return

    def scan_split(loader, split_name):
        total_samples = 0
        total_images = 0
        total_texts = 0
        
        print(f"⏳ Scanning all batches in the {split_name} Split...")
        for batch in loader:
            b_size = batch["input_ids"].size(0)
            total_samples += b_size
            total_texts += b_size
            
            # Count valid images
            if batch.get("images") is not None:
                total_images += batch["images"].size(0)

        # Deep Inspection: Find MINE vs FANE counts directly from our custom MultimodalSample objects
        mine_count = 0
        fane_count = 0
        
        dataset = loader.dataset
        if hasattr(dataset, 'samples'):
            for sample in dataset.samples:
                if sample.source_dataset == "MINE_Llama_Curated":
                    mine_count += 1
                elif sample.source_dataset == "FANE_Distilled":
                    fane_count += 1

        return {
            "name": split_name,
            "total": total_samples,
            "images": total_images,
            "texts": total_texts,
            "mine": mine_count,
            "fane": fane_count
        }

    # Run the scanner on all three splits
    train_data = scan_split(train_loader, "TRAIN")
    val_data = scan_split(val_loader, "VALIDATION")
    test_data = scan_split(test_loader, "TEST")

    # Safely calculate grand totals
    grand_mine = train_data['mine'] + val_data['mine'] + test_data['mine']
    grand_fane = train_data['fane'] + val_data['fane'] + test_data['fane']
    grand_total = train_data['total'] + val_data['total'] + test_data['total']
    grand_images = train_data['images'] + val_data['images'] + test_data['images']
    grand_texts = train_data['texts'] + val_data['texts'] + test_data['texts']

    print("\n" + "═"*80)
    print(" 📊 DETAILED BREAKDOWN PER SPLIT")
    print("═"*80)

    splits = [train_data, val_data, test_data]
    for s in splits:
        print(f" 📂 {s['name']} SPLIT:")
        print(f"    ├─ Source Breakdown:")
        print(f"    │  ├─ MINE Curated Dataset: {s['mine']:>8} samples")
        print(f"    │  └─ FANE Distilled Dataset:{s['fane']:>8} samples")
        print(f"    ├─ Modality Breakdown:")
        print(f"    │  ├─ Text Inputs (RoBERTa): {s['texts']:>8} valid texts")
        print(f"    │  └─ Image Inputs (DINOv2): {s['images']:>8} valid images")
        print(f"    └─ 🔹 TOTAL SAMPLES:         {s['total']:>8}")
        print("-" * 80)

    print(" 🏆 GRAND TOTALS (ALL SPLITS COMBINED)")
    print("═"*80)
    print(f" 🔹 TOTAL MINE CURATED:    {grand_mine:>9}")
    print(f" 🔹 TOTAL FANE DISTILLED:  {grand_fane:>9}")
    print(f" 🔹 TOTAL TEXTS PROCESSED: {grand_texts:>9}")
    print(f" 🔹 TOTAL IMAGES PROCESSED:{grand_images:>9}")
    print(f" 🚀 ABSOLUTE TOTAL SAMPLES:{grand_total:>9}")
    print("═"*80)
    
if __name__ == "__main__":
    generate_detailed_audit()
