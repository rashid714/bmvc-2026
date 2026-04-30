import sys
from pathlib import Path

# Add project root to path so we can import our data module
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

def verify():
    print("🔍 Spinning up Dataloaders for a Sanity Check...\n")
    
    # We set batch_size huge just to count fast
    train_dl, val_dl, test_dl = get_cloud_dataloaders(batch_size=500, num_workers=0)

    def analyze_split(name, dataloader):
        total_samples = 0
        source_counts = {"MINE_Llama_Curated": 0, "FANE_Distilled": 0}

        print(f"\n📊 Analyzing {name} Split...")
        # Loop through the actual PyTorch dataloader exactly how the training loop does
        for batch in dataloader:
            sources = batch["source"]
            total_samples += len(sources)
            
            for s in sources:
                if s in source_counts:
                    source_counts[s] += 1
                else:
                    source_counts[s] = 1

        print(f"   ├─ Total Images: {total_samples}")
        for source, count in source_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"   ├─ {source}: {count} ({percentage:.1f}%)")

    analyze_split("TRAIN", train_dl)
    analyze_split("VALIDATION", val_dl)
    analyze_split("TEST", test_dl)
    
    print("\n✅ Verification Complete. If Validation shows FANE images, you are ready to train!")

if __name__ == "__main__":
    verify()
