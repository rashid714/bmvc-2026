#!/usr/bin/env python3

import sys
import torch
import torchvision
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.advanced_multimodal_bear import AdvancedBEARModel
from data.cloud_datasets import get_cloud_dataloaders

# Your exact taxonomies (Using shortened names for cleaner files)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTIONS = ["Informing", "Seeking_Info", "Req_Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Gratitude", "Confusion"]
ACTIONS = ["Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing", "Shouting", "Crying", "Smiling", "Holding", "Looking_Away", "Gesturing", "Waving", "Reading"]

def sanitize(name):
    """Replaces slashes and spaces to prevent OS folder-creation crashes."""
    return str(name).replace("/", "-").replace(" ", "_")

def save_error_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Create directories to save the images safely
    output_dir = project_root / "error_analysis"
    for category in ["Emotions", "Intentions", "Actions"]:
        (output_dir / category).mkdir(parents=True, exist_ok=True)

    print("📦 Loading Test Dataset (Batch Size 1 for precise extraction)...")
    _, _, test_loader = get_cloud_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=2, distributed=False, sources=["mine_curated", "fane"]
    )

    print("🧠 Loading Model Weights...")
    model = AdvancedBEARModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "professor-run" / "seed_41" / "best_model.pt"
    
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {model_path}. Did the new training finish?")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Track how many images we've saved per class (Max 5)
    MAX_SAVES = 5
    saved_counts = {
        "Emotions": {name: 0 for name in EMOTIONS},
        "Intentions": {name: 0 for name in INTENTIONS},
        "Actions": {name: 0 for name in ACTIONS}
    }

    print("📸 Hunting for misclassifications... (Scanning Test Set)")

    total_saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch.get("images")
            if images is None:
                continue # Skip text-only samples

            images = images.to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            emo_lbl = batch["emotion_labels"].item()
            int_lbls = batch["intention_labels"][0].cpu().numpy()
            act_lbls = batch["action_labels"][0].cpu().numpy()

            out = model(input_ids, attention_mask, images=images)

            # Get Predictions
            emo_pred = torch.argmax(out["emotion_logits"], dim=1).item()
            int_preds = (torch.sigmoid(out["intention_logits"]) > 0.4)[0].cpu().numpy()
            act_preds = (torch.sigmoid(out["action_logits"]) > 0.4)[0].cpu().numpy()

            # 1. Check Emotion Errors
            if emo_pred != emo_lbl and emo_lbl < len(EMOTIONS):
                true_name = EMOTIONS[emo_lbl]
                pred_name = EMOTIONS[emo_pred] if emo_pred < len(EMOTIONS) else "Unknown"
                
                if saved_counts["Emotions"][true_name] < MAX_SAVES:
                    filename = output_dir / "Emotions" / f"True_{sanitize(true_name)}_Pred_{sanitize(pred_name)}_{batch_idx}.png"
                    torchvision.utils.save_image(images[0], filename, normalize=True)
                    saved_counts["Emotions"][true_name] += 1
                    total_saved += 1

            # 2. Check Intention Errors (False Negatives - missed the action)
            for i, (true_val, pred_val) in enumerate(zip(int_lbls, int_preds)):
                if true_val == 1 and pred_val == 0 and i < len(INTENTIONS):
                    name = INTENTIONS[i]
                    if saved_counts["Intentions"][name] < MAX_SAVES:
                        filename = output_dir / "Intentions" / f"Missed_{sanitize(name)}_{batch_idx}.png"
                        torchvision.utils.save_image(images[0], filename, normalize=True)
                        saved_counts["Intentions"][name] += 1
                        total_saved += 1

            # 3. Check Action Errors (False Negatives - missed the action)
            for i, (true_val, pred_val) in enumerate(zip(act_lbls, act_preds)):
                if true_val == 1 and pred_val == 0 and i < len(ACTIONS):
                    name = ACTIONS[i]
                    if saved_counts["Actions"][name] < MAX_SAVES:
                        filename = output_dir / "Actions" / f"Missed_{sanitize(name)}_{batch_idx}.png"
                        torchvision.utils.save_image(images[0], filename, normalize=True)
                        saved_counts["Actions"][name] += 1
                        total_saved += 1

            # Print a progress update every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"   ... Scanned {batch_idx} batches. Captured {total_saved} error images so far.")

    print(f"\n✅ Done! Successfully extracted {total_saved} error images.")
    print(f"📁 Check the '{output_dir}' folder to see the visual evidence!")

if __name__ == "__main__":
    save_error_images()
