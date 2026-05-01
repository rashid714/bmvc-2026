#!/usr/bin/env python3

import sys
import torch
import numpy as np
import warnings
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score, accuracy_score, f1_score

# Silence zero-division warnings 
warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.advanced_multimodal_bear import AdvancedBEARModel
from data.cloud_datasets import get_cloud_dataloaders

def run_quantitative_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    print("📦 Loading Full Test Dataset...")
    _, _, test_loader = get_cloud_dataloaders(
        batch_size=8, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    print("📖 Loading RoBERTa Tokenizer...")
    hf_cache = project_root / "models" / "hf_hub"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=str(hf_cache))

    # Pre-compute a completely blank text input to "blind" the text pipeline
    blank_text = tokenizer("", return_tensors="pt").to(device)

    print("🧠 Loading the Best Model (Seed 42)...")
    model = AdvancedBEARModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "professor-run" / "seed_42" / "best_model.pt"
    
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    def evaluate_condition(modality_name, blind_text=False, blind_vision=False):
        print(f"\n⏳ Evaluating: {modality_name} (Running full test set...)")
        all_emo_preds, all_emo_labels = [], []
        all_int_preds, all_int_probs, all_int_labels = [], [], []
        all_act_preds, all_act_probs, all_act_labels = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch.get("images")
                if images is not None: images = images.to(device)

                # APPLY THE BLINDING
                if blind_text:
                    b_size = input_ids.size(0)
                    input_ids = blank_text["input_ids"].expand(b_size, -1).to(device)
                    attention_mask = blank_text["attention_mask"].expand(b_size, -1).to(device)
                if blind_vision:
                    images = None

                out = model(input_ids, attention_mask, images=images)

                # 1. Emotion Predictions
                emo_preds = torch.argmax(out["emotion_logits"], dim=1).cpu().numpy()
                all_emo_preds.extend(emo_preds)
                all_emo_labels.extend(batch["emotion_labels"].cpu().numpy())

                # 2. Intention Predictions
                int_probs = torch.sigmoid(out["intention_logits"]).cpu().numpy()
                int_preds = (int_probs > 0.4).astype(int)
                all_int_probs.extend(int_probs)
                all_int_preds.extend(int_preds)
                all_int_labels.extend(batch["intention_labels"].cpu().numpy())

                # 3. Action Predictions
                act_probs = torch.sigmoid(out["action_logits"]).cpu().numpy()
                act_preds = (act_probs > 0.4).astype(int)
                all_act_probs.extend(act_probs)
                all_act_preds.extend(act_preds)
                all_act_labels.extend(batch["action_labels"].cpu().numpy())

        # CALCULATE FINAL METRICS
        emo_acc = accuracy_score(all_emo_labels, all_emo_preds) * 100
        int_f1 = f1_score(all_int_labels, all_int_preds, average="macro", zero_division=0) * 100
        int_map = average_precision_score(all_int_labels, all_int_probs, average="macro") * 100
        act_f1 = f1_score(all_act_labels, all_act_preds, average="macro", zero_division=0) * 100
        act_map = average_precision_score(all_act_labels, all_act_probs, average="macro") * 100

        print(f"✅ {modality_name} Final Numbers:")
        print(f"   ├─ Emotion Acc:  {emo_acc:.2f}%")
        print(f"   ├─ Intention F1: {int_f1:.2f}%  |  mAP: {int_map:.2f}%")
        print(f"   └─ Action F1:    {act_f1:.2f}%  |  mAP: {act_map:.2f}%")

    # Run the two blinded studies
    evaluate_condition("🖼️ VISION ONLY (DINOv2) [Text Blinded]", blind_text=True, blind_vision=False)
    evaluate_condition("💬 TEXT ONLY (RoBERTa) [Vision Blinded]", blind_text=False, blind_vision=True)

if __name__ == "__main__":
    run_quantitative_ablation()
