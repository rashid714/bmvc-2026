#!/usr/bin/env python3

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.advanced_multimodal_bear import AdvancedBEARModel
from data.cloud_datasets import get_cloud_dataloaders

# Taxonomies
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTIONS = ["Informing", "Seeking_Info", "Req_Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Gratitude", "Confusion"]
ACTIONS = ["Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing", "Shouting", "Crying", "Smiling", "Holding", "Looking_Away", "Gesturing", "Waving", "Reading"]

def stylized_print(step, description, data_shape, human_meaning):
    print(f"\n" + "═"*80)
    print(f" 🧱 STEP {step}: {description}")
    print("═"*80)
    print(f" 📐 Tensor Shape moving through pipes: {list(data_shape)}")
    print(f" 💡 Human Meaning: {human_meaning}")
    print("-" * 80)

def step_by_step_walkthrough():
    # Keep on CPU for easier data inspection
    device = torch.device("cpu")
    print(f"\n🚀 Booting up the BEAR Methodology Engine (Device: {device})...")
    
    output_dir = project_root / "test_set_evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading necessary components strictly from Test/Models folder
    print("📦 Loading 1 Sample from Test Set...")
    _, _, test_loader = get_cloud_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=0, distributed=False, sources=["mine_curated", "fane"]
    )
    
    print("📖 Loading RoBERTa Tokenizer...")
    hf_cache = project_root / "models" / "hf_hub"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=str(hf_cache))

    print("🧠 Loading the Trained 9-Emotion BEAR Model 'Brain'...")
    model = AdvancedBEARModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "professor-run" / "seed_41" / "best_model.pt"
    if not model_path.exists():
        print(f"❌ ERROR: Cannot find {model_path}.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Get exactly one sample
    batch = next(iter(test_loader))
    images = batch.get("images")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    if images is None: print("❌ No image found in sample."); return

    print("\n" + "="*80)
    print(" 🎉 WALKTHROUGH START: Inside the Mind of BEAR")
    print("="*80)

    # =========================================================================
    # PHASE 1: THE INPUTS
    # =========================================================================
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Save the input image so you know what the model is looking at
    img_filename = output_dir / "walkthrough_input_image.png"
    import torchvision
    torchvision.utils.save_image(images[0], img_filename, normalize=True)
    
    print(f"\n 🟢 RAW INPUT RECEIVED:")
    print(f"   ├─ 💬 Text Input: \"{input_text}\"")
    print(f"   └─ 🖼️ Image Input Saved to: {img_filename}")

    stylized_print(1, "Multimodal Input Loading", images.shape, 
                   "We have received [1 Image, with 3 Colors (RGB), sized 224x224 pixels]. This is Step 1 in your diagram.")

    # =========================================================================
    # PHASE 2: THE BACKBONES (FEATURE EXTRACTION)
    # =========================================================================
    print("\n[Methodology Branching: Parallel Processing]")

    with torch.no_grad():
        # Text Pipeline (RoBERTa)
        # Manually executing forward() steps
        text_out = model.text_backbone.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the CLS token (index 0) which summarizes the sentence
        text_cls = text_out.last_hidden_state[:, 0, :] 
        
        stylized_print("2A", "RoBERTa-Large Text Backbone", text_cls.shape,
                       "RoBERTa read the text input. It output a vector of 1024 numbers that summarize the sentence's grammar, tone, and meaning.")

        # Vision Pipeline (DINOv2)
        # DINOv2 returns the [B, 768] CLS token directly
        img_feats = model.vision_backbone(images)
        
        stylized_print("2B", "DINOv2 Vision Backbone", img_feats.shape,
                       "DINOv2 looked at the input image. It output a vector of 768 numbers summarizing the visual patterns (faces, posture, objects).")

    # =========================================================================
    # PHASE 3: MODALITY ENCODERS (DIMENSION ALIGNMENT)
    # =========================================================================
    print("\n[Methodology Step: Connecting parallel branches]")
    
    with torch.no_grad():
        # RoBERTa CLS (1024) is already hidden_dim size. This encoder adds LayerNorm/Dropout
        text_embed = model.text_encoder(text_cls)
        # DINOv2 CLS (768) must be projected to hidden_dim (1024)
        image_embed = model.image_encoder(img_feats)
        
        stylized_print(3, "Modality Encoders", text_embed.shape,
                       "We used linear layers to map both the Text (1024) and Vision (768) features to the exact same 'hidden size' of 1024. Now they can mathematically interact!")

    # =========================================================================
    # PHASE 4: THE FUSION GATE (RELIABILITY & ATTENTION)
    # =========================================================================
    print("\n[Methodology Step: Fusion Engine]")
    
    with torch.no_grad():
        modalities = [text_embed, image_embed, torch.zeros_like(text_embed), torch.zeros_like(text_embed)] # Audio/Video placeholders
        
        # 1. Calculate Reliability
        reliability_scores, uncertainties = model.reliability_module(modalities)
        text_rel = reliability_scores[0, 0].item() * 100
        image_rel = reliability_scores[0, 1].item() * 100
        
        stylized_print("4A", "Reliability Module (Gating)", reliability_scores.shape,
                       f"CRITICAL STEP: The model calculated how much to trust each modality. For this sample:\n"
                       f"      - Text Reliability:  {text_rel:.1f}%\n"
                       f"      - Image Reliability: {image_rel:.1f}%")

        # 2. Apply Fusion
        fused_embed = model.fusion(modalities, reliability_scores)
        
        stylized_print("4B", "Dual-Layer Attention Fusion", fused_embed.shape,
                       "The network fused the Text and Image features into one unified 'thought' vector, weighting the Text pipeline more heavily based on the scores from Step 4A.")

    # =========================================================================
    # PHASE 5: TRI-TASK HEADS (THE FINAL PREDICTION)
    # =========================================================================
    print("\n[Methodology Branching: Tri-Task Splitting]")
    
    with torch.no_grad():
        # 1. Shared Task Representation
        shared = model.task_heads.shared_encoder(fused_embed)
        
        stylized_print(5, "Shared Task Representation", shared.shape,
                       "The unified multimodal thought passes through two final deepening layers to prepare it for task splitting.")

        # 2. Final Classification Heads
        # We manually executing the parts of TaskHeads to show shapes
        emo_logits = model.task_heads.emotion_head(shared)
        int_logits = model.task_heads.intention_head(shared)
        act_logits = model.task_heads.action_head(shared)
        
        # Emo (9), Intention (12), Action (15)
        out_shapes = f"[Emo: {emo_logits.shape}, Int: {int_logits.shape}, Act: {act_logits.shape}]"
        
        stylized_print(6, "Tri-Task Heads (Output Splitting)", out_shapes,
                       "The Shared thought was split into 3 paths. Each path analyzed the thought for its specific task. These outputs are raw 'Logits' (unprocessed numbers).")

    # =========================================================================
    # PHASE 6: DECODING (TURNING NUMBERS INTO BEHAVIOR)
    # =========================================================================
    print("\n[Methodology Step: Decoding & Human Interpretation]")
    
    with torch.no_grad():
        # Temperature scaling applied in model forward, applying here for accuracy
        temperature = torch.clamp(model.temperature, min=1e-3)
        emo_scaled = emo_logits / temperature
        
        # 1. Turn Emotion Logits into % (Softmax)
        emo_probs = torch.softmax(emo_scaled, dim=1)[0]
        # 2. Turn Intention/Action Logits into Checklists (Sigmoid)
        int_probs = torch.sigmoid(int_logits)[0]
        act_probs = torch.sigmoid(act_logits)[0]

        # Get final English words
        best_emo_idx = torch.argmax(emo_probs).item()
        pred_emo = EMOTIONS[best_emo_idx]
        emo_conf = emo_probs[best_emo_idx].item() * 100

        pred_ints = [INTENTIONS[i] for i in range(len(INTENTIONS)) if int_probs[i] > 0.4]
        pred_acts = [ACTIONS[i] for i in range(len(ACTIONS)) if act_probs[i] > 0.4]

        # Ground Truth
        true_emo = EMOTIONS[batch["emotion_labels"].item()]
        true_ints = [INTENTIONS[i] for i, val in enumerate(batch["intention_labels"][0]) if val == 1]
        true_acts = [ACTIONS[i] for i, val in enumerate(batch["action_labels"][0]) if val == 1]

        stylized_print(7, "Output Decoding", "[Decoding Numbers to Taxonomy]",
                       "We applied Softmax/Sigmoid to turn the raw logits (Step 6) into useful behavior percentages. This represents the last block in your diagram.")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print(" 🏆 WALKTHROUGH COMPLETE: Final Network Decision")
    print("="*80)
    print(f" 🗣️ \"{input_text}\"")
    print("-" * 80)
    print(f" 🎭 EMOTION PREDICTION:\n     - Guess: {pred_emo} (True: {true_emo})\n     - Confidence: {emo_conf:.1f}%")
    print("-" * 80)
    print(f" 🎯 INTENTION PREDICTIONS:\n     - Guesses: {', '.join(pred_ints) if pred_ints else 'None'}\n     - True GT: {', '.join(true_ints) if true_ints else 'None'}")
    print("-" * 80)
    print(f" 🏃 ACTION PREDICTIONS:\n     - Guesses: {', '.join(pred_acts) if pred_acts else 'None'}\n     - True GT: {', '.join(true_acts) if true_acts else 'None'}")
    print("="*80 + "\n")
    print("🎨 Diagram Hint: Every 'stylized_print' in this script should be a box in your methodology diagram, linked by arrows!")

if __name__ == "__main__":
    step_by_step_walkthrough()
