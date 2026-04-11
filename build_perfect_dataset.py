#!/usr/bin/env python3
"""
BMVC 2026 - DATASET ARCHITECT (UNATTENDED ULTRA MODE)
Teacher: Meta Llama 3.2 11B Vision-Instruct
Features: Auto-run, 1000-Sample Translated Audits, Safe Checkpointing
"""

import os
import sys
import json
import torch
import logging
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# ==============================================================================
# 1. STRICT LOCAL FOLDER MANAGEMENT
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR 

DATA_DIR = PROJECT_ROOT / "data" / "mine_gdrive"
CURATED_DIR = PROJECT_ROOT / "data" / "mine_curated"
IMG_OUT_DIR = CURATED_DIR / "images"
OUTPUT_JSON = CURATED_DIR / "mine_perfect_annotations.json"
CHECKPOINT_FILE = CURATED_DIR / "checkpoint.jsonl"
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "hf_hub"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. TAXONOMIES & PROMPT
# ==============================================================================
EMOTIONS = ["0: Angry", "1: Disgust", "2: Fear", "3: Happy", "4: Neutral", "5: Sad", "6: Surprise", "7: Confused", "8: Shy", "9: Frustrated", "10: Excited"]
INTENTIONS = ["0: Informing/Stating", "1: Seeking Information", "2: Requesting Help", "3: Complaining", "4: Agreeing", "5: Disagreeing", "6: Warning", "7: Greeting", "8: Apologizing", "9: Suggesting", "10: Expressing Gratitude", "11: Expressing Confusion", "12: Denying", "13: Confirming", "14: Instructing/Commanding", "15: Inquiring (Status)", "16: Threatening", "17: Consoling/Comforting", "18: Persuading", "19: Promising"]
ACTIONS = ["0: No Action/Still", "1: Standing", "2: Sitting", "3: Walking", "4: Running", "5: Pointing", "6: Typing/Texting", "7: Shouting/Yelling", "8: Crying", "9: Smiling/Laughing", "10: Holding an Object", "11: Looking Away", "12: Gesturing", "13: Waving", "14: Reading/Examining"]

def build_prompt(user_text):
    system_prompt = (
        "You are an elite behavioral analyst for a computer vision dataset.\n"
        "Analyze the provided image and accompanying text.\n\n"
        "CRITICAL RULE 1: If the image is a poster, graphic, scenery, or DOES NOT contain a distinct human, "
        "you MUST set 'is_human_present' to false and leave the rest empty.\n\n"
        "CRITICAL RULE 2: If a human IS present, analyze them using EXACTLY these taxonomies:\n"
        f"- Emotions (Choose exactly 1): {', '.join(EMOTIONS)}\n"
        f"- Intentions (Choose 1 to 3): {', '.join(INTENTIONS)}\n"
        f"- Actions (Choose 1 to 3): {', '.join(ACTIONS)}\n\n"
        "You MUST respond ONLY with a raw JSON object. No markdown formatting. Format:\n"
        "{\n"
        "  \"is_human_present\": true,\n"
        "  \"emotion\": [4],\n"
        "  \"intentions\": [2, 3],\n"
        "  \"actions\": [1, 10]\n"
        "}\n\n"
        f"Text Context: \"{user_text}\"\n"
        "Output ONLY JSON:"
    )
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": system_prompt}]}]

# ==============================================================================
# 3. TRANSLATED AUDIT UTILITY
# ==============================================================================
def print_translated_audit(samples):
    print("\n" + "="*70)
    print(" 🛑 1000-SAMPLE AUDIT CHECK (Showing last 3 successful)")
    print("="*70)
    
    for s in samples:
        # Translate the raw integers back into human-readable strings
        emo_str = EMOTIONS[s['emotion_label']] if s['emotion_label'] < len(EMOTIONS) else str(s['emotion_label'])
        int_strs = [INTENTIONS[i] for i in s['intention_labels'] if i < len(INTENTIONS)]
        act_strs = [ACTIONS[i] for i in s['action_labels'] if i < len(ACTIONS)]
        
        print(f"🖼️  Image : {Path(s['image_path']).name}")
        print(f"📝 Text  : '{s['text']}'")
        print(f"🧠 Emo   : [{emo_str}]")
        print(f"🎯 Intent: {int_strs}")
        print(f"🏃 Action: {act_strs}")
        print("-" * 50)
    print("="*70 + "\n")

# ==============================================================================
# 4. CORE LOGIC
# ==============================================================================
def discover_data(data_dir: Path):
    dataset_paths = []
    for root, _, files in os.walk(data_dir):
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        txt_files = [f for f in files if f.lower().endswith('.txt')]
        
        for img_file in img_files:
            img_path = Path(root) / img_file
            text_content = ""
            if txt_files:
                try: text_content = (Path(root) / txt_files[0]).read_text(errors='ignore').strip()
                except Exception: pass
            dataset_paths.append({"image_path": str(img_path), "text": text_content})
    return dataset_paths

def process_single_image(model, processor, item):
    img_path = Path(item["image_path"])
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return None, None
        
    prompt = processor.apply_chat_template(build_prompt(item["text"]), add_generation_prompt=True)
    inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    for attempt in range(3):
        try:
            output_ids = model.generate(**inputs, max_new_tokens=150, temperature=0.1)
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_text)
            return result, img_path
        except json.JSONDecodeError:
            continue
    return None, img_path

def save_valid_sample(result, img_path, user_text):
    new_img_path = IMG_OUT_DIR / img_path.name
    counter = 1
    while new_img_path.exists():
        new_img_path = IMG_OUT_DIR / f"{img_path.stem}_{counter}{img_path.suffix}"
        counter += 1
        
    shutil.copy2(img_path, new_img_path)
    
    annotation = {
        "image_path": f"images/{new_img_path.name}",
        "text": user_text,
        "emotion_label": result.get("emotion", [4])[0],
        "intention_labels": result.get("intentions", [0]),
        "action_labels": result.get("actions", [0])
    }
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(json.dumps(annotation) + "\n")
    return annotation

# ==============================================================================
# 5. EXECUTION ENGINE
# ==============================================================================
def main():
    print("="*60)
    print(" 🎓 BMVC 2026 - DATASET ARCHITECT (UNATTENDED MODE)")
    print("="*60)
    print(f"📥 READING FROM: {DATA_DIR}")
    print(f"📤 SAVING TO: {CURATED_DIR}")
    print("="*60)

    raw_data = discover_data(DATA_DIR)
    
    processed_images = set()
    valid_samples = []
    recent_audits = []
    
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            for line in f:
                item = json.loads(line)
                processed_images.add(Path(item["image_path"]).name)
                valid_samples.append(item)
                recent_audits.append(item)
    
    pending_data = [d for d in raw_data if Path(d["image_path"]).name not in processed_images]
    print(f"\n✅ Found {len(pending_data)} pending images. (Already processed {len(valid_samples)})")

    if not pending_data:
        print("Dataset generation is already complete!")
        return

    print("\n🧠 Loading Llama 3.2 11B into GPU memory...")
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=str(MODEL_CACHE_DIR))
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(MODEL_CACHE_DIR))

    print("\n🚀 STARTING UNATTENDED BATCH PROCESSING...")
    
    valid_count = len(valid_samples)

    for item in tqdm(pending_data, desc="Annotating Data"):
        result, img_path = process_single_image(model, processor, item)
        
        if result and result.get("is_human_present", False):
            saved_sample = save_valid_sample(result, img_path, item["text"])
            valid_samples.append(saved_sample)
            recent_audits.append(saved_sample)
            valid_count += 1
            
            # TRIGGER AUDIT EVERY 1000 SAMPLES
            if valid_count % 1000 == 0:
                print_translated_audit(recent_audits[-3:])

    # Final Compilation
    with open(OUTPUT_JSON, "w") as f:
        json.dump(valid_samples, f, indent=2)
        
    print(f"\n🎉 Distillation Complete! {len(valid_samples)} clean samples saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
