#!/usr/bin/env python3
"""
BMVC 2026 - Elite Offline Knowledge Distillation
Teacher Model: Meta Llama 3.2 11B Vision-Instruct
Upgrades: Image-Only Fallback, Chain-of-Thought, Auto-Retry, Local Model Caching
"""

import os
import sys
import json
import csv
import torch
import signal
import logging
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# ==============================================================================
# 1. STRICT LOCAL FOLDER MANAGEMENT (Everything stays in BMVC-2026)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 

# Data Paths
DATA_DIR = PROJECT_ROOT / "data" / "mine_gdrive"
OUTPUT_JSON = PROJECT_ROOT / "data" / "distilled_annotations.json"
LOG_FILE = PROJECT_ROOT / "data" / "distillation.log"

# Model Cache Path (Forces Hugging Face to download the 11B model HERE)
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "hf_hub"

# Create all necessary folders immediately
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Setup Audit Logging
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global safe-state for Ctrl+C
global_results = []

def save_results_safely():
    if global_results:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(global_results, f, indent=2)
        logging.info(f"Safely saved {len(global_results)} annotations.")

def signal_handler(sig, frame):
    print("\n\n🛑 [INTERRUPT DETECTED] Securing data before shutdown...")
    save_results_safely()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==============================================================================
# 2. AUTO-DATA LOCATOR (WITH IMAGE-ONLY FALLBACK)
# ==============================================================================
def load_real_dataset(data_dir: Path):
    dataset_paths = []
    print(f"🔍 Scanning for dataset files deeply in: {data_dir.resolve()}...")
    
    json_files = list(data_dir.rglob("metadata.json"))
    csv_files = list(data_dir.rglob("dataset.csv"))
    
    # SCENARIO A: Text mappings exist
    if json_files or csv_files:
        for json_path in json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    img_path = json_path.parent / item["image_file"]
                    dataset_paths.append({
                        "id": str(item.get("id", len(dataset_paths))),
                        "image_path": str(img_path),
                        "text": item.get("text", "")
                    })

        for csv_path in csv_files:
            with open(csv_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    img_path = csv_path.parent / row["image_file"]
                    dataset_paths.append({
                        "id": str(row.get("id", len(dataset_paths))),
                        "image_path": str(img_path),
                        "text": row.get("text", "")
                    })
        print(f"✅ Loaded {len(dataset_paths)} multimodal samples with text maps.")
    
    # SCENARIO B: No text maps found -> Switch to IMAGE-ONLY mode
    else:
        print("⚠️ No metadata.json or dataset.csv found. Switching to IMAGE-ONLY fallback mode.")
        logging.info("Initiating Image-Only Fallback Mode.")
        
        image_extensions = ('*.jpg', '*.jpeg', '*.png')
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_dir.rglob(ext))
            
        if not image_files:
            raise FileNotFoundError(f"❌ Could not find ANY images (.jpg, .png) in {data_dir} or its subfolders!")
            
        for idx, img_path in enumerate(image_files):
            dataset_paths.append({
                "id": f"img_{idx}",
                "image_path": str(img_path),
                "text": "Analyze the visual expression and body language in this image."
            })
        print(f"✅ Loaded {len(dataset_paths)} raw images for visual-only distillation.")

    return dataset_paths

# ==============================================================================
# 3. LLAMA MODEL & CHAIN-OF-THOUGHT PROMPTING
# ==============================================================================
def load_teacher_model():
    print(f"🧠 Downloading/Loading Llama 3.2 11B into: {MODEL_CACHE_DIR}")
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=str(MODEL_CACHE_DIR)
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(MODEL_CACHE_DIR))
    return model, processor

def build_cot_prompt(user_text):
    """CHAIN OF THOUGHT UPGRADE: Forces Llama to think logically."""
    system_prompt = (
        "You are an elite psychological annotator for a computer vision research paper.\n"
        "Analyze the image and text. You must output ONLY a valid JSON object. No markdown, no chat.\n\n"
        "The JSON MUST follow this exact format with three keys:\n"
        "1. 'reasoning': A brief, 1-sentence logical explanation of the user's state.\n"
        "2. 'intention_classes': Array of integers (0-19) for the user's underlying goal.\n"
        "3. 'action_classes': Array of integers (0-14) for the best system response.\n\n"
        "--- FEW-SHOT EXAMPLES ---\n"
        "Example 1 (Angry face + 'My flight was canceled!'):\n"
        "{\"reasoning\": \"The user is visibly angry and experiencing a severe service failure, requiring compensation.\", \"intention_classes\": [4, 18], \"action_classes\": [2, 14]}\n\n"
        "Example 2 (Happy face + 'This coffee is amazing.'):\n"
        "{\"reasoning\": \"The user is displaying joy and praising the product, requiring no intervention.\", \"intention_classes\": [0, 2], \"action_classes\": [0]}\n\n"
        "--- REAL TASK ---\n"
        f"Text to consider (if applicable): \"{user_text}\"\n"
        "Output ONLY the JSON object:"
    )
    
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": system_prompt}]}]

# ==============================================================================
# 4. AUTO-RETRY DISTILLATION ENGINE
# ==============================================================================
def generate_annotations(model, processor, dataset_paths):
    global global_results
    processed_ids = set()
    
    # CRASH RECOVERY
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            try:
                global_results = json.load(f)
                processed_ids = {item["id"] for item in global_results}
                print(f"🔄 Resuming from sample {len(processed_ids)}...")
            except json.JSONDecodeError:
                pass

    remaining_tasks = [item for item in dataset_paths if item["id"] not in processed_ids]
    print(f"🚀 Distilling {len(remaining_tasks)} samples with Chain-of-Thought...\n")

    for item in tqdm(remaining_tasks, desc="Distilling Data"):
        image_path, user_text, sample_id = item["image_path"], item["text"], item["id"]

        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            continue
        
        conversation = build_cot_prompt(user_text)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            temp = 0.1 if attempt == 0 else 0.3
            
            output_ids = model.generate(**inputs, max_new_tokens=150, temperature=temp, do_sample=(temp > 0.1))
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                parsed_json = json.loads(clean_text)
                
                if "intention_classes" in parsed_json and "action_classes" in parsed_json:
                    global_results.append({
                        "id": sample_id,
                        "image_path": image_path,
                        "text": user_text,
                        "reasoning_log": parsed_json.get("reasoning", ""),
                        "intention_labels": parsed_json["intention_classes"],
                        "action_labels": parsed_json["action_classes"]
                    })
                    success = True
                    break 
            except json.JSONDecodeError:
                pass 
        
        if not success:
            logging.error(f"Sample {sample_id} failed after {max_retries} attempts. Output: {response_text}")

        # Save State and Clear Memory every 20 images
        if len(global_results) % 20 == 0:
            save_results_safely()
            torch.cuda.empty_cache()

    save_results_safely()
    print(f"\n🏆 Distillation Complete! Data saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 BMVC 2026 - Elite Llama 3.2 Knowledge Distillation")
    print("="*70)
    
    try:
        real_dataset = load_real_dataset(data_dir=DATA_DIR)
        if len(real_dataset) > 0:
            teacher_model, text_processor = load_teacher_model()
            generate_annotations(teacher_model, text_processor, real_dataset)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        logging.error(f"Fatal execution error: {e}")
