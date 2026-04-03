#!/usr/bin/env python3
"""
BMVC 2026 - Master Offline Knowledge Distillation
Teacher Model: Meta Llama 3.2 11B Vision-Instruct
Features: Auto-Data Locator, Crash Recovery, Ctrl+C Protection, Memory Management, Audit Logging
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
# 1. DYNAMIC PATH RESOLUTION (Ensures everything stays in BMVC-2026)
# ==============================================================================
# This automatically finds your BMVC-2026 root folder, no matter where you run the script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 
DATA_DIR = PROJECT_ROOT / "data" / "mine_gdrive"
OUTPUT_JSON = PROJECT_ROOT / "data" / "distilled_annotations.json"
LOG_FILE = PROJECT_ROOT / "data" / "distillation.log"

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Setup Logging for Research Audit
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global variable to hold results for safe Ctrl+C exiting
global_results = []

def save_results_safely():
    """Saves the JSON immediately. Used during checkpoints and Ctrl+C."""
    if global_results:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(global_results, f, indent=2)
        logging.info(f"Safely saved {len(global_results)} annotations to {OUTPUT_JSON}")
        print(f"\n💾 [SAFE SAVE] Successfully secured {len(global_results)} annotations.")

def signal_handler(sig, frame):
    """Catches Ctrl+C so you don't lose data if you stop the script manually."""
    print("\n\n🛑 [INTERRUPT DETECTED] Saving current progress before exiting...")
    save_results_safely()
    sys.exit(0)

# Register the Ctrl+C handler
signal.signal(signal.SIGINT, signal_handler)


# ==============================================================================
# 2. AUTO-DATA LOCATOR (DEEP RECURSIVE SEARCH)
# ==============================================================================
def load_real_dataset(data_dir: Path):
    dataset_paths = []
    print(f"🔍 Recursively scanning for dataset files in: {data_dir.resolve()}...")
    logging.info(f"Scanning for data in {data_dir}")
    
    json_files = list(data_dir.rglob("metadata.json"))
    csv_files = list(data_dir.rglob("dataset.csv"))
    
    if not json_files and not csv_files:
        raise FileNotFoundError(f"❌ Could not find any metadata.json or dataset.csv in {data_dir}")

    for json_path in json_files:
        logging.info(f"Found JSON mapping: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                img_path = json_path.parent / item["image_file"]
                dataset_paths.append({
                    "id": str(item.get("id", len(dataset_paths))),
                    "image_path": str(img_path),
                    "text": item.get("text", "")
                })

    for csv_path in csv_files:
        logging.info(f"Found CSV mapping: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = csv_path.parent / row["image_file"]
                dataset_paths.append({
                    "id": str(row.get("id", len(dataset_paths))),
                    "image_path": str(img_path),
                    "text": row.get("text", "")
                })

    print(f"✅ Successfully loaded {len(dataset_paths)} multimodal samples.")
    return dataset_paths

# ==============================================================================
# 3. THE LLAMA MODEL & PROMPT ENGINEERING
# ==============================================================================
def load_teacher_model():
    print("🧠 Loading Llama 3.2 11B Vision (Spreading across GPUs...)")
    logging.info("Loading Teacher Model: meta-llama/Llama-3.2-11B-Vision-Instruct")
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def build_few_shot_prompt(user_text):
    system_prompt = (
        "You are an expert psychological data annotator for a computer vision research paper.\n"
        "Analyze the provided image and text. You must output ONLY a valid JSON object. "
        "Do not include markdown formatting, backticks, or conversational text. Just the JSON.\n\n"
        "The JSON must have two keys:\n"
        "1. 'intention_classes': A list of integers (0-19) representing the user's goals.\n"
        "2. 'action_classes': A list of integers (0-14) representing how the system should respond.\n\n"
        "--- FEW-SHOT EXAMPLES ---\n"
        "Example 1 (Angry Face + 'My flight was canceled!'):\n"
        "{\"intention_classes\": [4, 18], \"action_classes\": [2, 14]}\n"
        "Example 2 (Happy Face + 'This coffee is amazing.'):\n"
        "{\"intention_classes\": [0, 2], \"action_classes\": [0]}\n\n"
        "--- REAL TASK ---\n"
        f"Text to analyze: \"{user_text}\"\n"
        "Output ONLY the JSON object for the image and text above:"
    )
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": system_prompt}
            ]
        }
    ]
    return conversation

# ==============================================================================
# 4. CRASH-PROOF DISTILLATION ENGINE
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
                print(f"🔄 Crash Recovery: Resuming from sample {len(processed_ids)}...")
                logging.info(f"Resumed distillation. {len(processed_ids)} already completed.")
            except json.JSONDecodeError:
                print("⚠️ Output file corrupted. Starting fresh.")

    remaining_tasks = [item for item in dataset_paths if item["id"] not in processed_ids]
    print(f"🚀 Starting Distillation for {len(remaining_tasks)} remaining samples...\n")

    for idx, item in enumerate(tqdm(remaining_tasks, desc="Distilling Data")):
        image_path = item["image_path"]
        user_text = item["text"]
        sample_id = item["id"]

        try:
            if not os.path.exists(image_path):
                logging.warning(f"File missing: {image_path}. Skipping.")
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
            except UnidentifiedImageError:
                logging.warning(f"Image corrupted: {image_path}. Skipping.")
                continue
            
            conversation = build_few_shot_prompt(user_text)
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            
            # Deterministic, cold generation
            output_ids = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False)
            
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                parsed_json = json.loads(clean_text)
                
                global_results.append({
                    "id": sample_id,
                    "image_path": image_path,
                    "text": user_text,
                    "intention_labels": parsed_json.get("intention_classes", []),
                    "action_labels": parsed_json.get("action_classes", [])
                })
                
                # SAVE STATE AND CLEAR MEMORY every 50 images
                if len(global_results) % 50 == 0:
                    save_results_safely()
                    torch.cuda.empty_cache() # Prevent VRAM memory leaks

            except json.JSONDecodeError:
                logging.error(f"Sample {sample_id} failed JSON parsing. Output: {response_text}")

        except Exception as e:
            logging.error(f"Failed to process sample {sample_id}: {e}")

    # Final Save
    save_results_safely()
    print(f"\n🏆 Distillation 100% Complete! Final dataset saved to {OUTPUT_JSON}")
    logging.info("Distillation pipeline completely finished.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 BMVC 2026 - Llama 3.2 Knowledge Distillation Engine")
    print("="*70)
    
    # 1. Load the real dataset using dynamic paths
    real_dataset = load_real_dataset(data_dir=DATA_DIR)
    
    if len(real_dataset) > 0:
        # 2. Load Teacher
        teacher_model, text_processor = load_teacher_model()
        
        # 3. Generate the Paper-Ready Data
        generate_annotations(
            model=teacher_model, 
            processor=text_processor, 
            dataset_paths=real_dataset
        )
    else:
        print("❌ Stopping. No images found. Check your data folders.")
