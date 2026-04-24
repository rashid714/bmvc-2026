"""
BEAR BMVC 2026 - SILVER STANDARD DATASET ARCHITECTURE
Spotlight Academic Version: Pure MINE Curated + FANE Integration + RandAugment
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Prevents the entire training run from crashing if an image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------------------
# Logging & Paths
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SilverDataset")

def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_data_root() -> Path:
    return (get_repo_root() / "data").resolve()

# ------------------------------------------------------------------------------
# 1. Core Data Structure
# ------------------------------------------------------------------------------
@dataclass
class MultimodalSample:
    text: str
    image_path: Optional[str] = None
    emotion_label: int = 0
    intention_labels: List[int] = field(default_factory=list)
    action_labels: List[int] = field(default_factory=list)
    source_dataset: str = "unknown"

# ------------------------------------------------------------------------------
# 2. The FANE Emotion Loader (JSON Distilled Version)
# ------------------------------------------------------------------------------
class FANELoader:
    @staticmethod
    def load_split(split: str = "train") -> List[MultimodalSample]:
        try:
            fane_dir = get_data_root() / "fane"
            json_path = fane_dir / "distilled_annotations.json"
            
            if not json_path.exists():
                logger.warning(f"⚠️ FANE JSON not found at {json_path}")
                return []

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Filter data strictly by the requested split (train, validation, test)
            split_data = [item for item in data if item.get("split", "train") == split]

            samples: List[MultimodalSample] = []
            
            for item in split_data:
                # The JSON uses "images_processed/angry/angry1.jpg"
                raw_img_path = item.get("image_path", "")
                final_img_path = str((fane_dir / raw_img_path).resolve())

                # Safely parse your CLIP arrays
                intent_lbl = item.get("intention_labels", [0])
                action_lbl = item.get("action_labels", [0])
                emo_lbl = item.get("emotion_label", 4)

                samples.append(MultimodalSample(
                    text="", # FANE is vision-only
                    image_path=final_img_path,
                    emotion_label=emo_lbl,
                    intention_labels=intent_lbl,
                    action_labels=action_lbl,
                    source_dataset="FANE_Distilled"
                ))
            
            logger.info(f"✅ Loaded {len(samples)} FANE Distilled samples for {split} split.")
            return samples
        except Exception as e:
            logger.error(f"❌ Failed to load FANE Distilled Data: {e}")
            return []

# ------------------------------------------------------------------------------
# 3. 🌟 The Curated MINE Loader (Llama Silver Standard)
# ------------------------------------------------------------------------------
class MINECuratedLoader:
    @staticmethod
    def load_split(split: str = "train") -> List[MultimodalSample]:
        try:
            curated_dir = get_data_root() / "mine_curated"
            json_path = curated_dir / "mine_perfect_annotations.json"
            
            if not json_path.exists():
                logger.error(f"❌ Curated JSON not found at {json_path}")
                return []

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Split logic to avoid data leakage
            n_total = len(data)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)

            if split == "train": data = data[:n_train]
            elif split == "validation": data = data[n_train:n_train+n_val]
            else: data = data[n_train+n_val:]

            samples: List[MultimodalSample] = []
            
            for item in data:
                # The JSON saves path as "images/mine_XXXX.jpg", we need to map it absolutely
                raw_img_path = item.get("image_path", "")
                final_img_path = str((curated_dir / raw_img_path).resolve())

                # Safely parse Llama arrays
                intent_lbl = item.get("intention_labels", [0])
                action_lbl = item.get("action_labels", [0])
                emo_lbl = item.get("emotion_label", 4)

                samples.append(MultimodalSample(
                    text=str(item.get("text", "")),
                    image_path=final_img_path,
                    emotion_label=emo_lbl,
                    intention_labels=intent_lbl,
                    action_labels=action_lbl,
                    source_dataset="MINE_Llama_Curated"
                ))
                
            logger.info(f"✅ Loaded {len(samples)} MINE Curated samples for {split} split.")
            return samples
        except Exception as e:
            logger.error(f"❌ Failed to load MINE Curated Data: {e}")
            return []

# ------------------------------------------------------------------------------
# 4. PyTorch Dataset Engine
# ------------------------------------------------------------------------------
class CloudMultimodalDataset(Dataset):
    def __init__(self, samples: List[MultimodalSample], tokenizer, max_text_len: int = 128, is_train: bool = False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.is_train = is_train
        
        # Symmetrical Dropout Probabilities
        self.text_dropout_prob = 0.2
        self.image_dropout_prob = 0.1
        self.word_dropout_prob = 0.05 
        
        if self.is_train:
            self.img_transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandAugment(num_ops=2, magnitude=9), # 🌟 SOTA Vision Augmentation
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int: 
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        safe_text = sample.text if sample.text and str(sample.text).strip() else "[NO TEXT]"
        
        if self.is_train:
            # 1. Text Dropout
            if random.random() < self.text_dropout_prob:
                safe_text = ""
            # 2. NLP Word-Level Augmentation
            elif safe_text != "[NO TEXT]" and random.random() < 0.5:
                words = safe_text.split()
                safe_text = " ".join([w for w in words if random.random() > self.word_dropout_prob])

        text_encoding = self.tokenizer(safe_text, max_length=self.max_text_len, truncation=True, padding="max_length", return_tensors="pt")

        batch: Dict[str, Any] = {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "source": sample.source_dataset,
        }

        # 🌟 STRICT CLASS TRUNCATION (9, 12, 15)
        # Emotion
        emo_idx = max(0, int(sample.emotion_label))
        batch["emotion_labels"] = torch.tensor(emo_idx if emo_idx < 9 else 4, dtype=torch.long)

        # Intention (Filtered to 12)
        intention_target = torch.zeros(12, dtype=torch.float32)
        for intent_idx in sample.intention_labels:
            if isinstance(intent_idx, int) and 0 <= intent_idx < 12:
                intention_target[intent_idx] = 1.0
        batch["intention_labels"] = intention_target

        # Action (Filtered to 15)
        action_target = torch.zeros(15, dtype=torch.float32)
        for action_idx in sample.action_labels:
            if isinstance(action_idx, int) and 0 <= action_idx < 15:
                action_target[action_idx] = 1.0
        batch["action_labels"] = action_target

        # Image Loading & Dropout
        if self.is_train and random.random() < self.image_dropout_prob:
            batch["images"] = torch.zeros(3, 224, 224)
        elif sample.image_path and os.path.exists(sample.image_path):
            try:
                img = Image.open(sample.image_path).convert("RGB")
                batch["images"] = self.img_transform(img)
            except Exception:
                batch["images"] = torch.zeros(3, 224, 224)
        else:
            batch["images"] = torch.zeros(3, 224, 224)

        return batch

# ------------------------------------------------------------------------------
# 5. Dataloader Builder
# ------------------------------------------------------------------------------
def get_cloud_dataloaders(batch_size: int = 16, eval_batch_size: Optional[int] = None, num_workers: int = 4, distributed: bool = False, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from transformers import AutoTokenizer
    
    # Set up HuggingFace cache
    models_root = get_repo_root() / "models"
    hf_model_cache = (models_root / "hf_hub").resolve()
    hf_model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_model_cache)

    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=str(hf_model_cache))

    logger.info("🚀 Compiling Silver Standard Dataset...")
    
    train_samples = MINECuratedLoader.load_split("train") + FANELoader.load_split("train")
    val_samples = MINECuratedLoader.load_split("validation") + FANELoader.load_split("validation")
    test_samples = MINECuratedLoader.load_split("test") + FANELoader.load_split("test")

    # Shuffle training data thoroughly to mix MINE and FANE
    random.shuffle(train_samples)

    train_ds = CloudMultimodalDataset(train_samples, tokenizer, is_train=True)
    val_ds = CloudMultimodalDataset(val_samples, tokenizer, is_train=False)
    test_ds = CloudMultimodalDataset(test_samples, tokenizer, is_train=False)

    train_sampler = DistributedSampler(train_ds) if distributed else None

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size or batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size or batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl
