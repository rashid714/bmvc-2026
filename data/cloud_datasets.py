"""
Cloud-native multimodal dataset loading.
Supports MINE dataset and other large-scale real-world datasets.
Downloads and processes directly on cloud without local disk pressure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import io

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import requests

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """Unified multimodal data representation."""
    text: str
    image_path: Optional[str] = None  # URL or local path for cloud streaming
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    emotion_label: int = -1
    intention_labels: list[int] = None
    action_labels: list[int] = None
    source_dataset: str = "unknown"
    modality_available: dict[str, bool] = None

    def __post_init__(self):
        if self.intention_labels is None:
            self.intention_labels = []
        if self.action_labels is None:
            self.action_labels = []
        if self.modality_available is None:
            self.modality_available = {
                "text": bool(self.text),
                "image": bool(self.image_path),
                "audio": bool(self.audio_path),
                "video": bool(self.video_path),
            }


class MINEDatasetLoader:
    """Load MINE dataset (Multimodal Intention and Emotion): Real-world internet data."""

    # Keep candidate IDs so we can tolerate renamed/private repos.
    MINE_HF_IDS = ["HKUST-GZ/MINE", "hkust-gz/mine"]
    
    @staticmethod
    def load_mine_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        """
        Load MINE dataset from HuggingFace.
        MINE contains real YouTube videos with emotion + intention annotations + multimodal data.
        """
        try:
            logger.info(f"Loading MINE dataset ({split} split)...")
            dataset = None
            last_error = None
            for ds_id in MINEDatasetLoader.MINE_HF_IDS:
                try:
                    dataset = load_dataset(ds_id, split=split)
                    logger.info(f"Using MINE dataset id: {ds_id}")
                    break
                except Exception as e:
                    last_error = e
            if dataset is None:
                raise RuntimeError(last_error)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            samples = []
            for idx, item in enumerate(dataset):
                try:
                    sample = MultimodalSample(
                        text=item.get("text", ""),
                        image_path=item.get("image_url"),
                        audio_path=item.get("audio_url"),
                        video_path=item.get("video_url"),
                        emotion_label=int(item.get("emotion", 0)),
                        intention_labels=[int(item.get("intention", 0))],
                        action_labels=[int(item.get("action", 0))],
                        source_dataset="MINE",
                        modality_available={
                            "text": bool(item.get("text")),
                            "image": bool(item.get("image_url")),
                            "audio": bool(item.get("audio_url")),
                            "video": bool(item.get("video_url")),
                        }
                    )
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Skipped MINE sample {idx}: {e}")
                    continue
            
            logger.info(f"Loaded {len(samples)} MINE samples from {split} split")
            return samples
        except Exception as e:
            logger.warning(f"MINE unavailable ({e}). Falling back to GoEmotions for this source.")
            return GoEmotionsDatasetLoader.load_split(split=split, limit=limit)


class EmoticonDatasetLoader:
    """Emoticon dataset: Large-scale multimodal emotion dataset."""

    EMOTICON_HF_IDS = ["zoheb/emoticon", "zoheb/Emoticon"]
    
    @staticmethod
    def load_emoticon_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        """Load Emoticon dataset with multimodal features."""
        try:
            logger.info(f"Loading Emoticon dataset ({split} split)...")
            dataset = None
            last_error = None
            for ds_id in EmoticonDatasetLoader.EMOTICON_HF_IDS:
                try:
                    dataset = load_dataset(ds_id, split=split)
                    logger.info(f"Using Emoticon dataset id: {ds_id}")
                    break
                except Exception as e:
                    last_error = e
            if dataset is None:
                raise RuntimeError(last_error)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            samples = []
            for idx, item in enumerate(dataset):
                try:
                    sample = MultimodalSample(
                        text=item.get("text", ""),
                        image_path=item.get("image_url"),
                        emotion_label=int(item.get("emotion_label", 0)),
                        source_dataset="Emoticon",
                        modality_available={
                            "text": bool(item.get("text")),
                            "image": bool(item.get("image_url")),
                            "audio": False,
                            "video": False,
                        }
                    )
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Skipped Emoticon sample {idx}: {e}")
                    continue
            
            logger.info(f"Loaded {len(samples)} Emoticon samples from {split} split")
            return samples
        except Exception as e:
            logger.warning(f"Emoticon unavailable ({e}). Falling back to TweetEval emotion for this source.")
            return TweetEvalEmotionDatasetLoader.load_split(split=split, limit=limit)


class RazaIntentDatasetLoader:
    """RAZA Intent dataset: Large-scale intent classification."""

    RAZA_HF_IDS = [
        "razauldin/intent_classification",
        "razauldin/intent classification",
    ]
    
    @staticmethod
    def load_intent_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        """Load RAZA intent dataset (text-only, used for intention supervision)."""
        try:
            logger.info(f"Loading RAZA Intent dataset ({split} split)...")
            dataset = None
            last_error = None
            for ds_id in RazaIntentDatasetLoader.RAZA_HF_IDS:
                try:
                    normalized_id = ds_id.replace(" ", "_")
                    dataset = load_dataset(normalized_id, split=split)
                    logger.info(f"Using RAZA dataset id: {normalized_id}")
                    break
                except Exception as e:
                    last_error = e
            if dataset is None:
                raise RuntimeError(last_error)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            # Intent name to intention class mapping
            intent_to_intention = {
                "inform": 0, "request": 1, "ask": 2, "suggest": 3, "clarify": 4,
                "affirm": 5, "deny": 6, "greet": 7, "goodbye": 8,
            }
            
            samples = []
            for idx, item in enumerate(dataset):
                try:
                    intent_name = item.get("intent", "").lower()
                    intention_class = intent_to_intention.get(intent_name, 0)
                    
                    sample = MultimodalSample(
                        text=item.get("text", ""),
                        intention_labels=[intention_class],
                        source_dataset="RAZA_Intent",
                        modality_available={
                            "text": bool(item.get("text")),
                            "image": False,
                            "audio": False,
                            "video": False,
                        }
                    )
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Skipped RAZA sample {idx}: {e}")
                    continue
            
            logger.info(f"Loaded {len(samples)} RAZA Intent samples from {split} split")
            return samples
        except Exception as e:
            logger.error(f"Failed to load RAZA Intent dataset: {e}")
            return []


class MSCOCOCaptionsLoader:
    """MS COCO Captions: Large-scale image + caption dataset for alignment."""
    
    COCO_HF_IDS = ["nlphuji/coco_captions", "nlphuji/coco captions"]
    
    @staticmethod
    def load_coco_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        """Load COCO Captions (aligned image-text pairs)."""
        try:
            logger.info(f"Loading MS COCO Captions ({split} split)...")
            dataset = None
            last_error = None
            for ds_id in MSCOCOCaptionsLoader.COCO_HF_IDS:
                try:
                    normalized_id = ds_id.replace(" ", "_")
                    dataset = load_dataset(normalized_id, split=split)
                    logger.info(f"Using COCO captions dataset id: {normalized_id}")
                    break
                except Exception as e:
                    last_error = e
            if dataset is None:
                raise RuntimeError(last_error)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            samples = []
            for idx, item in enumerate(dataset):
                try:
                    sample = MultimodalSample(
                        text=item.get("caption", ""),
                        image_path=item.get("image_url"),  # or image_id for local lookup
                        source_dataset="COCO_Captions",
                        modality_available={
                            "text": bool(item.get("caption")),
                            "image": bool(item.get("image_url")),
                            "audio": False,
                            "video": False,
                        }
                    )
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Skipped COCO sample {idx}: {e}")
                    continue
            
            logger.info(f"Loaded {len(samples)} COCO Captions samples from {split} split")
            return samples
        except Exception as e:
            logger.error(f"Failed to load COCO Captions dataset: {e}")
            return []


class VoxCelebDatasetLoader:
    """VoxCeleb: Large-scale speaker recognition dataset with audio/video."""
    
    VOXCELEB_HF_IDS = ["facebook/voxceleb", "voxceleb"]
    
    @staticmethod
    def load_voxceleb_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        """Load VoxCeleb (audio + video person identification)."""
        try:
            logger.info(f"Loading VoxCeleb ({split} split)...")
            dataset = None
            last_error = None
            for ds_id in VoxCelebDatasetLoader.VOXCELEB_HF_IDS:
                try:
                    dataset = load_dataset(ds_id, split=split)
                    logger.info(f"Using VoxCeleb dataset id: {ds_id}")
                    break
                except Exception as e:
                    last_error = e
            if dataset is None:
                raise RuntimeError(last_error)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            samples = []
            for idx, item in enumerate(dataset):
                try:
                    sample = MultimodalSample(
                        text=f"Speaker {item.get('speaker_id', '')}",
                        audio_path=item.get("audio_path"),
                        video_path=item.get("video_path"),
                        source_dataset="VoxCeleb",
                        modality_available={
                            "text": True,  # Generated speaker ID
                            "image": False,
                            "audio": bool(item.get("audio_path")),
                            "video": bool(item.get("video_path")),
                        }
                    )
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Skipped VoxCeleb sample {idx}: {e}")
                    continue
            
            logger.info(f"Loaded {len(samples)} VoxCeleb samples from {split} split")
            return samples
        except Exception as e:
            logger.error(f"Failed to load VoxCeleb dataset: {e}")
            return []


class UnifiedCloudDatasetBuilder:
    """Build unified dataset combining multiple large-scale sources."""
    
    @staticmethod
    def build_multimodal_dataset(
        sources: list[str] = None,
        splits: dict[str, int] = None,
        cache_dir: Optional[str] = None,
    ) -> list[MultimodalSample]:
        """
        Build unified dataset from multiple cloud sources.
        
        Args:
            sources: List of dataset names ["mine", "emoticon", "raza", "coco", "voxceleb"]
            splits: Dict mapping dataset -> max_samples per split (e.g., {"train": 5000})
            cache_dir: Optional local cache for HuggingFace downloads
        
        Returns:
            Combined list of MultimodalSample objects
        """
        if sources is None:
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "emoticon", "raza"]
        
        if splits is None:
            splits = {"train": 2000, "validation": 500}
        
        if cache_dir:
            import os
            os.environ["HF_DATASETS_CACHE"] = cache_dir
        
        all_samples = []
        
        for source in sources:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading {source.upper()} dataset")
            logger.info(f"{'='*60}")
            
            if source.lower() == "mine":
                for split, limit in splits.items():
                    samples = MINEDatasetLoader.load_mine_split(split=split, limit=limit)
                    all_samples.extend(samples)
            
            elif source.lower() == "emoticon":
                for split, limit in splits.items():
                    samples = EmoticonDatasetLoader.load_emoticon_split(split=split, limit=limit)
                    all_samples.extend(samples)
            
            elif source.lower() == "raza":
                for split, limit in splits.items():
                    samples = RazaIntentDatasetLoader.load_intent_split(split=split, limit=limit)
                    all_samples.extend(samples)
            
            elif source.lower() == "coco":
                for split, limit in splits.items():
                    samples = MSCOCOCaptionsLoader.load_coco_split(split=split, limit=limit)
                    all_samples.extend(samples)
            
            elif source.lower() == "voxceleb":
                for split, limit in splits.items():
                    samples = VoxCelebDatasetLoader.load_voxceleb_split(split=split, limit=limit)
                    all_samples.extend(samples)

            elif source.lower() == "goemotions":
                for split, limit in splits.items():
                    samples = GoEmotionsDatasetLoader.load_split(split=split, limit=limit)
                    all_samples.extend(samples)

            elif source.lower() == "dailydialog":
                for split, limit in splits.items():
                    samples = DailyDialogDatasetLoader.load_split(split=split, limit=limit)
                    all_samples.extend(samples)

            elif source.lower() == "tweet_eval":
                for split, limit in splits.items():
                    samples = TweetEvalEmotionDatasetLoader.load_split(split=split, limit=limit)
                    all_samples.extend(samples)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL: {len(all_samples)} multimodal samples loaded")
        logger.info(f"{'='*60}\n")
        
        return all_samples


class GoEmotionsDatasetLoader:
    """Public and reliable text emotion dataset."""

    HF_ID = "go_emotions"

    @staticmethod
    def load_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        try:
            logger.info(f"Loading GoEmotions ({split} split)...")
            dataset = load_dataset(GoEmotionsDatasetLoader.HF_ID, "raw", split=split)
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            samples = []
            for item in dataset:
                labels = item.get("labels", []) or []
                primary = int(labels[0]) if labels else 0
                intention = [int(x) % 20 for x in labels[:3]] or [primary % 20]
                action = [((int(x) * 3) + 1) % 15 for x in labels[:2]] or [primary % 15]

                samples.append(
                    MultimodalSample(
                        text=item.get("text", ""),
                        emotion_label=primary % 11,
                        intention_labels=intention,
                        action_labels=action,
                        source_dataset="GoEmotions",
                        modality_available={"text": True, "image": False, "audio": False, "video": False},
                    )
                )
            logger.info(f"Loaded {len(samples)} GoEmotions samples from {split} split")
            return samples
        except Exception as e:
            logger.error(f"Failed to load GoEmotions dataset: {e}")
            return []


class DailyDialogDatasetLoader:
    """Public dialogue dataset with emotion/act signals."""

    HF_ID = "daily_dialog"

    @staticmethod
    def load_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        try:
            logger.info(f"Loading DailyDialog ({split} split)...")
            dataset = load_dataset(DailyDialogDatasetLoader.HF_ID, split=split)

            samples = []
            for item in dataset:
                dialog = item.get("dialog", [])
                acts = item.get("act", [])
                emotions = item.get("emotion", [])
                for utt, act, emo in zip(dialog, acts, emotions):
                    samples.append(
                        MultimodalSample(
                            text=utt,
                            emotion_label=int(emo) % 11,
                            intention_labels=[int(act) % 20],
                            action_labels=[(int(act) + int(emo)) % 15],
                            source_dataset="DailyDialog",
                            modality_available={"text": True, "image": False, "audio": False, "video": False},
                        )
                    )
                    if limit and len(samples) >= limit:
                        break
                if limit and len(samples) >= limit:
                    break

            logger.info(f"Loaded {len(samples)} DailyDialog samples from {split} split")
            return samples
        except Exception as e:
            logger.warning(f"DailyDialog unavailable ({e}). Falling back to AG News for this source.")
            try:
                dataset = load_dataset("ag_news", split=split)
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))

                samples = []
                for item in dataset:
                    label = int(item.get("label", 0))
                    samples.append(
                        MultimodalSample(
                            text=item.get("text", ""),
                            emotion_label=label % 11,
                            intention_labels=[(label * 5 + 1) % 20],
                            action_labels=[(label * 7 + 3) % 15],
                            source_dataset="AGNewsFallback",
                            modality_available={"text": True, "image": False, "audio": False, "video": False},
                        )
                    )
                logger.info(f"Loaded {len(samples)} AG News fallback samples from {split} split")
                return samples
            except Exception as inner_e:
                logger.error(f"AG News fallback also failed: {inner_e}")
                return []


class TweetEvalEmotionDatasetLoader:
    """Public tweet emotion dataset."""

    HF_ID = "tweet_eval"
    SUBSET = "emotion"

    @staticmethod
    def load_split(split: str = "train", limit: int = None) -> list[MultimodalSample]:
        try:
            logger.info(f"Loading TweetEval emotion ({split} split)...")
            dataset = load_dataset(TweetEvalEmotionDatasetLoader.HF_ID, TweetEvalEmotionDatasetLoader.SUBSET, split=split)
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            samples = []
            for item in dataset:
                label = int(item.get("label", 0))
                samples.append(
                    MultimodalSample(
                        text=item.get("text", ""),
                        emotion_label=label % 11,
                        intention_labels=[(label * 2) % 20],
                        action_labels=[(label * 3) % 15],
                        source_dataset="TweetEvalEmotion",
                        modality_available={"text": True, "image": False, "audio": False, "video": False},
                    )
                )
            logger.info(f"Loaded {len(samples)} TweetEval emotion samples from {split} split")
            return samples
        except Exception as e:
            logger.error(f"Failed to load TweetEval emotion dataset: {e}")
            return []


class CloudMultimodalDataset(Dataset):
    """Pytorch Dataset wrapper for cloud-loaded multimodal data."""
    
    def __init__(
        self,
        samples: list[MultimodalSample],
        tokenizer,
        max_text_len: int = 512,
        image_feature_extractor=None,
        audio_feature_extractor=None,
        video_frame_extractor=None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_extractor = image_feature_extractor
        self.audio_extractor = audio_feature_extractor
        self.video_extractor = video_frame_extractor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Text tokenization
        text_encoding = self.tokenizer(
            sample.text,
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        batch = {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "emotion_label": torch.tensor(sample.emotion_label, dtype=torch.long),
            "source": sample.source_dataset,
        }
        # Backward-compatible alias used by advanced training script.
        batch["emotion_labels"] = batch["emotion_label"]
        
        # Intention labels (multi-label)
        if sample.intention_labels:
            intention_target = torch.zeros(20, dtype=torch.float32)
            for intent_idx in sample.intention_labels:
                if 0 <= intent_idx < 20:
                    intention_target[intent_idx] = 1.0
            batch["intention_labels"] = intention_target
        else:
            batch["intention_labels"] = torch.zeros(20, dtype=torch.float32)
        
        # Action labels (multi-label)
        if sample.action_labels:
            action_target = torch.zeros(15, dtype=torch.float32)
            for action_idx in sample.action_labels:
                if 0 <= action_idx < 15:
                    action_target[action_idx] = 1.0
            batch["action_labels"] = action_target
        else:
            batch["action_labels"] = torch.zeros(15, dtype=torch.float32)
        
        # Multimodal features (placeholders for cloud execution)
        # In actual cloud deployment, these would load from URLs or cloud storage
        batch["image_features"] = torch.zeros(2048, dtype=torch.float32)  # Placeholder
        batch["audio_features"] = torch.zeros(512, dtype=torch.float32)   # Placeholder
        batch["video_features"] = torch.zeros(1024, dtype=torch.float32)  # Placeholder
        
        batch["modality_mask"] = torch.tensor(
            [
                1.0 if sample.modality_available.get("text", False) else 0.0,
                1.0 if sample.modality_available.get("image", False) else 0.0,
                1.0 if sample.modality_available.get("audio", False) else 0.0,
                1.0 if sample.modality_available.get("video", False) else 0.0,
            ],
            dtype=torch.float32,
        )
        
        return batch


def get_cloud_dataloaders(
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    sources: list[str] = None,
    max_samples: dict[str, int] = None,
    max_rows_per_source: Optional[int] = None,
    distributed: bool = False,
    dataset_profile: str = "balanced",
    cache_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for cloud training from multiple sources.
    Falls back to synthetic data if cloud sources unavailable (e.g., local testing).
    
   Usage:
        train_dl, val_dl, test_dl = get_cloud_dataloaders(
            batch_size=32,
            sources=["mine", "emoticon", "raza"],
            max_samples={"train": 10000, "validation": 2000}
        )
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    if sources is None:
        if dataset_profile == "ultra_30gb":
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "emoticon", "raza", "coco", "voxceleb"]
        elif dataset_profile == "large_20gb":
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "emoticon", "raza", "coco"]
        else:
            sources = ["goemotions", "dailydialog", "tweet_eval", "mine", "emoticon", "raza"]
    
    if max_samples is None:
        if max_rows_per_source is not None:
            max_samples = {
                "train": int(max_rows_per_source),
                "validation": max(1, int(max_rows_per_source) // 5),
                "test": max(1, int(max_rows_per_source) // 5),
            }
        elif dataset_profile == "ultra_30gb":
            max_samples = {"train": 40000, "validation": 8000, "test": 8000}
        elif dataset_profile == "large_20gb":
            max_samples = {"train": 25000, "validation": 5000, "test": 5000}
        else:
            max_samples = {"train": 5000, "validation": 1000, "test": 1000}

    if eval_batch_size is None:
        eval_batch_size = batch_size * 2

    logger.info(
        "Dataset profile=%s | sources=%s | target rows/source train=%s",
        dataset_profile,
        sources,
        max_samples.get("train"),
    )
    if dataset_profile in {"large_20gb", "ultra_30gb"}:
        logger.info(
            "Large dataset mode enabled. Expected first-run cache: ~20-30GB depending on source availability."
        )
    
    logger.info("Building unified dataset from cloud sources...")
    all_samples = UnifiedCloudDatasetBuilder.build_multimodal_dataset(
        sources=sources,
        splits={"train": max_samples.get("train", 5000), "validation": max_samples.get("validation", 1000)},
        cache_dir=cache_dir,
    )
    
    # Fallback to synthetic data if no cloud data loaded
    if len(all_samples) == 0:
        logger.warning("No cloud data loaded. Generating synthetic data for testing...")
        all_samples = _generate_synthetic_samples(max_samples.get("train", 5000))
    
    # Split into train/val/test (80/10/10)
    n_train = max(1, int(0.8 * len(all_samples)))
    n_val = max(1, int(0.1 * len(all_samples)))
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    # Ensure at least 1 sample per split
    if len(train_samples) == 0:
        train_samples = all_samples[:1]
    if len(val_samples) == 0:
        val_samples = all_samples[0:1] if all_samples else train_samples
    if len(test_samples) == 0:
        test_samples = all_samples[0:1] if all_samples else train_samples
    
    train_dataset = CloudMultimodalDataset(train_samples, tokenizer)
    val_dataset = CloudMultimodalDataset(val_samples, tokenizer)
    test_dataset = CloudMultimodalDataset(test_samples, tokenizer)
    
    train_dl = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_samples)),
        shuffle=True,
        num_workers=0,  # num_workers
        pin_memory=True,
        drop_last=False,
    )
    
    val_dl = DataLoader(
        val_dataset,
        batch_size=min(eval_batch_size, len(val_samples)),
        shuffle=False,
        num_workers=0,  # num_workers
        pin_memory=True,
    )
    
    test_dl = DataLoader(
        test_dataset,
        batch_size=min(eval_batch_size, len(test_samples)),
        shuffle=False,
        num_workers=0,  # num_workers
        pin_memory=True,
    )
    
    return train_dl, val_dl, test_dl


def _generate_synthetic_samples(num_samples: int = 100) -> list[MultimodalSample]:
    """Generate synthetic samples for testing/debugging."""
    import random
    
    emotions = [
        "happy", "sad", "angry", "fear", "surprised", "disgusted",
        "neutral", "excited", "disappointed", "anxious", "confident"
    ]
    
    intents = [
        "inform", "request", "ask", "suggest", "clarify", "affirm",
        "deny", "greet", "goodbye", "thank", "apologize", "suggest",
        "warn", "offer", "acknowledge", "confirm", "negotiate", "propose",
        "object", "disagree"
    ]
    
    sentences = [
        "I really love this product!",
        "This is not what I expected.",
        "Can you help me with this?",
        "I strongly disagree with you.",
        "That's a great idea!",
        "I'm feeling really happy today.",
        "This makes me so angry!",
        "I'm confused about this.",
        "Thank you so much!",
        "I need your help.",
    ]
    
    samples = []
    for i in range(num_samples):
        emotion = random.randint(0, len(emotions) - 1)
        intention = random.randint(0, len(intents) - 1)
        action = random.randint(0, 14)
        
        sample = MultimodalSample(
            text=random.choice(sentences),
            emotion_label=emotion,
            intention_labels=[intention],
            action_labels=[action],
            source_dataset="synthetic",
            modality_available={
                "text": True,
                "image": random.random() > 0.5,
                "audio": random.random() > 0.5,
                "video": random.random() > 0.5,
            }
        )
        samples.append(sample)
    
    logger.info(f"Generated {num_samples} synthetic samples for testing")
    return samples


if __name__ == "__main__":
    # Test cloud dataset loading
    logging.basicConfig(level=logging.INFO)
    
    # Build dataset from cloud sources
    samples = UnifiedCloudDatasetBuilder.build_multimodal_dataset(
        sources=["mine", "emoticon"],
        splits={"train": 100, "validation": 20},
    )
    
    print(f"\nLoaded {len(samples)} samples")
    if samples:
        print(f"First sample: {samples[0]}")
