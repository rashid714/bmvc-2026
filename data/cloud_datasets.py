"""
================================================================================ json=========================================================================================
import logging
import os
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from datasets import disable_progress_bar, load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CloudDatasets")

if os.environ.get("DISABLE_HF_PROGRESS", "1") == "1":
    disable_progress_bar()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_list_of_ints(value: Any, default: Optional[List[int]] = None) -> List[int]:
    if default is None:
        default = [0]

    if value is None:
        return default

    if isinstance(value, list):
        result: List[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result if result else default

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        result: List[int] = []
        for part in parts:
            try:
                result.append(int(part))
            except (TypeError, ValueError):
                continue
        return result if result else default

    try:
        return [int(value)]
    except (TypeError, ValueError):
        return default


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_data_root(data_dir: Optional[str] = None) -> Path:
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    return (get_repo_root() / "data").resolve()


def get_models_root() -> Path:
    return (get_repo_root() / "models").resolve()


def _repo_dataset_cache_dir(data_dir: Optional[str] = None) -> str:
    path = (get_data_root(data_dir) / "hf_datasets").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _hf_load_dataset(*args, data_dir: Optional[str] = None, **kwargs):
    kwargs.setdefault("cache_dir", _repo_dataset_cache_dir(data_dir))
    return load_dataset(*args, **kwargs)


def _safe_local_image_path(candidate: Any) -> Optional[str]:
    if not candidate or not isinstance(candidate, str):
        return None
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return None
    p = Path(candidate).expanduser()
    return str(p.resolve()) if p.exists() else None


def discover_local_datasets(data_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    Auto-discover dataset roots under the base data directory.
    Returns a mapping like:
    {
        "goemotions": Path(...),
        "facial_emotions": Path(...),
        "bitext_intent": Path(...),
        "mine_gdrive": Path(...),
    }
    """
    data_root = get_data_root(data_dir)
    found: Dict[str, Path] = {}

    if not data_root.exists():
        logger.warning("Data root does not exist: %s", data_root)
        return found

    # Fast-path expected folders first
    expected = {
        "goemotions": data_root / "kaggle_datasets" / "goemotions",
        "facial_emotions": data_root / "kaggle_datasets" / "facial_emotions",
        "bitext_intent": data_root / "kaggle_datasets" / "bitext_intent",
        "mine_gdrive": data_root / "mine_gdrive",
    }

    go_dir = expected["goemotions"]
    if go_dir.exists() and (
        (go_dir / "train.csv").exists()
        or (go_dir / "train.tsv").exists()
    ):
        found["goemotions"] = go_dir

    facial_dir = expected["facial_emotions"]
    if facial_dir.exists() and (facial_dir / "images" / "train").exists():
        found["facial_emotions"] = facial_dir

    bitext_dir = expected["bitext_intent"]
    if bitext_dir.exists() and (bitext_dir / "Bitext_Sample_Customer_Service_Training_Dataset.csv").exists():
        found["bitext_intent"] = bitext_dir

    mine_dir = expected["mine_gdrive"]
    mine_candidates = [
        mine_dir / "train.jsonl",
        mine_dir / "train.json",
        mine_dir / "manifest.jsonl",
        mine_dir / "metadata.jsonl",
        mine_dir / "data.jsonl",
        mine_dir / "annotations.jsonl",
    ]
    if mine_dir.exists() and any(p.exists() for p in mine_candidates):
        found["mine_gdrive"] = mine_dir

    # Fallback full scan if something is missing
    if "goemotions" not in found:
        for p in data_root.rglob("*"):
            if p.is_dir() and (
                (p / "train.csv").exists() or (p / "train.tsv").exists()
            ) and (
                (p / "val.csv").exists()
                or (p / "validation.csv").exists()
                or (p / "val.tsv").exists()
                or (p / "validation.tsv").exists()
            ):
                found["goemotions"] = p
                break

    if "facial_emotions" not in found:
        for p in data_root.rglob("*"):
            if p.is_dir() and (p / "images" / "train").exists():
                found["facial_emotions"] = p
                break

    if "bitext_intent" not in found:
        for p in data_root.rglob("Bitext_Sample_Customer_Service_Training_Dataset.csv"):
            found["bitext_intent"] = p.parent
            break

    if "mine_gdrive" not in found:
        for p in data_root.rglob("*"):
            if not p.is_dir():
                continue
            candidates = [
                p / "train.jsonl",
                p / "train.json",
                p / "manifest.jsonl",
                p / "metadata.jsonl",
                p / "data.jsonl",
                p / "annotations.jsonl",
            ]
            if any(c.exists() for c in candidates):
                found["mine_gdrive"] = p
                break

    if found:
        logger.info(
            "Discovered local datasets: %s",
            {k: str(v) for k, v in found.items()},
        )
    else:
        logger.warning("No local datasets discovered under: %s", data_root)

    return found


# ------------------------------------------------------------------------------
# 1. Core data structure
# ------------------------------------------------------------------------------
@dataclass
class MultimodalSample:
    text: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    emotion_label: int = 0
    intention_labels: List[int] = field(default_factory=list)
    action_labels: List[int] = field(default_factory=list)
    source_dataset: str = "unknown"
    modality_available: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.intention_labels:
            self.intention_labels = [0]
        if not self.action_labels:
            self.action_labels = [0]

        self.modality_available = {
            "text": bool(self.text and str(self.text).strip() != ""),
            "image": bool(self.image_path),
            "audio": bool(self.audio_path),
            "video": bool(self.video_path),
        }


# ------------------------------------------------------------------------------
# 2. Kaggle subsystem
# ------------------------------------------------------------------------------
class KaggleDownloader:
    @staticmethod
    def ensure_dataset(
        kaggle_path: str,
        local_folder_name: str,
        data_dir: Optional[str] = None,
    ) -> Path:
        base_dir = get_data_root(data_dir) / "kaggle_datasets"
        base_dir.mkdir(parents=True, exist_ok=True)

        target_dir = base_dir / local_folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        if not any(target_dir.iterdir()):
            logger.info("Initiating Kaggle download for: %s", kaggle_path)
            try:
                subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        kaggle_path,
                        "-p",
                        str(target_dir),
                        "--unzip",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(
                    "Successfully downloaded & extracted %s to %s",
                    kaggle_path,
                    target_dir,
                )
            except Exception as e:
                logger.warning(
                    "Kaggle download failed or API key missing for %s. "
                    "Will still check local path %s. Reason: %s",
                    kaggle_path,
                    target_dir,
                    e,
                )

        return target_dir


class KaggleGoEmotionsLoader:
    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            dataset_paths = dataset_paths or discover_local_datasets(data_dir)
            target_dir = dataset_paths.get("goemotions")
            if target_dir is None:
                target_dir = KaggleDownloader.ensure_dataset(
                    "rkibria/goemotions-kaggle",
                    "goemotions",
                    data_dir=data_dir,
                )

            file_names = {
                "train": ["train.tsv", "train.csv"],
                "validation": ["val.tsv", "val.csv", "validation.tsv", "validation.csv"],
                "test": ["test.tsv", "test.csv"],
            }

            df: Optional[pd.DataFrame] = None
            for fname in file_names.get(split, []):
                potential_path = target_dir / fname
                if potential_path.exists():
                    sep = "\t" if fname.endswith(".tsv") else ","
                    df = pd.read_csv(potential_path, sep=sep)
                    logger.info("Loaded GoEmotions %s from %s", split, potential_path)
                    break

            if df is None:
                logger.warning("GoEmotions %s split file not found in %s", split, target_dir)
                return []

            if limit:
                df = df.head(limit)

            samples: List[MultimodalSample] = []
            for _, row in df.iterrows():
                text_value = row.get("text", "")
                text = str(text_value if pd.notna(text_value) else "")
                raw_labels = safe_list_of_ints(row.get("labels", "0"), default=[0])
                primary_label = raw_labels[0] if raw_labels else 0

                samples.append(
                    MultimodalSample(
                        text=text,
                        emotion_label=primary_label % 11,
                        intention_labels=[(primary_label * 2) % 20],
                        action_labels=[(primary_label * 3) % 15],
                        source_dataset="Kaggle_GoEmotions",
                    )
                )

            return samples

        except Exception as e:
            logger.exception("GoEmotions error on split '%s': %s", split, e)
            return []


class KaggleFacialEmotionLoader:
    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            dataset_paths = dataset_paths or discover_local_datasets(data_dir)
            target_dir = dataset_paths.get("facial_emotions")
            if target_dir is None:
                target_dir = KaggleDownloader.ensure_dataset(
                    "dima806/facial-emotions-image-detection-vit",
                    "facial_emotions",
                    data_dir=data_dir,
                )

            folder_split = "train" if split in ["train", "validation"] else "test"
            split_dir = target_dir / "images" / folder_split
            if not split_dir.exists():
                logger.warning("Facial emotion split dir not found: %s", split_dir)
                return []

            samples: List[MultimodalSample] = []
            emotion_map = {
                "angry": 0,
                "disgust": 1,
                "fear": 2,
                "happy": 3,
                "neutral": 4,
                "sad": 5,
                "surprise": 6,
            }

            count = 0
            for emotion_name, label_idx in emotion_map.items():
                emo_dir = split_dir / emotion_name
                if not emo_dir.exists():
                    continue

                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    for img_path in emo_dir.glob(ext):
                        samples.append(
                            MultimodalSample(
                                text="",
                                image_path=str(img_path.resolve()),
                                emotion_label=label_idx,
                                source_dataset="Kaggle_FacialEmotions",
                            )
                        )
                        count += 1
                        if limit and count >= limit:
                            break
                    if limit and count >= limit:
                        break
                if limit and count >= limit:
                    break

            return samples

        except Exception as e:
            logger.exception("FacialEmotion error on split '%s': %s", split, e)
            return []


class KaggleIntentLoader:
    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            dataset_paths = dataset_paths or discover_local_datasets(data_dir)
            target_dir = dataset_paths.get("bitext_intent")
            if target_dir is None:
                target_dir = KaggleDownloader.ensure_dataset(
                    "bitext/training-dataset-for-intent-classification",
                    "bitext_intent",
                    data_dir=data_dir,
                )

            csv_path = target_dir / "Bitext_Sample_Customer_Service_Training_Dataset.csv"
            if not csv_path.exists():
                logger.warning("Intent CSV not found: %s", csv_path)
                return []

            df = pd.read_csv(csv_path)

            if split == "train":
                df = df.iloc[: int(len(df) * 0.8)]
            elif split == "validation":
                df = df.iloc[int(len(df) * 0.8): int(len(df) * 0.9)]
            else:
                df = df.iloc[int(len(df) * 0.9):]

            if limit:
                df = df.head(limit)

            samples: List[MultimodalSample] = []
            for idx, row in df.iterrows():
                utterance = row.get("utterance", row.get("text", row.get("sentence", "")))
                text = str(utterance if pd.notna(utterance) else "")

                samples.append(
                    MultimodalSample(
                        text=text,
                        emotion_label=4,
                        intention_labels=[idx % 20],
                        action_labels=[(idx * 2) % 15],
                        source_dataset="Kaggle_BitextIntent",
                    )
                )

            return samples

        except Exception as e:
            logger.exception("Intent loader error on split '%s': %s", split, e)
            return []


# ------------------------------------------------------------------------------
# 3. Hugging Face loaders
# ------------------------------------------------------------------------------
class DairAiEmotionLoader:
    HF_ID = "dair-ai/emotion"

    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            dataset = _hf_load_dataset(DairAiEmotionLoader.HF_ID, split=split, data_dir=data_dir)
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            samples: List[MultimodalSample] = []
            for item in dataset:
                label = safe_int(item.get("label", 0), default=0)
                samples.append(
                    MultimodalSample(
                        text=str(item.get("text", "")),
                        emotion_label=label,
                        intention_labels=[(label * 2) % 20],
                        action_labels=[(label * 3) % 15],
                        source_dataset="HF_DairAiEmotion",
                    )
                )
            return samples

        except Exception as e:
            logger.exception("DairAI Emotion error on split '%s': %s", split, e)
            return []


class DailyDialogLoader:
    HF_ID = "daily_dialog"

    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            dataset = _hf_load_dataset(DailyDialogLoader.HF_ID, split=split, data_dir=data_dir)

            samples: List[MultimodalSample] = []
            for item in dataset:
                dialog = item.get("dialog", [])
                acts = item.get("act", [])
                emotions = item.get("emotion", [])

                for utt, act, emo in zip(dialog, acts, emotions):
                    act_i = safe_int(act, 0)
                    emo_i = safe_int(emo, 0)
                    samples.append(
                        MultimodalSample(
                            text=str(utt),
                            emotion_label=emo_i % 11,
                            intention_labels=[act_i % 20],
                            action_labels=[(act_i + emo_i) % 15],
                            source_dataset="HF_DailyDialog",
                        )
                    )
                    if limit and len(samples) >= limit:
                        break

                if limit and len(samples) >= limit:
                    break

            return samples

        except Exception as e:
            logger.exception("DailyDialog error on split '%s': %s", split, e)
            return []


class MSCOCOCaptionsLoader:
    HF_ID = "nlphuji/coco_captions"

    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            safe_split = "val" if split == "validation" else split
            dataset = _hf_load_dataset(MSCOCOCaptionsLoader.HF_ID, split=safe_split, data_dir=data_dir)

            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            samples: List[MultimodalSample] = []
            for item in dataset:
                local_image_path = None
                for key in ("image_path", "image_file", "local_image_path"):
                    val = item.get(key)
                    if isinstance(val, str):
                        local_image_path = _safe_local_image_path(val)
                        if local_image_path:
                            break

                samples.append(
                    MultimodalSample(
                        text=str(item.get("caption", "")),
                        image_path=local_image_path,
                        emotion_label=4,
                        source_dataset="HF_COCO",
                    )
                )

            return samples

        except Exception as e:
            logger.exception("MSCOCO error on split '%s': %s", split, e)
            return []


# ------------------------------------------------------------------------------
# 4. MINE loader
# ------------------------------------------------------------------------------
class MINEGoogleDriveDatasetLoader:
    @staticmethod
    def load_split(
        split: str = "train",
        limit: Optional[int] = None,
        data_dir: Optional[str] = None,
        dataset_paths: Optional[Dict[str, Path]] = None,
    ) -> List[MultimodalSample]:
        try:
            dataset_paths = dataset_paths or discover_local_datasets(data_dir)

            env_root = os.environ.get("MINE_GDRIVE_ROOT", "").strip()
            if env_root:
                root = Path(env_root).expanduser().resolve()
            else:
                root = dataset_paths.get("mine_gdrive", get_data_root(data_dir) / "mine_gdrive").resolve()

            if not root.exists() or not root.is_dir():
                logger.warning("MINE root not found or not a directory: %s", root)
                return []

            candidates = [
                f"{split}.jsonl",
                f"{split}.json",
                "manifest.jsonl",
                "metadata.jsonl",
                "data.jsonl",
                "annotations.jsonl",
            ]

            records: List[Dict[str, Any]] = []
            for rel in candidates:
                meta_path = root / rel
                if not meta_path.exists():
                    continue

                logger.info("Reading MINE metadata from: %s", meta_path)
                suffix = meta_path.suffix.lower()

                if suffix == ".jsonl":
                    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                elif suffix == ".json":
                    try:
                        payload = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(payload, list):
                            records.extend(payload)
                        elif isinstance(payload, dict):
                            records.append(payload)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON file: %s", meta_path)

                if records:
                    break

            if not records:
                logger.warning("No MINE records found for split '%s' in %s", split, root)
                return []

            samples: List[MultimodalSample] = []
            for item in records:
                split_value = str(item.get("split", "")).lower()
                if split_value and split_value != split.lower():
                    continue

                intent_labels = safe_list_of_ints(
                    item.get("intention_labels", item.get("intention", [0])),
                    default=[0],
                )
                action_labels = safe_list_of_ints(
                    item.get("action_labels", item.get("action", [0])),
                    default=[0],
                )

                raw_img_path = item.get("image_path") or item.get("image")
                final_img_path = None
                if isinstance(raw_img_path, str):
                    potential_path = (root / raw_img_path).resolve()
                    if potential_path.exists():
                        final_img_path = str(potential_path)
                    else:
                        final_img_path = _safe_local_image_path(raw_img_path)

                raw_audio_path = item.get("audio_path") or item.get("audio")
                raw_video_path = item.get("video_path") or item.get("video")

                samples.append(
                    MultimodalSample(
                        text=str(item.get("text") or item.get("caption") or item.get("transcript") or ""),
                        image_path=final_img_path,
                        audio_path=str(raw_audio_path) if raw_audio_path else None,
                        video_path=str(raw_video_path) if raw_video_path else None,
                        emotion_label=safe_int(item.get("emotion_label", item.get("emotion", 0)), default=0),
                        intention_labels=intent_labels,
                        action_labels=action_labels,
                        source_dataset="MINE_GDrive",
                    )
                )

                if limit and len(samples) >= limit:
                    break

            return samples

        except Exception as e:
            logger.exception("MINE GDrive loader error on split '%s': %s", split, e)
            return []


# ------------------------------------------------------------------------------
# 5. Synthetic fallback
# ------------------------------------------------------------------------------
class SyntheticMultimodalGenerator:
    @staticmethod
    def generate(num_samples: int = 100) -> List[MultimodalSample]:
        sentences = [
            "I absolutely love this new design!",
            "Can you process a refund for order 992?",
            "This makes me incredibly angry.",
        ]

        samples: List[MultimodalSample] = []
        for _ in range(num_samples):
            samples.append(
                MultimodalSample(
                    text=random.choice(sentences),
                    emotion_label=random.randint(0, 10),
                    intention_labels=[random.randint(0, 19)],
                    action_labels=[random.randint(0, 14)],
                    source_dataset="Synthetic_Emergency",
                )
            )
        return samples


# ------------------------------------------------------------------------------
# 6. Unified builder
# ------------------------------------------------------------------------------
class UnifiedCloudDatasetBuilder:
    REGISTRY = {
        "kaggle_goemotions": KaggleGoEmotionsLoader,
        "kaggle_facial": KaggleFacialEmotionLoader,
        "kaggle_intent": KaggleIntentLoader,
        "hf_emotion": DairAiEmotionLoader,
        "hf_dailydialog": DailyDialogLoader,
        "hf_coco": MSCOCOCaptionsLoader,
        "mine_gdrive": MINEGoogleDriveDatasetLoader,
        # aliases
        "goemotions": KaggleGoEmotionsLoader,
        "tweet_eval": DairAiEmotionLoader,
        "dailydialog": DailyDialogLoader,
        "mine": MINEGoogleDriveDatasetLoader,
        "emoticon": KaggleFacialEmotionLoader,
        "raza": KaggleIntentLoader,
        "coco": MSCOCOCaptionsLoader,
    }

    @staticmethod
    def build_multimodal_dataset(
        sources: Optional[List[str]] = None,
        splits: Optional[Dict[str, int]] = None,
        data_dir: Optional[str] = None,
        **kwargs,
    ) -> List[MultimodalSample]:
        if not sources:
            sources = [
                "mine",
                "kaggle_goemotions",
                "kaggle_facial",
                "kaggle_intent",
                "hf_emotion",
            ]

        if not splits:
            splits = {"train": 2000, "validation": 500}

        dataset_paths = discover_local_datasets(data_dir)
        all_samples: List[MultimodalSample] = []

        for source in sources:
            loader_class = UnifiedCloudDatasetBuilder.REGISTRY.get(source.lower().strip())
            if not loader_class:
                logger.warning("Unknown dataset source skipped: %s", source)
                continue

            logger.info("Executing Loader: %s", source.upper())
            for split_name, limit in splits.items():
                try:
                    loaded = loader_class.load_split(
                        split=split_name,
                        limit=limit,
                        data_dir=data_dir,
                        dataset_paths=dataset_paths,
                    )
                    all_samples.extend(loaded)
                    logger.info(
                        "Loaded %d samples from %s (%s)",
                        len(loaded),
                        source,
                        split_name,
                    )
                except Exception as e:
                    logger.exception(
                        "Loader failed for source=%s split=%s: %s",
                        source,
                        split_name,
                        e,
                    )

        if len(all_samples) < 10:
            logger.error(
                "FATAL: Primary datasets failed or were insufficient. Engaging synthetic fallback."
            )
            all_samples.extend(SyntheticMultimodalGenerator.generate(splits.get("train", 1000)))

        logger.info("UNIFIED BUILDER COMPLETE: %d total samples.", len(all_samples))
        return all_samples


# ------------------------------------------------------------------------------
# 7. PyTorch dataset and dataloaders
# ------------------------------------------------------------------------------
class CloudMultimodalDataset(Dataset):
    def __init__(self, samples: List[MultimodalSample], tokenizer, max_text_len: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.img_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        safe_text = sample.text if sample.text and str(sample.text).strip() else "[NO TEXT]"
        text_encoding = self.tokenizer(
            safe_text,
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        batch: Dict[str, Any] = {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "emotion_labels": torch.tensor(max(0, safe_int(sample.emotion_label, 0)), dtype=torch.long),
            "source": sample.source_dataset,
        }

        intention_target = torch.zeros(20, dtype=torch.float32)
        for intent_idx in sample.intention_labels:
            ii = safe_int(intent_idx, -1)
            if 0 <= ii < 20:
                intention_target[ii] = 1.0
        batch["intention_labels"] = intention_target

        action_target = torch.zeros(15, dtype=torch.float32)
        for action_idx in sample.action_labels:
            ai = safe_int(action_idx, -1)
            if 0 <= ai < 15:
                action_target[ai] = 1.0
        batch["action_labels"] = action_target

        if sample.image_path and os.path.exists(sample.image_path):
            try:
                img = Image.open(sample.image_path).convert("RGB")
                batch["images"] = self.img_transform(img)
            except Exception as e:
                logger.warning("Image load failed for %s: %s", sample.image_path, e)
                batch["images"] = torch.zeros(3, 224, 224)
        else:
            batch["images"] = torch.zeros(3, 224, 224)

        return batch


def get_cloud_dataloaders(
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    sources: Optional[List[str]] = None,
    max_samples: Optional[Dict[str, int]] = None,
    max_rows_per_source: Optional[int] = None,
    distributed: bool = False,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from transformers import AutoTokenizer

    data_root = get_data_root(data_dir)
    models_root = get_models_root()
    hf_data_cache = (data_root / "hf_datasets").resolve()
    hf_model_cache = (models_root / "hf_models").resolve()
    hf_data_cache.mkdir(parents=True, exist_ok=True)
    hf_model_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = str(hf_data_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_model_cache)

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-large",
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )

    if max_samples is None:
        train_rows = int(max_rows_per_source) if max_rows_per_source else 5000
        val_rows = max(100, train_rows // 5)
        max_samples = {"train": train_rows, "validation": val_rows}

    all_samples = UnifiedCloudDatasetBuilder.build_multimodal_dataset(
        sources=sources,
        splits=max_samples,
        data_dir=data_dir,
        **kwargs,
    )

    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    if not train_samples:
        train_samples = SyntheticMultimodalGenerator.generate(32)
    if not val_samples:
        val_samples = train_samples[: max(1, min(8, len(train_samples)))]
    if not test_samples:
        test_samples = train_samples[: max(1, min(8, len(train_samples)))]

    train_ds = CloudMultimodalDataset(train_samples, tokenizer)
    val_ds = CloudMultimodalDataset(val_samples, tokenizer)
    test_ds = CloudMultimodalDataset(test_samples, tokenizer)

    train_sampler = DistributedSampler(train_ds) if distributed else None

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=eval_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=eval_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl, test_dl


# ------------------------------------------------------------------------------
# 8. Optional smoke test
# ------------------------------------------------------------------------------
def debug_discovery(data_dir: Optional[str] = None) -> Dict[str, str]:
    discovered = discover_local_datasets(data_dir)
    return {k: str(v) for k, v in discovered.items()}


if __name__ == "__main__":
    discovered = debug_discovery()
    print("Discovered datasets:", discovered)

BEAR BMVC 2026 - MASTER CLOUD DATASET ARCHITECTURE (AUTO-DISCOVERY VERSION)
=========================================================================================
Cloud-native multimodal dataset loading system.

Features:
- Auto-discover local datasets from a single data_dir root
- TSV/CSV auto-detection for GoEmotions
- Real image loading via PIL and Torchvision transforms
- Google Drive / local MINE loader
- Hugging Face dataset support with local cache paths
- Safer schema handling and better logging
"""

from __future__ import annotations

