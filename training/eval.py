"""
BMVC 2026 - Multimodal Evaluation Engine
Spotlight Upgrades: Macro F1, mean Average Precision (mAP), and safe Zero-Division.
"""

from __future__ import annotations

import warnings
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
)

# 🌟 SOTA FIX: Silence the expected sklearn warnings when rare classes miss a validation batch
warnings.filterwarnings("ignore", message="No positive class found in y_true.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

def evaluate_tritask(
    emotion_preds: torch.Tensor,
    intention_preds: torch.Tensor,
    action_preds: torch.Tensor,
    emotion_labels: torch.Tensor,
    intention_labels: torch.Tensor,
    action_labels: torch.Tensor,
    threshold: float = 0.4,  # BMVC Dynamic Threshold
) -> dict[str, float]:
    """
    Evaluate multi-task predictions using strict academic standards.
    """
    metrics = {}
    
    # -------------------------------------------------------------------------
    # 1. Emotion (Single-Label Classification)
    # -------------------------------------------------------------------------
    if emotion_preds.dim() > 1 and emotion_preds.size(1) > 1:
        emotion_preds = torch.argmax(emotion_preds, dim=1)
        
    emotion_preds_np = emotion_preds.cpu().numpy()
    emotion_labels_np = emotion_labels.cpu().numpy()
    
    metrics["emotion_accuracy"] = accuracy_score(emotion_labels_np, emotion_preds_np)
    metrics["emotion_macro_f1"] = f1_score(emotion_labels_np, emotion_preds_np, average="macro", zero_division=0)
    
    # -------------------------------------------------------------------------
    # 2. Intention (Multi-Label Classification)
    # -------------------------------------------------------------------------
    if intention_preds.min() < 0 or intention_preds.max() > 1:
        intention_probs = torch.sigmoid(intention_preds)
    else:
        intention_probs = intention_preds
        
    intention_probs_np = intention_probs.cpu().numpy()
    intention_binary_np = (intention_probs_np > threshold).astype(int)
    intention_labels_np = intention_labels.cpu().numpy()
    
    # Macro F1 is threshold dependent
    metrics["intention_macro_f1"] = f1_score(intention_labels_np, intention_binary_np, average="macro", zero_division=0)
    
    # 🌟 SOTA UPGRADE: mAP evaluates the entire Precision-Recall curve independent of the 0.4 threshold
    try:
        metrics["intention_mAP"] = average_precision_score(intention_labels_np, intention_probs_np, average="macro")
    except ValueError:
        metrics["intention_mAP"] = 0.0 # Fallback if a batch lacks positive samples
        
    # -------------------------------------------------------------------------
    # 3. Action (Multi-Label Classification)
    # -------------------------------------------------------------------------
    if action_preds.min() < 0 or action_preds.max() > 1:
        action_probs = torch.sigmoid(action_preds)
    else:
        action_probs = action_preds
        
    action_probs_np = action_probs.cpu().numpy()
    action_binary_np = (action_probs_np > threshold).astype(int)
    action_labels_np = action_labels.cpu().numpy()
    
    metrics["action_macro_f1"] = f1_score(action_labels_np, action_binary_np, average="macro", zero_division=0)
    
    try:
        metrics["action_mAP"] = average_precision_score(action_labels_np, action_probs_np, average="macro")
    except ValueError:
        metrics["action_mAP"] = 0.0
        
    return metrics
