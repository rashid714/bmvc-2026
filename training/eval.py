"""Evaluation metrics and utilities for multimodal classification."""

from __future__ import annotations

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_tritask(
    emotion_preds: torch.Tensor,
    intention_preds: torch.Tensor,
    action_preds: torch.Tensor,
    emotion_labels: torch.Tensor,
    intention_labels: torch.Tensor,
    action_labels: torch.Tensor,
) -> dict[str, float]:
    """
    Evaluate multi-task predictions.
    
    Args:
        emotion_preds: [batch] class predictions
        intention_preds: [batch, num_classes] binary predictions
        action_preds: [batch, num_classes] binary predictions
        emotion_labels: [batch] class labels
        intention_labels: [batch, num_classes] binary labels
        action_labels: [batch, num_classes] binary labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Emotion (single-label classification)
    emotion_preds_np = emotion_preds.cpu().numpy()
    emotion_labels_np = emotion_labels.cpu().numpy()
    
    metrics["emotion_accuracy"] = accuracy_score(emotion_labels_np, emotion_preds_np)
    metrics["emotion_macro_f1"] = f1_score(emotion_labels_np, emotion_preds_np, average="macro", zero_division=0)
    metrics["emotion_weighted_f1"] = f1_score(emotion_labels_np, emotion_preds_np, average="weighted", zero_division=0)
    
    # Intention (multi-label classification)
    intention_preds_np = intention_preds.cpu().numpy()
    intention_labels_np = intention_labels.cpu().numpy()
    
    metrics["intention_micro_f1"] = f1_score(intention_labels_np, intention_preds_np, average="micro", zero_division=0)
    metrics["intention_macro_f1"] = f1_score(intention_labels_np, intention_preds_np, average="macro", zero_division=0)
    metrics["intention_hamming"] = 1.0 - (intention_preds_np != intention_labels_np).mean()
    
    # Action (multi-label classification)
    action_preds_np = action_preds.cpu().numpy()
    action_labels_np = action_labels.cpu().numpy()
    
    metrics["action_micro_f1"] = f1_score(action_labels_np, action_preds_np, average="micro", zero_division=0)
    metrics["action_macro_f1"] = f1_score(action_labels_np, action_preds_np, average="macro", zero_division=0)
    metrics["action_hamming"] = 1.0 - (action_preds_np != action_labels_np).mean()
    
    return metrics
