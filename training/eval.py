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
    threshold: float = 0.4,  # 🌟 BMVC DYNAMIC THRESHOLD
) -> dict[str, float]:
    """
    Evaluate multi-task predictions using academic standards.
    
    Args:
        emotion_preds: [batch, num_classes] logits or [batch] class predictions
        intention_preds: [batch, num_classes] raw logits or probabilities
        action_preds: [batch, num_classes] raw logits or probabilities
        emotion_labels: [batch] class labels
        intention_labels: [batch, num_classes] binary labels
        action_labels: [batch, num_classes] binary labels
        threshold: Decision boundary for multi-label classification (default: 0.4)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # -------------------------------------------------------------------------
    # 1. Emotion (single-label classification)
    # -------------------------------------------------------------------------
    # If logits are passed instead of class indices, convert to class predictions
    if emotion_preds.dim() > 1 and emotion_preds.size(1) > 1:
        emotion_preds = torch.argmax(emotion_preds, dim=1)
        
    emotion_preds_np = emotion_preds.cpu().numpy()
    emotion_labels_np = emotion_labels.cpu().numpy()
    
    metrics["emotion_accuracy"] = accuracy_score(emotion_labels_np, emotion_preds_np)
    metrics["emotion_macro_f1"] = f1_score(emotion_labels_np, emotion_preds_np, average="macro", zero_division=0)
    metrics["emotion_weighted_f1"] = f1_score(emotion_labels_np, emotion_preds_np, average="weighted", zero_division=0)
    
    # -------------------------------------------------------------------------
    # 2. Intention (multi-label classification)
    # -------------------------------------------------------------------------
    # Convert logits to probabilities if they haven't been already
    if intention_preds.min() < 0 or intention_preds.max() > 1:
        intention_probs = torch.sigmoid(intention_preds)
    else:
        intention_probs = intention_preds
        
    intention_binary_np = (intention_probs > threshold).cpu().numpy().astype(int)
    intention_labels_np = intention_labels.cpu().numpy()
    
    metrics["intention_micro_f1"] = f1_score(intention_labels_np, intention_binary_np, average="micro", zero_division=0)
    # 🌟 CRITICAL: Track the Macro F1 for the paper
    metrics["intention_macro_f1"] = f1_score(intention_labels_np, intention_binary_np, average="macro", zero_division=0)
    metrics["intention_hamming"] = 1.0 - (intention_binary_np != intention_labels_np).mean()
    
    # -------------------------------------------------------------------------
    # 3. Action (multi-label classification)
    # -------------------------------------------------------------------------
    # Convert logits to probabilities if they haven't been already
    if action_preds.min() < 0 or action_preds.max() > 1:
        action_probs = torch.sigmoid(action_preds)
    else:
        action_probs = action_preds
        
    action_binary_np = (action_probs > threshold).cpu().numpy().astype(int)
    action_labels_np = action_labels.cpu().numpy()
    
    metrics["action_micro_f1"] = f1_score(action_labels_np, action_binary_np, average="micro", zero_division=0)
    # 🌟 CRITICAL: Track the Macro F1 for the paper
    metrics["action_macro_f1"] = f1_score(action_labels_np, action_binary_np, average="macro", zero_division=0)
    metrics["action_hamming"] = 1.0 - (action_binary_np != action_labels_np).mean()
    
    return metrics
