"""
BMVC 2026 - Multimodal Loss Engines
Spotlight Upgrades: Weighted Multi-Label Focal Loss & Label Smoothing
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

class WeightedMultiLabelFocalLoss(nn.Module):
    """
    🌟 The Ultimate Multi-Label Engine: 
    Fuses dynamic inverse class weights (pos_weight) with Focal Loss modulating factors.
    Prevents rare classes from vanishing into zero-gradients.
    """
    def __init__(self, pos_weight: torch.Tensor = None, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Calculate standard BCE with the dynamic inverse weights
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets.float(), 
            pos_weight=self.pos_weight, 
            reduction="none"
        )
        
        # 2. Calculate the probability of the true class
        pt = torch.exp(-bce_loss)
        
        # 3. Apply Focal Modulator: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SilverStandardLoss(nn.Module):
    """
    Centralized Loss Engine for the BMVC pipeline.
    Combines Smoothed CrossEntropy for Emotion and Weighted Focal for Intention/Action.
    """
    def __init__(
        self,
        pos_weight_intent: torch.Tensor = None,
        pos_weight_action: torch.Tensor = None,
        emotion_weight: float = 1.0,
        intention_weight: float = 2.0, 
        action_weight: float = 2.0, 
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.intention_weight = intention_weight
        self.action_weight = action_weight

        # 🌟 Label Smoothing prevents RoBERTa from becoming overconfident on simple images
        self.emotion_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 🌟 Weighted Focal Loss automatically handles the extreme Kaggle/MINE imbalances
        self.intention_loss_fn = WeightedMultiLabelFocalLoss(pos_weight=pos_weight_intent, gamma=2.0)
        self.action_loss_fn = WeightedMultiLabelFocalLoss(pos_weight=pos_weight_action, gamma=2.0)

    def forward(
        self,
        emotion_logits: torch.Tensor,
        intention_logits: torch.Tensor,
        action_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        intention_labels: torch.Tensor,
        action_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_labels)
        intention_loss = self.intention_loss_fn(intention_logits, intention_labels)
        action_loss = self.action_loss_fn(action_logits, action_labels)

        total_loss = (
            self.emotion_weight * emotion_loss
            + self.intention_weight * intention_loss
            + self.action_weight * action_loss
        )

        return {
            "total_loss": total_loss,
            "emotion_loss": emotion_loss,
            "intention_loss": intention_loss,
            "action_loss": action_loss,
        }
