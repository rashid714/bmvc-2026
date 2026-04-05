from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal loss for single-label emotion classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return loss.mean()

# 🌟 THE BMVC 80% UPGRADE: Multi-Label Focal Loss
class MultiLabelFocalLoss(nn.Module):
    """Focal loss specifically designed for highly imbalanced multi-label arrays."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCEWithLogitsLoss expects float targets for multi-label
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """Combines emotion, intention, action losses using the Focal Engine."""

    def __init__(
        self,
        emotion_weight: float = 1.0,
        intention_weight: float = 2.0, # 🌟 Increased to force learning rare classes
        action_weight: float = 2.0,    # 🌟 Increased to force learning rare classes
        use_focal: bool = True,
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.intention_weight = intention_weight
        self.action_weight = action_weight

        if use_focal:
            self.emotion_loss_fn = FocalLoss()
        else:
            self.emotion_loss_fn = nn.CrossEntropyLoss()

        # 🌟 Replaced standard BCE with Multi-Label Focal Loss
        self.intention_loss_fn = MultiLabelFocalLoss(gamma=2.0)
        self.action_loss_fn = MultiLabelFocalLoss(gamma=2.0)

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


class ConsistencyLoss(nn.Module):
    """Encourages semantic consistency across modalities (counterfactual regularization)."""

    def __init__(self, lambda_consistency: float = 0.1):
        super().__init__()
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        emotion_probs: torch.Tensor,
        intention_probs: torch.Tensor,
        action_probs: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = emotion_probs.shape[0]
        consistency_loss = 0.0

        for i in range(batch_size):
            mask = modality_mask[i]
            if mask.sum() >= 2:
                consistency_loss += torch.std(emotion_probs[i]) + torch.std(intention_probs[i])

        return self.lambda_consistency * consistency_loss / batch_size


class ContrastiveRelationLoss(nn.Module):
    """Contrastive learning between emotion-intention pairs for implicit correlation mining."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emotion_features: torch.Tensor,
        intention_features: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = emotion_features.shape[0]

        emotion_norm = F.normalize(emotion_features, dim=1)
        intention_norm = F.normalize(intention_features, dim=1)

        similarity_matrix = torch.mm(emotion_norm, intention_norm.t()) / self.temperature

        labels = torch.arange(batch_size, device=emotion_features.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
