from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ReliabilityModule(nn.Module):
    """Estimates modality reliability and calibrated confidence."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 modalities
            nn.Sigmoid(),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.reliability_head(hidden_state)


class TriTaskHead(nn.Module):
    """Shared + task-specific heads for Emotion, Intention, Action."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_emotion_classes: int = 11,
        num_intention_classes: int = 20,
        num_action_classes: int = 15,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_emotion_classes),
        )

        self.intention_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_intention_classes),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_action_classes),
        )

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_layer(hidden_state)
        emotion_logits = self.emotion_head(shared)
        intention_logits = self.intention_head(shared)
        action_logits = self.action_head(shared)
        return emotion_logits, intention_logits, action_logits


class CalibrationModule(nn.Module):
    """Temperature scaling and focal loss for calibrated predictions."""

    def __init__(self, num_tasks: int = 3, learnable: bool = True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.ones(num_tasks))
        else:
            self.register_buffer("temperature", torch.ones(num_tasks))

    def forward(self, logits: torch.Tensor, task_idx: int) -> torch.Tensor:
        return logits / (self.temperature[task_idx].clamp(min=0.1) + 1e-8)

    def set_temperature(self, temps: list[float]) -> None:
        if self.temperature.requires_grad:
            self.temperature.data = torch.tensor(temps, device=self.temperature.device)


class ModalityEncoder(nn.Module):
    """Universal modality encoder with projection to shared space."""

    def __init__(self, input_dim: int, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MultimodalFusionModule(nn.Module):
    """Advanced fusion combining attention + gating + reliability weights."""

    def __init__(self, hidden_dim: int = 768, num_modalities: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # Cross-modal attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, modality_embeds: list[torch.Tensor], reliability_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            modality_embeds: List of [batch, hidden_dim] tensors (4 modalities)
            reliability_weights: [batch, 4] reliability scores per modality
        """
        batch_size = modality_embeds[0].size(0)

        # Stack modalities and apply reliability weighting
        stacked = torch.stack(modality_embeds, dim=1)  # [batch, 4, hidden_dim]
        weighted = stacked * reliability_weights.unsqueeze(-1)

        # Cross-modal attention
        attn_out, _ = self.multihead_attn(
            weighted, weighted, weighted
        )

        # Flatten for gating
        flattened = attn_out.reshape(batch_size, -1)  # [batch, 4*hidden_dim]
        gate_logits = self.gate(flattened)  # [batch, hidden_dim]
        
        # Apply gate to each modality's contribution
        gated_embeds = []
        for i in range(4):
            gated_embeds.append(attn_out[:, i, :] * gate_logits)
        
        # Fused representation: weighted sum
        fusion = torch.stack(gated_embeds, dim=1).mean(dim=1)  # [batch, hidden_dim]

        return fusion


class MINEModel(nn.Module):
    """
    World-class multimodal Emotion-Intention-Action model with:
    - Multimodal fusion (text, image, audio, video)
    - Reliability-aware gating mechanism
    - Tri-task heads with calibration
    - Production-ready for cloud deployment
    """

    def __init__(
        self,
        text_backbone: str = "distilroberta-base",
        hidden_dim: int = 768,
        num_emotion_classes: int = 11,
        num_intention_classes: int = 20,
        num_action_classes: int = 15,
        dropout_rate: float = 0.2,
        use_calibration: bool = True,
        use_multimodal: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_multimodal = use_multimodal

        # Text encoder
        self.text_backbone = AutoModel.from_pretrained(text_backbone)
        self.text_backbone_config = self.text_backbone.config

        # Multimodal encoders (for cloud deployment with full data)
        self.image_encoder = ModalityEncoder(2048, hidden_dim, dropout_rate)  # ResNet feature size
        self.audio_encoder = ModalityEncoder(512, hidden_dim, dropout_rate)   # Audio feature size
        self.video_encoder = ModalityEncoder(1024, hidden_dim, dropout_rate)  # Video feature size

        # Fusion module
        self.fusion = MultimodalFusionModule(hidden_dim, num_modalities=4)

        # Task-specific components
        self.reliability_module = ReliabilityModule(hidden_dim)
        self.tri_task_head = TriTaskHead(
            hidden_dim=hidden_dim,
            num_emotion_classes=num_emotion_classes,
            num_intention_classes=num_intention_classes,
            num_action_classes=num_action_classes,
            dropout_rate=dropout_rate,
        )
        self.calibration = CalibrationModule(num_tasks=3, learnable=use_calibration)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass supporting multimodal inputs with graceful fallback to text-only.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            image_features: [batch, 2048] (optional)
            audio_features: [batch, 512] (optional)
            video_features: [batch, 1024] (optional)
            modality_mask: [batch, 4] indicating which modalities available
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Text encoding
        text_out = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embed = text_out.last_hidden_state[:, 0, :]  # CLS token

        if self.use_multimodal and (image_features is not None or audio_features is not None or video_features is not None):
            # Initialize modality embeddings
            modality_embeds = [text_embed]

            # Handle optional modalities with zero embedding as fallback
            if image_features is not None:
                image_embed = self.image_encoder(image_features)
            else:
                image_embed = torch.zeros_like(text_embed)

            if audio_features is not None:
                audio_embed = self.audio_encoder(audio_features)
            else:
                audio_embed = torch.zeros_like(text_embed)

            if video_features is not None:
                video_embed = self.video_encoder(video_features)
            else:
                video_embed = torch.zeros_like(text_embed)

            modality_embeds.extend([image_embed, audio_embed, video_embed])

            # Reliability weighting
            reliability_scores = self.reliability_module(text_embed)

            # Fused multimodal representation
            fused_hidden = self.fusion(modality_embeds, reliability_scores)
        else:
            # Text-only fallback (when no multimodal data available)
            reliability_scores = self.reliability_module(text_embed)
            fused_hidden = text_embed

        # Task predictions
        emotion_logits, intention_logits, action_logits = self.tri_task_head(fused_hidden)

        # Calibrated predictions
        emotion_logits = self.calibration(emotion_logits, task_idx=0)
        intention_logits = self.calibration(intention_logits, task_idx=1)
        action_logits = self.calibration(action_logits, task_idx=2)

        if modality_mask is None:
            modality_mask = torch.ones(batch_size, 4, device=device)

        return {
            "emotion_logits": emotion_logits,
            "intention_logits": intention_logits,
            "action_logits": action_logits,
            "reliability_scores": reliability_scores,
            "modality_mask": modality_mask,
            "fused_representation": fused_hidden,
        }

    def get_predictions(self, model_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        emotion_probs = F.softmax(model_output["emotion_logits"], dim=1)
        intention_probs = torch.sigmoid(model_output["intention_logits"])
        action_probs = torch.sigmoid(model_output["action_logits"])

        return {
            "emotion_probs": emotion_probs,
            "emotion_preds": emotion_probs.argmax(dim=1),
            "intention_probs": intention_probs,
            "intention_preds": (intention_probs > 0.5).long(),
            "action_probs": action_probs,
            "action_preds": (action_probs > 0.5).long(),
        }
