"""
Advanced Top-Tier Multimodal BEAR Model (BMVC 2026)
Fixed version for compatibility with train_advanced_multimodal.py

Key fixes:
- Proper class-level forward() method
- Returns a plain dict (not a tuple)
- Real image support via ResNet50 backbone
- Safe fallback if pretrained vision weights are unavailable
- Added get_predictions() helper for compatibility with other training scripts
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

logger = logging.getLogger(__name__)


class AdvancedModalityEncoder(nn.Module):
    """
    Enhanced modality encoder.
    Input: feature vector -> Output: normalized hidden_dim embedding
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class DualLayerAttention(nn.Module):
    """
    Dual-layer cross-modal attention mechanism.
    Layer 1: low-level feature fusion
    Layer 2: high-level semantic fusion
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16):
        super().__init__()

        self.layer1_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.layer1_norm = nn.LayerNorm(hidden_dim)
        self.layer1_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.layer2_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.layer2_norm = nn.LayerNorm(hidden_dim)
        self.layer2_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.gating = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embeddings_list: list[torch.Tensor],
        reliability_scores: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack(embeddings_list, dim=1)  # [B, M, H]

        attn_out1, _ = self.layer1_attention(stacked, stacked, stacked)
        attn_out1 = self.layer1_norm(attn_out1 + stacked)
        ffn_out1 = self.layer1_ffn(attn_out1)
        layer1_out = self.layer1_norm(ffn_out1 + attn_out1)

        weighted1 = layer1_out * reliability_scores.unsqueeze(-1)
        layer1_fused = torch.mean(weighted1, dim=1)  # [B, H]

        attn_out2, _ = self.layer2_attention(stacked, stacked, stacked)
        attn_out2 = self.layer2_norm(attn_out2 + stacked)
        ffn_out2 = self.layer2_ffn(attn_out2)
        layer2_out = self.layer2_norm(ffn_out2 + attn_out2)

        weighted2 = layer2_out * (reliability_scores ** 1.5).unsqueeze(-1)
        layer2_fused = torch.mean(weighted2, dim=1)  # [B, H]

        combined_fused = torch.cat([layer1_fused, layer2_fused], dim=-1)
        gate = self.gating(combined_fused)
        final_fused = gate * layer1_fused + (1.0 - gate) * layer2_fused
        return final_fused


class AdvancedReliabilityModule(nn.Module):
    """
    Modality reliability scoring with uncertainty estimation.
    """

    def __init__(self, hidden_dim: int = 1024, num_modalities: int = 4):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1),
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, text_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        confidence = self.confidence_net(text_embed)
        uncertainty = self.uncertainty_net(text_embed)
        reliability = confidence * (1.0 - uncertainty)
        return reliability, uncertainty


class AdvancedLLMBackbone(nn.Module):
    """
    Dual-layer LLM backbone combining two encoders.
    Layer 1: RoBERTa-large
    Layer 2: DistilRoBERTa-base
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.roberta = AutoModel.from_pretrained("roberta-large")
        self.roberta_proj = nn.Linear(1024, hidden_dim)

        self.distilroberta = AutoModel.from_pretrained("distilroberta-base")
        self.distilroberta_proj = nn.Linear(768, hidden_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        roberta_cls = roberta_out.last_hidden_state[:, 0, :]
        roberta_embed = self.roberta_proj(roberta_cls)

        distilroberta_out = self.distilroberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        distilroberta_cls = distilroberta_out.last_hidden_state[:, 0, :]
        distilroberta_embed = self.distilroberta_proj(distilroberta_cls)

        combined = torch.cat([roberta_embed, distilroberta_embed], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * roberta_embed + (1.0 - gate) * distilroberta_embed
        return self.layer_norm(fused)


class AdvancedTriTaskHead(nn.Module):
    """
    Task heads for emotion, intention, and action.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_emotion: int = 11,
        num_intention: int = 20,
        num_action: int = 15,
    ):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotion),
        )
        self.intention_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_intention),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_action),
        )

    def forward(self, fused_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_encoder(fused_embed)
        emotion_logits = self.emotion_head(shared)
        intention_logits = self.intention_head(shared)
        action_logits = self.action_head(shared)
        return emotion_logits, intention_logits, action_logits


class AdvancedBEARModel(nn.Module):
    """
    Advanced multimodal BEAR model.

    Modalities:
    - Text via dual-LLM backbone
    - Images via ResNet50
    - Audio placeholder encoder
    - Video placeholder encoder
    """

    def __init__(self, hidden_dim: int = 1024, use_pretrained_vision: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_backbone = AdvancedLLMBackbone(hidden_dim)
        self.text_encoder = AdvancedModalityEncoder(hidden_dim, hidden_dim)

        # Vision backbone
        resnet = self._build_resnet50(use_pretrained_vision)
        for param in resnet.parameters():
            param.requires_grad = False
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.image_encoder = AdvancedModalityEncoder(2048, hidden_dim)

        # Placeholder encoders for future audio/video features
        self.audio_encoder = AdvancedModalityEncoder(512, hidden_dim)
        self.video_encoder = AdvancedModalityEncoder(1024, hidden_dim)

        self.reliability_module = AdvancedReliabilityModule(hidden_dim, num_modalities=4)
        self.fusion = DualLayerAttention(hidden_dim, num_heads=16)
        self.task_heads = AdvancedTriTaskHead(
            hidden_dim,
            num_emotion=11,
            num_intention=20,
            num_action=15,
        )
        self.temperature = nn.Parameter(torch.ones(1))

    @staticmethod
    def _build_resnet50(use_pretrained_vision: bool) -> nn.Module:
        try:
            if use_pretrained_vision:
                # torchvision >= 0.13 style
                weights = getattr(models, "ResNet50_Weights").IMAGENET1K_V2
                return models.resnet50(weights=weights)
            return models.resnet50(weights=None)
        except Exception as e:
            logger.warning("Falling back to uninitialized ResNet50 weights: %s", e)
            try:
                # older torchvision fallback
                return models.resnet50(pretrained=False)
            except TypeError:
                return models.resnet50(weights=None)

    def _encode_images(self, images: Optional[torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
        if images is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        if images.dim() != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")

        with torch.no_grad():
            img_feats = self.vision_backbone(images).flatten(1)  # [B, 2048]

        image_embed = self.image_encoder(img_feats)

        # Zero out embeddings for placeholder zero-images from the dataloader.
        image_present_mask = (images.abs().sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        image_embed = image_embed * image_present_mask
        return image_embed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        batch_size = input_ids.size(0)

        text_embed = self.text_backbone(input_ids, attention_mask)
        text_embed = self.text_encoder(text_embed)

        reliability_scores, uncertainties = self.reliability_module(text_embed)

        modalities: list[torch.Tensor] = [text_embed]

        image_embed = self._encode_images(images, device=device, batch_size=batch_size)
        modalities.append(image_embed)

        if audio_features is not None:
            audio_embed = self.audio_encoder(audio_features)
        else:
            audio_embed = torch.zeros(batch_size, self.hidden_dim, device=device)
        modalities.append(audio_embed)

        if video_features is not None:
            video_embed = self.video_encoder(video_features)
        else:
            video_embed = torch.zeros(batch_size, self.hidden_dim, device=device)
        modalities.append(video_embed)

        fused_embed = self.fusion(modalities, reliability_scores)
        emotion_logits, intention_logits, action_logits = self.task_heads(fused_embed)

        temperature = torch.clamp(self.temperature, min=1e-3)
        emotion_logits = emotion_logits / temperature
        intention_logits = intention_logits / temperature
        action_logits = action_logits / temperature

        return {
            "emotion_logits": emotion_logits,
            "intention_logits": intention_logits,
            "action_logits": action_logits,
            "reliability_scores": reliability_scores,
            "uncertainties": uncertainties,
            "fused_embed": fused_embed,
            "text_embed": text_embed,
            "image_embed": image_embed,
        }

    @staticmethod
    def get_predictions(model_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "emotion_preds": torch.argmax(model_output["emotion_logits"], dim=1),
            "intention_preds": (torch.sigmoid(model_output["intention_logits"]) > 0.5).long(),
            "action_preds": (torch.sigmoid(model_output["action_logits"]) > 0.5).long(),
        }
