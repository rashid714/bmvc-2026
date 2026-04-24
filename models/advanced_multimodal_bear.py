"""
Advanced Top-Tier Multimodal BEAR Model (BMVC 2026)
Spotlight Version: DINOv2 Vision (Top-Layers Unfrozen) + RoBERTa-Large + Strict Local Caching
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

# ==============================================================================
# STRICT LOCAL DIRECTORY ENFORCEMENT
# ==============================================================================
def get_project_root() -> Path:
    """Dynamically finds the BMVC 2026 main folder to lock all downloads locally."""
    return Path(__file__).resolve().parent.parent

# Force all massive model weights to save strictly inside the project folder
MODELS_DIR = get_project_root() / "models"
HF_CACHE = MODELS_DIR / "hf_hub"
TORCH_CACHE = MODELS_DIR / "torch_hub"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
TORCH_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_HOME"] = str(HF_CACHE)
torch.hub.set_dir(str(TORCH_CACHE))

# ==============================================================================
# MODALITY ENCODERS & FUSION
# ==============================================================================
class AdvancedModalityEncoder(nn.Module):
    """
    Enhanced modality encoder.
    Input: raw backbone feature vector -> Output: normalized hidden_dim embedding
    """
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class DualLayerAttention(nn.Module):
    """
    Dual-layer cross-modal attention mechanism with reliability gating.
    """
    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.layer1_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.layer1_norm = nn.LayerNorm(hidden_dim)
        self.layer1_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim * 4, hidden_dim))

        self.layer2_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.layer2_norm = nn.LayerNorm(hidden_dim)
        self.layer2_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim * 4, hidden_dim))

        self.gating = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, embeddings_list: list[torch.Tensor], reliability_scores: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(embeddings_list, dim=1)  # [B, M, H]

        attn_out1, _ = self.layer1_attention(stacked, stacked, stacked)
        attn_out1 = self.layer1_norm(attn_out1 + stacked)
        ffn_out1 = self.layer1_ffn(attn_out1)
        layer1_out = self.layer1_norm(ffn_out1 + attn_out1)

        weighted1 = layer1_out * reliability_scores.unsqueeze(-1)
        layer1_fused = torch.mean(weighted1, dim=1) 

        attn_out2, _ = self.layer2_attention(stacked, stacked, stacked)
        attn_out2 = self.layer2_norm(attn_out2 + stacked)
        ffn_out2 = self.layer2_ffn(attn_out2)
        layer2_out = self.layer2_norm(ffn_out2 + attn_out2)

        weighted2 = layer2_out * (reliability_scores ** 1.5).unsqueeze(-1)
        layer2_fused = torch.mean(weighted2, dim=1) 

        combined_fused = torch.cat([layer1_fused, layer2_fused], dim=-1)
        gate = self.gating(combined_fused)
        return gate * layer1_fused + (1.0 - gate) * layer2_fused


class AdvancedReliabilityModule(nn.Module):
    """
    Independent Modality Reliability Scoring.
    Every modality evaluates its own confidence and uncertainty mathematically.
    """
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.confidence_scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.uncertainty_scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, modalities_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked_mods = torch.stack(modalities_list, dim=1) 
        
        raw_confidence = self.confidence_scorer(stacked_mods).squeeze(-1) 
        confidence = self.softmax(raw_confidence)
        uncertainty = self.uncertainty_scorer(stacked_mods).squeeze(-1) 
        
        reliability = confidence * (1.0 - uncertainty)
        reliability = reliability / (reliability.sum(dim=-1, keepdim=True) + 1e-6)
        return reliability, uncertainty

# ==============================================================================
# SOTA BACKBONES (DINOv2 & RoBERTa)
# ==============================================================================
class TextBackboneRoBERTa(nn.Module):
    """
    Pure RoBERTa-Large Backbone.
    Strategically freezes bottom layers to save massive VRAM while preserving grammar.
    """
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-large", cache_dir=str(HF_CACHE))
        
        # 🌟 VRAM OPTIMIZATION: Freeze the bottom 16 layers (out of 24)
        if hasattr(self.roberta, "encoder"):
            for layer in self.roberta.encoder.layer[:16]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.layer_norm(cls_token)


class AdvancedTriTaskHead(nn.Module):
    """
    Strict dimension enforcement to match curated Silver Standard dataset.
    """
    def __init__(self, hidden_dim: int = 1024, num_emotion: int = 9, num_intention: int = 12, num_action: int = 15):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
        )
        self.emotion_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, num_emotion))
        self.intention_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, num_intention))
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, num_action))

    def forward(self, fused_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_encoder(fused_embed)
        return self.emotion_head(shared), self.intention_head(shared), self.action_head(shared)

# ==============================================================================
# MAIN BEAR MODEL
# ==============================================================================
class AdvancedBEARModel(nn.Module):
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Text Pipeline (RoBERTa-Large -> 1024)
        self.text_backbone = TextBackboneRoBERTa(hidden_dim)
        self.text_encoder = AdvancedModalityEncoder(1024, hidden_dim)

        # 2. Vision Pipeline (DINOv2 -> 768)
        self.vision_backbone = self._build_dinov2()
        self.image_encoder = AdvancedModalityEncoder(768, hidden_dim)

        # 3. Audio/Video Placeholders
        self.audio_encoder = AdvancedModalityEncoder(512, hidden_dim)
        self.video_encoder = AdvancedModalityEncoder(1024, hidden_dim)

        # 4. Fusion & Output Engines
        self.reliability_module = AdvancedReliabilityModule(hidden_dim)
        self.fusion = DualLayerAttention(hidden_dim, num_heads=16)
        self.task_heads = AdvancedTriTaskHead(hidden_dim, num_emotion=9, num_intention=12, num_action=15)
        self.temperature = nn.Parameter(torch.ones(1))

    @staticmethod
    def _build_dinov2() -> nn.Module:
        """Loads Meta's DINOv2 and unfreezes the top layers for domain adaptation."""
        logger.info("Downloading/Loading DINOv2 ViT-B/14 Backbone...")
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # 1. Freeze the entire model to save VRAM initially
        for param in dinov2.parameters():
            param.requires_grad = False
            
        # 2. 🌟 STRATEGY A: Unfreeze the final 2 Transformer blocks! 
        # This allows DINOv2 to adapt its pre-trained knowledge specifically to human faces.
        if hasattr(dinov2, 'blocks'):
            for block in dinov2.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
                    
        # 3. Unfreeze the final normalization layer
        if hasattr(dinov2, 'norm'):
            for param in dinov2.norm.parameters():
                param.requires_grad = True
                
        return dinov2

    def _encode_images(self, images: Optional[torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
        if images is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        if images.dim() != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")

        # 🌟 CRITICAL FIX: Removed torch.no_grad() so gradients flow into the unfrozen DINOv2 layers
        # DINOv2 returns the [B, 768] CLS token directly
        img_feats = self.vision_backbone(images)

        image_embed = self.image_encoder(img_feats)
        image_present_mask = (images.abs().sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        return image_embed * image_present_mask

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

        # Extract & Encode Modalities
        text_embed = self.text_backbone(input_ids, attention_mask)
        text_embed = self.text_encoder(text_embed)
        image_embed = self._encode_images(images, device=device, batch_size=batch_size)
        
        audio_embed = self.audio_encoder(audio_features) if audio_features is not None else torch.zeros(batch_size, self.hidden_dim, device=device)
        video_embed = self.video_encoder(video_features) if video_features is not None else torch.zeros(batch_size, self.hidden_dim, device=device)

        # Gate & Fuse
        modalities = [text_embed, image_embed, audio_embed, video_embed]
        reliability_scores, uncertainties = self.reliability_module(modalities)
        fused_embed = self.fusion(modalities, reliability_scores)

        # Task Heads & Temperature Scaling
        emotion_logits, intention_logits, action_logits = self.task_heads(fused_embed)
        temperature = torch.clamp(self.temperature, min=1e-3)
        
        return {
            "emotion_logits": emotion_logits / temperature,
            "intention_logits": intention_logits / temperature,
            "action_logits": action_logits / temperature,
            "reliability_scores": reliability_scores,
            "uncertainties": uncertainties,
            "fused_embed": fused_embed,
            "text_embed": text_embed,
            "image_embed": image_embed,
        }

    @staticmethod
    def get_predictions(model_output: Dict[str, torch.Tensor], threshold: float = 0.4) -> Dict[str, torch.Tensor]:
        """BMVC Dynamic Multi-Label Thresholding Strategy"""
        return {
            "emotion_preds": torch.argmax(model_output["emotion_logits"], dim=1),
            "intention_preds": (torch.sigmoid(model_output["intention_logits"]) > threshold).long(),
            "action_preds": (torch.sigmoid(model_output["action_logits"]) > threshold).long(),
        }
