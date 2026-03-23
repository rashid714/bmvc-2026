"""
Advanced Top-Tier Multimodal BEAR Model (BMVC 2026)
Dual-Layer LLM Architecture with Maximum Model Capacity
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


class AdvancedModalityEncoder(nn.Module):
    """
    Enhanced modality encoder with deeper architecture.
    Input: feature vector → Output: normalized 1024-dim embedding
    """
    def __init__(self, input_dim, hidden_dim=1024):
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
    
    def forward(self, x):
        return self.norm(self.net(x))


class DualLayerAttention(nn.Module):
    """
    Dual-layer cross-modal attention mechanism.
    Layer 1: Low-level feature fusion
    Layer 2: High-level semantic fusion
    """
    def __init__(self, hidden_dim=1024, num_heads=16):
        super().__init__()
        
        # Layer 1: Low-level fusion
        self.layer1_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer1_norm = nn.LayerNorm(hidden_dim)
        self.layer1_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Layer 2: High-level fusion
        self.layer2_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
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
    
    def forward(self, embeddings_list, reliability_scores):
        """
        Args:
            embeddings_list: List of [batch, hidden_dim] embeddings
            reliability_scores: [batch, num_modalities] confidence scores
        """
        batch_size = embeddings_list[0].size(0)
        
        # Stack embeddings: [batch, num_modalities, hidden_dim]
        stacked = torch.stack(embeddings_list, dim=1)
        
        # Layer 1: Low-level cross-modal fusion
        attn_out1, _ = self.layer1_attention(stacked, stacked, stacked)
        attn_out1 = self.layer1_norm(attn_out1 + stacked)
        ffn_out1 = self.layer1_ffn(attn_out1)
        layer1_out = self.layer1_norm(ffn_out1 + attn_out1)
        
        # Weight by reliability
        weighted1 = layer1_out * reliability_scores.unsqueeze(-1)
        layer1_fused = torch.mean(weighted1, dim=1)  # [batch, hidden_dim]
        
        # Layer 2: High-level semantic fusion
        layer1_repeated = layer1_fused.unsqueeze(1).expand_as(stacked)
        combined = torch.cat([stacked, layer1_repeated], dim=-1)  # [batch, num_mod, 2*hidden_dim]
        
        attn_out2, _ = self.layer2_attention(stacked, stacked, stacked)
        attn_out2 = self.layer2_norm(attn_out2 + stacked)
        ffn_out2 = self.layer2_ffn(attn_out2)
        layer2_out = self.layer2_norm(ffn_out2 + attn_out2)
        
        # Weight by reliability (squared for emphasis)
        weighted2 = layer2_out * (reliability_scores ** 1.5).unsqueeze(-1)
        layer2_fused = torch.mean(weighted2, dim=1)  # [batch, hidden_dim]
        
        # Combine both layers with gating
        combined_fused = torch.cat([layer1_fused, layer2_fused], dim=-1)
        gate = self.gating(combined_fused)
        final_fused = gate * layer1_fused + (1 - gate) * layer2_fused
        
        return final_fused


class AdvancedReliabilityModule(nn.Module):
    """
    Advanced modality reliability scoring with uncertainty estimation.
    """
    def __init__(self, hidden_dim=1024, num_modalities=4):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Sigmoid()
        )
    
    def forward(self, text_embed):
        """
        Args:
            text_embed: [batch, hidden_dim]
        Returns:
            reliability_scores: [batch, num_modalities]
            uncertainty: [batch, num_modalities]
        """
        confidence = self.confidence_net(text_embed)
        uncertainty = self.uncertainty_net(text_embed)
        reliability = confidence * (1 - uncertainty)
        return reliability, uncertainty


class AdvancedLLMBackbone(nn.Module):
    """
    Dual-layer LLM backbone combining two powerful language models.
    Layer 1: RoBERTa-large (high-capacity semantic understanding)
    Layer 2: DistilRoBERTa-base (lighter complementary encoder)
    """
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Layer 1: RoBERTa-large (355M params)
        self.roberta = AutoModel.from_pretrained("roberta-large")
        self.roberta_proj = nn.Linear(1024, hidden_dim)
        
        # Layer 2: DistilRoBERTa-base (82M params)
        # Keeps tokenizer compatibility with RoBERTa family token IDs.
        self.distilroberta = AutoModel.from_pretrained("distilroberta-base")
        self.distilroberta_proj = nn.Linear(768, hidden_dim)
        
        # Fusion layer
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            combined_embed: [batch, hidden_dim]
        """
        # Layer 1: RoBERTa
        roberta_out = self.roberta(input_ids, attention_mask=attention_mask)
        roberta_cls = roberta_out.last_hidden_state[:, 0, :]  # [batch, 1024]
        roberta_embed = self.roberta_proj(roberta_cls)  # [batch, hidden_dim]
        
        # Layer 2: DistilRoBERTa
        distilroberta_out = self.distilroberta(input_ids, attention_mask=attention_mask)
        distilroberta_cls = distilroberta_out.last_hidden_state[:, 0, :]  # [batch, 768]
        distilroberta_embed = self.distilroberta_proj(distilroberta_cls)  # [batch, hidden_dim]
        
        # Fusion with gating
        combined = torch.cat([roberta_embed, distilroberta_embed], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * roberta_embed + (1 - gate) * distilroberta_embed
        
        return self.layer_norm(fused)


class AdvancedTriTaskHead(nn.Module):
    """
    Advanced tri-task prediction heads with hierarchical structure.
    """
    def __init__(self, hidden_dim=1024, num_emotion=11, num_intention=20, num_action=15):
        super().__init__()
        
        # Shared encoder
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
        
        # Task-specific heads
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotion)
        )
        
        self.intention_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_intention)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_action)
        )
    
    def forward(self, fused_embed):
        """
        Args:
            fused_embed: [batch, hidden_dim]
        Returns:
            emotion_logits: [batch, 11]
            intention_logits: [batch, 20]
            action_logits: [batch, 15]
        """
        shared = self.shared_encoder(fused_embed)
        
        emotion_logits = self.emotion_head(shared)
        intention_logits = self.intention_head(shared)
        action_logits = self.action_head(shared)
        
        return emotion_logits, intention_logits, action_logits


class AdvancedBEARModel(nn.Module):
    """
    Advanced Top-Tier Multimodal BEAR Model
    Combines dual-layer LLM + 4 modality encoders + dual-layer attention
    """
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Text: Dual-layer LLM backbone
        self.text_backbone = AdvancedLLMBackbone(hidden_dim)
        self.text_encoder = AdvancedModalityEncoder(hidden_dim, hidden_dim)
        
        # Vision: Image encoder
        self.image_encoder = AdvancedModalityEncoder(2048, hidden_dim)
        
        # Audio: Audio encoder
        self.audio_encoder = AdvancedModalityEncoder(512, hidden_dim)
        
        # Video: Video encoder
        self.video_encoder = AdvancedModalityEncoder(1024, hidden_dim)
        
        # Reliability module
        self.reliability_module = AdvancedReliabilityModule(hidden_dim, num_modalities=4)
        
        # Dual-layer attention fusion
        self.fusion = DualLayerAttention(hidden_dim, num_heads=16)
        
        # Task-specific heads
        self.task_heads = AdvancedTriTaskHead(hidden_dim, num_emotion=11, num_intention=20, num_action=15)
        
        # Calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, input_ids, attention_mask, image_features=None, audio_features=None, video_features=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            image_features: [batch, 2048] or None
            audio_features: [batch, 512] or None
            video_features: [batch, 1024] or None
        
        Returns:
            Dict with emotion/intention/action logits
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Text encoding via dual-layer LLM
        text_embed = self.text_backbone(input_ids, attention_mask)
        text_embed = self.text_encoder(text_embed)
        
        # Get reliability scores
        reliability_scores, uncertainties = self.reliability_module(text_embed)
        
        # Initialize modality embeddings
        modalities = [text_embed]
        
        # Image
        if image_features is not None:
            image_embed = self.image_encoder(image_features)
            modalities.append(image_embed)
        else:
            modalities.append(torch.zeros(batch_size, self.hidden_dim, device=device))
        
        # Audio
        if audio_features is not None:
            audio_embed = self.audio_encoder(audio_features)
            modalities.append(audio_embed)
        else:
            modalities.append(torch.zeros(batch_size, self.hidden_dim, device=device))
        
        # Video
        if video_features is not None:
            video_embed = self.video_encoder(video_features)
            modalities.append(video_embed)
        else:
            modalities.append(torch.zeros(batch_size, self.hidden_dim, device=device))
        
        # Dual-layer attention fusion
        fused_embed = self.fusion(modalities, reliability_scores)
        
        # Task-specific predictions
        emotion_logits, intention_logits, action_logits = self.task_heads(fused_embed)
        
        # Temperature scaling for calibration
        emotion_logits = emotion_logits / self.temperature
        intention_logits = intention_logits / self.temperature
        action_logits = action_logits / self.temperature
        
        return {
            "emotion_logits": emotion_logits,
            "intention_logits": intention_logits,
            "action_logits": action_logits,
            "reliability_scores": reliability_scores,
            "uncertainties": uncertainties,
            "fused_embed": fused_embed,
        }
