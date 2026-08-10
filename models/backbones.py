"""
Geospatial Foundation Model Loaders and Feature Extraction Pipelines.

Supports:
- GeoRSCLIP (ViT-B/32 pre-trained on RS5M)
- Standard CLIP (ViT-B/32, ViT-L/14)
- Satellite DINOv3 (SAT-493M ViT-L/16 patch & CLS tokens)
- MultiModalHydroEncoder (Dual-stream Optical + SAR fusion encoder)
"""
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Hydrological and Earth Observation Prompt Templates
WATER_PROMPT_TEMPLATES = [
    "a centered satellite photo of a {}.",
    "a remote sensing image of a {}.",
    "a multi-spectral Earth observation image showing a {}.",
    "a satellite photo of a water body categorized as {}.",
    "an aerial remote sensing image of a hydrological feature: {}.",
]

VLM_BACKBONE_REGISTRY = {
    'clip_b32': 'ViT-B/32',
    'clip_l14': 'ViT-L/14',
    'georsclip_b32': 'GeoRSCLIP-ViT-B/32',
}

_GEO_SNAP = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Zilun--GeoRSCLIP/snapshots/'
    '4920188e6eba4e711ef9848cfd7cb77e874ee33f/ckpt/'
)

GEORSCLIP_CHECKPOINTS = {
    'georsclip_b32': _GEO_SNAP + 'RS5M_ViT-B-32.pt',
}


def load_vlm_backbone(
    backbone_name: str = 'georsclip_b32',
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> Tuple[nn.Module, object, callable]:
    """Loads vision-language foundation backbone, transform, and tokenizer.
    
    Falls back gracefully to standard CLIP ViT-B/32 if GeoRSCLIP checkpoint is not cached locally.
    """
    import clip
    if backbone_name in GEORSCLIP_CHECKPOINTS and os.path.exists(GEORSCLIP_CHECKPOINTS[backbone_name]):
        try:
            import open_clip
            arch = 'ViT-B-32'
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='openai')
            state = torch.load(GEORSCLIP_CHECKPOINTS[backbone_name], map_location='cpu')
            model.load_state_dict(state, strict=False)
            model.to(device).eval()
            tokenizer = open_clip.get_tokenizer(arch)
            tokenize_fn = lambda texts: tokenizer(texts).to(device)
            return model, preprocess, tokenize_fn
        except Exception as e:
            print(f"Warning: GeoRSCLIP load failed ({e}). Falling back to standard CLIP ViT-B/32.")

    # Standard CLIP fallback
    clip_arch = VLM_BACKBONE_REGISTRY.get(backbone_name, 'ViT-B/32')
    if 'georsclip' in backbone_name:
        clip_arch = 'ViT-B/32'
    model, preprocess = clip.load(clip_arch, download_root='./caches/clip', device=device)
    model.eval()
    tokenize_fn = lambda texts: clip.tokenize(texts).to(device)
    return model, preprocess, tokenize_fn


def build_text_classifier_weights(
    class_names: List[str],
    templates: List[str],
    model: nn.Module,
    tokenize_fn: callable,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> Tensor:
    """Builds L2-normalized zero-shot text classification prototype matrix W_text in R^{d x K}."""
    with torch.no_grad():
        zeroshot_weights = []
        for classname in class_names:
            texts = [template.format(classname.replace('_', ' ').lower()) for template in templates]
            text_tokens = tokenize_fn(texts)
            if hasattr(model, 'encode_text'):
                text_features = model.encode_text(text_tokens)
            else:
                text_features = model(text_tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
            mean_text_feature = text_features.mean(dim=0)
            mean_text_feature = F.normalize(mean_text_feature, dim=-1)
            zeroshot_weights.append(mean_text_feature)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device) # [d, K]
    return zeroshot_weights


class MultiModalHydroEncoder(nn.Module):
    """Multi-Modal Dual-Stream Foundation Encoder for Optical + SAR Water Remote Sensing.
    
    Stream 1: Optical Semantic Encoder (GeoRSCLIP / CLIP visual encoder)
    Stream 2: Structural / Microwave Radar Encoder (DINOv3 / Sentinel-1 SAR backscatter)
    """

    def __init__(self, opt_dim: int = 512, sar_dim: int = 768, proj_dim: int = 512):
        super().__init__()
        self.opt_dim = opt_dim
        self.sar_dim = sar_dim
        self.proj_dim = proj_dim

        self.opt_proj = nn.Sequential(
            nn.Linear(opt_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.sar_proj = nn.Sequential(
            nn.Linear(sar_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.gating_network = nn.Sequential(
            nn.Linear(proj_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        opt_features: Tensor,
        sar_features: Tensor,
        cloud_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Fuses optical and SAR embeddings with adaptive reliability gating.
        
        Args:
            opt_features: [N, opt_dim] optical feature vectors.
            sar_features: [N, sar_dim] SAR / structural feature vectors.
            cloud_mask: Optional [N, 1] cloud attenuation factor in [0, 1] (1 = clear, 0 = overcast).
            
        Returns:
            fused_features: [N, proj_dim] L2-normalized multi-source embedding.
            fusion_weights: [N, 2] modality weights [w_opt, w_sar].
        """
        h_opt = self.opt_proj(opt_features)
        h_sar = self.sar_proj(sar_features)

        concat = torch.cat([h_opt, h_sar], dim=-1)
        weights = self.gating_network(concat) # [N, 2]

        if cloud_mask is not None:
            # Re-weight optical modality by cloud transmission factor
            w_opt = weights[:, 0:1] * cloud_mask
            w_sar = weights[:, 1:2] + weights[:, 0:1] * (1.0 - cloud_mask)
            total_w = w_opt + w_sar + 1e-8
            weights = torch.cat([w_opt / total_w, w_sar / total_w], dim=-1)

        fused = weights[:, 0:1] * h_opt + weights[:, 1:2] * h_sar
        fused = F.normalize(fused, dim=-1)
        return fused, weights
