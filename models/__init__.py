"""
Model Backbones and Vision-Language Foundation Model Loaders for Remote Sensing & Water Resources.
"""
from models.backbones import (
    load_vlm_backbone,
    build_text_classifier_weights,
    MultiModalHydroEncoder,
    VLM_BACKBONE_REGISTRY,
    WATER_PROMPT_TEMPLATES,
)

__all__ = [
    'load_vlm_backbone',
    'build_text_classifier_weights',
    'MultiModalHydroEncoder',
    'VLM_BACKBONE_REGISTRY',
    'WATER_PROMPT_TEMPLATES',
]
