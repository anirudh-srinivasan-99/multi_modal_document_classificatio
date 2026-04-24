from enum import StrEnum


class ModelName(StrEnum):
    MOBILE_NET_V2: str = 'mobilenetv3_large_100'
    EFFICIENT_NET_V2: str = 'efficientnetv2_rw_m'
    VIT_BASE: str = 'vit_base_patch16_224'
    DINO_V2_BASE: str = 'vit_base_patch14_dinov2'

    DISTILBERT_BASE: str = 'distilbert-base-uncased'
    ROBERTA_BASE: str = 'FacebookAI/roberta-base'
    MODERNBERT_BASE: str = 'answerdotai/ModernBERT-base'
