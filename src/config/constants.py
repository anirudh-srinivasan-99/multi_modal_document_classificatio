from enum import StrEnum


class ModelName(StrEnum):
    MOBILE_NET_V2: str = 'mobilenetv3_large_100'
    EFFICIENT_NET_V2: str = 'efficientnetv2_rw_m'
    VIT_BASE: str = 'vit_base_patch16_224'
    DINO_V2_BASE: str = 'vit_base_patch14_dinov2'

    DISTILBERT_BASE: str = 'distilbert-base-uncased'
    ROBERTA_BASE: str = 'FacebookAI/roberta-base'
    MODERNBERT_BASE: str = 'answerdotai/ModernBERT-base'


class DefaultTrainerArgs:
    BATCH_SIZE: int = 16
    EPOCHS: int = 15
    LR: float = 0.01
    USE_GPU: bool = False
    LOAD_MODEL: bool = False
    PATH_MODEL_DIR: str = 'models/pretrained/run_{run_number}'

    MLFLOW_EXP: str = 'Multi Modal Document Classification'
    LOG_N_EPOCH: int = 3

    VISION_MODEL_NAME: str = ModelName.MOBILE_NET_V2
    VISION_PROJECTION_DIMENSION: int = 1024
    VISION_BACKBONE_TRAINABLE: bool = False

    LANGUAGE_MODEL_NAME: str = ModelName.ROBERTA_BASE
    LANGUAGE_PROJECTION_DIMENSION: int = 896
    LANGUAGE_BACKBONE_TRAINABLE: bool = False
    MAX_SEQ_LENGTH: int = 1024

    NUM_CLASSES: int = 10

    DATASET_MEAN: tuple[float, float, float] = (0.9342018365859985, 0.9342018365859985, 0.9342018365859985)
    DATASET_STD: tuple[float, float, float] = (0.2343880981206894, 0.2343880981206894, 0.2343880981206894)
