from pathlib import Path

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from src.config.constants import DefaultTrainerArgs as dargs
from src.config.env_loader import DefaultPaths as DP, HFConfig
from src.data_processor.multi_modal_dataloader import MultiModalDataLoader
from src.model_builder.multi_modal_classifier import MultiModalClassifier
from src.utils.custom_mlflow_logger import CustomMLFlowLogger


RANDOM_SEED = 0
BATCH_SIZE = dargs.BATCH_SIZE
EPOCHS = dargs.EPOCHS
LR = dargs.LR
USE_GPU = dargs.USE_GPU
RUN_VERSION = 2
LOAD_MODEL = False
PATH_MODEL_DIR = DP.MODEL_CHECKPOINT_DIR / Path(f'run_{RUN_VERSION}')
# PATH_MODEL = Path(f'models/checkpoints/multimodal/run_{RUN_VERSION - 1}/last.ckpt')

MLFLOW_EXP = 'MultiModal Document Classification'
MLFLOW_RUN_NAME = f'run_{RUN_VERSION}'


VISION_MODEL_NAME = dargs.VISION_MODEL_NAME
VISION_PROJECTION_DIMENSION = dargs.VISION_PROJECTION_DIMENSION
VISION_BACKBONE_TRAINABLE = dargs.VISION_BACKBONE_TRAINABLE

LANGUAGE_MODEL_NAME = dargs.LANGUAGE_MODEL_NAME
LANGUAGE_PROJECTION_DIMENSION = dargs.LANGUAGE_PROJECTION_DIMENSION
LANGUAGE_BACKBONE_TRAINABLE = dargs.LANGUAGE_BACKBONE_TRAINABLE
MAX_SEQ_LENGTH = dargs.MAX_SEQ_LENGTH

NUM_CLASSES = dargs.NUM_CLASSES

DATASET_MEAN = dargs.DATASET_MEAN
DATASET_STD = dargs.DATASET_STD


def main() -> None:
    PATH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)

    mlflow_logger = CustomMLFlowLogger(
        experiment_name=MLFLOW_EXP,
        tracking_uri=f'file:{DP.MLFLOW_DIR.as_posix()}',
        synchronous=False,
        run_name=MLFLOW_RUN_NAME,
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_MODEL_DIR,
        save_last=True,
        save_on_exception=True,
        save_top_k=2,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=True,
        monitor='val_loss',
        mode='min'
    )
    multi_modal_classifier = MultiModalClassifier(
        vision_model_name=VISION_MODEL_NAME,
        vision_projection_dimension=VISION_PROJECTION_DIMENSION,
        vision_backbone_trainable=VISION_BACKBONE_TRAINABLE,
        language_model_name=LANGUAGE_MODEL_NAME,
        language_projection_dimension=LANGUAGE_PROJECTION_DIMENSION,
        language_backbone_trainable=LANGUAGE_BACKBONE_TRAINABLE,
        max_seq_len=min(MAX_SEQ_LENGTH, tokenizer.model_max_length),
        num_classes=NUM_CLASSES,
        learning_rate=LR
    )
    data_module = MultiModalDataLoader(
        hf_repo_id=HFConfig.HF_REPO_ID,
        batch_size=BATCH_SIZE,
        image_size=multi_modal_classifier.vision_fe.input_size,
        max_seq_length=multi_modal_classifier.language_fe.max_seq_len,
        tokenizer=AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME),
        dataset_mean=DATASET_MEAN,
        dataset_std=DATASET_STD

    )

    # trainer = L.Trainer(
    #     accelerator='gpu' if USE_GPU else 'cpu',
    #     devices=1,
    #     max_epochs=EPOCHS,
    #     callbacks=[checkpoint_callback],
    #     logger=mlflow_logger,
    # )
    trainer = L.Trainer(
        overfit_batches=1,  # Use only 1 batch to see if loss goes to ~0
        max_epochs=1,     # Usually needs more epochs to fully overfit
        accelerator="auto",
        logger=False        # Often disabled during quick sanity tests
    )
    trainer.fit(
        multi_modal_classifier, datamodule=data_module,
    )
    # trainer.test(
    #     multi_modal_classifier, datamodule=data_module,
    #     ckpt_path="best"
    # )


if __name__ == '__main__':
    seed_everything(RANDOM_SEED)
    main()
