import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class HFConfig:
    HF_TOKEN: str = os.getenv('HF_TOKEN')
    HF_REPO_ID: str = os.getenv('HF_REPO_ID')
    HF_FORCE_DOWNLOAD: bool = os.getenv('HF_FORCE_DOWNLOAD', '').lower() == 'true'
    DL_NUM_WORKERS: int = min(int(os.getenv('DL_NUM_WORKERS')), os.cpu_count())


class DefaultPaths:
    BASE_PATH: Path = Path(os.getenv('BASE_PATH'))
    HF_CACHE_DIR: Path = BASE_PATH / 'data' / '.cache' / 'huggingface'
    MODEL_CHECKPOINT_DIR: Path = BASE_PATH / 'models' / 'checkpoints' / 'multimodal_model'
    MLFLOW_DIR: Path = BASE_PATH / 'mlruns'


os.environ['HF_HOME'] = str(DefaultPaths.HF_CACHE_DIR.resolve())
