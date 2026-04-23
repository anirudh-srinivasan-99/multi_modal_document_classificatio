import os
from typing import NamedTuple

from dotenv import load_dotenv

load_dotenv()


class HFConfig(NamedTuple):
    HF_TOKEN: str
    HF_REPO_ID: str

HF_CONFIG = HFConfig(
    HF_TOKEN=os.getenv('HF_TOKEN'),
    HF_REPO_ID=os.getenv('HF_REPO_ID')
)
