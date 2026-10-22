"""
Model resolution
================
Weights are not stored in git. At startup the API resolves a model directory:

  1. MODEL_DIR       — a local directory (e.g. runs/mtl after training), if set
  2. MODEL_REPO_ID   — otherwise the checkpoint is downloaded from the
                       Hugging Face Hub (cached under HF_HOME), pinned to
                       MODEL_REVISION (a tag or commit sha; default "main")
"""

import logging
import os

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from model.mtl_model import MTLIntentEmotionModel

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Desmondonam/intent-emotion-mtl-distilroberta"


def resolve_model_dir() -> str:
    local_dir = os.getenv("MODEL_DIR")
    if local_dir:
        logger.info(f"Using local model directory {local_dir}")
        return local_dir

    repo_id = os.getenv("MODEL_REPO_ID", DEFAULT_REPO_ID)
    revision = os.getenv("MODEL_REVISION", "main")
    logger.info(f"Downloading model {repo_id}@{revision} from the Hugging Face Hub")
    return snapshot_download(repo_id=repo_id, revision=revision, token=os.getenv("HF_TOKEN"))


def load_model_and_tokenizer(model_dir: str):
    model = MTLIntentEmotionModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer
