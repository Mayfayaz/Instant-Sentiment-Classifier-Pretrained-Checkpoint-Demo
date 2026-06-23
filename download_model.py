"""
download_model.py

Downloads pretrained weights from the Hugging Face Hub and caches them
locally so that predict.py can run instantly (and offline) afterwards.

Usage:
    python download_model.py
    python download_model.py --model-id <hf-repo-id> --local-dir models/my-model
"""

import argparse
import os

from huggingface_hub import snapshot_download

# Default pretrained checkpoint used by this project.
# Swap this for any model on the Hugging Face Hub to reuse this template.
DEFAULT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_LOCAL_DIR = os.path.join("models", "sentiment-distilbert")


def download(model_id: str = DEFAULT_MODEL_ID, local_dir: str = DEFAULT_LOCAL_DIR) -> str:
    """Download (or reuse the cache of) a model repo from the HF Hub.

    Args:
        model_id: Hugging Face Hub repo id, e.g. "distilbert-base-uncased-finetuned-sst-2-english".
        local_dir: Local directory to store the checkpoint in.

    Returns:
        The local path where the checkpoint is stored.
    """
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading '{model_id}' from the Hugging Face Hub...")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Done. Checkpoint cached at: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a pretrained model checkpoint from the Hugging Face Hub."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face Hub repo id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Local directory to cache the weights in (default: {DEFAULT_LOCAL_DIR})",
    )
    args = parser.parse_args()
    download(args.model_id, args.local_dir)
