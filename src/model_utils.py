"""
src/model_utils.py

Thin wrapper around a Hugging Face `pipeline` so predict.py stays simple.
Loads the locally cached checkpoint downloaded by download_model.py.
"""

import os

from transformers import pipeline

# Path to the locally cached checkpoint (populated by download_model.py)
MODEL_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "sentiment-distilbert",
)

# Pipeline task type. Change this if you swap in a different kind of model
# (e.g. "text-classification", "image-classification", "automatic-speech-recognition").
PIPELINE_TASK = "sentiment-analysis"

_pipeline_cache = None


def load_pipeline():
    """Load (and cache in-process) the inference pipeline from the local checkpoint."""
    global _pipeline_cache
    if _pipeline_cache is None:
        if not os.path.isdir(MODEL_LOCAL_DIR) or not os.listdir(MODEL_LOCAL_DIR):
            raise FileNotFoundError(
                f"No checkpoint found at '{MODEL_LOCAL_DIR}'. "
                "Run `python download_model.py` first, or call predict.py "
                "directly (it downloads automatically)."
            )
        _pipeline_cache = pipeline(
            PIPELINE_TASK,
            model=MODEL_LOCAL_DIR,
            tokenizer=MODEL_LOCAL_DIR,
        )
    return _pipeline_cache
