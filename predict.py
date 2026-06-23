"""
predict.py

CLI entry point for instant inference using the pretrained checkpoint.
Downloads the checkpoint automatically on first run if it isn't cached yet.

Usage:
    python predict.py --text "This is great!"
    python predict.py --file examples/sample_input.txt
"""

import argparse
import os
import sys

from download_model import DEFAULT_MODEL_ID, download
from src.model_utils import MODEL_LOCAL_DIR, load_pipeline


def ensure_model_available() -> None:
    """Download the checkpoint if it isn't cached locally yet."""
    if not os.path.isdir(MODEL_LOCAL_DIR) or not os.listdir(MODEL_LOCAL_DIR):
        print("Checkpoint not found locally — downloading now (one-time setup)...")
        download(DEFAULT_MODEL_ID, MODEL_LOCAL_DIR)


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run instant inference with the pretrained sentiment model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A single text string to classify.")
    group.add_argument(
        "--file", type=str, help="Path to a local .txt file (one example per line)."
    )
    args = parser.parse_args()

    ensure_model_available()
    classifier = load_pipeline()

    if args.text:
        inputs = [args.text]
    else:
        if not os.path.isfile(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        inputs = read_lines(args.file)
        if not inputs:
            print(f"No non-empty lines found in: {args.file}")
            sys.exit(1)

    results = classifier(inputs)

    for text, result in zip(inputs, results):
        print(f"\nInput:  {text}")
        print(f"Result: {result['label']} (confidence: {result['score']:.4f})")


if __name__ == "__main__":
    main()
