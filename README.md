# Instant Sentiment Classifier — Pretrained Checkpoint Demo

A minimal, reproducible template that shows the right way to ship a model
project on GitHub: **don't make people train from scratch to try it out.**

This repo downloads a pretrained checkpoint from the Hugging Face Hub and
gives you instant inference via a single `predict.py` script — no GPU,
no training run, no waiting.

> Swap the model ID in `download_model.py` and you have a reusable template
> for shipping *any* pretrained Hugging Face model with the same UX.

---

## Why this matters

Most "research code" repos on GitHub force users to:
1. Clone the repo
2. Find a dataset
3. Train for hours (or days) on a GPU
4. *Then* finally see if the model works

This repo skips straight to step 4. The pretrained weights are pulled
automatically from the **[Hugging Face Hub](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)**
the first time you run anything, then cached locally for instant reuse.

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/<your-username>/pretrained-model-demo.git
cd pretrained-model-demo
pip install -r requirements.txt

# 2. (Optional) Pre-download the checkpoint explicitly
python download_model.py

# 3. Run inference — instantly, no training required
python predict.py --text "This project saved me so much time, thank you!"
```

Expected output:
```
Input:  This project saved me so much time, thank you!
Result: POSITIVE (confidence: 0.9998)
```

If you skip step 2, `predict.py` will detect the missing checkpoint and
download it automatically the first time it's needed.

---

## Usage

### Classify a single string
```bash
python predict.py --text "I really did not enjoy this at all."
```

### Classify every line in a local file
```bash
python predict.py --file examples/sample_input.txt
```

Each line in the file is treated as one example and classified independently.

---

## Project structure

```
pretrained-model-demo/
├── README.md
├── requirements.txt
├── download_model.py      # Downloads + caches the pretrained checkpoint
├── predict.py              # CLI entry point: instant inference
├── src/
│   ├── __init__.py
│   └── model_utils.py      # Model loading / pipeline wrapper
├── examples/
│   └── sample_input.txt    # Sample inputs for --file mode
├── tests/
│   └── test_predict.py     # Smoke tests
├── .gitignore
└── LICENSE
```

---

## Model details

| | |
|---|---|
| **Model** | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) |
| **Task** | Binary sentiment classification (POSITIVE / NEGATIVE) |
| **Source** | Hugging Face Hub |
| **Size** | ~268 MB |
| **Hardware to run inference** | CPU is sufficient |

The checkpoint is downloaded once into `models/sentiment-distilbert/` (git-ignored)
and reused on every subsequent run — no repeated downloads, no retraining.

### Using your own checkpoint instead

To swap in your own fine-tuned model:
1. Push your model to a Hugging Face Hub repo (or keep it local).
2. Update `DEFAULT_MODEL_ID` in `download_model.py` (or point `--local-dir`
   at your existing local weights).
3. Adjust the pipeline task name in `src/model_utils.py` if it isn't
   `sentiment-analysis` (e.g. `text-classification`, `image-classification`, etc.).

---

## Requirements

* Python 3.9+
* See `requirements.txt` for pinned package versions

---

## License

MIT — see [LICENSE](LICENSE).
