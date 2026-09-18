#!/usr/bin/env python3
"""Extract resumable RoBERTa-large token features into a NumPy memmap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


ADAPTATION_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ADAPTATION_ROOT / "data/benchmark_manifest.csv"
FEATURE_DIR = ADAPTATION_ROOT / "features"
FEATURES = FEATURE_DIR / "roberta_large_features_fp16.npy"
LENGTHS = FEATURE_DIR / "token_lengths.npy"
PROGRESS = FEATURE_DIR / "extraction_progress.json"
METADATA = FEATURE_DIR / "feature_metadata.json"
MODEL_NAME = "roberta-large"
MAX_LENGTH = 512
HIDDEN_SIZE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    n = len(frame)
    if args.force:
        for path in (FEATURES, LENGTHS, PROGRESS, METADATA):
            path.unlink(missing_ok=True)

    if FEATURES.exists():
        features = np.lib.format.open_memmap(FEATURES, mode="r+")
        if features.shape != (n, MAX_LENGTH, HIDDEN_SIZE):
            raise RuntimeError(f"Unexpected existing feature shape: {features.shape}")
    else:
        features = np.lib.format.open_memmap(
            FEATURES,
            mode="w+",
            dtype=np.float16,
            shape=(n, MAX_LENGTH, HIDDEN_SIZE),
        )
    if LENGTHS.exists():
        lengths = np.lib.format.open_memmap(LENGTHS, mode="r+")
    else:
        lengths = np.lib.format.open_memmap(
            LENGTHS, mode="w+", dtype=np.uint16, shape=(n,)
        )

    start = 0
    if PROGRESS.exists():
        start = int(json.loads(PROGRESS.read_text())["next_index"])
    if start >= n and METADATA.exists():
        print(f"Feature extraction already complete: {n:,} samples")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    abstracts = frame["abstract"].astype(str).tolist()
    for begin in tqdm(range(start, n, args.batch_size), initial=start // args.batch_size):
        end = min(begin + args.batch_size, n)
        encoded = tokenizer(
            abstracts[begin:end],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch_lengths = encoded["attention_mask"].sum(dim=1).cpu().numpy()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            hidden = model(**encoded).last_hidden_state
        hidden_np = hidden.float().cpu().numpy().astype(np.float16)
        features[begin:end] = 0
        features[begin:end, : hidden_np.shape[1], :] = hidden_np
        lengths[begin:end] = batch_lengths.astype(np.uint16)
        features.flush()
        lengths.flush()
        PROGRESS.write_text(
            json.dumps({"next_index": end, "total": n}, indent=2), encoding="utf-8"
        )

    metadata = {
        "model_name": MODEL_NAME,
        "samples": n,
        "shape": [n, MAX_LENGTH, HIDDEN_SIZE],
        "dtype": "float16",
        "max_length": MAX_LENGTH,
        "minimum_token_length": int(lengths.min()),
        "maximum_token_length": int(lengths.max()),
        "mean_token_length": float(lengths.mean()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
