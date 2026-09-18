#!/usr/bin/env python3
"""Apply the adapted APT-Det classifier to an abstract CSV.

The script writes a separate prediction file and never mutates its input. The
CheckGPT label convention is retained internally: class 0 is detector-positive
and class 1 is detector-negative.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, required=True,
        help="CSV containing a unique id column and an abstract/text column.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "models/APT-Det_full_head.pth",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "predictions/apt_det_predictions.csv",
    )
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--text-column", default="abstract")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


args = parse_args()
if args.device.startswith("cuda:"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":", 1)[1])

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CheckGPT  # noqa: E402


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def load_input() -> pd.DataFrame:
    frame = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    if args.id_column not in frame or args.text_column not in frame:
        raise KeyError(
            f"Input must contain {args.id_column!r} and {args.text_column!r}; "
            f"available columns={list(frame.columns)}"
        )
    frame = frame[[args.id_column, args.text_column]].copy()
    frame.columns = ["id", "abstract"]
    frame["id"] = frame["id"].astype("string").str.strip()
    frame["abstract"] = frame["abstract"].map(clean_text)
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Input IDs must be non-missing and unique")
    if frame["abstract"].eq("").any():
        raise ValueError(f"Blank abstracts: {int(frame['abstract'].eq('').sum())}")
    return frame


def load_completed() -> pd.DataFrame:
    if args.resume and args.output.exists():
        completed = pd.read_csv(args.output, dtype={"id": "string"})
        if completed["id"].duplicated().any():
            raise ValueError("Existing output contains duplicate IDs")
        return completed
    return pd.DataFrame(columns=[
        "id", "detector_positive", "detector_positive_probability",
        "detector_negative_probability", "threshold",
    ])


def main() -> None:
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device("cuda" if args.device.startswith("cuda") else "cpu")
    data = load_input()
    completed = load_completed()
    pending = data.loc[~data["id"].isin(set(completed["id"]))].reset_index(drop=True)
    print(json.dumps({
        "input_rows": len(data), "completed_rows": len(completed),
        "pending_rows": len(pending), "threshold": args.threshold,
    }, indent=2))

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    encoder = RobertaModel.from_pretrained("roberta-large").to(device).eval()
    classifier = CheckGPT(
        input_size=1024, hidden_size=256, batch_first=True, dropout=0.5,
        bidirectional=True, num_layers=2,
        device="cuda" if device.type == "cuda" else "cpu", v1=1,
    ).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state, strict=True)

    rows = []
    for start in tqdm(range(0, len(pending), args.batch_size), desc="APT-Det"):
        batch = pending.iloc[start:start + args.batch_size]
        encoded = tokenizer(
            batch["abstract"].tolist(), padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1).cpu()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            features = encoder(**encoded).last_hidden_state
            logits = classifier(features, lengths=lengths)
            probabilities = F.softmax(logits.float(), dim=1).cpu().numpy()
        for paper_id, (positive_probability, negative_probability) in zip(
            batch["id"], probabilities
        ):
            rows.append({
                "id": paper_id,
                "detector_positive": int(positive_probability >= args.threshold),
                "detector_positive_probability": float(positive_probability),
                "detector_negative_probability": float(negative_probability),
                "threshold": args.threshold,
            })

    result = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
    result = data[["id"]].merge(result, on="id", how="left", validate="one_to_one")
    if result.isna().any().any():
        raise RuntimeError("Predictions are incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
