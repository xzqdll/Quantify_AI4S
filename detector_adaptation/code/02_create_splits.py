#!/usr/bin/env python3
"""Create leakage-free paper-level splits for the CheckGPT benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ADAPTATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE = ADAPTATION_ROOT / "data/our_benchmark_unified_index.csv"
EXP = ADAPTATION_ROOT
SPLITS = ADAPTATION_ROOT / "data/splits"
SEED = 42


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig")
    required = {
        "sample_id", "paper_key", "id", "title", "abstract",
        "label", "label_name", "generator", "source_file",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("sample_id is not unique")
    if frame[list(required)].isna().any().any():
        raise RuntimeError("Required fields contain missing values")

    grouped = frame.groupby("paper_key", sort=False)
    bad = grouped.agg(
        n=("sample_id", "size"),
        human=("label", lambda x: int((x == 1).sum())),
        ai=("label", lambda x: int((x == 0).sum())),
        generators=("generator", "nunique"),
    )
    bad = bad[
        (bad["n"] != 5)
        | (bad["human"] != 1)
        | (bad["ai"] != 4)
        | (bad["generators"] != 5)
    ]
    if len(bad):
        raise RuntimeError(f"{len(bad)} malformed paper groups")

    paper_keys = frame["paper_key"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(paper_keys)
    n = len(paper_keys)
    n_train = int(np.floor(n * 0.70))
    n_validation = int(np.floor(n * 0.15))
    split_map = {
        **{key: "train" for key in paper_keys[:n_train]},
        **{
            key: "validation"
            for key in paper_keys[n_train : n_train + n_validation]
        },
        **{key: "test" for key in paper_keys[n_train + n_validation :]},
    }
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    frame["split"] = frame["paper_key"].map(split_map)
    if frame["split"].isna().any():
        raise RuntimeError("Some rows were not assigned to a split")

    manifest = ADAPTATION_ROOT / "data/benchmark_manifest.csv"
    frame.to_csv(manifest, index=False, encoding="utf-8-sig")
    for split in ("train", "validation", "test"):
        frame.loc[frame["split"].eq(split)].to_csv(
            SPLITS / f"{split}.csv", index=False, encoding="utf-8-sig"
        )

    paper_overlap = {
        f"{a}_{b}": len(
            set(frame.loc[frame["split"].eq(a), "paper_key"])
            & set(frame.loc[frame["split"].eq(b), "paper_key"])
        )
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    summary = (
        frame.groupby(["split", "label_name", "generator"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    summary.to_csv(SPLITS / "split_summary.csv", index=False, encoding="utf-8-sig")
    audit = {
        "seed": SEED,
        "source_rows": int(len(frame)),
        "unique_papers": int(frame["paper_key"].nunique()),
        "unique_samples": int(frame["sample_id"].nunique()),
        "paper_counts": {
            split: int(frame.loc[frame["split"].eq(split), "paper_key"].nunique())
            for split in ("train", "validation", "test")
        },
        "sample_counts": frame["split"].value_counts().sort_index().astype(int).to_dict(),
        "paper_overlap": paper_overlap,
        "label_convention": {"0": "AI-generated", "1": "human-written"},
    }
    (SPLITS / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
