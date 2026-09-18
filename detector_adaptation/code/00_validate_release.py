#!/usr/bin/env python3
"""Validate the distributed APT-Det benchmark, split and checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/benchmark_manifest.csv":
        "c809315833e22219efdf6f149a2f4ee352fc42cacb537feed0360e221ef746de",
    "models/CS_Task3.pth":
        "a4ea7fe39ebe99c8d5bf6860937390e012fd5935fc29df2894f2e49e07253653",
    "models/APT-Det_full_head.pth":
        "894e2e561a8fe2426aeafb66f451dd6adb0f108781172e112473fc83fd29d460",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {relative}: {actual}")

    manifest = pd.read_csv(
        ROOT / "data/benchmark_manifest.csv", encoding="utf-8-sig", low_memory=False
    )
    required = {
        "row_index", "sample_id", "paper_key", "abstract", "label",
        "label_name", "generator", "split",
    }
    if not required.issubset(manifest):
        raise RuntimeError(f"Missing columns: {sorted(required - set(manifest))}")
    if len(manifest) != 42_170 or manifest["sample_id"].nunique() != 42_170:
        raise RuntimeError("Unexpected benchmark row or sample count")
    grouped = manifest.groupby("paper_key", sort=False)
    if manifest["paper_key"].nunique() != 8_434:
        raise RuntimeError("Unexpected source-paper count")
    if not grouped.size().eq(5).all() or not grouped["label"].sum().eq(1).all():
        raise RuntimeError("Each paper must contain one human and four polished texts")
    paper_splits = grouped["split"].nunique()
    if not paper_splits.eq(1).all():
        raise RuntimeError("Source-paper versions leak across splits")

    expected_papers = {"train": 5_903, "validation": 1_265, "test": 1_266}
    actual_papers = (
        manifest[["paper_key", "split"]].drop_duplicates()["split"]
        .value_counts().to_dict()
    )
    if actual_papers != expected_papers:
        raise RuntimeError(f"Unexpected split counts: {actual_papers}")

    report = {
        "samples": len(manifest),
        "source_papers": manifest["paper_key"].nunique(),
        "paper_counts": actual_papers,
        "paper_level_leakage": False,
        "checksums_verified": list(EXPECTED),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
