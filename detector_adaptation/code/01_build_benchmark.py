#!/usr/bin/env python3
"""Build a deduplicated long-form benchmark index for CheckGPT experiments."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


ADAPTATION_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ADAPTATION_ROOT / "data"
RAW_DIR = BENCHMARK_DIR / "raw"
HUMAN_CSV = RAW_DIR / "all_target_conferences_2019_with_abstract_recovered.csv"
OUTPUT_CSV = BENCHMARK_DIR / "our_benchmark_unified_index.csv"
SUMMARY_CSV = BENCHMARK_DIR / "benchmark_source_summary.csv"
DUPLICATES_CSV = BENCHMARK_DIR / "duplicate_source_records.csv"

SOURCES = [
    ("human", 1, "human-written", HUMAN_CSV),
    (
        "gemini-2.5-flash",
        0,
        "AI-generated",
        RAW_DIR / "2019_abstract_8440_gemini-2.5-flash_0.csv",
    ),
    (
        "glm-4-flash",
        0,
        "AI-generated",
        RAW_DIR / "2019_abstract_8440_glm-4-flash_0.csv",
    ),
    (
        "glm-5.1",
        0,
        "AI-generated",
        RAW_DIR / "2019_abstract_8440_glm-5.1_0.csv",
    ),
    (
        "gpt-4o",
        0,
        "AI-generated",
        RAW_DIR / "2019_abstract_8440_gpt-4o_0.csv",
    ),
]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def record_key(row: dict) -> tuple[str, str]:
    return clean_text(row.get("id", "")), normalized_title(row.get("title", ""))


def paper_key(key: tuple[str, str]) -> str:
    digest = hashlib.sha256(f"{key[0]}\t{key[1]}".encode("utf-8")).hexdigest()
    return f"paper_{digest[:20]}"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "title", "abstract"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{path} lacks required fields {sorted(required)}")
        return list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BENCHMARK_DIR / "backups" / f"before_unified_index_{timestamp}"
    existing = [path for path in (OUTPUT_CSV, SUMMARY_CSV, DUPLICATES_CSV) if path.exists()]
    if existing:
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, backup_dir / path.name)

    source_data = []
    canonical_order: list[tuple[str, str]] = []
    canonical_keys: set[tuple[str, str]] | None = None
    duplicates_report = []
    summary = []

    for generator, label, label_name, path in SOURCES:
        raw_rows = read_rows(path)
        unique: dict[tuple[str, str], dict] = {}
        duplicate_count: dict[tuple[str, str], int] = {}
        for source_row_number, row in enumerate(raw_rows, start=2):
            key = record_key(row)
            if not key[0] or not key[1] or not clean_text(row.get("abstract", "")):
                raise RuntimeError(
                    f"Blank id/title/abstract in {path.name}, source row {source_row_number}"
                )
            if key in unique:
                duplicate_count[key] = duplicate_count.get(key, 1) + 1
                duplicates_report.append(
                    {
                        "generator": generator,
                        "source_file": path.name,
                        "source_row_number": source_row_number,
                        "id": clean_text(row["id"]),
                        "title": clean_text(row["title"]),
                        "duplicate_of_first_occurrence": True,
                    }
                )
                continue
            unique[key] = {
                "id": clean_text(row["id"]),
                "title": clean_text(row["title"]),
                "abstract": clean_text(row["abstract"]),
            }
            if generator == "human":
                canonical_order.append(key)

        keys = set(unique)
        if canonical_keys is None:
            canonical_keys = keys
        elif keys != canonical_keys:
            missing = canonical_keys - keys
            extra = keys - canonical_keys
            raise RuntimeError(
                f"Key-set mismatch for {path.name}: missing={len(missing)}, extra={len(extra)}"
            )

        source_data.append((generator, label, label_name, path, unique))
        summary.append(
            {
                "generator": generator,
                "label": label,
                "label_name": label_name,
                "source_file": path.name,
                "source_rows": len(raw_rows),
                "unique_id_title": len(unique),
                "duplicates_removed": len(raw_rows) - len(unique),
                "blank_abstracts": 0,
            }
        )

    assert canonical_keys is not None
    unified = []
    for generator, label, label_name, path, unique in source_data:
        for key in canonical_order:
            row = unique[key]
            pkey = paper_key(key)
            unified.append(
                {
                    "sample_id": f"{pkey}::{generator}",
                    "paper_key": pkey,
                    "id": row["id"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "label": label,
                    "label_name": label_name,
                    "generator": generator,
                    "source_file": path.name,
                }
            )

    output_fields = [
        "sample_id",
        "paper_key",
        "id",
        "title",
        "abstract",
        "label",
        "label_name",
        "generator",
        "source_file",
    ]
    write_rows(OUTPUT_CSV, output_fields, unified)
    write_rows(SUMMARY_CSV, list(summary[0]), summary)
    duplicate_fields = [
        "generator",
        "source_file",
        "source_row_number",
        "id",
        "title",
        "duplicate_of_first_occurrence",
    ]
    write_rows(DUPLICATES_CSV, duplicate_fields, duplicates_report)

    expected = len(canonical_order) * len(SOURCES)
    if len(unified) != expected:
        raise RuntimeError(f"Unified rows={len(unified)}, expected={expected}")
    if len({row["sample_id"] for row in unified}) != len(unified):
        raise RuntimeError("sample_id is not unique")
    if any("\n" in value or "\r" in value for row in unified for value in row.values() if isinstance(value, str)):
        raise RuntimeError("Embedded newline remains in unified output")

    print(f"Unique papers: {len(canonical_order):,}")
    print(f"Versions per paper: {len(SOURCES)}")
    print(f"Unified samples: {len(unified):,}")
    print(f"Unique sample_id: {len({row['sample_id'] for row in unified}):,}")
    print(f"Duplicate source rows removed: {len(duplicates_report):,}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Duplicates: {DUPLICATES_CSV}")
    if existing:
        print(f"Previous outputs backed up to: {backup_dir}")


if __name__ == "__main__":
    main()
