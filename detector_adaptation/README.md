# APT-Det adaptation and inference

This directory reproduces the target-domain adaptation of CheckGPT used in
the study. APT-Det classifies an abstract as **detector-positive** when it
contains textual traces consistent with LLM-assisted polishing. The output is
not evidence that an identifiable author used an LLM.

## Benchmark

The benchmark contains 8,434 source papers from 2019. Each paper contributes
one original human-written abstract and four polished versions generated with
GPT-4o, Gemini-2.5-Flash, GLM-4-Flash and GLM-5.1, yielding 42,170 texts. The
split is performed at source-paper level (seed 42): 5,903 papers for training,
1,265 for validation and 1,266 for testing. Consequently, versions of the same
source paper never occur in different splits.

`data/benchmark_manifest.csv` is the analysis-ready long-form benchmark. The
label convention inherited from CheckGPT is `0 = LLM-polished` and
`1 = human-written`. The inference script exposes the less ambiguous Boolean
field `detector_positive`.

The original source CSVs are not duplicated in this repository. To rebuild the
manifest, place the five files listed by `code/01_build_benchmark.py` in
`data/raw/`, then run scripts 01 and 02.

## Reproduction order

Run commands from this directory:

```bash
python code/00_validate_release.py
python code/03_extract_features.py --device cuda --batch-size 16
python code/04_run_adaptation.py --method all --device cuda --batch-size 32
python code/05_aggregate_results.py
```

Feature extraction creates a resumable float16 memmap of approximately 43 GB
under `features/`; this generated cache is intentionally not distributed. The
three evaluated configurations are direct application of CheckGPT, adaptation
of the final classification layer using 150 texts, and adaptation of the full
CheckGPT classification head using the complete training set. Validation
Macro-F1 determines early stopping and checkpoint selection. The held-out test
set is used only for final evaluation. Reported primary classifications use the
default 0.5 threshold.

## Applying APT-Det

```bash
python code/06_apply_apt_det.py \
  --input /path/to/abstracts.csv \
  --text-column abstract \
  --device cuda:0 \
  --threshold 0.5
```

Predictions are written to `predictions/apt_det_predictions.csv`; the input is
never modified. Use `--resume` to continue an interrupted inference run.

## Directory contents

- `code/`: portable benchmark, training, evaluation and inference programs.
- `data/`: benchmark manifest and paper-level split audit.
- `models/`: base CheckGPT head and the selected adapted APT-Det head.
- `results/`: compact evaluation tables used to report detector performance.
- `features/`: generated RoBERTa-large feature cache (not distributed).
- `predictions/`: generated corpus predictions (not distributed here).

See `models/README.md` for checkpoint provenance and SHA-256 checksums.
