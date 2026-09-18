#!/usr/bin/env python3
"""Run direct validation and two CheckGPT adaptation strategies."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ADAPTATION_ROOT = Path(__file__).resolve().parents[1]
CHECKGPT_DIR = Path(__file__).resolve().parent
EXP = ADAPTATION_ROOT
MANIFEST = ADAPTATION_ROOT / "data/benchmark_manifest.csv"
FEATURES = EXP / "features/roberta_large_features_fp16.npy"
LENGTHS = EXP / "features/token_lengths.npy"
REPRESENTATIONS = EXP / "features/checkgpt_pretrained_representations_fp32.npy"
REP_PROGRESS = EXP / "features/representation_progress.json"
WEIGHTS = ADAPTATION_ROOT / "models/CS_Task3.pth"
METHOD1 = ADAPTATION_ROOT / "results/method1_direct_validation"
METHOD2 = ADAPTATION_ROOT / "results/method2_last_layer_150"
METHOD3 = ADAPTATION_ROOT / "results/method3_full_head"
AI_GENERATORS = ["gemini-2.5-flash", "glm-4-flash", "glm-5.1", "gpt-4o"]
DEFAULT_THRESHOLD = 0.5
CALIBRATION_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
SENSITIVITY_THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.05), 2)

sys.path.insert(0, str(CHECKGPT_DIR))
from model import CheckGPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["representations", "method1", "method2", "method3", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--preload-features",
        action="store_true",
        help="Copy the feature memmap into RAM to avoid random-I/O contention.",
    )
    parser.add_argument("--method2-runs", type=int, default=5)
    parser.add_argument("--method3-runs", type=int, default=3)
    parser.add_argument(
        "--method3-run-numbers",
        default="",
        help="Optional comma-separated run numbers, e.g. 2,3, for parallel execution.",
    )
    parser.add_argument("--method3-max-epochs", type=int, default=15)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_head(device: torch.device) -> CheckGPT:
    model = CheckGPT(
        input_size=1024,
        hidden_size=256,
        batch_first=True,
        dropout=0.5,
        bidirectional=True,
        num_layers=2,
        device="cuda" if device.type == "cuda" else "cpu",
        v1=1,
    )
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def forward_representation(model: CheckGPT, x: torch.Tensor, lengths: torch.Tensor):
    packed1 = pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    out1, _ = model.lstm1(packed1)
    out1, _ = pad_packed_sequence(out1, batch_first=True)
    first, _ = model.atten1(out1)
    packed2 = pack_padded_sequence(
        out1, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    out2, _ = model.lstm2(packed2)
    out2, _ = pad_packed_sequence(out2, batch_first=True)
    second, _ = model.atten2(out2)
    return torch.cat([first, second], dim=1)


class FeatureDataset(Dataset):
    def __init__(self, features, lengths, frame: pd.DataFrame):
        self.features = features
        self.lengths = lengths
        self.indices = frame["row_index"].to_numpy(dtype=np.int64)
        self.labels = frame["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        length = int(self.lengths[index])
        feature = np.array(self.features[index], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(feature),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[item], dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


def feature_loader(features, lengths, frame, batch_size, shuffle=False):
    return DataLoader(
        FeatureDataset(features, lengths, frame),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def metric_dict(
    y_label: np.ndarray,
    ai_probability: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    # CheckGPT convention: label 0 is AI, label 1 is human.
    y_ai = (y_label == 0).astype(int)
    predicted_ai = (ai_probability >= threshold).astype(int)
    predicted_label = np.where(predicted_ai == 1, 0, 1)
    return {
        "threshold": float(threshold),
        "n": int(len(y_label)),
        "n_ai": int(y_ai.sum()),
        "n_human": int((1 - y_ai).sum()),
        "accuracy": float(accuracy_score(y_label, predicted_label)),
        "balanced_accuracy": float(balanced_accuracy_score(y_ai, predicted_ai)),
        "precision_ai": float(precision_score(y_ai, predicted_ai, zero_division=0)),
        "recall_ai": float(recall_score(y_ai, predicted_ai, zero_division=0)),
        "f1_ai": float(f1_score(y_ai, predicted_ai, zero_division=0)),
        "macro_f1": float(f1_score(y_ai, predicted_ai, average="macro")),
        "roc_auc": float(roc_auc_score(y_ai, ai_probability)),
        "pr_auc": float(average_precision_score(y_ai, ai_probability)),
        "tn_human": int(confusion_matrix(y_ai, predicted_ai, labels=[0, 1])[0, 0]),
        "fp_human_as_ai": int(
            confusion_matrix(y_ai, predicted_ai, labels=[0, 1])[0, 1]
        ),
        "fn_ai_as_human": int(
            confusion_matrix(y_ai, predicted_ai, labels=[0, 1])[1, 0]
        ),
        "tp_ai": int(confusion_matrix(y_ai, predicted_ai, labels=[0, 1])[1, 1]),
    }


def threshold_curve(
    frame: pd.DataFrame,
    ai_probability: np.ndarray,
    thresholds: np.ndarray,
    dataset: str,
) -> pd.DataFrame:
    rows = []
    labels = frame["label"].to_numpy()
    for threshold in thresholds:
        rows.append(
            {
                "dataset": dataset,
                **metric_dict(labels, ai_probability, float(threshold)),
            }
        )
    return pd.DataFrame(rows)


def select_calibrated_threshold(
    validation: pd.DataFrame,
    validation_ai_probability: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Select a threshold exclusively on validation Macro-F1.

    Ties are resolved by validation balanced accuracy and then proximity to 0.5.
    """
    curve = threshold_curve(
        validation,
        validation_ai_probability,
        CALIBRATION_THRESHOLDS,
        "validation",
    )
    ranked = curve.assign(
        distance_from_default=(curve["threshold"] - DEFAULT_THRESHOLD).abs()
    ).sort_values(
        ["macro_f1", "balanced_accuracy", "distance_from_default", "threshold"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["threshold"]), curve


def evaluate_and_save(
    frame: pd.DataFrame,
    ai_probability: np.ndarray,
    output_dir: Path,
    method: str,
    run: str,
    validation: pd.DataFrame | None = None,
    validation_ai_probability: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = frame[
        ["row_index", "sample_id", "paper_key", "id", "title", "label",
         "label_name", "generator", "split"]
    ].copy()
    result["ai_probability"] = ai_probability
    result["human_probability"] = 1.0 - ai_probability
    result["predicted_label"] = np.where(ai_probability >= DEFAULT_THRESHOLD, 0, 1)
    result["predicted_label_name"] = np.where(
        result["predicted_label"].eq(0), "AI-generated", "human-written"
    )
    result.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metric_rows = [
        {
            "method": method,
            "run": run,
            "evaluation_scope": "overall",
            "generator": "all",
            **metric_dict(frame["label"].to_numpy(), ai_probability),
        }
    ]
    humans = frame["generator"].eq("human").to_numpy()
    for generator in AI_GENERATORS:
        mask = humans | frame["generator"].eq(generator).to_numpy()
        metric_rows.append(
            {
                "method": method,
                "run": run,
                "evaluation_scope": "generator_vs_human",
                "generator": generator,
                **metric_dict(
                    frame.loc[mask, "label"].to_numpy(), ai_probability[mask]
                ),
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )

    if validation is None or validation_ai_probability is None:
        return

    calibrated_threshold, validation_curve = select_calibrated_threshold(
        validation, validation_ai_probability
    )
    validation_curve["is_selected"] = validation_curve["threshold"].eq(
        calibrated_threshold
    )
    validation_curve.to_csv(
        output_dir / "validation_threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibrated = result.drop(
        columns=["predicted_label", "predicted_label_name"]
    ).copy()
    calibrated["calibrated_threshold"] = calibrated_threshold
    calibrated["predicted_label"] = np.where(
        ai_probability >= calibrated_threshold, 0, 1
    )
    calibrated["predicted_label_name"] = np.where(
        calibrated["predicted_label"].eq(0), "AI-generated", "human-written"
    )
    calibrated.to_csv(
        output_dir / "predictions_threshold_calibrated.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibrated_metric_rows = [
        {
            "method": method,
            "run": run,
            "threshold_strategy": "validation_macro_f1",
            "evaluation_scope": "overall",
            "generator": "all",
            **metric_dict(
                frame["label"].to_numpy(), ai_probability, calibrated_threshold
            ),
        }
    ]
    humans = frame["generator"].eq("human").to_numpy()
    for generator in AI_GENERATORS:
        mask = humans | frame["generator"].eq(generator).to_numpy()
        calibrated_metric_rows.append(
            {
                "method": method,
                "run": run,
                "threshold_strategy": "validation_macro_f1",
                "evaluation_scope": "generator_vs_human",
                "generator": generator,
                **metric_dict(
                    frame.loc[mask, "label"].to_numpy(),
                    ai_probability[mask],
                    calibrated_threshold,
                ),
            }
        )
    pd.DataFrame(calibrated_metric_rows).to_csv(
        output_dir / "metrics_threshold_calibrated.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity = threshold_curve(
        frame, ai_probability, SENSITIVITY_THRESHOLDS, "test"
    )
    sensitivity["threshold_strategy"] = "prespecified_sensitivity"
    sensitivity.to_csv(
        output_dir / "test_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "threshold_calibration_metadata.json").write_text(
        json.dumps(
            {
                "selection_dataset": "validation",
                "selection_metric": "macro_f1",
                "tie_breakers": ["balanced_accuracy", "distance_to_0.5"],
                "threshold_grid_start": 0.01,
                "threshold_grid_stop": 0.99,
                "threshold_grid_step": 0.01,
                "default_threshold": DEFAULT_THRESHOLD,
                "selected_threshold": calibrated_threshold,
                "test_set_used_for_threshold_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_representations(
    manifest: pd.DataFrame,
    features,
    lengths,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n = len(manifest)
    if REPRESENTATIONS.exists():
        representations = np.lib.format.open_memmap(REPRESENTATIONS, mode="r+")
        if representations.shape != (n, 1024):
            raise RuntimeError(f"Bad representation shape: {representations.shape}")
    else:
        representations = np.lib.format.open_memmap(
            REPRESENTATIONS, mode="w+", dtype=np.float32, shape=(n, 1024)
        )
    start = 0
    if REP_PROGRESS.exists():
        start = int(json.loads(REP_PROGRESS.read_text())["next_index"])
    if start >= n:
        return representations

    model = load_head(device).eval()
    subset = manifest.loc[manifest["row_index"].ge(start)].copy()
    loader = feature_loader(features, lengths, subset, batch_size, shuffle=False)
    completed = start
    with torch.inference_mode():
        for x, batch_lengths, _, indices in tqdm(loader, desc="CheckGPT representations"):
            x = x.to(device, non_blocking=True)
            representation = forward_representation(model, x, batch_lengths)
            idx = indices.numpy()
            representations[idx] = representation.cpu().numpy()
            completed += len(idx)
            representations.flush()
            REP_PROGRESS.write_text(
                json.dumps({"next_index": completed, "total": n}, indent=2),
                encoding="utf-8",
            )
    return representations


def method1(
    manifest: pd.DataFrame,
    representations: np.ndarray,
    device: torch.device,
) -> None:
    METHOD1.mkdir(parents=True, exist_ok=True)
    model = load_head(device).eval()
    validation = manifest.loc[manifest["split"].eq("validation")].copy()
    test = manifest.loc[manifest["split"].eq("test")].copy()
    val_idx = validation["row_index"].to_numpy(dtype=np.int64)
    idx = test["row_index"].to_numpy(dtype=np.int64)
    with torch.inference_mode():
        val_logits = model.fc(
            torch.from_numpy(np.array(representations[val_idx], copy=True)).to(device)
        )
        logits = model.fc(
            torch.from_numpy(np.array(representations[idx], copy=True)).to(device)
        )
        validation_ai_probability = F.softmax(val_logits, dim=1)[:, 0].cpu().numpy()
        ai_probability = F.softmax(logits, dim=1)[:, 0].cpu().numpy()
    evaluate_and_save(
        test,
        ai_probability,
        METHOD1,
        "direct_validation",
        "seed_0",
        validation=validation,
        validation_ai_probability=validation_ai_probability,
    )
    metadata = {
        "checkpoint": str(WEIGHTS),
        "checkpoint_loaded_strictly": True,
        "trained_parameters": [],
        "test_samples": int(len(test)),
    }
    (METHOD1 / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def select_method2_samples(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = []
    used_papers = set()
    ai_counts = dict(zip(AI_GENERATORS, [19, 19, 19, 18]))
    for generator, count in ai_counts.items():
        candidates = train.loc[
            train["generator"].eq(generator)
            & ~train["paper_key"].isin(used_papers)
        ]
        chosen = candidates.iloc[rng.choice(len(candidates), size=count, replace=False)]
        selected.append(chosen)
        used_papers.update(chosen["paper_key"])
    humans = train.loc[
        train["generator"].eq("human") & ~train["paper_key"].isin(used_papers)
    ]
    chosen_human = humans.iloc[rng.choice(len(humans), size=75, replace=False)]
    selected.append(chosen_human)
    result = pd.concat(selected, ignore_index=True)
    if len(result) != 150 or result["paper_key"].nunique() != 150:
        raise RuntimeError("Method 2 sample selection is not 150 unique papers")
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def method2(
    manifest: pd.DataFrame,
    representations: np.ndarray,
    device: torch.device,
    runs: int,
) -> None:
    METHOD2.mkdir(parents=True, exist_ok=True)
    train = manifest.loc[manifest["split"].eq("train")].copy()
    validation = manifest.loc[manifest["split"].eq("validation")].copy()
    test = manifest.loc[manifest["split"].eq("test")].copy()
    val_idx = validation["row_index"].to_numpy(dtype=np.int64)
    test_idx = test["row_index"].to_numpy(dtype=np.int64)
    x_val = torch.from_numpy(np.array(representations[val_idx], copy=True)).to(device)
    y_val = torch.tensor(validation["label"].to_numpy(), dtype=torch.long, device=device)
    x_test = torch.from_numpy(np.array(representations[test_idx], copy=True)).to(device)
    base = load_head(device)

    for run_number in range(1, runs + 1):
        seed = 100 + run_number
        seed_everything(seed)
        run_dir = METHOD2 / f"run_{run_number:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        selected = select_method2_samples(train, seed)
        selected.to_csv(
            run_dir / "training_samples_150.csv", index=False, encoding="utf-8-sig"
        )
        idx = selected["row_index"].to_numpy(dtype=np.int64)
        x_train = torch.from_numpy(np.array(representations[idx], copy=True))
        y_train = torch.tensor(selected["label"].to_numpy(), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)

        layer = copy.deepcopy(base.fc).to(device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=2e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        best_state = copy.deepcopy(layer.state_dict())
        best_score = -1.0
        patience = 0
        history = []
        for epoch in range(1, 51):
            layer.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(layer(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            layer.eval()
            with torch.inference_mode():
                val_ai = F.softmax(layer(x_val), dim=1)[:, 0].cpu().numpy()
            score = metric_dict(validation["label"].to_numpy(), val_ai)["macro_f1"]
            history.append(
                {"epoch": epoch, "train_loss": np.mean(losses), "validation_macro_f1": score}
            )
            if score > best_score + 1e-6:
                best_score = score
                best_state = copy.deepcopy(layer.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 5:
                break
        layer.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_fc.pth")
        pd.DataFrame(history).to_csv(
            run_dir / "training_history.csv", index=False, encoding="utf-8-sig"
        )
        layer.eval()
        with torch.inference_mode():
            best_val_ai = F.softmax(layer(x_val), dim=1)[:, 0].cpu().numpy()
            test_ai = F.softmax(layer(x_test), dim=1)[:, 0].cpu().numpy()
        evaluate_and_save(
            test,
            test_ai,
            run_dir,
            "last_layer_150",
            f"seed_{seed}",
            validation=validation,
            validation_ai_probability=best_val_ai,
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "training_samples": 150,
                    "human_samples": 75,
                    "ai_samples": 75,
                    "trainable_parameters": ["fc.weight", "fc.bias"],
                    "best_validation_macro_f1": best_score,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def predict_full_head(
    model, features, lengths, frame, device, batch_size
) -> np.ndarray:
    model.eval()
    loader = feature_loader(features, lengths, frame, batch_size, shuffle=False)
    probabilities = []
    with torch.inference_mode():
        for x, batch_lengths, _, _ in loader:
            logits = model(x.to(device, non_blocking=True), batch_lengths)
            probabilities.append(F.softmax(logits, dim=1)[:, 0].cpu().numpy())
    return np.concatenate(probabilities)


def method3(
    manifest: pd.DataFrame,
    features,
    lengths,
    device: torch.device,
    batch_size: int,
    run_numbers: list[int],
    max_epochs: int,
) -> None:
    METHOD3.mkdir(parents=True, exist_ok=True)
    train = manifest.loc[manifest["split"].eq("train")].copy()
    validation = manifest.loc[manifest["split"].eq("validation")].copy()
    test = manifest.loc[manifest["split"].eq("test")].copy()
    counts = train["label"].value_counts()
    class_weights = torch.tensor(
        [len(train) / (2 * counts[0]), len(train) / (2 * counts[1])],
        dtype=torch.float32,
        device=device,
    )

    for run_number in run_numbers:
        seed = 200 + run_number
        seed_everything(seed)
        run_dir = METHOD3 / f"run_{run_number:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = load_head(device)
        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            FeatureDataset(features, lengths, train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_score = -1.0
        patience = 0
        history = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            losses = []
            for x, batch_lengths, labels, _ in tqdm(
                loader, desc=f"Full head run {run_number} epoch {epoch}", leave=False
            ):
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, batch_lengths)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            val_ai = predict_full_head(
                model, features, lengths, validation, device, batch_size
            )
            score = metric_dict(validation["label"].to_numpy(), val_ai)["macro_f1"]
            history.append(
                {"epoch": epoch, "train_loss": np.mean(losses), "validation_macro_f1": score}
            )
            pd.DataFrame(history).to_csv(
                run_dir / "training_history.csv", index=False, encoding="utf-8-sig"
            )
            if score > best_score + 1e-6:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(best_state, run_dir / "best_classifier_head.pth")
                patience = 0
            else:
                patience += 1
            if patience >= 3:
                break
        model.load_state_dict(best_state)
        model.to(device)
        best_val_ai = predict_full_head(
            model, features, lengths, validation, device, batch_size
        )
        test_ai = predict_full_head(
            model, features, lengths, test, device, batch_size
        )
        evaluate_and_save(
            test,
            test_ai,
            run_dir,
            "full_classification_head",
            f"seed_{seed}",
            validation=validation,
            validation_ai_probability=best_val_ai,
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "training_samples": int(len(train)),
                    "class_weights": class_weights.cpu().tolist(),
                    "trainable_parameters": "entire CheckGPT classification head",
                    "roberta_parameters": "frozen via cached features",
                    "best_validation_macro_f1": best_score,
                    "epochs_completed": len(history),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    features = np.load(FEATURES, mmap_mode="r")
    lengths = np.load(LENGTHS, mmap_mode="r")
    if args.preload_features:
        print("Preloading feature cache into RAM...", flush=True)
        features = np.array(features, copy=True)
        lengths = np.array(lengths, copy=True)
        print("Feature cache preloaded.", flush=True)
    if len(manifest) != len(features) or len(manifest) != len(lengths):
        raise RuntimeError("Manifest and feature sizes differ")

    representations = None
    if args.method in ("representations", "method1", "method2", "all"):
        representations = build_representations(
            manifest, features, lengths, device, args.batch_size
        )
    if args.method in ("method1", "all"):
        method1(manifest, representations, device)
    if args.method in ("method2", "all"):
        method2(manifest, representations, device, args.method2_runs)
    if args.method in ("method3", "all"):
        run_numbers = (
            [int(item) for item in args.method3_run_numbers.split(",") if item.strip()]
            if args.method3_run_numbers
            else list(range(1, args.method3_runs + 1))
        )
        method3(
            manifest,
            features,
            lengths,
            device,
            args.batch_size,
            run_numbers,
            args.method3_max_epochs,
        )


if __name__ == "__main__":
    main()
