#!/usr/bin/env python3
"""Aggregate CheckGPT benchmark runs and create publication-ready comparisons."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


ADAPTATION_ROOT = Path(__file__).resolve().parents[1]
EXP = ADAPTATION_ROOT / "results"
COMPARISON = EXP / "comparison"
METHOD_CONFIG = {
    "Direct validation": [EXP / "method1_direct_validation"],
    "Last layer (150)": sorted((EXP / "method2_last_layer_150").glob("run_*")),
    "Full head": sorted((EXP / "method3_full_head").glob("run_*")),
}
COLORS = {
    "Direct validation": "#9CA3AF",
    "Last layer (150)": "#6BAED6",
    "Full head": "#3182BD",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def collect():
    metric_frames = []
    predictions = {}
    for display, directories in METHOD_CONFIG.items():
        if not directories:
            raise RuntimeError(f"No completed runs found for {display}")
        predictions[display] = []
        for directory in directories:
            metric_path = directory / "metrics.csv"
            prediction_path = directory / "predictions.csv"
            if not metric_path.exists() or not prediction_path.exists():
                raise RuntimeError(f"Incomplete run: {directory}")
            metrics = pd.read_csv(metric_path, encoding="utf-8-sig")
            metrics.insert(0, "method_display", display)
            metrics.insert(1, "run_directory", directory.name)
            metric_frames.append(metrics)
            predictions[display].append(
                pd.read_csv(prediction_path, encoding="utf-8-sig")
            )
    return pd.concat(metric_frames, ignore_index=True), predictions


def main() -> None:
    COMPARISON.mkdir(parents=True, exist_ok=True)
    metrics, predictions = collect()
    metrics.to_csv(COMPARISON / "all_run_metrics.csv", index=False, encoding="utf-8-sig")

    value_columns = [
        "accuracy", "balanced_accuracy", "precision_ai", "recall_ai",
        "f1_ai", "macro_f1", "roc_auc", "pr_auc",
    ]
    summary = (
        metrics.groupby(
            ["method_display", "evaluation_scope", "generator"], sort=False
        )[value_columns]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in col if item).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    summary.to_csv(
        COMPARISON / "method_comparison.csv", index=False, encoding="utf-8-sig"
    )
    for display, root in (
        ("Last layer (150)", EXP / "method2_last_layer_150"),
        ("Full head", EXP / "method3_full_head"),
    ):
        metrics.loc[metrics["method_display"].eq(display)].to_csv(
            root / "aggregate_metrics.csv", index=False, encoding="utf-8-sig"
        )

    overall = metrics.loc[metrics["evaluation_scope"].eq("overall")].copy()
    plot_metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
        ("roc_auc", "ROC–AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 55 / 25.4))
    for ax, (metric, label) in zip(axes, plot_metrics):
        stats = overall.groupby("method_display", sort=False)[metric].agg(["mean", "std"])
        order = list(METHOD_CONFIG)
        means = stats.reindex(order)["mean"]
        errors = stats.reindex(order)["std"].fillna(0)
        x = np.arange(len(order))
        ax.bar(
            x, means, yerr=errors, color=[COLORS[item] for item in order],
            edgecolor="none", capsize=3, width=0.68,
        )
        ax.set_xticks(x, ["Direct", "Last layer\n(150)", "Full head"])
        ax.set_ylabel(label)
        ax.set_ylim(max(0, float((means - errors).min()) - 0.08), 1.01)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, COMPARISON / "method_performance_comparison")

    fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    grid = np.linspace(0, 1, 501)
    for display, runs in predictions.items():
        interpolated = []
        aucs = []
        for run in runs:
            y_ai = run["label"].eq(0).astype(int).to_numpy()
            fpr, tpr, _ = roc_curve(y_ai, run["ai_probability"].to_numpy())
            interpolated.append(np.interp(grid, fpr, tpr))
            aucs.append(
                overall.loc[overall["method_display"].eq(display), "roc_auc"].iloc[
                    len(aucs)
                ]
            )
        curves = np.vstack(interpolated)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(
            grid, mean, color=COLORS[display], linewidth=1.7,
            label=f"{display} ({np.mean(aucs):.3f})",
        )
        if len(runs) > 1:
            ax.fill_between(
                grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                color=COLORS[display], alpha=0.16, linewidth=0,
            )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(fig, COMPARISON / "roc_curves")

    per_generator = metrics.loc[
        metrics["evaluation_scope"].eq("generator_vs_human")
    ].copy()
    stats = (
        per_generator.groupby(["method_display", "generator"], sort=False)["roc_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(150 / 25.4, 70 / 25.4))
    generators = ["gemini-2.5-flash", "glm-4-flash", "glm-5.1", "gpt-4o"]
    x = np.arange(len(generators))
    offsets = [-0.22, 0, 0.22]
    for offset, display in zip(offsets, METHOD_CONFIG):
        subset = stats.loc[stats["method_display"].eq(display)].set_index("generator")
        ax.errorbar(
            x + offset,
            subset.reindex(generators)["mean"],
            yerr=subset.reindex(generators)["std"].fillna(0),
            marker="o",
            markersize=4.5,
            linewidth=1.2,
            capsize=2.5,
            color=COLORS[display],
            label=display,
        )
    ax.set_xticks(x, ["Gemini 2.5 Flash", "GLM-4 Flash", "GLM-5.1", "GPT-4o"])
    ax.set_ylabel("ROC–AUC")
    ax.set_ylim(max(0, stats["mean"].min() - 0.08), 1.01)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    save_figure(fig, COMPARISON / "performance_by_generator")

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 58 / 25.4))
    for ax, display in zip(axes, METHOD_CONFIG):
        matrices = []
        for run in predictions[display]:
            y_ai = run["label"].eq(0).astype(int).to_numpy()
            pred_ai = run["predicted_label"].eq(0).astype(int).to_numpy()
            matrices.append(confusion_matrix(y_ai, pred_ai, normalize="true"))
        matrix = np.mean(matrices, axis=0)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=sns.light_palette(COLORS[display], as_cmap=True),
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            ax=ax,
        )
        ax.set_title(display, fontsize=9)
        ax.set_xticklabels(["Human", "AI"], rotation=0)
        ax.set_yticklabels(["Human", "AI"], rotation=0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, COMPARISON / "confusion_matrices")
    print(f"Saved comparison outputs to {COMPARISON}")


if __name__ == "__main__":
    main()
