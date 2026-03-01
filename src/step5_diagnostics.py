"""Step 5 additional diagnostics (validation-only).

Adds reliability/error/training-behaviour diagnostics after tuning, while
keeping the test set untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

try:
    from src.step3_preprocessing import ID_COL, TARGET_COL, load_default_dataset
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.step3_preprocessing import ID_COL, TARGET_COL, load_default_dataset  # type: ignore


TEXT_COLOR = "#111111"
GRID_COLOR = "#D9D9D9"
SPINE_COLOR = "#666666"


@dataclass
class Step5DiagnosticsArtifacts:
    metrics_dir: str
    figures_dir: str
    summary_path: str


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.tick_params(axis="both", colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor(SPINE_COLOR)
            for t in legend.get_texts():
                t.set_color(TEXT_COLOR)
            if legend.get_title():
                legend.get_title().set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)


def _load_splits(df: pd.DataFrame, split_assignments_path: Path) -> dict[str, pd.Index]:
    """Load split assignments and verify row/ID/target alignment."""
    split_df = pd.read_csv(split_assignments_path)
    if len(split_df) != len(df):
        raise ValueError("split assignments row count mismatch")
    if not (split_df["ID"].to_numpy() == df[ID_COL].to_numpy()).all():
        raise ValueError("split assignments ID mismatch")
    if not (split_df["target"].to_numpy() == df[TARGET_COL].to_numpy()).all():
        raise ValueError("split assignments target mismatch")
    return {
        "train": pd.Index(split_df.loc[split_df["split"] == "train", "row_index"].to_list()),
        "validation": pd.Index(split_df.loc[split_df["split"] == "validation", "row_index"].to_list()),
        "test": pd.Index(split_df.loc[split_df["split"] == "test", "row_index"].to_list()),
    }


def _set_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "legend.facecolor": "white",
            "legend.edgecolor": SPINE_COLOR,
        },
    )


def _bin_limit_bal(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 50000, 100000, 200000, 500000, np.inf]
    labels = ["<=50k", "50k-100k", "100k-200k", "200k-500k", ">500k"]
    return pd.cut(series, bins=bins, labels=labels)


def _bin_age(series: pd.Series) -> pd.Series:
    bins = [0, 25, 35, 45, 55, np.inf]
    labels = ["<=25", "26-35", "36-45", "46-55", "56+"]
    return pd.cut(series, bins=bins, labels=labels)


def _extract_training_curves(model) -> dict[str, list[float] | None]:
    """Extract available curve attributes from fitted MLP/HGB estimators."""
    curves: dict[str, list[float] | None] = {
        "loss_curve": None,
        "validation_scores": None,
        "train_score": None,
        "validation_score": None,
    }
    if hasattr(model, "loss_curve_") and getattr(model, "loss_curve_") is not None:
        curves["loss_curve"] = [float(x) for x in list(model.loss_curve_)]
    if hasattr(model, "validation_scores_") and getattr(model, "validation_scores_") is not None:
        curves["validation_scores"] = [float(x) for x in list(model.validation_scores_)]
    if hasattr(model, "train_score_") and getattr(model, "train_score_") is not None:
        curves["train_score"] = [float(x) for x in list(model.train_score_)]
    if hasattr(model, "validation_score_") and getattr(model, "validation_score_") is not None:
        v = getattr(model, "validation_score_")
        try:
            curves["validation_score"] = [float(x) for x in list(v)]
        except TypeError:
            curves["validation_score"] = [float(v)]
    return curves


def run_step5_diagnostics(
    df: pd.DataFrame,
    outputs_dir: str | Path,
) -> Step5DiagnosticsArtifacts:
    """Run validation-only diagnostics for the validation-selected setup.

    This step does not alter model selection. It documents calibration,
    failure modes, and training dynamics to strengthen the evaluation narrative.
    """
    _set_theme()
    outputs_dir = Path(outputs_dir)
    fig_dir = outputs_dir / "figures" / "step5_diagnostics"
    metrics_dir = outputs_dir / "metrics" / "step5_diagnostics"
    for p in [fig_dir, metrics_dir]:
        p.mkdir(parents=True, exist_ok=True)

    step5_summary = json.loads((outputs_dir / "metrics" / "step5_tuning" / "step5_tuning_summary.json").read_text())
    final_model_name = step5_summary["best_model_by_validation_pr_auc_after_tuning"]
    final_threshold = float(step5_summary["selected_thresholds"][final_model_name]["selected_threshold_for_f1"])

    splits = _load_splits(df, outputs_dir / "metrics" / "step3_preprocessing" / "split_assignments.csv")
    val_idx = splits["validation"]
    X_val = df.loc[val_idx, [c for c in df.columns if c not in {TARGET_COL, ID_COL}]]
    y_val = df.loc[val_idx, TARGET_COL]

    # Load tuned validation-fit models from Step 5 (train fit only; valid for validation diagnostics).
    hgb_pipe = joblib.load(outputs_dir / "models" / "step5_tuning" / "best_hist_gradient_boosting.joblib")
    mlp_pipe = joblib.load(outputs_dir / "models" / "step5_tuning" / "best_mlp_neural_network.joblib")
    final_pipe = hgb_pipe if final_model_name == "hist_gradient_boosting" else mlp_pipe

    y_prob = final_pipe.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= final_threshold).astype(int)

    pred_df = df.loc[val_idx, [ID_COL, TARGET_COL, "LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE", "PAY_0"]].copy()
    pred_df["row_index"] = val_idx.to_list()
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df["error_type"] = np.select(
        [
            (pred_df[TARGET_COL] == 1) & (pred_df["y_pred"] == 1),
            (pred_df[TARGET_COL] == 0) & (pred_df["y_pred"] == 0),
            (pred_df[TARGET_COL] == 0) & (pred_df["y_pred"] == 1),
            (pred_df[TARGET_COL] == 1) & (pred_df["y_pred"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    pred_df["is_error"] = pred_df["error_type"].isin(["FP", "FN"])
    pred_df["LIMIT_BAL_BIN"] = _bin_limit_bal(pred_df["LIMIT_BAL"])
    pred_df["AGE_BIN"] = _bin_age(pred_df["AGE"])
    pred_df.to_csv(metrics_dir / "validation_predictions_with_errors_final_model.csv", index=False)

    # Calibration table + plot (validation only)
    frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=10, strategy="quantile")
    calib_df = pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positive": frac_pos})
    calib_df.to_csv(metrics_dir / "validation_calibration_table_final_model.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1, label="Perfectly calibrated")
    axes[0].plot(mean_pred, frac_pos, marker="o", color="#4C78A8", label=final_model_name)
    axes[0].set_title("Validation Calibration Curve (Final Model)")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed default rate")
    axes[0].legend()

    axes[1].hist(y_prob, bins=25, color="#F58518", edgecolor="white")
    axes[1].axvline(final_threshold, linestyle="--", color="black", linewidth=1.2, label=f"threshold={final_threshold:.2f}")
    axes[1].set_title("Validation Predicted Probability Distribution")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    _save_fig(fig, fig_dir / "validation_calibration_and_probability_distribution_final_model.png")

    # Confidence by outcome/error type
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["TN", "FP", "FN", "TP"]
    sns.boxplot(data=pred_df, x="error_type", y="y_prob", order=order, ax=ax, color="#72B7B2", showfliers=False)
    ax.axhline(final_threshold, linestyle="--", color="black", linewidth=1)
    ax.set_title("Validation Predicted Probabilities by Outcome/Error Type")
    ax.set_xlabel("Outcome / error type")
    ax.set_ylabel("Predicted default probability")
    _save_fig(fig, fig_dir / "validation_probability_by_error_type_final_model.png")

    # Segment-level error rates / failure modes
    segment_tables: dict[str, pd.DataFrame] = {}
    for seg_col, out_name in [
        ("PAY_0", "segment_error_rates_by_pay0.csv"),
        ("LIMIT_BAL_BIN", "segment_error_rates_by_limit_bal_bin.csv"),
        ("AGE_BIN", "segment_error_rates_by_age_bin.csv"),
        ("EDUCATION", "segment_error_rates_by_education.csv"),
    ]:
        seg = (
            pred_df.groupby(seg_col, dropna=False)
            .agg(
                n=("is_error", "size"),
                error_rate=("is_error", "mean"),
                fp_rate=("error_type", lambda s: (s == "FP").mean()),
                fn_rate=("error_type", lambda s: (s == "FN").mean()),
                actual_default_rate=(TARGET_COL, "mean"),
                predicted_positive_rate=("y_pred", "mean"),
            )
            .reset_index()
        )
        segment_tables[seg_col] = seg
        seg.to_csv(metrics_dir / out_name, index=False)

    # Plot a few failure-mode views
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, seg_col, title in [
        (axes[0], "PAY_0", "Error Rate by PAY_0"),
        (axes[1], "LIMIT_BAL_BIN", "Error Rate by LIMIT_BAL Bin"),
        (axes[2], "AGE_BIN", "Error Rate by Age Bin"),
    ]:
        d = segment_tables[seg_col].copy()
        d = d[d["n"] >= 30] if "BIN" not in seg_col else d
        sns.barplot(data=d, x=seg_col, y="error_rate", ax=ax, color="#E45756")
        ax.set_title(title)
        ax.set_xlabel(seg_col)
        ax.set_ylabel("Validation error rate")
        ax.tick_params(axis="x", rotation=30)
    _save_fig(fig, fig_dir / "validation_failure_modes_segment_error_rates_final_model.png")

    # Hard-case tables (highest-confidence mistakes)
    top_fp = (
        pred_df[pred_df["error_type"] == "FP"]
        .sort_values("y_prob", ascending=False)
        .head(25)
        .copy()
    )
    top_fn = (
        pred_df[pred_df["error_type"] == "FN"]
        .sort_values("y_prob", ascending=True)
        .head(25)
        .copy()
    )
    top_fp.to_csv(metrics_dir / "top_false_positives_final_model_validation.csv", index=False)
    top_fn.to_csv(metrics_dir / "top_false_negatives_final_model_validation.csv", index=False)

    # Confusion matrix on validation for the locked threshold (diagnostic only; threshold chosen on validation in Step 5)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Validation Confusion Matrix ({final_model_name}, thr={final_threshold:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"], rotation=0)
    _save_fig(fig, fig_dir / "validation_confusion_matrix_final_model_locked_threshold.png")

    # Training curves from tuned models (validation-only diagnostics)
    hgb_curves = _extract_training_curves(hgb_pipe.named_steps["model"])
    mlp_curves = _extract_training_curves(mlp_pipe.named_steps["model"])

    curves_summary = {
        "hgb_curve_lengths": {k: (len(v) if isinstance(v, list) else 0) for k, v in hgb_curves.items()},
        "mlp_curve_lengths": {k: (len(v) if isinstance(v, list) else 0) for k, v in mlp_curves.items()},
    }

    # MLP training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if mlp_curves["loss_curve"]:
        axes[0].plot(range(1, len(mlp_curves["loss_curve"]) + 1), mlp_curves["loss_curve"], color="#4C78A8")
        axes[0].set_title("MLP Training Loss Curve")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
    else:
        axes[0].text(0.5, 0.5, "No loss_curve_ available", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("MLP Training Loss Curve")
    if mlp_curves["validation_scores"]:
        axes[1].plot(range(1, len(mlp_curves["validation_scores"]) + 1), mlp_curves["validation_scores"], color="#54A24B")
        axes[1].set_title("MLP Internal Validation Score (Early Stopping)")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Score")
    else:
        axes[1].text(0.5, 0.5, "No validation_scores_ available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("MLP Internal Validation Score")
    _save_fig(fig, fig_dir / "training_curves_mlp_best_tuned_model.png")

    # HGB training curves (if available)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if hgb_curves["train_score"]:
        axes[0].plot(range(1, len(hgb_curves["train_score"]) + 1), hgb_curves["train_score"], color="#F58518")
        axes[0].set_title("HGB train_score_ by Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Score")
    else:
        axes[0].text(0.5, 0.5, "No train_score_ available", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("HGB train_score_")
    if hgb_curves["validation_score"]:
        axes[1].plot(range(1, len(hgb_curves["validation_score"]) + 1), hgb_curves["validation_score"], color="#E45756")
        axes[1].set_title("HGB validation_score_ by Iteration")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Score")
    else:
        axes[1].text(0.5, 0.5, "No validation_score_ available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("HGB validation_score_")
    _save_fig(fig, fig_dir / "training_curves_hgb_best_tuned_model.png")

    # Export raw curve arrays for reproducibility
    curve_export = {
        "hist_gradient_boosting": hgb_curves,
        "mlp_neural_network": mlp_curves,
    }
    (metrics_dir / "training_curves_raw.json").write_text(json.dumps(curve_export, indent=2))

    # Compact summary
    error_summary = {
        "final_model_name": final_model_name,
        "locked_validation_threshold": final_threshold,
        "n_validation": int(len(val_idx)),
        "n_errors": int(pred_df["is_error"].sum()),
        "error_rate": float(pred_df["is_error"].mean()),
        "fp_count": int((pred_df["error_type"] == "FP").sum()),
        "fn_count": int((pred_df["error_type"] == "FN").sum()),
        "top_error_segments_pay0": segment_tables["PAY_0"].sort_values("error_rate", ascending=False).head(10).to_dict(orient="records"),
        "curve_availability": curves_summary,
    }
    summary_path = metrics_dir / "step5_diagnostics_summary.json"
    summary_path.write_text(json.dumps(error_summary, indent=2, default=str))

    return Step5DiagnosticsArtifacts(
        metrics_dir=str(metrics_dir),
        figures_dir=str(fig_dir),
        summary_path=str(summary_path),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_default_dataset(root / "data" / "raw" / "default_of_credit_card_clients.xls")
    artifacts = run_step5_diagnostics(df, root / "outputs")
    print(json.dumps({"metrics_dir": artifacts.metrics_dir, "figures_dir": artifacts.figures_dir, "summary_path": artifacts.summary_path}, indent=2))


if __name__ == "__main__":
    main()
