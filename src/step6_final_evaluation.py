"""Step 6 final model refit and locked one-time test evaluation.

Reads the validation-selected model/threshold from Step 5, refits on
train+validation, and evaluates the test set exactly once for the locked
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import shutil
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from src.step3_preprocessing import ID_COL, TARGET_COL, build_preprocessor, load_default_dataset
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.step3_preprocessing import ID_COL, TARGET_COL, build_preprocessor, load_default_dataset  # type: ignore


@dataclass
class Step6Artifacts:
    metrics_dir: str
    figures_dir: str
    models_dir: str
    final_model_name: str
    final_threshold: float
    test_summary_path: str


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_splits(df: pd.DataFrame, split_assignments_path: Path) -> dict[str, pd.Index]:
    """Load split assignments and verify row/ID/target alignment."""
    split_df = pd.read_csv(split_assignments_path)
    if len(split_df) != len(df):
        raise ValueError("split assignments row count mismatch.")
    if not (split_df["ID"].to_numpy() == df[ID_COL].to_numpy()).all():
        raise ValueError("split assignments ID mismatch.")
    if not (split_df["target"].to_numpy() == df[TARGET_COL].to_numpy()).all():
        raise ValueError("split assignments target mismatch.")
    return {
        "train": pd.Index(split_df.loc[split_df["split"] == "train", "row_index"].to_list()),
        "validation": pd.Index(split_df.loc[split_df["split"] == "validation", "row_index"].to_list()),
        "test": pd.Index(split_df.loc[split_df["split"] == "test", "row_index"].to_list()),
    }


def _supports_sample_weight(pipeline: Pipeline) -> bool:
    return "sample_weight" in inspect.signature(pipeline.named_steps["model"].fit).parameters


def _normalize_params(model_name: str, params: dict) -> dict:
    """Normalize serialized params back to estimator-friendly types."""
    p = dict(params)
    if model_name == "mlp_neural_network" and "hidden_layer_sizes" in p and isinstance(p["hidden_layer_sizes"], list):
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    return p


def _make_pipeline(model_name: str, params: dict, random_state: int) -> Pipeline:
    """Build the final model pipeline from locked model name + params."""
    params = _normalize_params(model_name, params)
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            **params,
        )
    elif model_name == "mlp_neural_network":
        model = MLPClassifier(
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            **params,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def _compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute final test metrics at a single locked threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_prediction_rate": float(y_pred.mean()),
    }


def _plot_final_model_curves(y_prob: np.ndarray, y_test: pd.Series, model_name: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("Test ROC Curve (Final Locked Model)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _save_fig(fig, fig_dir / "test_roc_curve_final_model.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = float((y_test == 1).mean())
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"Baseline={baseline:.3f}")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    ax.set_title("Test Precision-Recall Curve (Final Locked Model)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    _save_fig(fig, fig_dir / "test_pr_curve_final_model.png")


def _plot_final_confusion_locked(y_test: pd.Series, y_prob: np.ndarray, threshold: float, model_name: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm = confusion_matrix(y_test, (y_prob >= threshold).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name}\nTest Confusion Matrix (Locked Threshold {threshold:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"], rotation=0)
    _save_fig(fig, fig_dir / "test_confusion_matrix_final_model_locked_threshold.png")


def run_step6_final_evaluation(
    df: pd.DataFrame,
    outputs_dir: str | Path,
    *,
    random_state: int = 42,
) -> Step6Artifacts:
    """Execute locked final evaluation protocol on the held-out test set.

    Protocol:
    - lock model/hyperparameters/threshold from Step 5 summary,
    - refit locked model on train+validation,
    - evaluate test set once and persist final artefacts.
    """
    sns.set_theme(context="notebook", style="whitegrid")
    outputs_dir = Path(outputs_dir)
    fig_dir = outputs_dir / "figures" / "step6_final_evaluation"
    metrics_dir = outputs_dir / "metrics" / "step6_final_evaluation"
    models_dir = outputs_dir / "models" / "step6_final_evaluation"
    for p in [fig_dir, metrics_dir, models_dir]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    step5_summary_path = outputs_dir / "metrics" / "step5_tuning" / "step5_tuning_summary.json"
    step5_summary = json.loads(step5_summary_path.read_text())
    best_configs: dict = step5_summary["best_configs"]
    selected_thresholds: dict = step5_summary["selected_thresholds"]
    final_model_name = step5_summary["best_model_by_validation_pr_auc_after_tuning"]

    splits = _load_splits(df, outputs_dir / "metrics" / "step3_preprocessing" / "split_assignments.csv")
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, ID_COL}]
    X = df[feature_cols]
    y = df[TARGET_COL]
    ids = df[ID_COL]

    # Refit uses all non-test rows after configuration is locked on validation.
    train_val_idx = splits["train"].append(splits["validation"])
    test_idx = splits["test"]

    X_train_val = X.loc[train_val_idx]
    y_train_val = y.loc[train_val_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_val)

    final_model_threshold = float(selected_thresholds[final_model_name]["selected_threshold_for_f1"])
    params = best_configs[final_model_name]
    pipeline = _make_pipeline(final_model_name, params=params, random_state=random_state)
    fit_kwargs = {}
    if _supports_sample_weight(pipeline):
        fit_kwargs["model__sample_weight"] = sample_weights
    t0 = time.perf_counter()
    pipeline.fit(X_train_val, y_train_val, **fit_kwargs)
    fit_seconds = time.perf_counter() - t0

    # One locked test pass for the single final configuration.
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    locked_metrics = _compute_metrics(y_test, y_prob, threshold=final_model_threshold)

    final_metrics_row = {
        "model_name": final_model_name,
        "threshold_label": "locked_threshold_from_validation",
        "label": f"{final_model_name} | locked threshold from validation",
        "fit_seconds_train_plus_val": float(fit_seconds),
        "used_sample_weight": bool("model__sample_weight" in fit_kwargs),
        **locked_metrics,
    }
    pd.DataFrame([final_metrics_row]).to_csv(metrics_dir / "final_model_test_metrics_locked.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "row_index": test_idx.to_list(),
            "ID": ids.loc[test_idx].to_list(),
            "split": "test",
            "y_true": y_test.to_list(),
            "y_pred_proba": y_prob.tolist(),
            "y_pred_label_locked_threshold": (y_prob >= final_model_threshold).astype(int).tolist(),
            "locked_threshold_used": final_model_threshold,
        }
    )
    pred_df.to_csv(metrics_dir / "test_predictions_final_model_locked.csv", index=False)

    model_obj = pipeline.named_steps["model"]
    model_training_meta = {
        final_model_name: {
            "fit_seconds_train_plus_val": float(fit_seconds),
            "used_sample_weight": bool("model__sample_weight" in fit_kwargs),
            "params": model_obj.get_params(),
            "locked_threshold_from_validation": final_model_threshold,
        }
    }

    joblib.dump(pipeline, models_dir / f"final_refit_{final_model_name}.joblib")
    joblib.dump(pipeline, models_dir / "selected_final_model_pipeline.joblib")

    final_summary = {
        "final_model_name": final_model_name,
        "final_model_selected_on": "validation_pr_auc_after_tuning",
        "final_model_hyperparameters": best_configs[final_model_name],
        "threshold_source": "validation_set_f1_maximization",
        "final_threshold": final_model_threshold,
        "fit_data_for_final_model": "train_plus_validation",
        "test_set_size": int(len(test_idx)),
        "test_positive_rate": float((y_test == 1).mean()),
        "test_metrics_at_final_threshold": final_metrics_row,
        "test_set_used_once_for_final_evaluation": True,
    }
    test_summary_path = metrics_dir / "final_test_evaluation_summary.json"
    test_summary_path.write_text(json.dumps(final_summary, indent=2, default=str))

    (metrics_dir / "model_training_metadata_train_plus_val.json").write_text(json.dumps(model_training_meta, indent=2, default=str))

    # Save final chosen model + threshold metadata bundle
    (models_dir / "selected_final_model_threshold.json").write_text(
        json.dumps(
            {
                "model_name": final_model_name,
                "threshold": final_model_threshold,
                "threshold_selected_on": "validation",
                "selection_metric": "f1",
            },
            indent=2,
        )
    )

    _plot_final_model_curves(y_prob, y_test, final_model_name, fig_dir)
    _plot_final_confusion_locked(y_test, y_prob, final_model_threshold, final_model_name, fig_dir)

    return Step6Artifacts(
        metrics_dir=str(metrics_dir),
        figures_dir=str(fig_dir),
        models_dir=str(models_dir),
        final_model_name=str(final_model_name),
        final_threshold=float(final_model_threshold),
        test_summary_path=str(test_summary_path),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_default_dataset(root / "data" / "raw" / "default_of_credit_card_clients.xls")
    artifacts = run_step6_final_evaluation(df, root / "outputs", random_state=42)
    print(
        json.dumps(
            {
                "metrics_dir": artifacts.metrics_dir,
                "figures_dir": artifacts.figures_dir,
                "models_dir": artifacts.models_dir,
                "final_model_name": artifacts.final_model_name,
                "final_threshold": artifacts.final_threshold,
                "test_summary_path": artifacts.test_summary_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
