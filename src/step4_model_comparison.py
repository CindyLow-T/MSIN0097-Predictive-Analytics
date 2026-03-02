"""Step 4 model development and validation comparison.

Compares baseline and alternative models on the validation split only using a
shared preprocessing pipeline. Outputs tables/plots/models for reproducible
shortlisting before tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
    from src.step3_preprocessing import (
        ID_COL,
        TARGET_COL,
        build_preprocessor,
        load_default_dataset,
    )
except ModuleNotFoundError:
    # Support direct execution: `python src/step4_model_comparison.py`
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.step3_preprocessing import (  # type: ignore
        ID_COL,
        TARGET_COL,
        build_preprocessor,
        load_default_dataset,
    )


@dataclass
class Step4Artifacts:
    comparison_table_path: str
    metrics_dir: str
    figure_dir: str
    models_dir: str
    best_model_by_pr_auc: str
    validation_metrics: list[dict]


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_splits(df: pd.DataFrame, split_assignments_path: str | Path) -> dict[str, pd.Index]:
    """Load split assignments and verify row/ID/target alignment."""
    split_df = pd.read_csv(split_assignments_path)
    required_cols = {"row_index", "ID", "target", "split"}
    if not required_cols.issubset(split_df.columns):
        raise KeyError(f"split_assignments missing columns: {sorted(required_cols - set(split_df.columns))}")

    # Basic reproducibility checks: align to current dataset.
    if len(split_df) != len(df):
        raise ValueError("split_assignments row count does not match dataset.")
    if not (split_df["ID"].to_numpy() == df[ID_COL].to_numpy()).all():
        raise ValueError("split_assignments ID order does not match dataset.")
    if not (split_df["target"].to_numpy() == df[TARGET_COL].to_numpy()).all():
        raise ValueError("split_assignments target order does not match dataset.")

    return {
        "train": pd.Index(split_df.loc[split_df["split"] == "train", "row_index"].to_list()),
        "validation": pd.Index(split_df.loc[split_df["split"] == "validation", "row_index"].to_list()),
        "test": pd.Index(split_df.loc[split_df["split"] == "test", "row_index"].to_list()),
    }


def build_step4_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Define candidate model pipelines with identical preprocessing."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=40,
                        l2_regularization=0.0,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "mlp_neural_network": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        batch_size=256,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _supports_sample_weight(pipeline: Pipeline) -> bool:
    model = pipeline.named_steps["model"]
    return "sample_weight" in inspect.signature(model.fit).parameters


def _compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute ranking and threshold metrics for a probability vector."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
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


def _save_predictions(
    path: Path,
    row_index: pd.Index,
    ids: pd.Series,
    y_true: pd.Series,
    y_prob: np.ndarray,
    split_name: str,
) -> None:
    pred_df = pd.DataFrame(
        {
            "row_index": row_index.to_list(),
            "ID": ids.loc[row_index].to_list(),
            "split": split_name,
            "y_true": y_true.loc[row_index].to_list(),
            "y_pred_proba": y_prob.tolist(),
            "y_pred_label_0p5": (y_prob >= 0.5).astype(int).tolist(),
        }
    )
    pred_df.to_csv(path, index=False)


def _plot_metric_bars(results_df: pd.DataFrame, figure_dir: Path) -> None:
    metric_cols = ["pr_auc", "roc_auc", "balanced_accuracy", "f1", "recall", "precision"]
    long_df = results_df.melt(
        id_vars=["model_name"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long_df, x="metric", y="score", hue="model_name", ax=ax)
    ax.set_title("Validation Metrics by Model")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_fig(fig, figure_dir / "validation_metric_comparison_bar.png")


def _plot_roc_pr_curves(
    curve_payloads: dict[str, dict[str, np.ndarray]],
    y_val: pd.Series,
    figure_dir: Path,
) -> None:
    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, payload in curve_payloads.items():
        fpr, tpr, _ = roc_curve(y_val, payload["y_prob"])
        auc = roc_auc_score(y_val, payload["y_prob"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Validation ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _save_fig(fig, figure_dir / "validation_roc_curves.png")

    # PR
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = float((y_val == 1).mean())
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"Baseline={baseline:.3f}")
    for model_name, payload in curve_payloads.items():
        precision, recall, _ = precision_recall_curve(y_val, payload["y_prob"])
        ap = average_precision_score(y_val, payload["y_prob"])
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    ax.set_title("Validation Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    _save_fig(fig, figure_dir / "validation_pr_curves.png")


def _plot_confusion_matrices(
    y_val: pd.Series,
    curve_payloads: dict[str, dict[str, np.ndarray]],
    figure_dir: Path,
) -> None:
    model_names = list(curve_payloads.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        y_prob = curve_payloads[model_name]["y_prob"]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{model_name}\nConfusion Matrix (0.5)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)
    _save_fig(fig, figure_dir / "validation_confusion_matrices.png")


def run_step4_model_comparison(
    df: pd.DataFrame,
    outputs_dir: str | Path,
    *,
    random_state: int = 42,
) -> Step4Artifacts:
    """Train/compare candidate models on train->validation workflow.

    Selection policy:
    - Primary ranking metric: PR-AUC on validation.
    - Test split is not touched in this step.
    """
    sns.set_theme(context="notebook", style="whitegrid")
    outputs_dir = Path(outputs_dir)
    figure_dir = outputs_dir / "figures" / "step4_model_comparison"
    metrics_dir = outputs_dir / "metrics" / "step4_model_comparison"
    models_dir = outputs_dir / "models" / "step4_model_comparison"
    for p in [figure_dir, metrics_dir, models_dir]:
        p.mkdir(parents=True, exist_ok=True)

    split_assignments_path = outputs_dir / "metrics" / "step3_preprocessing" / "split_assignments.csv"
    splits = _load_splits(df, split_assignments_path)

    feature_cols = [c for c in df.columns if c not in {TARGET_COL, ID_COL}]
    X = df[feature_cols]
    y = df[TARGET_COL]
    ids = df[ID_COL]

    train_idx = splits["train"]
    val_idx = splits["validation"]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    # Sample weights for imbalance-aware training where supported.
    balanced_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model_pipelines = build_step4_models(random_state=random_state)
    results: list[dict] = []
    curve_payloads: dict[str, dict[str, np.ndarray]] = {}
    run_metadata: dict[str, dict] = {}

    for model_name, pipeline in model_pipelines.items():
        # Clone to keep model definitions immutable across runs/configs.
        estimator = clone(pipeline)
        fit_kwargs = {}
        if _supports_sample_weight(estimator):
            fit_kwargs["model__sample_weight"] = balanced_weights
        t0 = time.perf_counter()
        estimator.fit(X_train, y_train, **fit_kwargs)
        fit_seconds = time.perf_counter() - t0

        y_train_prob = estimator.predict_proba(X_train)[:, 1]
        y_val_prob = estimator.predict_proba(X_val)[:, 1]

        train_metrics = _compute_metrics(y_train, y_train_prob)
        val_metrics = _compute_metrics(y_val, y_val_prob)

        result_row = {
            "model_name": model_name,
            "fit_seconds": float(fit_seconds),
            "supports_sample_weight": bool(_supports_sample_weight(estimator)),
            "used_sample_weight": bool("model__sample_weight" in fit_kwargs),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics,
        }
        results.append(result_row)
        curve_payloads[model_name] = {"y_prob": y_val_prob}

        _save_predictions(
            metrics_dir / f"validation_predictions_{model_name}.csv",
            row_index=val_idx,
            ids=ids,
            y_true=y,
            y_prob=y_val_prob,
            split_name="validation",
        )

        joblib.dump(estimator, models_dir / f"{model_name}.joblib")

        model_obj = estimator.named_steps["model"]
        run_metadata[model_name] = {
            "model_class": type(model_obj).__name__,
            "fit_seconds": float(fit_seconds),
            "supports_sample_weight": bool(_supports_sample_weight(estimator)),
            "used_sample_weight": bool("model__sample_weight" in fit_kwargs),
            "params": model_obj.get_params(),
        }

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv(metrics_dir / "validation_model_comparison.csv", index=False)

    # Rankings table focused on validation metrics
    ranking_cols = [
        "model_name",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "f1",
        "recall",
        "precision",
        "accuracy",
        "log_loss",
        "brier_score",
        "fit_seconds",
    ]
    results_df[ranking_cols].to_csv(metrics_dir / "validation_model_ranking_table.csv", index=False)

    summary = {
        "random_state": random_state,
        "evaluation_split": "validation",
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "validation_positive_rate": float((y_val == 1).mean()),
        "primary_ranking_metric": "pr_auc",
        "best_model_by_pr_auc": str(results_df.iloc[0]["model_name"]),
        "models_compared": results_df["model_name"].tolist(),
    }
    (metrics_dir / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    (metrics_dir / "model_run_metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str))

    _plot_metric_bars(results_df[ranking_cols], figure_dir)
    _plot_roc_pr_curves(curve_payloads, y_val, figure_dir)
    _plot_confusion_matrices(y_val, curve_payloads, figure_dir)

    return Step4Artifacts(
        comparison_table_path=str(metrics_dir / "validation_model_comparison.csv"),
        metrics_dir=str(metrics_dir),
        figure_dir=str(figure_dir),
        models_dir=str(models_dir),
        best_model_by_pr_auc=str(results_df.iloc[0]["model_name"]),
        validation_metrics=results_df[ranking_cols].to_dict(orient="records"),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_default_dataset(root / "data" / "default_of_credit_card_clients_raw.xls")
    artifacts = run_step4_model_comparison(df, root / "outputs", random_state=42)
    print(
        json.dumps(
            {
                "comparison_table_path": artifacts.comparison_table_path,
                "metrics_dir": artifacts.metrics_dir,
                "figure_dir": artifacts.figure_dir,
                "models_dir": artifacts.models_dir,
                "best_model_by_pr_auc": artifacts.best_model_by_pr_auc,
                "validation_metrics": artifacts.validation_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
