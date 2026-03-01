"""Step 5 hyperparameter tuning and validation-only threshold selection.

This step tunes shortlisted models on training data, ranks configurations on
validation PR-AUC, and then selects operating thresholds on validation F1.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
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
class Step5Artifacts:
    metrics_dir: str
    figures_dir: str
    models_dir: str
    best_configs: dict
    selected_thresholds: dict
    tuned_model_ranking_path: str


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_splits(df: pd.DataFrame, split_assignments_path: Path) -> dict[str, pd.Index]:
    """Load split assignments and verify row/ID/target alignment."""
    split_df = pd.read_csv(split_assignments_path)
    if len(split_df) != len(df):
        raise ValueError("split_assignments row count does not match dataset.")
    if not (split_df["ID"].to_numpy() == df[ID_COL].to_numpy()).all():
        raise ValueError("split_assignments IDs do not align with dataset.")
    if not (split_df["target"].to_numpy() == df[TARGET_COL].to_numpy()).all():
        raise ValueError("split_assignments targets do not align with dataset.")
    return {
        "train": pd.Index(split_df.loc[split_df["split"] == "train", "row_index"].to_list()),
        "validation": pd.Index(split_df.loc[split_df["split"] == "validation", "row_index"].to_list()),
        "test": pd.Index(split_df.loc[split_df["split"] == "test", "row_index"].to_list()),
    }


def _supports_sample_weight(pipeline: Pipeline) -> bool:
    return "sample_weight" in inspect.signature(pipeline.named_steps["model"].fit).parameters


def _compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute threshold-dependent and ranking metrics from probabilities."""
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


def _make_pipeline(model_name: str, params: dict, random_state: int) -> Pipeline:
    """Build one candidate model pipeline for tuning."""
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
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def _param_grid() -> dict[str, list[dict]]:
    """Return constrained search grids for shortlisted models."""
    return {
        "hist_gradient_boosting": [
            {"learning_rate": 0.03, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 0.0},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 0.0},
            {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.0},
            {"learning_rate": 0.08, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 80, "l2_regularization": 0.1},
            {"learning_rate": 0.03, "max_iter": 500, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 0.1},
        ],
        "mlp_neural_network": [
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 256, "max_iter": 250},
            {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 256, "max_iter": 250},
            {"hidden_layer_sizes": (128,), "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 256, "max_iter": 250},
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "learning_rate_init": 1e-3, "batch_size": 256, "max_iter": 250},
            {"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "learning_rate_init": 5e-4, "batch_size": 256, "max_iter": 300},
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 5e-4, "batch_size": 128, "max_iter": 300},
        ],
    }


def _threshold_sweep(y_true: pd.Series, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Evaluate metrics across candidate thresholds on validation only."""
    rows = [_compute_metrics(y_true, y_prob, threshold=float(t)) for t in thresholds]
    return pd.DataFrame(rows)


def _plot_hyperparam_search(search_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model_name in zip(axes, ["hist_gradient_boosting", "mlp_neural_network"]):
        d = search_df[search_df["model_name"] == model_name].copy().sort_values("validation_pr_auc", ascending=False)
        d["config_rank"] = range(1, len(d) + 1)
        sns.barplot(data=d, x="config_rank", y="validation_pr_auc", ax=ax, color="#4C78A8")
        ax.set_title(f"{model_name}: Validation PR-AUC by Config")
        ax.set_xlabel("Config rank (sorted by PR-AUC)")
        ax.set_ylabel("Validation PR-AUC")
        ax.set_ylim(0, max(0.6, d["validation_pr_auc"].max() * 1.1))
    _save_fig(fig, fig_dir / "hyperparameter_search_pr_auc.png")


def _plot_threshold_curves(threshold_df: pd.DataFrame, model_name: str, fig_dir: Path) -> None:
    metrics = ["f1", "balanced_accuracy", "precision", "recall"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in metrics:
        ax.plot(threshold_df["threshold"], threshold_df[m], label=m)
    best_f1_row = threshold_df.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0]
    ax.axvline(best_f1_row["threshold"], linestyle="--", color="black", linewidth=1, label=f"best_f1={best_f1_row['threshold']:.2f}")
    ax.set_title(f"{model_name}: Validation Threshold Tuning")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    _save_fig(fig, fig_dir / f"threshold_tuning_{model_name}.png")


def _plot_tuned_vs_default(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = metrics_df.melt(
        id_vars=["model_name", "metric_set", "label"],
        value_vars=["pr_auc", "roc_auc", "balanced_accuracy", "f1", "recall", "precision"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="label", ax=ax)
    ax.set_title("Shortlisted Models: Default (0.5) vs Tuned Threshold Metrics")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    _save_fig(fig, fig_dir / "tuned_vs_default_threshold_metrics.png")


def run_step5_tuning(
    df: pd.DataFrame,
    outputs_dir: str | Path,
    *,
    random_state: int = 42,
) -> Step5Artifacts:
    """Tune shortlisted models and lock validation-selected thresholds.

    Discipline:
    - Hyperparameters selected by validation PR-AUC.
    - Threshold selected by validation F1.
    - Test set is not used.
    """
    sns.set_theme(context="notebook", style="whitegrid")
    outputs_dir = Path(outputs_dir)
    fig_dir = outputs_dir / "figures" / "step5_tuning"
    metrics_dir = outputs_dir / "metrics" / "step5_tuning"
    models_dir = outputs_dir / "models" / "step5_tuning"
    for p in [fig_dir, metrics_dir, models_dir]:
        p.mkdir(parents=True, exist_ok=True)

    splits = _load_splits(df, outputs_dir / "metrics" / "step3_preprocessing" / "split_assignments.csv")
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, ID_COL}]
    X = df[feature_cols]
    y = df[TARGET_COL]
    ids = df[ID_COL]

    train_idx = splits["train"]
    val_idx = splits["validation"]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    balanced_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    search_rows: list[dict] = []
    best_estimators: dict[str, Pipeline] = {}
    best_val_probs: dict[str, np.ndarray] = {}
    best_configs: dict[str, dict] = {}
    default_threshold_metrics_rows: list[dict] = []
    tuned_threshold_metrics_rows: list[dict] = []
    selected_thresholds: dict[str, dict] = {}

    for model_name, configs in _param_grid().items():
        # Keep a per-model best candidate by validation ranking metrics.
        best_key = None
        best_estimator_for_model = None
        best_prob_for_model = None
        best_config_for_model = None

        for config_idx, params in enumerate(configs, start=1):
            pipeline = _make_pipeline(model_name, params=params, random_state=random_state)
            fit_kwargs = {}
            if _supports_sample_weight(pipeline):
                fit_kwargs["model__sample_weight"] = balanced_weights

            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train, **fit_kwargs)
            fit_seconds = time.perf_counter() - t0

            y_train_prob = pipeline.predict_proba(X_train)[:, 1]
            y_val_prob = pipeline.predict_proba(X_val)[:, 1]
            train_metrics = _compute_metrics(y_train, y_train_prob, threshold=0.5)
            val_metrics = _compute_metrics(y_val, y_val_prob, threshold=0.5)

            row = {
                "model_name": model_name,
                "config_idx": config_idx,
                "fit_seconds": float(fit_seconds),
                "supports_sample_weight": bool(_supports_sample_weight(pipeline)),
                "used_sample_weight": bool("model__sample_weight" in fit_kwargs),
                "params_json": json.dumps(params, default=str, sort_keys=True),
                **{f"param_{k}": (str(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "threshold"},
                **{f"validation_{k}": v for k, v in val_metrics.items() if k != "threshold"},
            }
            search_rows.append(row)

            key = (
                val_metrics["pr_auc"],
                val_metrics["roc_auc"],
                val_metrics["balanced_accuracy"],
                -val_metrics["brier_score"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_estimator_for_model = clone(pipeline)
                # reuse already fit estimator to avoid refit immediately
                best_estimator_for_model = pipeline
                best_prob_for_model = y_val_prob
                best_config_for_model = params

        if best_estimator_for_model is None or best_prob_for_model is None or best_config_for_model is None:
            raise RuntimeError(f"No best estimator selected for {model_name}")

        best_estimators[model_name] = best_estimator_for_model
        best_val_probs[model_name] = best_prob_for_model
        best_configs[model_name] = best_config_for_model

    search_df = pd.DataFrame(search_rows)
    search_df = search_df.sort_values(["model_name", "validation_pr_auc"], ascending=[True, False]).reset_index(drop=True)
    search_df.to_csv(metrics_dir / "hyperparameter_search_results.csv", index=False)

    # Best config summary per shortlisted model
    best_rows = []
    threshold_compare_rows = []
    for model_name, estimator in best_estimators.items():
        y_val_prob = best_val_probs[model_name]

        default_metrics = _compute_metrics(y_val, y_val_prob, threshold=0.5)
        default_threshold_metrics_rows.append({"model_name": model_name, "metric_set": "default_0.5", **default_metrics})

        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
        # Threshold scan is performed after ranking-model selection.
        thr_df = _threshold_sweep(y_val, y_val_prob, thresholds)
        thr_df.to_csv(metrics_dir / f"threshold_sweep_{model_name}.csv", index=False)
        _plot_threshold_curves(thr_df, model_name, fig_dir)

        best_f1_row = thr_df.sort_values(["f1", "balanced_accuracy", "precision"], ascending=False).iloc[0]
        best_bal_row = thr_df.sort_values(["balanced_accuracy", "f1", "recall"], ascending=False).iloc[0]
        selected_thresholds[model_name] = {
            "threshold_selection_rule": "maximize_f1_on_validation",
            "selected_threshold_for_f1": float(best_f1_row["threshold"]),
            "best_f1_metrics": {k: float(v) for k, v in best_f1_row.to_dict().items()},
            "best_balanced_accuracy_threshold": float(best_bal_row["threshold"]),
            "best_balanced_accuracy_metrics": {k: float(v) for k, v in best_bal_row.to_dict().items()},
        }

        tuned_metrics = _compute_metrics(y_val, y_val_prob, threshold=float(best_f1_row["threshold"]))
        tuned_threshold_metrics_rows.append({"model_name": model_name, "metric_set": "tuned_threshold_f1", **tuned_metrics})

        # Save validation predictions for best config
        pred_df = pd.DataFrame(
            {
                "row_index": val_idx.to_list(),
                "ID": ids.loc[val_idx].to_list(),
                "split": "validation",
                "y_true": y_val.to_list(),
                "y_pred_proba": y_val_prob.tolist(),
                "y_pred_label_default_0p5": (y_val_prob >= 0.5).astype(int).tolist(),
                "y_pred_label_tuned_f1": (y_val_prob >= float(best_f1_row["threshold"])).astype(int).tolist(),
            }
        )
        pred_df.to_csv(metrics_dir / f"validation_predictions_best_{model_name}.csv", index=False)

        # Confusion matrices at default and tuned thresholds
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, label, thr in [
            (axes[0], "Default 0.50", 0.5),
            (axes[1], f"Tuned {float(best_f1_row['threshold']):.2f}", float(best_f1_row["threshold"])),
        ]:
            cm = confusion_matrix(y_val, (y_val_prob >= thr).astype(int), labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{model_name}\n{label}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"], rotation=0)
        _save_fig(fig, fig_dir / f"confusion_default_vs_tuned_{model_name}.png")

        joblib.dump(estimator, models_dir / f"best_{model_name}.joblib")

        best_rows.append(
            {
                "model_name": model_name,
                "selected_by": "validation_pr_auc",
                "best_params_json": json.dumps(best_configs[model_name], default=str, sort_keys=True),
                **{f"val_default0p5_{k}": v for k, v in default_metrics.items()},
                **{f"val_tunedf1_{k}": v for k, v in tuned_metrics.items()},
            }
        )

        threshold_compare_rows.extend(
            [
                {
                    "model_name": model_name,
                    "metric_set": "default_0.5",
                    "label": f"{model_name} | default 0.5",
                    **default_metrics,
                },
                {
                    "model_name": model_name,
                    "metric_set": "tuned_threshold_f1",
                    "label": f"{model_name} | tuned F1",
                    **tuned_metrics,
                },
            ]
        )

    best_tuned_df = pd.DataFrame(best_rows).sort_values("val_default0p5_pr_auc", ascending=False).reset_index(drop=True)
    best_tuned_df.to_csv(metrics_dir / "best_tuned_models_validation_summary.csv", index=False)

    threshold_metrics_df = pd.DataFrame(threshold_compare_rows)
    threshold_metrics_df.to_csv(metrics_dir / "threshold_tuning_comparison_metrics.csv", index=False)
    _plot_tuned_vs_default(threshold_metrics_df[["model_name", "metric_set", "label", "pr_auc", "roc_auc", "balanced_accuracy", "f1", "recall", "precision"]], fig_dir)

    # Comparison table for tuned shortlisted models (rank by validation PR-AUC; threshold-independent)
    tuned_ranking_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "best_params_json": json.dumps(best_configs[model_name], default=str, sort_keys=True),
                "selection_metric": "validation_pr_auc",
                "validation_pr_auc": _compute_metrics(y_val, best_val_probs[model_name], 0.5)["pr_auc"],
                "validation_roc_auc": _compute_metrics(y_val, best_val_probs[model_name], 0.5)["roc_auc"],
                "validation_default_threshold": 0.5,
                "validation_selected_threshold_f1": selected_thresholds[model_name]["selected_threshold_for_f1"],
                "validation_default_f1": _compute_metrics(y_val, best_val_probs[model_name], 0.5)["f1"],
                "validation_tuned_f1": _compute_metrics(
                    y_val, best_val_probs[model_name], selected_thresholds[model_name]["selected_threshold_for_f1"]
                )["f1"],
                "validation_default_balanced_accuracy": _compute_metrics(y_val, best_val_probs[model_name], 0.5)["balanced_accuracy"],
                "validation_tuned_balanced_accuracy": _compute_metrics(
                    y_val, best_val_probs[model_name], selected_thresholds[model_name]["selected_threshold_for_f1"]
                )["balanced_accuracy"],
            }
            for model_name in best_estimators.keys()
        ]
    ).sort_values("validation_pr_auc", ascending=False)
    tuned_ranking_path = metrics_dir / "tuned_shortlisted_model_ranking.csv"
    tuned_ranking_df.to_csv(tuned_ranking_path, index=False)

    summary = {
        "random_state": random_state,
        "tuning_split": "validation",
        "shortlisted_models": list(best_estimators.keys()),
        "hyperparameter_selection_metric": "validation_pr_auc",
        "threshold_selection_metric": "validation_f1",
        "best_configs": best_configs,
        "selected_thresholds": selected_thresholds,
        "best_model_by_validation_pr_auc_after_tuning": str(tuned_ranking_df.iloc[0]["model_name"]),
        "test_set_used": False,
    }
    (metrics_dir / "step5_tuning_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    _plot_hyperparam_search(search_df, fig_dir)

    return Step5Artifacts(
        metrics_dir=str(metrics_dir),
        figures_dir=str(fig_dir),
        models_dir=str(models_dir),
        best_configs=best_configs,
        selected_thresholds=selected_thresholds,
        tuned_model_ranking_path=str(tuned_ranking_path),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_default_dataset(root / "data" / "raw" / "default_of_credit_card_clients.xls")
    artifacts = run_step5_tuning(df, root / "outputs", random_state=42)
    print(
        json.dumps(
            {
                "metrics_dir": artifacts.metrics_dir,
                "figures_dir": artifacts.figures_dir,
                "models_dir": artifacts.models_dir,
                "best_configs": artifacts.best_configs,
                "selected_thresholds": artifacts.selected_thresholds,
                "tuned_model_ranking_path": artifacts.tuned_model_ranking_path,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
