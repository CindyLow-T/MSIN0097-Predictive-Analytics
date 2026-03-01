"""Step 2 focused EDA.

This module generates imbalance-aware EDA artefacts used to justify downstream
preprocessing and model choices. It saves all plots/tables to disk so notebook
cells can stay lightweight and fully reproducible.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TARGET_COL = "default payment next month"
CATEGORICAL_CODE_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CONTINUOUS_COLS = ["LIMIT_BAL", "AGE", *BILL_AMT_COLS, *PAY_AMT_COLS]
TEXT_COLOR = "#111111"
GRID_COLOR = "#D9D9D9"
SPINE_COLOR = "#666666"


def _save_fig(fig: plt.Figure, path: Path) -> None:
    # Force readable light theme regardless of any global dark rcParams.
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
            if legend.get_title() is not None:
                legend.get_title().set_color(TEXT_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
        # Keep subtle grid visible on white background for chart axes.
        ax.grid(True, color=GRID_COLOR, alpha=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)


def _iqr_outlier_rate(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((s < lower) | (s > upper)).mean())


def _prepare_dirs(figures_dir: Path, metrics_dir: Path) -> tuple[Path, Path]:
    """Create and return step-specific output directories."""
    fig_dir = Path(figures_dir) / "step2_eda"
    met_dir = Path(metrics_dir) / "step2_eda"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, met_dir


def run_focused_eda(
    df: pd.DataFrame,
    figures_dir: str | Path,
    metrics_dir: str | Path,
    target_col: str = TARGET_COL,
) -> dict:
    """Run focused EDA and persist figures/metrics used in the report.

    Notes:
    - This step is descriptive only (no train/validation/test usage).
    - Outputs are saved under `outputs/figures/step2_eda` and
      `outputs/metrics/step2_eda`.
    """
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
        },
    )
    fig_dir, met_dir = _prepare_dirs(Path(figures_dir), Path(metrics_dir))

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    missing_cols = [c for c in (CATEGORICAL_CODE_COLS + PAY_STATUS_COLS + CONTINUOUS_COLS) if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")

    y = df[target_col]
    n = len(df)
    positive_rate = float((y == 1).mean())
    class_counts = y.value_counts().sort_index()
    majority_class = int(class_counts.idxmax())
    majority_accuracy = float(class_counts.max() / n)

    imbalance_summary = {
        "n_samples": int(n),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "positive_rate": positive_rate,
        "majority_class": majority_class,
        "majority_accuracy": majority_accuracy,
        "baseline_pr_auc": positive_rate,
        "majority_baseline_balanced_accuracy": 0.5,
    }
    (met_dir / "class_imbalance_summary.json").write_text(json.dumps(imbalance_summary, indent=2))

    # Category code quality and target relationship
    category_tables: dict[str, pd.DataFrame] = {}
    unexpected_codes: dict[str, list[int]] = {}
    expected_code_sets = {
        "SEX": {1, 2},
        "EDUCATION": {1, 2, 3, 4},
        "MARRIAGE": {1, 2, 3},
    }
    for col in CATEGORICAL_CODE_COLS:
        grp = (
            df.groupby(col, dropna=False)[target_col]
            .agg(count="size", default_rate="mean")
            .reset_index()
            .sort_values(col)
        )
        grp["default_rate_pct"] = grp["default_rate"] * 100
        grp.to_csv(met_dir / f"{col.lower()}_default_rate.csv", index=False)
        category_tables[col] = grp
        codes = set(df[col].dropna().astype(int).unique().tolist())
        unexpected_codes[col] = sorted(codes - expected_code_sets.get(col, set()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col in zip(axes, CATEGORICAL_CODE_COLS):
        plot_df = category_tables[col]
        sns.barplot(data=plot_df, x=col, y="default_rate", ax=ax, color="#4C78A8")
        ax.set_title(f"{col}: Default Rate by Code")
        ax.set_ylabel("Default rate")
        ax.set_ylim(0, max(0.35, plot_df["default_rate"].max() * 1.15))
        for p, (_, row) in zip(ax.patches, plot_df.iterrows()):
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.005,
                f"{row['default_rate']:.1%}\n(n={int(row['count']):,})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    _save_fig(fig, fig_dir / "categorical_code_default_rates.png")

    # PAY_* status relationships with target
    pay_rate_frames = []
    for col in PAY_STATUS_COLS:
        tmp = (
            df.groupby(col)[target_col]
            .agg(count="size", default_rate="mean")
            .reset_index()
            .rename(columns={col: "status"})
        )
        tmp["pay_col"] = col
        pay_rate_frames.append(tmp)
    pay_rate_df = pd.concat(pay_rate_frames, ignore_index=True)
    pay_rate_df["status"] = pay_rate_df["status"].astype(int)
    pay_rate_df.to_csv(met_dir / "pay_status_default_rates_long.csv", index=False)

    pay_plot_df = pay_rate_df[pay_rate_df["count"] >= 50].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, col in zip(axes.flatten(), PAY_STATUS_COLS):
        d = pay_plot_df[pay_plot_df["pay_col"] == col].sort_values("status")
        sns.lineplot(data=d, x="status", y="default_rate", marker="o", ax=ax, color="#F58518")
        ax.set_title(f"{col}: status vs default rate")
        ax.set_xlabel("Repayment status code")
        ax.set_ylabel("Default rate")
        ax.set_ylim(0, min(1.0, pay_plot_df["default_rate"].max() * 1.1))
    _save_fig(fig, fig_dir / "pay_status_default_rates_by_code.png")

    # Continuous feature diagnostics (distribution, zeros, negatives, skew, outliers)
    continuous_diag = pd.DataFrame(index=CONTINUOUS_COLS)
    continuous_diag["mean"] = df[CONTINUOUS_COLS].mean()
    continuous_diag["median"] = df[CONTINUOUS_COLS].median()
    continuous_diag["std"] = df[CONTINUOUS_COLS].std()
    continuous_diag["min"] = df[CONTINUOUS_COLS].min()
    continuous_diag["max"] = df[CONTINUOUS_COLS].max()
    continuous_diag["skew"] = df[CONTINUOUS_COLS].skew()
    continuous_diag["zero_rate"] = (df[CONTINUOUS_COLS] == 0).mean()
    continuous_diag["negative_rate"] = (df[CONTINUOUS_COLS] < 0).mean()
    continuous_diag["iqr_outlier_rate"] = [ _iqr_outlier_rate(df[c]) for c in CONTINUOUS_COLS ]
    continuous_diag.rename_axis("feature").reset_index().to_csv(
        met_dir / "continuous_feature_diagnostics.csv", index=False
    )

    outlier_plot_df = (
        continuous_diag.rename_axis("feature").reset_index()
        .sort_values("iqr_outlier_rate", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=outlier_plot_df, x="iqr_outlier_rate", y="feature", ax=ax, color="#54A24B")
    ax.set_title("Top IQR Outlier Rates (Continuous Features)")
    ax.set_xlabel("Outlier rate")
    ax.set_ylabel("")
    for p, (_, row) in zip(ax.patches, outlier_plot_df.iterrows()):
        ax.text(p.get_width() + 0.003, p.get_y() + p.get_height() / 2, f"{row['iqr_outlier_rate']:.1%}", va="center", fontsize=8)
    _save_fig(fig, fig_dir / "top_iqr_outlier_rates.png")

    # Example feature distributions (money features are log1p-transformed for visualization only)
    plot_features = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), plot_features):
        plot_df = df[[col, target_col]].copy()
        if col in {"LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"}:
            plot_df[f"{col}_plot"] = np.log1p(plot_df[col].clip(lower=0))
            x_col = f"{col}_plot"
            xlabel = f"log1p({col})"
        else:
            x_col = col
            xlabel = col
        sns.histplot(
            data=plot_df,
            x=x_col,
            hue=target_col,
            stat="density",
            common_norm=False,
            bins=40,
            element="step",
            fill=False,
            ax=ax,
        )
        ax.set_title(f"Distribution by Target: {col}")
        ax.set_xlabel(xlabel)
    _save_fig(fig, fig_dir / "feature_distributions_by_target.png")

    # Target-wise boxplots (clip to 99th percentile for readability)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), plot_features):
        plot_df = df[[col, target_col]].copy()
        upper = plot_df[col].quantile(0.99)
        lower = plot_df[col].quantile(0.01) if col in {"AGE"} else None
        plot_df[col] = plot_df[col].clip(lower=lower, upper=upper)
        sns.boxplot(data=plot_df, x=target_col, y=col, ax=ax, color="#72B7B2", showfliers=False)
        ax.set_title(f"{col} by Target (clipped for display)")
        ax.set_xlabel(target_col)
    _save_fig(fig, fig_dir / "boxplots_by_target.png")

    # Target-wise summary for continuous features
    cont_target_summary = (
        df.groupby(target_col)[CONTINUOUS_COLS]
        .agg(["mean", "median"])
        .T
        .reset_index()
    )
    cont_target_summary.columns = [
        "feature",
        "stat",
        "target_0",
        "target_1",
    ]
    cont_target_summary.to_csv(met_dir / "continuous_summary_by_target.csv", index=False)

    # Correlations with target and heatmap for top features
    corr = df.drop(columns=["ID"], errors="ignore").corr(numeric_only=True)
    target_corr = (
        corr[target_col]
        .drop(labels=[target_col])
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("corr_with_target")
        .rename_axis("feature")
        .reset_index()
    )
    target_corr.to_csv(met_dir / "feature_target_correlations.csv", index=False)

    top_heatmap_features = target_corr.head(12)["feature"].tolist()
    heatmap_cols = top_heatmap_features + [target_col]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr.loc[heatmap_cols, heatmap_cols],
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        annot=False,
        fmt=".2f",
    )
    ax.set_title("Correlation Heatmap (Top |corr| Features with Target)")
    _save_fig(fig, fig_dir / "correlation_heatmap_top_features.png")

    # Compact summary JSON for notebook and reporting
    insights = {
        "class_imbalance": imbalance_summary,
        "unexpected_category_codes": unexpected_codes,
        "top_positive_correlations": target_corr.head(10).to_dict(orient="records"),
        "continuous_high_skew_features": (
            continuous_diag.rename_axis("feature").reset_index()
            .assign(abs_skew=lambda d: d["skew"].abs())
            .sort_values("abs_skew", ascending=False)
            .head(10)[["feature", "skew", "zero_rate", "negative_rate", "iqr_outlier_rate"]]
            .to_dict(orient="records")
        ),
    }
    (met_dir / "eda_insights_summary.json").write_text(json.dumps(insights, indent=2))

    return {
        "figure_dir": str(fig_dir),
        "metrics_dir": str(met_dir),
        "class_imbalance_summary": imbalance_summary,
        "unexpected_category_codes": unexpected_codes,
        "top_correlations": target_corr.head(10).to_dict(orient="records"),
    }


def load_default_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the UCI credit-default spreadsheet using the second row header."""
    return pd.read_excel(Path(data_path), header=1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "default_of_credit_card_clients.xls"
    outputs_dir = root / "outputs"
    df = load_default_dataset(data_path)
    result = run_focused_eda(df, outputs_dir / "figures", outputs_dir / "metrics")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
