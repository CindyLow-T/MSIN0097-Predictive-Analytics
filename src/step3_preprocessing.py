"""Step 3 preprocessing and split discipline.

This module owns:
- stratified train/validation/test splitting,
- preprocessing pipeline construction, and
- persistence of split/preprocessor artefacts for downstream steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


TARGET_COL = "default payment next month"
ID_COL = "ID"

CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


@dataclass
class Step3Artifacts:
    split_summary: dict
    preprocessing_summary: dict
    feature_names: list[str]
    split_assignment_path: str
    preprocessor_path: str
    output_dir: str


def load_default_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the UCI credit-default spreadsheet using the second row header."""
    return pd.read_excel(Path(data_path), header=1)


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the shared preprocessing transformer used by all model steps.

    Design choices:
    - Numeric: median imputation + robust scaling.
    - Categorical/code columns: most-frequent imputation + one-hot encoding.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
    val_size: float = 0.20,
) -> dict[str, pd.Index]:
    """Create stratified 60/20/20-style index splits with fixed random_state."""
    if not (0 < test_size < 1 and 0 < val_size < 1):
        raise ValueError("test_size and val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    train_val_idx, test_idx = train_test_split(
        X.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y.loc[X.index],
    )

    # Derive validation fraction relative to the remaining train+val pool.
    val_relative = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=random_state,
        stratify=y.loc[train_val_idx],
    )

    return {
        "train_idx": pd.Index(train_idx),
        "val_idx": pd.Index(val_idx),
        "test_idx": pd.Index(test_idx),
    }


def _class_distribution(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts().sort_index()
    return {
        "count_0": int(counts.get(0, 0)),
        "count_1": int(counts.get(1, 0)),
        "positive_rate": float((series == 1).mean()),
    }


def _validate_feature_lists(df: pd.DataFrame) -> None:
    """Ensure expected columns exist and feature groups do not overlap."""
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, ID_COL])
    missing = sorted(c for c in expected if c not in df.columns)
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    overlap = set(NUMERIC_FEATURES).intersection(CATEGORICAL_FEATURES)
    if overlap:
        raise ValueError(f"Overlapping feature assignments: {sorted(overlap)}")


def run_step3_preprocessing(
    df: pd.DataFrame,
    outputs_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
    val_size: float = 0.20,
) -> Step3Artifacts:
    """Run Step 3 and persist reproducible split/preprocessor artefacts.

    Leakage control:
    - Split first.
    - Fit preprocessor on train only.
    - Apply the fitted transformer to validation/test unchanged.
    """
    _validate_feature_lists(df)

    outputs_dir = Path(outputs_dir)
    fig_dir = outputs_dir / "figures" / "step3_preprocessing"
    metrics_dir = outputs_dir / "metrics" / "step3_preprocessing"
    models_dir = outputs_dir / "models" / "step3_preprocessing"
    for p in [fig_dir, metrics_dir, models_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Exclude target and ID from model features; keep original row index for reproducible splits.
    modeling_feature_cols = [c for c in df.columns if c not in {TARGET_COL, ID_COL}]
    X = df[modeling_feature_cols].copy()
    y = df[TARGET_COL].copy()

    splits = stratified_train_val_test_split(
        X,
        y,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
    )

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # Build and fit preprocessor on train only to prevent leakage.
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    transformed_mats = {"train": X_train_t, "val": X_val_t, "test": X_test_t}
    transformed_shapes = {k: [int(v.shape[0]), int(v.shape[1])] for k, v in transformed_mats.items()}
    nan_counts = {
        k: int(np.isnan(np.asarray(v)).sum())
        for k, v in transformed_mats.items()
    }

    if X_train.shape[1] != len(modeling_feature_cols):
        raise RuntimeError("Unexpected feature count mismatch before preprocessing.")
    if any(count > 0 for count in nan_counts.values()):
        raise RuntimeError(f"NaNs found after preprocessing transform: {nan_counts}")

    split_assignments = pd.DataFrame(
        {
            "row_index": df.index,
            "ID": df[ID_COL].values,
            "target": y.values,
            "split": "unassigned",
        }
    )
    split_assignments.loc[train_idx, "split"] = "train"
    split_assignments.loc[val_idx, "split"] = "validation"
    split_assignments.loc[test_idx, "split"] = "test"
    if (split_assignments["split"] == "unassigned").any():
        raise RuntimeError("Some rows were not assigned to a split.")
    split_assignment_path = metrics_dir / "split_assignments.csv"
    split_assignments.to_csv(split_assignment_path, index=False)

    split_summary = {
        "random_state": random_state,
        "test_size": test_size,
        "val_size": val_size,
        "train_size": float(1 - test_size - val_size),
        "n_total": int(len(df)),
        "n_features_raw_model_input": int(len(modeling_feature_cols)),
        "id_excluded_from_modeling": True,
        "target_column": TARGET_COL,
        "feature_columns_used": modeling_feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_distribution": {
            "overall": _class_distribution(y),
            "train": _class_distribution(y_train),
            "validation": _class_distribution(y_val),
            "test": _class_distribution(y_test),
        },
    }
    (metrics_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2))

    feature_names_path = metrics_dir / "preprocessor_feature_names.txt"
    feature_names_path.write_text("\n".join(feature_names))

    # Save a compact transformed feature sample for sanity checking (train only).
    sample_n = min(5, X_train_t.shape[0])
    sample_df = pd.DataFrame(
        np.asarray(X_train_t[:sample_n]),
        columns=feature_names,
    )
    sample_df.insert(0, "row_index", X_train.index[:sample_n].to_list())
    sample_df.to_csv(metrics_dir / "transformed_train_sample.csv", index=False)

    preprocessing_summary = {
        "preprocessor": {
            "numeric_pipeline": ["SimpleImputer(strategy='median')", "RobustScaler()"],
            "categorical_pipeline": ["SimpleImputer(strategy='most_frequent')", "OneHotEncoder(handle_unknown='ignore')"],
        },
        "transformed_shapes": transformed_shapes,
        "transformed_feature_count": int(len(feature_names)),
        "nan_counts_after_transform": nan_counts,
        "train_only_fit": True,
        "feature_names_file": str(feature_names_path),
    }
    (metrics_dir / "preprocessing_summary.json").write_text(json.dumps(preprocessing_summary, indent=2))

    # Persist fitted preprocessor for downstream modeling.
    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    return Step3Artifacts(
        split_summary=split_summary,
        preprocessing_summary=preprocessing_summary,
        feature_names=feature_names,
        split_assignment_path=str(split_assignment_path),
        preprocessor_path=str(preprocessor_path),
        output_dir=str(metrics_dir),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "default_of_credit_card_clients.xls"
    outputs_dir = root / "outputs"
    df = load_default_dataset(data_path)
    result = run_step3_preprocessing(df, outputs_dir)
    print(
        json.dumps(
            {
                "split_summary": result.split_summary,
                "preprocessing_summary": result.preprocessing_summary,
                "split_assignment_path": result.split_assignment_path,
                "preprocessor_path": result.preprocessor_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
