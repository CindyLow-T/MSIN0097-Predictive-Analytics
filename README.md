# Credit Default Prediction (Validation-Disciplined ML Pipeline)

This project builds a reproducible binary classification pipeline for predicting `default payment next month` using the UCI credit card default dataset.

## Project Goal

- Predict short-term credit default risk (`1` = default, `0` = non-default)
- Use strict train/validation/test discipline
- Compare models fairly, tune on validation only, and evaluate test once under a locked final configuration

## Main Notebook

- `notebooks/predictive_analytics_credit_default_final.ipynb` (latest coursework notebook with diagnostics)

## Quick Start

1. Create/activate Python environment.
2. Install dependencies (choose one):
   - Reproducible (pinned): `pip install -r requirements-lock.txt`
   - Flexible (minimum versions): `pip install -r requirements.txt`
3. Open and run:
   - `notebooks/predictive_analytics_credit_default_final.ipynb`
4. In Jupyter: `Kernel -> Restart Kernel and Run All Cells` (from-scratch run).

## Data

- Expected raw file path:
  - `data/raw/default_of_credit_card_clients.xls`
- Data source:
  - UCI Machine Learning Repository: Default of Credit Card Clients (Yeh & Lien, 2009)
  - URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- If the file is missing, download the dataset and place the Excel file at:
  - `data/raw/default_of_credit_card_clients.xls`

## Source Modules (`src/`)

- `step2_focused_eda.py`: focused EDA, class imbalance, feature diagnostics
- `step3_preprocessing.py`: stratified split + preprocessing pipeline
- `step4_model_comparison.py`: baseline/candidate model comparison on validation
- `step5_tuning.py`: hyperparameter tuning + threshold optimization (validation only)
- `step5_diagnostics.py`: additional validation diagnostics (calibration, failure modes, training curves)
- `step6_final_evaluation.py`: locked final model evaluation (one-time test protocol)
- `project_paths.py`: shared path constants

## Outputs

All generated artifacts are saved under `outputs/`:

- `outputs/figures/`: plots by workflow step
- `outputs/metrics/`: tables, JSON summaries, prediction exports
- `outputs/models/`: fitted preprocessors and model pipelines

Step-specific folders:

- `outputs/metrics/step2_eda`, `outputs/figures/step2_eda`
- `outputs/metrics/step3_preprocessing`, `outputs/models/step3_preprocessing`
- `outputs/metrics/step4_model_comparison`, `outputs/figures/step4_model_comparison`, `outputs/models/step4_model_comparison`
- `outputs/metrics/step5_tuning`, `outputs/figures/step5_tuning`, `outputs/models/step5_tuning`
- `outputs/metrics/step5_diagnostics`, `outputs/figures/step5_diagnostics`
- `outputs/metrics/step6_final_evaluation`, `outputs/figures/step6_final_evaluation`, `outputs/models/step6_final_evaluation`

## Final Locked Configuration (Current)

- Model: `HistGradientBoostingClassifier`
- Threshold: `0.57` (selected on validation via F1)
- Final test metrics summary:
  - `outputs/metrics/step6_final_evaluation/final_test_evaluation_summary.json`

## Reproducibility Notes

- Fixed random state used throughout (`42`)
- Preprocessing fit on training data only
- Validation used for model selection/tuning
- Test set evaluated once for the locked final model configuration
- Paths are project-root relative in code (no machine-specific hardcoded paths in `src/`)

## Notes

- Agent-assisted development was used for implementation speed, with manual verification of methodology and leakage controls.
