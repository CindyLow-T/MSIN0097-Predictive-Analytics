from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
# Backward-compatible alias name used by older code paths.
DATA_RAW_DIR = DATA_DIR
DATA_RAW_PATH = DATA_DIR / 'default_of_credit_card_clients_raw.xls'
DATA_PROCESSED_PATH = DATA_DIR / 'default_of_credit_card_clients_processed.csv'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
METRICS_DIR = OUTPUTS_DIR / 'metrics'
MODELS_DIR = OUTPUTS_DIR / 'models'
for _p in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
