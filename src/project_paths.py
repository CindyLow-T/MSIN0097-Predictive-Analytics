from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
METRICS_DIR = OUTPUTS_DIR / 'metrics'
MODELS_DIR = OUTPUTS_DIR / 'models'
for _p in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
