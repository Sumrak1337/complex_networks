import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
RESULTS_ROOT = PROJECT_ROOT / 'results'
DATA_ROOT = PROJECT_ROOT / 'data'
os.makedirs(RESULTS_ROOT, exist_ok=True)
