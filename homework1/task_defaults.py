import os
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent.absolute() / 'results'
CLEAR_DATA_ROOT = Path(__file__).parent.absolute() / 'clear_data'
os.makedirs(RESULTS_ROOT, exist_ok=True)
