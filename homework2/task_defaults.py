import os
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent.absolute() / 'results'
os.makedirs(RESULTS_ROOT, exist_ok=True)
