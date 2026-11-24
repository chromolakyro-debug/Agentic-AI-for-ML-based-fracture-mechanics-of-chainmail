"""Isolated continuation run; original model, sampler and FEM remain untouched."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIGINAL = ROOT / 'cfm_v0' / 'run'
RUN = HERE / 'run'
os.environ['CFM_V0_RUN_DIR'] = str(RUN)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
sys.path.insert(0, str(ROOT / 'cfm_v0'))

