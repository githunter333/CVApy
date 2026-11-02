#!/usr/bin/env python3
"""
runcvatext.py
=============
Minimal single-stage example: run CVA on one file and plot the eigenvalue
significance analysis. For the full three-stage pipeline use
run_cva_pipeline.py, or launch the GUI with cva_gui.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cva_run import CVAConfig, CVARun

DATA_FILE = Path(__file__).resolve().parent / 'bltimedatashort.txt'

cfg = CVAConfig(
    lags        = 32,      # ~12.5 ms at 2560 Hz, ~5 cycles at 400 Hz
    tol         = 1e-4,
    neighbors   = 800,
    eval_stride = 5,
    fs          = 2560.0,
    freq_min    = 50.0,
    freq_max    = 800.0,
)

run = CVARun(cfg)
run.load_data(str(DATA_FILE), col_time=0, col_force=1, col_accel_start=2, n_in=1)
run.build_database()
run.run(n_jobs=1)
run.eigenvalue_significance()
run.plot_significance()
