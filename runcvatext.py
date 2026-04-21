#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:50:57 2026

@author: nhunterjr
"""
import cva_run as run

from run import CVAConfig

cfg = CVAConfig(
    lags        = 32,    # ~12.5 ms at 2560 Hz, ~5 cycles at 400 Hz
    tol         = 1e-4,
    neighbors   = 800,
    eval_stride = 5,
    fs          = 2560.0,
    freq_min    = 50.0,
    freq_max    = 800.0,
)

run.plot_signifificance()