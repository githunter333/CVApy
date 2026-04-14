#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:51:09 2026

@author: nhunterjr
"""

"""
run_cva_pipeline.py
===================
Complete CVA pipeline for a single state data file.

Runs:
  1. CVARun        — load data, build database, run CVA, eigenvalue significance
  2. CVAPhiCluster — PCA, elbow detection, k-means clustering
  3. CVAModeShapes — physical mode shape extraction and plotting

All results are saved to HDF5 files in OUTPUT_DIR, named automatically
from the input filename (e.g. s16d240.txt -> s16d240_cva.h5 etc).

================================================================
QUICK START — running a new state
================================================================
1. Set DATA_FILE to the full path of your text data file
2. Set OUTPUT_DIR to where you want HDF5 results saved
3. Run the script — all three steps run automatically

================================================================
DATA FILE FORMAT
================================================================
Whitespace- or comma-delimited text, one row per sample:
  col 1       : time (s)
  col 2       : force input
  cols 3..N+2 : accelerometer outputs (N channels)

================================================================
KEY PARAMETERS TO TUNE
================================================================
LAGS        : time lags for Hankel embedding
                16  = fast, lower frequency resolution
                64  = slower, better mode separation
              Start with 16 to check results, then try 64.

TOL         : neighbor search tolerance
                1e-3 = loose (more neighbors, faster, less selective)
                1e-4 = recommended starting point
                5e-5 = tight (fewer neighbors, slower)

NEIGHBORS   : max neighbors to use
                200  = fast
                960  = better conditioned, use with 64 lags

EVAL_STRIDE : stride between evaluated windows
                1    = every window (most accurate, slowest)
                10   = 10x faster, good for initial exploration

N_CLUSTERS  : number of k-means clusters
                0    = auto-detect via elbow method (recommended)
                2-4  = manual override if elbow is ambiguous

MIN_IM_FRAC : filter for complex eigenvalues
                0.1  = strict (fewer, cleaner modes)
                0.05 = relaxed (more modes including higher freq)

N_MODES_MAX : max modes shown per cluster in plots
                6    = default, fast plots
                20   = show full frequency range

================================================================
RELOADING SAVED RESULTS (no recomputation)
================================================================
Set any combination of these to False to skip recomputation
and load from the saved HDF5 file instead:

  RUN_CVA     = False   # load from s??d??_cva.h5
  RUN_CLUSTER = False   # load from s??d??_clusters.h5
  RUN_MODES   = False   # load from s??d??_modes.h5

Useful for:
  - Adjusting plot settings without rerunning CVA
  - Trying different N_CLUSTERS on a completed run
  - Replotting after changing N_MODES_MAX or MIN_IM_FRAC

================================================================
REQUIRED FILES (must be in CVAPY_DIR)
================================================================
  cva_core.py         — cvaonestep, delaybuild, reduce_matrix
  cva_run.py          — CVAConfig, CVARun
  cva_phi_cluster.py  — CVAPhiCluster
  cva_mode_shapes.py  — CVAModeShapes

================================================================
Author: Norman / Claude
Date:   2026-04-13
"""

import sys
import os
from pathlib import Path

# ---- ensure CVApy directory is on path ----
CVAPY_DIR = Path('/Users/nhunterjr/Documents/Code/python/CVApy')
if str(CVAPY_DIR) not in sys.path:
    sys.path.insert(0, str(CVAPY_DIR))

from cva_run         import CVAConfig, CVARun
from cva_phi_cluster import CVAPhiCluster
from cva_mode_shapes import CVAModeShapes

# =============================================================================
# ===== USER SETTINGS — edit these =====
# =============================================================================

# Input data file
DATA_FILE = '/Users/nhunterjr/Documents/Code/shared/Nonlinear Data/Experimental_Data/r6dof/cva_formatted/r6dofcubic2_cva.txt'
# Output directory for HDF5 results
OUTPUT_DIR = Path('/Users/nhunterjr/Documents/Code/python/CVApy/results')

# --- CVA parameters ---
LAGS        = 16        # time lags (16 = fast, 64 = better resolution)
TOL         = 1e-4      # neighbor tolerance
NEIGHBORS   = 200       # max neighbors
RANKLOCK    = 0         # 0 = auto rank
EVAL_STRIDE = 1         # 1 = every window, higher = faster but coarser
FS          = 2048.0    # sample rate Hz

# --- Clustering parameters ---
K_MAX        = 8        # max clusters to try in elbow sweep
N_PCA_VAR    = 95.0     # % variance to retain in PCA
KMEANS_REP   = 20       # k-means replicates
N_CLUSTERS   = 0        # 0 = auto-detect, >0 = override

# --- Mode shape parameters ---
MIN_IM_FRAC  = 0.05     # min |Im(ev)|/|ev| for complex mode filter
F_MIN        = 0.5      # min frequency Hz
ZETA_MAX     = 0.5      # max damping ratio
N_MODES_MAX  = 12       # max modes to plot per cluster

# --- What to run ---
RUN_CVA         = True   # run CVA (False = load from saved file)
RUN_CLUSTER     = True   # run clustering
RUN_MODES       = True   # run mode shape extraction

# --- What to plot ---
PLOT_SIGNIFICANCE = True
PLOT_CLUSTERS     = True
PLOT_MODES        = True

# =============================================================================
# ===== SETUP =====
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Derive output filenames from input data filename
stem     = Path(DATA_FILE).stem                  # e.g. 's01d11'
h5_run   = OUTPUT_DIR / f'{stem}_cva.h5'
h5_clust = OUTPUT_DIR / f'{stem}_clusters.h5'
h5_modes = OUTPUT_DIR / f'{stem}_modes.h5'

print('=' * 60)
print(f'  CVA Pipeline — {stem}')
print('=' * 60)
print(f'  Data     : {DATA_FILE}')
print(f'  Output   : {OUTPUT_DIR}')
print(f'  lags={LAGS}  tol={TOL:.0e}  neighbors={NEIGHBORS}')
print('=' * 60)

# =============================================================================
# ===== STEP 1: CVA RUN =====
# =============================================================================

if RUN_CVA:
    print('\n--- Step 1: CVA Run ---')
    cfg = CVAConfig(
        lags        = LAGS,
        tol         = TOL,
        neighbors   = NEIGHBORS,
        ranklock    = RANKLOCK,
        eval_stride = EVAL_STRIDE,
        fs          = FS,
    )
    run = CVARun(cfg)
    run.load_data(DATA_FILE)
    run.build_database()
    run.run()
    run.eigenvalue_significance()
    run.save(str(h5_run))
    print(f'  Saved: {h5_run}')
else:
    print(f'\n--- Step 1: Loading CVA Run from {h5_run} ---')
    run = CVARun.load(str(h5_run))

print(run)

if PLOT_SIGNIFICANCE:
    run.plot_significance()

if PLOT_SIGNIFICANCE:
    run.plot_significance()
    run.plot_reconstruction(channel=0)
# =============================================================================
# ===== STEP 2: PHI CLUSTERING =====
# =============================================================================

if RUN_CLUSTER:
    print('\n--- Step 2: Phi Clustering ---')
    clust = CVAPhiCluster(
        run,
        k_max       = K_MAX,
        n_pca_var   = N_PCA_VAR,
        kmeans_rep  = KMEANS_REP,
    )
    clust.fit(n_clusters=N_CLUSTERS)
    clust.save(str(h5_clust))
    print(f'  Saved: {h5_clust}')
else:
    print(f'\n--- Step 2: Loading Clusters from {h5_clust} ---')
    clust = CVAPhiCluster.load(str(h5_clust), run)

print(clust)

if PLOT_CLUSTERS:
    clust.plot_all()

# =============================================================================
# ===== STEP 3: MODE SHAPES =====
# =============================================================================

if RUN_MODES:
    print('\n--- Step 3: Mode Shape Extraction ---')
    ms = CVAModeShapes(
        run,
        clust,
        min_im_frac = MIN_IM_FRAC,
        f_min       = F_MIN,
        zeta_max    = ZETA_MAX,
        n_modes_max = N_MODES_MAX,
    )
    ms.extract()
    ms.save(str(h5_modes))
    print(f'  Saved: {h5_modes}')
else:
    print(f'\n--- Step 3: Loading Mode Shapes from {h5_modes} ---')
    ms = CVAModeShapes.load(str(h5_modes), run, clust)

print(ms)

if PLOT_MODES:
    ms.plot_all()

# =============================================================================
# ===== SUMMARY =====
# =============================================================================

print('\n' + '=' * 60)
print(f'  Pipeline complete — {stem}')
print('=' * 60)
print(f'  CVA run   : {h5_run}')
print(f'  Clusters  : {h5_clust}')
print(f'  Modes     : {h5_modes}')

if clust.cluster_ids is not None:
    import numpy as np
    sizes = np.bincount(clust.cluster_ids)
    print(f'\n  Clusters  : {clust.n_clusters}  (regime A={clust.regime_A}, B={clust.regime_B})')
    for k, s in enumerate(sizes):
        tag = ' <- A' if k == clust.regime_A else (' <- B' if k == clust.regime_B else '')
        print(f'    Cluster {k}: {s} windows{tag}')

if ms.cluster_modes:
    print('\n  Modes per cluster:')
    for k, cm in enumerate(ms.cluster_modes):
        if cm is None:
            print(f'    Cluster {k}: no modes')
        else:
            print(f'    Cluster {k}: {cm.shapes.shape[1]} modes  '
                  f'({cm.freq[0]:.1f}–{cm.freq[-1]:.1f} Hz)')

print('\nDone.')