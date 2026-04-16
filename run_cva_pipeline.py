"""
run_cva_pipeline.py
===================
Complete CVA pipeline for a single state data file.

Runs:
  1. CVARun        — load data, build database, run CVA, eigenvalue significance
  2. CVAPhiCluster — PCA, elbow detection, k-means clustering
  3. CVAModeShapes — physical mode shape extraction and plotting

Results saved to:
  RESULTS_ROOT / <stem> / <stem>_cva.h5
  RESULTS_ROOT / <stem> / <stem>_clusters.h5
  RESULTS_ROOT / <stem> / <stem>_modes.h5
  RESULTS_ROOT / <stem> / figures / *.png

RESULTS_ROOT defaults to DEVONthink watched folder for auto-indexing.

================================================================
QUICK START
================================================================
1. Set DATA_FILE to your text data file path
2. Run — all three steps execute automatically
3. Results appear in RESULTS_ROOT/<stem>/

================================================================
DATA FILE FORMAT
================================================================
Whitespace- or comma-delimited text:
  col 1       : time (s)
  col 2       : force input
  cols 3..N+2 : accelerometer outputs

================================================================
KEY PARAMETERS
================================================================
LAGS        : 16=fast  64=better resolution
TOL         : 1e-3=loose  1e-4=recommended  5e-5=tight
NEIGHBORS   : 200=fast  960=better conditioned
EVAL_STRIDE : 1=every window  10=10x faster
N_CLUSTERS  : 0=auto-detect  >0=override
MIN_IM_FRAC : 0.1=strict  0.05=more modes
N_MODES_MAX : max modes plotted per cluster

================================================================
RELOAD WITHOUT RECOMPUTING
================================================================
Set False to load from saved HDF5:
  RUN_CVA = False  /  RUN_CLUSTER = False  /  RUN_MODES = False

================================================================
Author: Norman / Claude
Date:   2026-04-14
"""

import sys
import numpy as np
from pathlib import Path

CVAPY_DIR = Path('/Users/nhunterjr/Documents/Code/python/CVApy')
if str(CVAPY_DIR) not in sys.path:
    sys.path.insert(0, str(CVAPY_DIR))

from cva_run         import CVAConfig, CVARun
from cva_phi_cluster import CVAPhiCluster
from cva_mode_shapes import CVAModeShapes

# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_FILE = '/Users/nhunterjr/Documents/Code/matlab/CVA/experiments/formatted input data/dumbbell/dumbbellringdown.txt'
# Save settings
RESULTS_ROOT = Path('/Users/nhunterjr/Documents/DEVONthink/CVA_Results')
SAVE_PLOTS   = True
PLOT_DPI     = 150

# CVA parameters
LAGS        = 64
TOL         = 1e-4
NEIGHBORS   = 100
RANKLOCK    = 0
EVAL_STRIDE = 1
FS          = 2048.0

# Clustering
K_MAX        = 8
N_PCA_VAR    = 95.0
KMEANS_REP   = 20
N_CLUSTERS   = 0

# Frequency band filter — applied to both clustering and mode shapes
# Set F_MIN / F_MAX to None to disable (uses default 0.5 Hz lower bound only)
F_MIN        = 40  # Hz  lower bound  (e.g. 5.0 to strip near-DC)
F_MAX        = 500 # Hz  upper bound  (e.g. 100.0 for bookshelf data)

# Mode shapes
MIN_IM_FRAC  = 0.05
ZETA_MAX     = 0.5
N_MODES_MAX  = 12

# Run flags
RUN_CVA     = True
RUN_CLUSTER = True
RUN_MODES   = True

# Plot flags
PLOT_SIGNIFICANCE   = True
PLOT_RECONSTRUCTION = True
PLOT_CLUSTERS       = True
PLOT_MODES          = True

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

stem    = Path(DATA_FILE).stem
run_dir = RESULTS_ROOT / stem
fig_dir = run_dir / 'figures'
run_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

h5_run   = run_dir / f'{stem}_cva.h5'
h5_clust = run_dir / f'{stem}_clusters.h5'
h5_modes = run_dir / f'{stem}_modes.h5'

def figpath(name):
    return fig_dir / f'{name}.png' if SAVE_PLOTS else None

print('=' * 60)
print(f'  CVA Pipeline — {stem}')
print('=' * 60)
print(f'  Data    : {DATA_FILE}')
print(f'  Results : {run_dir}')
print(f'  Figures : {"saved to " + str(fig_dir) if SAVE_PLOTS else "display only"}')
print(f'  lags={LAGS}  tol={TOL:.0e}  neighbors={NEIGHBORS}')
print('=' * 60)

# =============================================================================
# STEP 1: CVA RUN
# =============================================================================

if RUN_CVA:
    print('\n--- Step 1: CVA Run ---')
    cfg = CVAConfig(lags=LAGS, tol=TOL, neighbors=NEIGHBORS,
                    ranklock=RANKLOCK, eval_stride=EVAL_STRIDE, fs=FS)
    run = CVARun(cfg)
    run.load_data(DATA_FILE)
    run.build_database()
    run.run()
    run.eigenvalue_significance()
    run.save(str(h5_run))
    print(f'  Saved: {h5_run}')
else:
    print(f'\n--- Step 1: Loading from {h5_run.name} ---')
    run = CVARun.load(str(h5_run))

print(run)

if PLOT_SIGNIFICANCE:
    run.plot_significance(save_path=figpath('significance'), dpi=PLOT_DPI)

if PLOT_RECONSTRUCTION:
    run.plot_reconstruction(channel=0,
                            save_path=figpath('reconstruction_ch1'),
                            dpi=PLOT_DPI)

# =============================================================================
# STEP 2: PHI CLUSTERING
# =============================================================================

if RUN_CLUSTER:
    print('\n--- Step 2: Phi Clustering ---')
    clust = CVAPhiCluster(run, k_max=K_MAX,
                          n_pca_var=N_PCA_VAR, kmeans_rep=KMEANS_REP,
                          f_min=F_MIN, f_max=F_MAX)
    clust.fit(n_clusters=N_CLUSTERS)
    clust.save(str(h5_clust))
    print(f'  Saved: {h5_clust}')
else:
    print(f'\n--- Step 2: Loading from {h5_clust.name} ---')
    clust = CVAPhiCluster.load(str(h5_clust), run)

print(clust)

if PLOT_CLUSTERS:
    clust.plot_pca_variance(save_path=figpath('clusters_pca_variance'),
                            dpi=PLOT_DPI)
    clust.plot_elbow(save_path=figpath('clusters_elbow'), dpi=PLOT_DPI)
    clust.plot_regime_timeseries(save_path=figpath('clusters_timeseries'),
                                 dpi=PLOT_DPI)
    clust.plot_eigenvalues_zplane(save_path=figpath('clusters_zplane'),
                                  dpi=PLOT_DPI)
    clust.plot_cluster_amplitude(save_path=figpath('clusters_amplitude'),
                                 dpi=PLOT_DPI)
    clust.plot_pca_scatter(save_path=figpath('clusters_pca_scatter'),
                           dpi=PLOT_DPI)

# =============================================================================
# STEP 3: MODE SHAPES
# =============================================================================

if RUN_MODES:
    print('\n--- Step 3: Mode Shape Extraction ---')
    ms = CVAModeShapes(run, clust, min_im_frac=MIN_IM_FRAC,
                       f_min=F_MIN, zeta_max=ZETA_MAX,
                       n_modes_max=N_MODES_MAX)
    ms.extract()
    ms.save(str(h5_modes))
    print(f'  Saved: {h5_modes}')
else:
    print(f'\n--- Step 3: Loading from {h5_modes.name} ---')
    ms = CVAModeShapes.load(str(h5_modes), run, clust)

print(ms)

if PLOT_MODES:
    ms.plot_regime_comparison(
        save_path=figpath('modes_regime_comparison'), dpi=PLOT_DPI)
    ms.plot_per_cluster(
        save_path=figpath('modes_per_cluster'), dpi=PLOT_DPI)
    ms.plot_magnitude_overlay(
        save_path=figpath('modes_magnitude_overlay'), dpi=PLOT_DPI)

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '=' * 60)
print(f'  Complete — {stem}')
print('=' * 60)

if SAVE_PLOTS:
    saved = sorted(fig_dir.glob('*.png'))
    print(f'  {len(saved)} figures saved to {fig_dir}')
    for f in saved:
        print(f'    {f.name}')

if clust.cluster_ids is not None:
    sizes = np.bincount(clust.cluster_ids)
    print(f'\n  Clusters: {clust.n_clusters}  (A={clust.regime_A}, B={clust.regime_B})')
    for k, s in enumerate(sizes):
        tag = ' <-A' if k == clust.regime_A else (' <-B' if k == clust.regime_B else '')
        print(f'    Cluster {k}: {s} windows{tag}')

if ms.cluster_modes:
    print('\n  Modes per cluster:')
    for k, cm in enumerate(ms.cluster_modes):
        if cm is None:
            print(f'    Cluster {k}: no modes')
        else:
            print(f'    Cluster {k}: {cm.shapes.shape[1]} modes'
                  f'  ({cm.freq[0]:.1f}–{cm.freq[-1]:.1f} Hz)')

print('\nDone.')
