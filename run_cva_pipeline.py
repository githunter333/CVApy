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

CVAPY_DIR = Path(__file__).resolve().parent
if str(CVAPY_DIR) not in sys.path:
    sys.path.insert(0, str(CVAPY_DIR))

import importlib
import cva_core, cva_run, cva_phi_cluster, cva_mode_shapes
importlib.reload(cva_core)
importlib.reload(cva_run)
importlib.reload(cva_phi_cluster)
importlib.reload(cva_mode_shapes)
from cva_record import RecordSpec, default_spec, sniff_record
from cva_run import CVAConfig, CVARun
from cva_phi_cluster import CVAPhiCluster
from cva_mode_shapes import CVAModeShapes

# =============================================================================
# USER SETTINGS
# =============================================================================

# Records live under ~/Waveforms/Waveforms 2 — the folder moved out of the
# home root, so the old '/Users/nhunterjr/Waveforms 2/...' paths no longer
# resolve. WAVEFORMS below keeps the rest of the path readable.
WAVEFORMS = Path('/Users/nhunterjr/Waveforms/Waveforms 2')

#DATA_FILE = WAVEFORMS / 'cvastructurehit1.txt'
DATA_FILE = WAVEFORMS / 'initial beam transient/square structure data/sstrdaa.txt'

# Set when the excitation and the responses are in separate files — the
# '<base>in.dat' / '<base>out.dat' records under nldata and r6dof. None
# means DATA_FILE holds every channel. With a pair, COL_* below are ignored:
# column 0 of the input file is the excitation and every column of the
# response file is a response.
RESPONSE_FILE = None
# e.g. RESPONSE_FILE = Path('.../r6dof/r6dofcubic2out.dat')
# Save settings
RESULTS_ROOT = Path('/Users/nhunterjr/Documents/DEVONthink/CVA_Results')
SAVE_PLOTS   = True
PLOT_DPI     = 150

SAVE_HDF5    = True

# CVA parameters
LAGS        = 50
TOL         = 1e-2
NEIGHBORS   =1000
RANKLOCK    = 0
EVAL_STRIDE = 1
# 0 = take the rate from the record's own time column, which is what most
# records carry. A non-zero value OVERRIDES the time column and is reported
# as a warning when the two disagree.
#
# NOTE: sstrdaa.txt's time column has dt = 0.0004 s, i.e. 2500 Hz, not the
# 5000 Hz set here — 50002 samples spanning 20.0004 s. One of the two is
# wrong, and which it is decides whether the reported frequencies are right
# or doubled. Left at 5000 so past results stay reproducible until it is
# settled.
FS          = 5000
N_JOBS      = 1    # 1=serial  -1=all cores (verified identical to serial)

# Signal conditioning, applied in this order: decimate, then cut the window.
DECIMATE    = 1      # 1 = none. 4 would take 10 kHz down to 2.5 kHz
T_START     = None   # seconds; None = start of record
T_END       = None   # seconds; None = end of record
                     # A transient usually wants the impact and its ringdown
                     # only — most records here hold seconds of pre-trigger
                     # noise and a long quiet tail, and the sweep cost is
                     # proportional to the samples kept.

# Data column layout
COL_TIME        = 0
COL_FORCE       = 1
COL_ACCEL_START = 2
N_IN            = 1

# Clustering
K_MAX        = 8
N_PCA_VAR    = 95.0
KMEANS_REP   = 20
N_CLUSTERS   = 0

# Frequency band filter — applied to both clustering and mode shapes
# Set F_MIN / F_MAX to None to disable (uses default 0.5 Hz lower bound only)
F_MIN        = 20  # Hz  lower bound  (e.g. 5.0 to strip near-DC)
F_MAX        = 400 # Hz  upper bound  (e.g. 100.0 for bookshelf data)

# Mode shapes
MIN_IM_FRAC  = 0.15
ZETA_MAX     = 0.5
N_MODES_MAX  = 12

# Run flags
RUN_CVA     = True
RUN_CLUSTER = True
RUN_MODES   = True

# Plot flags
PLOT_TIME_HISTORY   = True     # excitation / responses, with the window shaded
PLOT_PSD            = True     # excitation / response spectra, before the sweep
PLOT_SIGNIFICANCE   = True
PLOT_RECONSTRUCTION = True
PLOT_CLUSTERS       = True
PLOT_MODES          = True
PLOT_FREQ_DEVIATION = True

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

    info = sniff_record(DATA_FILE)
    if RESPONSE_FILE:
        spec = default_spec(info, fs=FS or None,
                            pair_info=sniff_record(RESPONSE_FILE))
    else:
        spec = RecordSpec(
            input_path=Path(DATA_FILE), time_col=COL_TIME,
            in_cols=list(range(COL_FORCE, COL_FORCE + N_IN)),
            out_cols=list(range(COL_ACCEL_START, info.n_col)),
            fs=FS or None)
    spec.decimate = DECIMATE
    spec.t_start, spec.t_end = T_START, T_END
    run.load_record(spec)

    if PLOT_TIME_HISTORY:
        run.plot_time_history(
            window=(T_START, T_END) if (T_START is not None
                                        and T_END is not None) else None,
            save_path=figpath('time_history'), dpi=PLOT_DPI)
    if PLOT_PSD:
        run.plot_spectral_density(save_path=figpath('spectral_density'),
                                  dpi=PLOT_DPI)
    run.build_database()
    run.run(n_jobs=N_JOBS)
    run.eigenvalue_significance()
    if SAVE_HDF5:
        run.save(str(h5_run))
        print(f'  Saved: {h5_run}')
else:
    print(f'\n--- Step 1: Loading from {h5_run.name} ---')
    run = CVARun.load(str(h5_run))

print(run)

if PLOT_SIGNIFICANCE:
   run.plot_significance(save_path=figpath('significance'), 
                      dpi=PLOT_DPI,
                      z_threshold=0.98)

if PLOT_FREQ_DEVIATION:
    run.plot_frequency_deviation(
    band_half=20.0,
    ma_window=20,
    save_path=figpath('freq_deviation'),
    dpi=PLOT_DPI)
    
if PLOT_RECONSTRUCTION:
    run.plot_reconstruction(channel=0,
                            save_path=figpath('reconstruction_ch1'),
                            dpi=PLOT_DPI)

# --- ADD THESE ---
print('\n--- Prediction Errors ---')
run.compute_one_step_error(stride=10)

run.plot_iterated_prediction(channel=0, iter_K=20, iter_stride=100,
                             save_path=figpath('iterated_prediction_ch1'),
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
    if SAVE_HDF5:
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
    if SAVE_HDF5:
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
