# CVApy

Local Canonical Variate Analysis (LCVA) for nonlinear structural system
identification. Fits a local linear state-space model in each sliding window of
a response record, then clusters the resulting Φ matrices to find distinct
dynamical regimes and extracts physical mode shapes for each.

## Environment

CVApy has its own conda environment, separate from `rattlesnake` / `rattlesnake4`:

```bash
conda activate cvapy
```

To recreate it from scratch:

```bash
conda create -y -n cvapy -c conda-forge python=3.13 numpy scipy scikit-learn h5py matplotlib joblib pyside6
```

Verified with Python 3.13.18, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1,
h5py 3.16.0, matplotlib 3.11.2, joblib 1.6.0, PySide6 6.11.3.

## Running

**GUI** — parameter panel, tabbed figures, live log, pipeline on a worker thread:

```bash
conda activate cvapy && python cva_gui.py
```

Picking a record (Browse …, or File ▸ Open Data Record) reads it straight away
and plots the excitation and response spectral densities, so the drive band and
the response can be checked before committing to the sweep. **Run Pipeline**
then reuses that already-parsed array instead of reading the file twice. The
status bar carries the live stage, window count and ETA; the progress bar runs
0–55 % over the CVA sweep (serial *and* parallel), then 60 % clustering, 85 %
mode shapes.

**Stop** is checked once per window, so a sweep gives up within one window
instead of at the end of the stage; partial results are discarded. Ctrl+C in
the launching terminal also quits. If the window is ever wedged beyond that,
`pkill -f cva_gui.py`.

**Script** — edit the settings block at the top of `run_cva_pipeline.py`
(`PLOT_PSD` plots the same spectra right after the record is read), then:

```bash
conda activate cvapy && python run_cva_pipeline.py
```

## Data format

Whitespace- or comma-delimited text:

| column | contents |
|---|---|
| `col_time` (default 0) | time (s) |
| `col_force` (default 1) | force input — `n_in` columns wide |
| `col_accel_start` (default 2) onward | accelerometer outputs |

Sample record: `bltimedatashort.txt` (3001 samples, 17 outputs, 1 input, 3200 Hz).

## Pipeline

| Stage | Class | Output |
|---|---|---|
| 1 | `CVARun` (`cva_run.py`) | windowed CVA, eigenvalue significance, prediction error → `<stem>_cva.h5` |
| 2 | `CVAPhiCluster` (`cva_phi_cluster.py`) | PCA → elbow → k-means on Φ → `<stem>_clusters.h5` |
| 3 | `CVAModeShapes` (`cva_mode_shapes.py`) | physical mode shapes per cluster → `<stem>_modes.h5` |

`cva_core.py` holds the MATLAB translation (`delaybuild`, `reduce_matrix`,
`cvaonestep`). `cva_diagnostics.py` adds cross-run comparisons.
`cva_plotting.py` controls figure display policy.

Each stage saves to and loads from HDF5, so any stage can be re-run without
recomputing the ones before it — in the GUI, untick the stage; in the script,
set `RUN_CVA` / `RUN_CLUSTER` / `RUN_MODES` to `False`.

### Key parameters

| name | meaning |
|---|---|
| `lags` | window length in samples. 16 = fast, 64 = better frequency resolution |
| `tol` | SVD rank tolerance. 1e-3 loose, 1e-4 recommended, 5e-5 tight |
| `neighbors` | nearest neighbours per local fit. 200 fast, 960 better conditioned |
| `eval_stride` | evaluate every Nth window. 1 = every window |
| `n_jobs` | 1 = serial, -1 = all cores (verified bit-identical to serial) |
| `f_min` / `f_max` | frequency band for clustering and mode shapes |
| `n_clusters` | 0 = auto-detect via elbow |

## Plotting

Every plot method builds a figure, optionally saves it, and **returns the
`Figure`**. `plot_per_cluster` returns a list, one figure per cluster.

`CVARun.plot_spectral_density()` needs `load_data()` only — no CVA results — so
a record can be judged before the sweep. Three stacked panels: excitation PSD,
response PSD with all channels overlaid, and response PSD per channel as a map.
Welch segment length (`nperseg=0` picks it from the record) and the plotted
`f_max` are exposed in the GUI's Plots group.

Script use pops a window as before. A GUI embeds the returned figure instead and
must first suppress the blocking show:

```python
import cva_plotting
cva_plotting.set_show(False)
```

## Parallelism

`CVARun.run(n_jobs=-1)` evaluates windows with joblib threads. Window 0 runs
serially first to fix the model rank, which is then passed explicitly to every
worker — so the parallel path is deterministic and produces results identical to
the serial path (verified: equal ranks, `max |Δeigenvalue| = 0.0`), at roughly
2× speedup on this hardware.

`run(progress_cb=...)` reports `(done, total, elapsed, eta)` roughly 100 times
per sweep on both paths; the GUI uses it to drive the progress bar, and scripts
can ignore it and read the printed percentages instead. `run(stop_cb=...)` is
polled once per window and raises `InterruptedError` when it returns True.

**Do not emit a Qt signal from `progress_cb`, `stop_cb`, or anything else the
parallel path calls.** Those run on joblib worker threads, which Qt did not
create; emitting from them hangs the event loop, and the updates arrive in one
batch after the sweep finishes — if they arrive at all. The GUI's workers write
their progress and log lines into plain attributes and a `deque`, and a
`QTimer` on the GUI thread moves them onto the screen.

## Repository notes

`cva_run_preparallel.py`, `cva_runs_pre_pred_err.py` and
`run_cva_pipeline (from backup).py` are superseded snapshots kept for reference.
They have drifted from the live modules and are not importable as a set — do not
develop against them.
