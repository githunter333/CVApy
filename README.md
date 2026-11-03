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

Picking a record (Browse …, or File ▸ Open Data Record) inspects it at once —
header rows, delimiter, channel names and units, whether there is a time column,
and whether an `in`/`out` sibling exists on disk — and fills the **Record**
panel from what it finds. Each column gets a role (Time / Excitation / Response
/ Ignore) that can be changed; `fs` and `decimate` sit under the table.

The record then loads in full and appears as two tabs: the **time history** and
the **spectral density**. A record with no time column — the `in.dat`/`out.dat`
pairs — cannot be read until you give it a rate, so it waits: type one into
`fs` and press Enter. The analysis window is chosen from the time history —
drag across any panel to set it, type `t start` / `t end` by hand, or press
**Suggest**, which picks a window to suit the detected signal type: a transient
gets its impact and ringdown, a stationary record gets as much as
`SUGGEST_MAX_WINDOWS` (5000) windows will cover, since for a stationary record
only the length matters and not where it starts. **Sweep
size** keeps a running count of the samples and CVA windows the current
settings imply, and warns when a sweep would run for hours or is too short to
fit `neighbors + lags`.

**Run Pipeline** applies the window and reuses the already-parsed arrays rather
than reading the files again. The status bar carries the live stage, window
count and ETA; the progress bar runs 0–55 % over the CVA sweep (serial *and*
parallel), then 60 % clustering, 85 % mode shapes.

**Stop** is checked once per window, so a sweep gives up within one window
instead of at the end of the stage; partial results are discarded. Ctrl+C in
the launching terminal also quits. If the window is ever wedged beyond that,
`pkill -f cva_gui.py`.

**Script** — edit the settings block at the top of `run_cva_pipeline.py`
(`RESPONSE_FILE` for a paired record, `DECIMATE`, `T_START` / `T_END`, and
`PLOT_TIME_HISTORY` / `PLOT_PSD` for the same two pre-sweep figures), then:

```bash
conda activate cvapy && python run_cva_pipeline.py
```

## Records

`cva_record.py` opens three shapes of record without being told which is which:

| shape | where | what it needs |
|---|---|---|
| combined, with a time column | PicoScope exports (`Time / Channel A …`, 3-line header), `*_cva.txt` | nothing — the rate comes from the time column |
| combined, no time column | single-column drive waveforms | a sample rate |
| **paired** `<base>in.dat` / `<base>out.dat` | `nldata`, `r6dof` | a sample rate; the sibling is found on disk |

`sniff_record()` reads a probe of the file, never all of it, and reports the
header depth, delimiter, channel names and units, column count, sample count,
and whether column 0 is a uniformly sampled time vector (which also gives fs).
A file written as one enormous row — `din.dat` is 100 001 columns on one line —
is recognised as a single channel and transposed.

Pairing covers the spellings in use: `in`/`out` at the end of the stem
(`r6dofcubic2in`), `input`/`output` (`ainput2`), and `in`/`out` before a suffix
(`icein38e`). Candidates are always checked against the disk, so a stem that
merely contains `in` is never mistaken for a pair.

Where a paired excitation is longer than its response — `mdofpulsein.dat` has
200 000 samples against `mdofpulseout.dat`'s 10 001 — it was simulated at a
finer time step, so it is decimated by the length ratio and both are trimmed to
the shorter. This is the step `saveformatteddata.m` and `r6dofcvaformatter.py`
did by hand; writing a `_cva.txt` intermediate is now optional rather than a
precondition.

**Sample rate.** `fs = 0` (or blank) means *take it from the time column*,
which is what 200 of the 203 records here carry. A non-zero `fs` overrides the
time column and logs a warning when the two disagree — worth reading, since a
record read at the wrong rate reports every frequency wrong by that factor.

**Decimation and window.** `decimate` is applied first, in stages for large
ratios (20 becomes 5 then 4), using the same order-8 Chebyshev filter run
forwards and backwards that MATLAB's `decimate` uses. `t_start` / `t_end` then
cut the analysis window, keeping the record's own clock, so a window on a
pre-triggered record is written in that record's negative-time terms.

Most of these records need a window rather than wanting one: 10 s at 10 kHz is
100 000 samples, which is ~100 000 CVA windows at `eval_stride = 1`, and 157 of
200 records begin before the trigger, so their opening stretch holds no
excitation at all.

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
| `decimate` | 1 = none. Applied before the window; divides fs |
| `t_start` / `t_end` | analysis window in seconds. None = the whole record |
| `n_jobs` | 1 = serial, -1 = all cores (verified bit-identical to serial) |
| `f_min` / `f_max` | frequency band for clustering and mode shapes |
| `n_clusters` | 0 = auto-detect via elbow |

## Plotting

Every plot method builds a figure, optionally saves it, and **returns the
`Figure`**. `plot_per_cluster` returns a list, one figure per cluster.

`CVARun.plot_time_history()` and `CVARun.plot_spectral_density()` need the
record only — no CVA results — so it can be judged before the sweep.

`plot_time_history()` draws one excitation channel and the first few responses
(`max_out`, default 3) rather than all of them, and shades the proposed window.
Long traces are drawn as min/max bands with the block mean through them, so the
single-sample peak of an impact still shows instead of being strided over. In
the GUI, dragging across any of its panels sets the window.

`plot_spectral_density()` needs the record only too, so
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
