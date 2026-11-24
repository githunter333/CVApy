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

### Setting up on another computer

Nothing here needs more than Python and the packages above.

1. Install Miniconda or Anaconda, and Git.
2. Get the code (or copy the folder across; `results/` is not needed):

   ```bash
   git clone -b nonlinearity-analysis git@github.com:githunter333/CVApy.git
   cd CVApy
   ```

   The nonlinearity analyses are on the `nonlinearity-analysis` branch until
   it is merged into `main`.
3. Create the environment with the `conda create` line above, then
   `conda activate cvapy`.
4. Check it: `conda install -y pytest`, then `python -m pytest tests -q`
   should end with every test passed.

To analyse your own data see *Running the nonlinearity analyses on your own
data* below.

## Running

**GUI** — parameter panel, tabbed figures, live log, with the CVA pipeline
and the nonlinearity analyses each on a worker thread:

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

## Nonlinearity indicators from spectra

`cva_spectra.py` is the first look at a record, before any model is fitted.
It splits each response's spectral density into the part linearly related to
the input (coherent) and the part that is not, measures the energy above the
excitation band, and reports skewness and excess kurtosis with standard
errors.

```python
from cva_spectra import analyze_record, plot_record, plot_comparison
res = analyze_record(inputd, datout, fs, label='0 dB')
res.table()
plot_record(res)                    # one record
plot_comparison([low, res])         # test levels, or baseline against case
```

With several inputs the coherence is the multiple coherence. Channels marked
*resolution-limited* have a resonance narrower than the frequency resolution;
their non-coherent share is an upper bound, so pass a longer `nperseg` or
judge them against a linear or low-level record of the same system.

`run_r6dof_spectra.py` runs it over the simulated six-mass records and
writes the figures to `results/step1/`. Those records are sampled at
2500 Hz, and the base acceleration actually applied is column 7 of each
`...out.dat` file.

## The non-coherent response in time

`cva_residual.py` turns the non-coherent part into a time history: the
measured input is passed through the estimated frequency response and the
result subtracted from the measured response. What is left shows *when* the
nonlinearity acts.

```python
from cva_residual import noncoherent_history, plot_residual, track_residual
rr = noncoherent_history(inputd, datout, fs, label='0 dB')
rr.table()                      # share, kurtosis, level exponent per channel
plot_residual(rr, channel=3)    # five views of one channel
track_residual(rr).summary()    # frequency and damping of the residual's peaks
```

The level exponent says whether the residual grows faster than the response
(positive, cubic-like), in step with it (tension/compression), or matters
most at low level (negative: free play). Its sign and ordering are reliable;
its size is understated, because the residual rings on after the level that
caused it has passed.

`run_r6dof_residual.py` writes these for the six-mass records to
`results/step2/`.

## Modes against level, and peaks above the excitation band

`cva_level.py` sets each tracked in-band mode's frequency and damping
against the response level in the same block. A linear system gives flat
lines; several records of one system (test levels) can share the axes.

```python
from cva_level import track_in_band, level_dependence, plot_level
trk = track_in_band(time, inputd, datout, fs, band_top)
lv = level_dependence(trk, label='0 dB')
lv.table()                      # % change per doubling of level, with s.e.
plot_level([lv_low, lv])
```

`cva_bands.py` takes each response peak above the excitation band and gives
its band-limited time history and envelope at every channel, its frequency
and apparent damping block by block, which channels carry it, and how its
level moves against the in-band level.

```python
from cva_bands import analyze_bands, plot_band
bb = analyze_bands(datout, fs, band_top=416.0, label='dead zone')
bb.table()
plot_band(bb, 0)
bb.bands[0].signal              # (N, n_out) time history of that band
```

Bands are found from the spectrum, or passed as `bands=[(centre, half), ...]`.
On synthetic modes of known properties the frequency reads within about
0.3% and the damping within about 5%. On real nonlinear records the damping
is *apparent*: frequency wander within a block reads as extra damping, so
the page also shows the value from the spectral peak width.

`run_r6dof_level_bands.py` writes both for the six-mass records to
`results/step3/`.

## Frequency coupling, and one state or two

`cva_hos.py` asks which frequencies are phase-locked to which. Quadratic
coupling (bicoherence: f1 and f2 locked to f1 + f2) marks an asymmetric
nonlinearity such as stiffness that differs in tension and compression.
Cubic coupling (f1, f1 and f2 locked to 2 f1 + f2) without quadratic marks
a symmetric one: a cubic spring, free play. Each cell is judged against its
own floor, worked out from the record itself.

```python
from cva_hos import analyze_hos, plot_hos
hh = analyze_hos(datout, fs, band_top=416.0)
hh.table()                  # per channel: asymmetric, symmetric or none
hh.pair_table(3)            # coupling at the in-band mode pairs
plot_hos(hh, channel=3)
```

Keep the frequency resolution no finer than the width of the modes
(`nperseg`); a mode spread over several bins reads weaker.

`cva_regimes.py` tests whether the system has two states at the same
response level (coexisting attractors) or one state that merely looks
different in the loud and quiet stretches of the record. It removes the
level dependence from each in-band mode frequency and from the non-coherent
level, and looks for two separated, persistent clusters in what is left.

```python
from cva_level import track_in_band
from cva_regimes import analyze_regimes, plot_regimes
rg = analyze_regimes(track_in_band(time, inputd, datout, fs, band_top), rr)
rg.summary()                # verdict, and each state's frequencies and damping
plot_regimes(rg)
```

`run_r6dof_hos.py` and `run_r6dof_regimes.py` write these for the six-mass
records to `results/step4/` and `results/step5/`.

### All of it in one call, and in the GUI

`cva_report.py` runs every analysis above on one record:

```python
from cva_report import full_report
full_report(inputd, datout, fs, 'results/mytest', names=channel_names,
            channel=3)          # writes the pages and tables.txt
```

`run_blockhead.py` does this for the Area III 6-DOF blockhead test, with the
six table motions as inputs:

```bash
python run_blockhead.py            # the 12 "target" channels on the block
python run_blockhead.py --full     # all 24 block channels
python run_blockhead.py --nut      # lower nut accelerometer, A10
python run_blockhead.py --base     # base plate corners, A1-A4
```

Several options can be given together. The six inputs are computed from the
base plate channels, so with `--base` the non-coherent figures are low by
construction; skewness, kurtosis, out-of-band energy and coupling do not
depend on the inputs.

`run_blockhead_time.py` asks a separate question of the same test: does the
spectral density change during the record? It splits the record into 1 s
blocks and, for all 45 channels, compares the power in three frequency
ranges and the first third of the record against the last third.

### Running the nonlinearity analyses on your own data

The analyses need **time histories**: the control accelerometer(s) as
inputs and the response channels, sampled together at one rate. Spectra
alone (PSD, coherence, FRF) are not enough; only the coherent / non-coherent
/ out-of-band split could be formed from them, and the code does not read
spectra.

**Text files, in the GUI.** Columns of numbers, with or without a time
column and header lines, or a paired `<name>in.dat` / `<name>out.dat` (see
*Records*). Run `python cva_gui.py`, open the record, set each control
accelerometer column to Excitation in the Record panel, give the sample rate
if there is no time column, and press **Analyse Nonlinearity**.

**MATLAB `.mat` files, or anything else, in a short script.** Load the
arrays yourself and hand them to `full_report`. Save this as, say,
`run_mytest.py` in the CVApy folder and edit the four marked lines:

```python
import scipy.io as sio
from cva_report import full_report

m = sio.loadmat('/path/to/mytest.mat')           # <- your file
inputs = m['control']        # <- (samples, n_inputs) control accelerometers
responses = m['response']    # <- (samples, n_responses)
fs = 12800.0                 # <- sample rate, Hz

names = [f"ch {c + 1}" for c in range(responses.shape[1])]
full_report(inputs, responses, fs, 'results/mytest', label='my test',
            names=names, channel=0)
```

```bash
MPLBACKEND=Agg python run_mytest.py
```

The pages (PNG) and `tables.txt` appear in `results/mytest/`.

- Arrays must have one **column** per channel. If yours are one row per
  channel, use `m['control'].T`.
- `channel` picks the response shown on the single-channel pages (residual
  and coupling); every table covers all channels.
- `scipy.io.loadmat` cannot read files saved with MATLAB's `-v7.3` option.
  Re-save without it, or read the file with `h5py`.
- With several inputs pass `nperseg` so that there are at least ten Welch
  segments per input (the blockhead run uses `nperseg=8192` for six inputs
  and 23 s at 12,800 Hz); otherwise the coherence reads high.
- For a text file in a script, `numpy.loadtxt` gives the array; split its
  columns into inputs and responses the same way.

In the GUI these are the **Nonlinearity analysis** group and the **Analyse
Nonlinearity** button. They need only a loaded record with an excitation
channel, not the CVA sweep, and work over the analysis window. The pages
appear as tabs after the time history and spectral density: Nonlinearity,
Residual, Level, one per out-of-band peak, Coupling and Regimes. The
single-channel pages follow the `channel` setting in the Plots group;
change it and press **Regenerate Plots**, with no need to re-run.

One window serves both jobs because they share everything up to the
analysis itself: reading the record, assigning channel roles, the sample
rate, and choosing the window. **Run Pipeline** remains the CVA sweep with
clustering and mode shapes; the two can be run in either order on the same
record.

## Time-block tracking

`cva_block.py` answers a different question from the pipeline above: how do
frequency, damping and mode shape *evolve along the record*? It fits a short
input/output model (4 lags by default) to each contiguous block of time
(0.5 s by default, half-overlapping) and matches the poles from block to block
by frequency and mode-shape correlation, so each mode becomes one track.

```python
from cva_block import BlockConfig, BlockTracker
trk = BlockTracker.from_run(run, BlockConfig(lags=4, block_sec=0.5))
trk.summary()          # mean, block-to-block scatter, trend per mode
fig = trk.plot_tracks()
```

It reports the *effective* properties over each block. A joint that changes
state within a cycle is averaged, not resolved into its separate states.
With a single response channel there is no mode shape to reject numerical
poles, so set `freq_max` to the band of interest and use more lags (16 or so).

`cva_sim.py` simulates a six-mass chain with a joint that is softer in
tension, optionally softening along the record, and keeps the true modal
values. `tests/test_cva_block.py` scores the tracker against it:

```bash
python -m pytest tests -q
```

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
