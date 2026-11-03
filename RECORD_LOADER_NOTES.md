# Record loading rework — notes

Commit `4f31bda` on `main`, 2 November 2026. Five files, +1577 / −114, with
`cva_record.py` as a new module.

---

## 1. What was wrong

`CVARun.load_data()` could open exactly one shape of file: plain numeric text
laid out as `[time, input, responses]`, read with a bare `np.loadtxt`.

A survey of every `.txt` and `.dat` record across the three libraries showed
how little of the data that covered.

### `~/Waveforms/Waveforms 2` — 203 records

| | |
|---|---|
| **unloadable** (3-line PicoScope header breaks `np.loadtxt`) | **178 of 203** |
| loadable before this work | 22 |
| time column present | 200 |
| no time column (single-column drive waveforms) | 3 |
| **pre-trigger data** (record starts at negative time) | **157 of 200** |
| sample rates | 10 kHz (99), 5 kHz (51), 9615 Hz (34), 20 kHz (7), other (9) |
| durations | 10 s (129), 20 s (43), 5 s (10) |

The header the loader rejected:

```
Time	Channel A	Channel B	Channel C
(s)	(V)	(V)	(V)
<blank>
-1.00020003	0.01434370	-0.01312296	-0.01571703
```

### `~/Code/shared/Nonlinear Data/Experimental_Data/nldata` — 1335 files

`*in.dat` / `*out.dat` pairs — excitation in one file, responses in another,
**no time column anywhere** — plus 746 `.mat`, 223 `.uff`, 123 `.unv`.

Some pairs do not match in length, because the excitation was simulated at a
finer time step:

| pair | excitation | response | ratio |
|---|---|---|---|
| `3modin.dat` / `3modout.dat` | 100 000 | 25 001 | 4× |
| `mdofpulsein.dat` / `mdofpulseout.dat` | 200 000 | 10 001 | 20× |

Some are written as a single enormous row — `din.dat` is 100 001 columns on
one line.

### `~/Code/shared/Nonlinear Data/Experimental_Data/r6dof`

More `in`/`out` pairs at 2048 Hz, plus `cva_formatted/*_cva.txt` — records
already converted into pipeline layout.

### The underlying problem

Three separate formatters already existed to bridge this gap, and they
disagreed with each other:

- `saveformatteddata.m` — interactive MATLAB: prompts for fs, pick input file,
  pick response file, choose channel count and length, optional `decimate`,
  writes tab-delimited `[time, input, resp…]`
- `r6dofcvaformatter.py` — hardcodes `FS = 2048`, `MAX_SAMPLES = 16384`,
  `N_OUTPUT_COLS = 6`
- manual preparation for everything else

So every record was converted to a throwaway `_cva.txt` intermediate before
CVApy saw it, and the decisions that determine what the analysis *means* —
sample rate, decimation ratio, channel selection, record length — lived in a
script header or in a MATLAB prompt answered months earlier, recorded nowhere.

---

## 2. What was built

### `cva_record.py` (new, 702 lines)

**`sniff_record(path)`** describes a file from a probe of its first rows plus a
line count — never a full read, so it runs the instant a file is picked. It
reports header depth, delimiter, channel names, units, column count, sample
count, and whether column 0 is a uniformly sampled time vector (which also
gives fs).

**`find_pair(path)`** locates the in/out sibling, covering every spelling in
use:

| stem | sibling | rule |
|---|---|---|
| `r6dofcubic2in` | `r6dofcubic2out` | trailing `in`/`out` |
| `ainput2` | `aoutput2` | `input`/`output` |
| `icein38e` | `iceout38e` | `in`/`out` before a suffix |

Candidates are always checked against the disk, so a stem that merely contains
the letters `in` is never mistaken for a pair.

**`RecordSpec`** is a complete description of how to read one record — both
files, the time column, which columns are excitation and which are response,
fs, decimation, and the time window. It is saved into the HDF5, so a result can
always say how its signals were assembled.

**`load_signals(spec)`** applies the steps in a fixed order:

1. read the file(s)
2. reconcile paired lengths — an excitation simulated at a finer step is
   decimated by the length ratio, then both are trimmed to the shorter
3. apply the user's decimation
4. build or decimate the time vector
5. cut the analysis window

**`decimate_safe()`** splits large ratios into stages (20 becomes 5 then 4),
using the order-8 Chebyshev type I filter run forwards and backwards that
MATLAB's `decimate()` uses by default.

**`classify()`** decides transient or random from the excitation's crest factor
and kurtosis plus whether the response envelope decays across the record.

**`suggest_window()`** proposes a window to suit that type. For a transient it
runs from just before the excitation peak to where the response envelope has
settled; for a stationary record it skips a short settling allowance and runs
to the end, capped at a sample budget the caller sets.

### `cva_run.py`

- `load_record(spec, cache=None)` — the general entry point
- `load_data(...)` keeps its column-index signature for existing callers, and
  now opens header'd files too; it gained `t_start`, `t_end`, `decimate`
- `plot_time_history(in_ch, out_ch, max_out, window)` — one excitation channel
  and the first few responses, drawn as min/max bands over a block mean, so the
  single-sample peak of an impact survives the reduction instead of being
  strided over. Shades the proposed window and reports the sample count.
- the record metadata round-trips through the HDF5 as JSON

### `cva_gui.py`

- the four column spin boxes became a **channel-role table**: one row per
  column, with its source file, name, units and a role (Time / Excitation /
  Response / Ignore), pre-filled from the sniff
- a **Paired with** field, auto-filled when a sibling is found
- `fs` defaults to `0`, meaning *take it from the time column*; a non-zero
  value is an explicit override and warns when it disagrees
- a **decimate** control
- **Analysis window**: signal type (Auto / Transient / Random), `t start` /
  `t end`, **Full record** and **Suggest** buttons, and drag-to-select on any
  panel of the time history
- **Sweep size** — a live readout of the samples and CVA windows the current
  settings imply, warning both when a sweep would run for hours and when the
  window is too short to fit `neighbors + lags`
- a record with no time column waits for a rate instead of auto-loading; Enter
  in the `fs` box loads it

### `run_cva_pipeline.py`

New settings: `RESPONSE_FILE`, `DECIMATE`, `T_START`, `T_END`,
`PLOT_TIME_HISTORY`.

---

## 3. Why the time window is not optional

Three findings, each independently sufficient:

1. **10 s at 10 kHz is 100 000 samples**, which is ~100 000 CVA windows at
   `eval_stride = 1`. At the measured ~0.28 s/window that is an eight-hour
   sweep. All 99 of the 10 kHz records were unrunnable in practice, not merely
   unloadable.
2. **157 of 200 records begin before the trigger**, so their opening stretch
   holds no excitation at all — nothing for CVA to fit.
3. A transient event is a small fraction of its record. `Bridge Tap 2.txt` is
   10 s long; the impact and its ringdown occupy 0.153 s.

The suggestion measures the **pre-trigger noise floor** to decide where a
ringdown has ended. A fixed fraction of peak does not work on real data: a
ringdown settles onto the instrument noise floor rather than onto zero, so
`Bridge Tap 2.txt` was initially offered the full 9 s because 2 % of peak was
never reached.

---

## 4. Testing

| test | result |
|---|---|
| sniff + spec across all three libraries | 353 files, **284 valid specs** |
| no response channel | 63 — single-column drive waveforms with no sibling; CVA cannot use these by definition, and they now say so |
| genuine failures | 6 (see below) |
| GUI, PicoScope transient | loads, Suggest → 1477 windows over −0.0029–0.1498 s |
| GUI, r6dof paired + decimate 4 | 1 excitation + 9 responses at 512 Hz, Suggest → 4999 windows |
| end to end | sweep of 76 windows over a windowed record; HDF5 saves and reloads the metadata |
| legacy `load_data()` | still works, now on header'd files too |

The six failures, none of them a loader gap:

- `3modin01.dat` — a 4-byte stub containing the text `rin1`
- `modaccvstime.05.dat`, `modfreqvstime.05.dat` — Mathematica `Graphics[...]`
  dumps, not time series
- `softening.dat` — a MATLAB `.mat` file wearing a `.dat` extension
- `softening.txt`, `modulated30a.txt` — Mathematica list format (`{0., 0., 0.}`
  per line); only two such files in the whole library, so no parser was written

`.mat`, `.uff` and `.unv` were explicitly out of scope.

---

## 5. One bug found during testing

Picking a record with no time column fired an automatic load, which failed and
threw an **app-modal** error dialog. On macOS that swallows all input to the
main window — so the `fs` field was untypeable until the dialog was dismissed,
and the dialog was sitting on top of the one box you needed to type into, to
tell you to type into it.

Fixed three ways: no auto-load when a rate is still needed (status bar says so,
cursor moves to the field); Enter in the `fs` box loads the record; and a
pressed **Load record** with `fs` blank gives a plain *"Check parameters"*
warning rather than *"Pipeline failed"* with a traceback.

---

## 6. Still open

**`sstrdaa.txt`'s sample rate.** Its time column gives `dt = 0.0004 s` →
**2500 Hz**, corroborated by 50 002 samples spanning 20.0004 s.
`run_cva_pipeline.py` sets `FS = 5000`. One of the two is wrong, and which it
is decides whether every frequency reported from that record is right or
doubled. `FS` was left at 5000 so existing results stay reproducible; a warning
now prints on every run.

**The README's Parallelism section** still credits a "roughly 2× speedup" to
`n_jobs`. Measurement says otherwise — the win comes from capping the inner
BLAS/OpenMP threads, not from joblib fan-out:

| configuration | time (149 windows) | vs default |
|---|---|---|
| `n_jobs=1`, BLAS unlimited (today's default) | 87.5 s | 1.00× |
| **`n_jobs=1`, BLAS capped to 1** | **42.1 s** | **2.08×** |
| `n_jobs=4`, BLAS capped to 1 | 51.3 s | 1.70× |
| `n_jobs=8`, BLAS capped to 1 | 50.8 s | 1.72× |
| `n_jobs=10`, BLAS capped to 1 | 50.9 s | 1.72× |

Serial with one BLAS thread beats every parallel configuration: the SVDs are
small enough (~900×544) that OpenBLAS spends more time synchronising ten
threads than computing. A `threadpool_limits(1)` around the sweep would be a
real speedup on every run.

One caveat on those numbers: the capped serial run differs from the unlimited
baseline by 3e-6 in eigenvalue magnitude, while the capped *parallel* runs
matched it to 7e-15 — which suggests the thread cap may not reach joblib's
worker threads, so the parallel rows may not be testing quite what they claim.
Ranks were identical in every configuration. "Bit-identical" holds only between
runs with the same BLAS thread count.

**Housekeeping.** The local `record-loader` branch is now redundant;
`mydata_ds.txt` shows as deleted in the working tree from before this work.

---

## 7. Where the data lives

| path | contents |
|---|---|
| `~/Waveforms/Waveforms 2` | 203 PicoScope `.txt` records |
| `~/Code/shared/Nonlinear Data/Experimental_Data/nldata` | 1335 files: in/out `.dat` pairs, `.mat`, `.uff`, `.unv` |
| `~/Code/shared/Nonlinear Data/Experimental_Data/r6dof` | in/out pairs @ 2048 Hz + `cva_formatted/` |
| `~/Code/shared/Nonlinear Data/Experimental_Data/8DOF Farrar` | 10 `.UNV` |
| `~/Code/shared/Nonlinear Data/Experimental_Data/Farrar 4DOF Nonlinear` | 340 `.txt`, 1643 `.xml` |
| `~/Code/shared/Nonlinear Data/Experimental_Data/cap data` | 38 `.dat`, 8 `.mat` |
| `~/Code/shared/Nonlinear Data/Bridge_Data` | 21 `.unv`, 6 `.dat`, 28 `.zip` |
| `~/Code/matlab/CVA/experiments/formatted input data` | dumbbell, farrar4dof, r6dof |
