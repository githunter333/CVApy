"""
cva_record.py
=============
Record inspection and reading for the CVA pipeline.

CVA needs one array of shape (N, 1 + n_in + n_out) — time, excitation,
responses — but the recorded data arrives in three shapes:

  combined + time   PicoScope exports and the *_cva.txt files already
                    formatted for the pipeline: every channel in one file
                    with a time column, so fs comes from the record itself.

  combined, no time every channel in one file but no time column; fs must
                    be supplied.

  paired            excitation and response in *separate* files, no time
                    column anywhere — the '<base>in.dat' / '<base>out.dat'
                    convention used throughout nldata and r6dof. The two
                    files may differ in length when the excitation was
                    simulated at a finer time step, in which case the
                    excitation is decimated to match.

sniff_record() reads only a probe of the file, so the GUI can describe a
record — channel names, units, sample rate, length — the moment it is
picked and before committing to a full read.

Functions
---------
sniff_record   describe a file without reading all of it
find_pair      locate the in/out sibling of a paired record
read_array     read a record to a 2-D array, honouring header and layout
decimate_safe  decimate by any ratio, in stages when the ratio is large
classify       transient or random, from crest factor and envelope decay
suggest_window a sensible analysis window for that signal type

Author: Norman / Claude
Date:   2026-11-02
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import decimate

# A file written as one very long row is a single channel, not N channels.
TRANSPOSE_MIN_COLS = 64

# Relative spread of successive differences below which a column is
# accepted as a uniformly sampled time vector.
TIME_UNIFORM_RTOL = 1e-3


# =============================================================================
# RecordInfo
# =============================================================================

@dataclass
class RecordInfo:
    """What a probe of the file can tell us, without reading all of it."""

    path:       Path
    skiprows:   int   = 0
    delimiter:  Optional[str] = None      # None = any whitespace
    names:      List[str] = field(default_factory=list)
    units:      List[str] = field(default_factory=list)
    n_col:      int   = 0
    n_samp:     int   = 0
    time_col:   Optional[int] = None      # None = no time column found
    fs:         Optional[float] = None    # from the time column, if any
    t0:         Optional[float] = None
    transposed: bool  = False             # one row, many columns
    pair:       Optional[Path] = None     # in/out sibling, when found
    kind:       str   = 'combined'        # 'combined' | 'input' | 'response'

    @property
    def n_data_col(self) -> int:
        """Columns that carry signal rather than time."""
        return self.n_col - (1 if self.time_col is not None else 0)

    def channel_names(self) -> List[str]:
        """One label per column, falling back to 'Column N'."""
        out = []
        for i in range(self.n_col):
            if i < len(self.names) and self.names[i].strip():
                out.append(self.names[i].strip())
            elif i == self.time_col:
                out.append('Time')
            else:
                out.append(f'Column {i}')
        return out

    def channel_units(self) -> List[str]:
        return [self.units[i].strip() if i < len(self.units) else ''
                for i in range(self.n_col)]

    def describe(self) -> str:
        bits = [f'{self.n_col} column{"s" if self.n_col != 1 else ""}',
                f'{self.n_samp} samples']
        if self.time_col is not None:
            bits.append(f'time col {self.time_col}, fs={self.fs:.1f} Hz')
        else:
            bits.append('no time column — fs required')
        if self.transposed:
            bits.append('single row (transposed)')
        if self.skiprows:
            bits.append(f'{self.skiprows} header row'
                        f'{"s" if self.skiprows != 1 else ""}')
        if self.pair:
            bits.append(f'pairs with {self.pair.name}')
        return '  |  '.join(bits)


# =============================================================================
# sniffing
# =============================================================================

def _is_numeric(line: str) -> bool:
    toks = line.replace(',', ' ').replace(';', ' ').split()
    if not toks:
        return False
    try:
        for t in toks:
            float(t)
    except ValueError:
        return False
    return True


def _split(line: str, delimiter: Optional[str]) -> List[str]:
    if delimiter is None:
        return line.split()
    return [t.strip() for t in line.split(delimiter)]


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            n += chunk.count(b'\n')
            last = chunk
    if n and not last.endswith(b'\n'):
        n += 1          # final line without a trailing newline
    return n


def sniff_record(path, probe_rows: int = 40) -> RecordInfo:
    """
    Describe a text record from a probe of its first rows.

    Finds the header depth, the delimiter, the channel names and units when
    the file carries them, and whether column 0 is a uniformly sampled time
    vector (which also gives fs). Reads a probe plus a line count, never the
    whole file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Not found: {p}')

    with open(p, 'r', errors='replace') as fh:
        probe = []
        for _ in range(probe_rows):
            ln = fh.readline()
            if not ln:
                break
            probe.append(ln.rstrip('\n'))

    first_num = next((i for i, ln in enumerate(probe) if _is_numeric(ln)), None)
    if first_num is None:
        raise ValueError(f'No numeric data found in the first {len(probe)} '
                         f'lines of {p.name}')

    data_line = probe[first_num]
    delim = None
    for cand in ('\t', ',', ';'):
        if cand in data_line:
            delim = cand
            break

    info = RecordInfo(path=p, skiprows=first_num, delimiter=delim)
    info.n_col = len(_split(data_line, delim))

    # ---- header text ------------------------------------------------
    head = [ln for ln in probe[:first_num] if ln.strip()]
    if head:
        info.names = _split(head[0], delim)
        if len(head) > 1 and re.search(r'\(.*\)', head[1]):
            info.units = _split(head[1], delim)

    # ---- layout -----------------------------------------------------
    n_rows = _count_lines(p) - first_num
    if n_rows == 1 and info.n_col >= TRANSPOSE_MIN_COLS:
        # One enormous row: a single channel written sideways.
        info.transposed = True
        info.n_samp = info.n_col
        info.n_col = 1
    else:
        info.n_samp = max(n_rows, 0)

    # ---- time column ------------------------------------------------
    if info.n_col > 1 and not info.transposed:
        probe_num = [ln for ln in probe[first_num:] if _is_numeric(ln)]
        try:
            col0 = np.array([float(_split(ln, delim)[0]) for ln in probe_num])
        except (ValueError, IndexError):
            col0 = np.array([])
        if col0.size >= 3:
            d = np.diff(col0)
            if np.all(d > 0) and np.std(d) / max(abs(np.mean(d)), 1e-30) \
                    < TIME_UNIFORM_RTOL:
                info.time_col = 0
                info.fs = float(1.0 / np.median(d))
                info.t0 = float(col0[0])

    # ---- in/out pairing ---------------------------------------------
    pair, kind = find_pair(p)
    info.pair, info.kind = pair, kind
    return info


# =============================================================================
# in / out pairing
# =============================================================================

def _pair_candidates(stem: str) -> Tuple[List[str], str]:
    """
    Candidate sibling stems for a paired record, and this file's role.

    The libraries use several spellings of the same convention — 'in'/'out'
    at the end (r6dofcubic2in), 'input'/'output' (ainput2), and 'in'/'out'
    buried before a suffix (icein38e) — so candidates are generated for all
    of them and the caller keeps whichever exists on disk.
    """
    low = stem.lower()
    cands: List[str] = []

    if 'input' in low:
        cands.append(_replace_ci(stem, 'input', 'output'))
        return cands, 'input'
    if 'output' in low:
        cands.append(_replace_ci(stem, 'output', 'input'))
        return cands, 'response'

    if low.endswith('in'):
        cands.append(stem[:-2] + 'out')
        return cands, 'input'
    if low.endswith('out'):
        cands.append(stem[:-3] + 'in')
        return cands, 'response'

    # 'in' or 'out' before a trailing suffix, e.g. icein38e / iceout38e.
    m = re.search(r'(?i)out', stem)
    if m:
        cands.append(stem[:m.start()] + 'in' + stem[m.end():])
        return cands, 'response'
    m = re.search(r'(?i)in', stem)
    if m:
        cands.append(stem[:m.start()] + 'out' + stem[m.end():])
        return cands, 'input'

    return [], 'combined'


def _replace_ci(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)


def find_pair(path) -> Tuple[Optional[Path], str]:
    """
    Locate the in/out sibling of a paired record.

    Returns (sibling_path, kind) where kind is 'input' when this file holds
    the excitation, 'response' when it holds the responses, and 'combined'
    when no sibling exists — in which case the file is assumed to hold
    every channel. Candidate names are always checked against the disk, so
    a stem that merely happens to contain 'in' is not mistaken for a pair.
    """
    p = Path(path)
    stems, kind = _pair_candidates(p.stem)
    if not stems:
        return None, 'combined'
    for stem in stems:
        for suffix in (p.suffix, '.dat', '.txt', '.csv'):
            cand = p.with_name(stem + suffix)
            if cand.exists() and cand != p:
                return cand, kind
    return None, 'combined'


# =============================================================================
# reading
# =============================================================================

def read_array(info: RecordInfo) -> np.ndarray:
    """Read a record to (n_samp, n_col), honouring header depth and layout."""
    a = np.loadtxt(str(info.path), skiprows=info.skiprows,
                   delimiter=info.delimiter)
    a = np.atleast_1d(a)
    if a.ndim == 1:
        a = a[:, None]
    if info.transposed:
        a = a.reshape(-1, 1)        # one row, many columns -> one channel
    return a


def decimate_safe(x: np.ndarray, q: int, axis: int = 0) -> np.ndarray:
    """
    Decimate by q, splitting large ratios into stages.

    scipy (like MATLAB) filters poorly for q much above 10, so a ratio such
    as 20 is applied as 5 then 4. Matches MATLAB decimate()'s default: an
    order-8 Chebyshev type I filter applied forwards and backwards.
    """
    q = int(q)
    if q <= 1:
        return x
    for step in _factor_stages(q):
        x = decimate(x, step, ftype='iir', zero_phase=True, axis=axis)
    return x


def _factor_stages(q: int, cap: int = 10) -> List[int]:
    """Split q into factors no larger than cap, largest first."""
    if q <= cap:
        return [q]
    stages = []
    rest = q
    for f in range(cap, 1, -1):
        while rest % f == 0 and rest > 1:
            stages.append(f)
            rest //= f
        if rest == 1:
            break
    if rest > 1:
        # q has a large prime factor; one oversized stage is unavoidable.
        stages.append(rest)
    return stages


# =============================================================================
# signal type
# =============================================================================

@dataclass
class SignalType:
    kind:     str            # 'transient' | 'random'
    crest:    float          # peak / rms of the excitation
    kurtosis: float
    decay:    float          # response rms, last tenth / first tenth
    reason:   str = ''

    def __str__(self):
        return (f'{self.kind}  (crest={self.crest:.1f}, '
                f'kurtosis={self.kurtosis:.1f}, decay={self.decay:.2f})'
                + (f' — {self.reason}' if self.reason else ''))


def classify(inputd: np.ndarray, datout: Optional[np.ndarray] = None
             ) -> SignalType:
    """
    Decide whether a record is a transient or a random excitation.

    An impact concentrates its energy in a few samples, so the excitation's
    crest factor and kurtosis are both large; a stationary random drive
    keeps both near Gaussian values (crest ~4, kurtosis ~3). The response
    envelope decaying from start to end is the third, strongest clue.
    """
    x = np.asarray(inputd, dtype=float).ravel()
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-30
    crest = float(np.max(np.abs(x)) / rms)
    kurt = float(np.mean(x ** 4) / rms ** 4)

    decay = 1.0
    if datout is not None and len(datout) >= 20:
        y = np.asarray(datout, dtype=float)
        n = len(y) // 10
        first = float(np.sqrt(np.mean(y[:n] ** 2))) or 1e-30
        last = float(np.sqrt(np.mean(y[-n:] ** 2)))
        decay = last / first

    if decay < 0.3:
        return SignalType('transient', crest, kurtosis=kurt, decay=decay,
                          reason='response decays over the record')
    if crest > 6.0 or kurt > 8.0:
        return SignalType('transient', crest, kurtosis=kurt, decay=decay,
                          reason='excitation energy is concentrated')
    return SignalType('random', crest, kurtosis=kurt, decay=decay,
                      reason='excitation is near-stationary')


def suggest_window(time: np.ndarray, inputd: np.ndarray,
                   datout: Optional[np.ndarray] = None,
                   kind: Optional[str] = None,
                   pre_roll: float = 0.005,
                   floor_frac: float = 0.02,
                   skip_frac: float = 0.01,
                   max_samples: Optional[int] = None) -> Tuple[float, float]:
    """
    A sensible analysis window, in seconds, for this record.

    transient : from just before the excitation peak to where the response
                envelope has fallen to floor_frac of its own peak. Most
                records in the library carry pre-trigger data, which has no
                excitation in it and nothing for CVA to fit.
    random    : stationary, so where the window starts hardly matters and
                the question is only how much record to analyse. Starts
                after a short settling allowance and runs to the end, or to
                max_samples if that is shorter.

    max_samples : cap on the samples kept, whatever the signal type. A 20 s
                  record at 10 kHz is 200 000 windows at stride 1, which is
                  not a sweep anyone waits for; the caller passes the budget
                  it is willing to spend.
    """
    t = np.asarray(time, dtype=float)
    if kind is None:
        kind = classify(inputd, datout).kind

    def cap(lo, hi):
        if max_samples is None:
            return lo, hi
        sel = np.nonzero((t >= lo) & (t <= hi))[0]
        if sel.size > max_samples:
            hi = float(t[sel[0] + int(max_samples) - 1])
        return lo, hi

    if kind != 'transient':
        return cap(float(t[0] + skip_frac * (t[-1] - t[0])), float(t[-1]))

    x = np.abs(np.asarray(inputd, dtype=float)).max(axis=1) \
        if np.ndim(inputd) == 2 else np.abs(np.asarray(inputd, dtype=float))
    ip = int(np.argmax(x))
    t_start = max(float(t[0]), float(t[ip]) - pre_roll)

    t_end = float(t[-1])
    if datout is not None and len(datout) > 100:
        y = np.asarray(datout, dtype=float)
        env = np.sqrt((y ** 2).mean(axis=1)) if y.ndim == 2 else np.abs(y)
        n_blk = 200
        blk = max(1, len(env) // n_blk)
        trimmed = env[:len(env) // blk * blk].reshape(-1, blk).mean(axis=1)
        peak = trimmed.max() or 1e-30
        after = max(1, ip // blk)

        # A ringdown settles onto the instrument noise floor, not onto zero,
        # so a fixed fraction of the peak is often never reached. Where the
        # record has pre-trigger data it measures that floor directly.
        floor = floor_frac * peak
        if after >= 2:
            noise = float(np.median(trimmed[:after]))
            floor = max(floor, 3.0 * noise)

        below = np.nonzero(trimmed[after:] < floor)[0]
        if below.size:
            idx = min(len(t) - 1, (after + int(below[0]) + 1) * blk)
            t_end = float(t[idx])

    if t_end <= t_start:
        t_end = float(t[-1])
    return cap(t_start, t_end)


# =============================================================================
# RecordSpec — everything needed to turn files into CVA signals
# =============================================================================

@dataclass
class RecordSpec:
    """
    A complete description of how to read one record.

    Saved alongside the results, so a run can always say where its signals
    came from and what was done to them — which of the two files, which
    columns, at what rate, decimated by how much, over which time window.
    """

    input_path:    Path
    response_path: Optional[Path] = None   # None = every channel in one file
    time_col:      Optional[int]  = None
    in_cols:       List[int] = field(default_factory=list)
    out_cols:      List[int] = field(default_factory=list)
    fs:            Optional[float] = None  # required when there is no time col
    decimate:      int = 1                 # user-requested, applied last
    t_start:       Optional[float] = None  # seconds; None = start of record
    t_end:         Optional[float] = None  # seconds; None = end of record
    align:         str = 'auto'            # 'auto' | 'trim' | 'none'

    @property
    def paired(self) -> bool:
        return self.response_path is not None

    def summary(self) -> str:
        bits = [f'in={self.input_path.name}']
        if self.paired:
            bits.append(f'out={self.response_path.name}')
        bits.append(f'in_cols={self.in_cols}  out_cols={self.out_cols}')
        if self.decimate > 1:
            bits.append(f'decimate={self.decimate}')
        if self.t_start is not None or self.t_end is not None:
            bits.append(f'window=[{self.t_start}, {self.t_end}] s')
        return '  '.join(bits)


def default_spec(info: RecordInfo, fs: Optional[float] = None,
                 pair_info: Optional[RecordInfo] = None) -> RecordSpec:
    """
    A first guess at how to read this record, for the GUI to show and edit.

    Combined file : first data column is the excitation, the rest responses.
    Paired files  : the excitation file supplies the input (its first column
                    when it has several, matching saveformatteddata.m), the
                    response file supplies every response channel.
    """
    in_info, out_info = info, None
    if info.kind == 'response' and info.pair is not None:
        in_info = pair_info if pair_info is not None else sniff_record(info.pair)
        out_info = info
    elif info.kind == 'input' and info.pair is not None:
        out_info = pair_info if pair_info is not None else sniff_record(info.pair)

    if out_info is None:                       # one combined file
        data = [c for c in range(info.n_col) if c != info.time_col]
        return RecordSpec(input_path=info.path,
                          time_col=info.time_col,
                          in_cols=data[:1],
                          out_cols=data[1:],
                          fs=fs if fs else info.fs)

    return RecordSpec(input_path=in_info.path,
                      response_path=out_info.path,
                      time_col=None,
                      in_cols=[0],
                      out_cols=list(range(out_info.n_col)),
                      fs=fs if fs else (in_info.fs or out_info.fs))


# =============================================================================
# load_signals
# =============================================================================

def load_signals(spec: RecordSpec, verbose: bool = True, cache: dict = None):
    """
    Read a record described by `spec` into (time, inputd, datout, meta).

    Order of operations matters and is fixed here:
      1. read the file(s)
      2. for a paired record, reconcile the two lengths — an excitation
         simulated at a finer time step is decimated by the length ratio
      3. apply the user's decimation to every channel
      4. build or decimate the time vector
      5. cut the analysis window

    meta records what was actually done, for titles and for the HDF5.

    cache : optional {path string: array} of records already read. Lets a
            caller that has parsed a file once — the GUI's preview, say —
            re-slice it without a second pass over the disk.
    """
    def say(msg):
        if verbose:
            print(msg, flush=True)

    def read(info):
        key = str(info.path)
        if cache is not None and key in cache:
            return cache[key]
        say(f'Reading: {info.path}')
        a = read_array(info)
        if cache is not None:
            cache[key] = a
        return a

    in_info = sniff_record(spec.input_path)
    a_in = read(in_info)

    if spec.paired:
        out_info = sniff_record(spec.response_path)
        a_out = read(out_info)
    else:
        out_info, a_out = in_info, a_in

    # ---- columns ----------------------------------------------------
    if not spec.out_cols and not spec.paired:
        raise ValueError(
            f'{spec.input_path.name} has {in_info.n_data_col} data column'
            f'{"s" if in_info.n_data_col != 1 else ""} and no in/out sibling '
            f'on disk, so there is nothing to use as a response. CVA needs '
            f'an excitation and at least one response channel.')
    for label, cols, arr in (('input', spec.in_cols, a_in),
                             ('response', spec.out_cols, a_out)):
        if not cols:
            raise ValueError(f'No {label} columns selected.')
        bad = [c for c in cols if c >= arr.shape[1]]
        if bad:
            raise ValueError(
                f'{label} column{"s" if len(bad) > 1 else ""} {bad} '
                f'out of range — the file has {arr.shape[1]}.')

    inputd = a_in[:, spec.in_cols].astype(float)
    datout = a_out[:, spec.out_cols].astype(float)
    t_col = (a_in[:, spec.time_col].astype(float)
             if spec.time_col is not None else None)

    # ---- reconcile paired lengths -----------------------------------
    align_q = 1
    if spec.paired and len(inputd) != len(datout) and spec.align != 'none':
        ratio = len(inputd) / len(datout)
        q = int(round(ratio))
        if spec.align == 'auto' and q >= 2 and abs(ratio - q) / q < 0.01:
            say(f'  Excitation is {ratio:.3f}x longer than the response — '
                f'decimating input by {q}')
            inputd = decimate_safe(inputd, q)
            align_q = q
        n = min(len(inputd), len(datout))
        if len(inputd) != n or len(datout) != n:
            say(f'  Trimming to the shorter record: {n} samples')
        inputd, datout = inputd[:n], datout[:n]
    elif len(inputd) != len(datout):
        n = min(len(inputd), len(datout))
        inputd, datout = inputd[:n], datout[:n]
        if t_col is not None:
            t_col = t_col[:n]

    # ---- sample rate ------------------------------------------------
    fs_raw = spec.fs
    if spec.time_col is not None and in_info.fs:
        if spec.fs and abs(spec.fs - in_info.fs) / in_info.fs > 1e-3:
            say(f'  WARNING: fs={spec.fs:.1f} Hz was given but the time '
                f'column says {in_info.fs:.1f} Hz — using the given value')
            fs_raw = spec.fs
        else:
            fs_raw = in_info.fs
    if not fs_raw:
        raise ValueError(
            f'{spec.input_path.name} has no time column, so a sample rate '
            f'must be supplied.')
    # The paired excitation was decimated to the response rate, so fs_raw
    # already describes the aligned signals.

    # ---- user decimation --------------------------------------------
    q = max(1, int(spec.decimate))
    if q > 1:
        stages = _factor_stages(q)
        say(f'  Decimating by {q}'
            + (f' (stages {stages})' if len(stages) > 1 else '')
            + f'  ->  fs {fs_raw:.1f} -> {fs_raw / q:.1f} Hz')
        inputd = decimate_safe(inputd, q)
        datout = decimate_safe(datout, q)
        if t_col is not None:
            t_col = t_col[::q]
    fs = fs_raw / q

    n = min(len(inputd), len(datout))
    inputd, datout = inputd[:n], datout[:n]

    # ---- time vector ------------------------------------------------
    if t_col is not None:
        time = t_col[:n]
    else:
        time = np.arange(n, dtype=float) / fs

    # ---- analysis window --------------------------------------------
    t_start = time[0] if spec.t_start is None else float(spec.t_start)
    t_end = time[-1] if spec.t_end is None else float(spec.t_end)
    if t_end <= t_start:
        raise ValueError(f'Empty time window: t_start={t_start} >= '
                         f't_end={t_end}')
    keep = (time >= t_start) & (time <= t_end)
    n_keep = int(keep.sum())
    if n_keep < 2:
        raise ValueError(
            f'Window {t_start:.4f}-{t_end:.4f} s selects {n_keep} samples; '
            f'the record spans {time[0]:.4f}-{time[-1]:.4f} s.')
    if n_keep < n:
        say(f'  Window {t_start:.4f}-{t_end:.4f} s  ->  {n_keep} of {n} '
            f'samples ({n_keep / fs:.3f} s)')
    time, inputd, datout = time[keep], inputd[keep], datout[keep]

    # ---- names ------------------------------------------------------
    in_names = [in_info.channel_names()[c] for c in spec.in_cols]
    out_names = [out_info.channel_names()[c] for c in spec.out_cols]
    in_units = [in_info.channel_units()[c] for c in spec.in_cols]
    out_units = [out_info.channel_units()[c] for c in spec.out_cols]

    meta = dict(
        input_path=str(spec.input_path),
        response_path=str(spec.response_path) if spec.paired else '',
        time_col=-1 if spec.time_col is None else int(spec.time_col),
        in_cols=list(spec.in_cols), out_cols=list(spec.out_cols),
        fs=float(fs), fs_raw=float(fs_raw),
        decimate=int(q), align_decimate=int(align_q),
        t_start=float(time[0]), t_end=float(time[-1]),
        n_samp=int(len(time)),
        in_names=in_names, out_names=out_names,
        in_units=in_units, out_units=out_units,
    )
    return time, inputd, datout, meta
