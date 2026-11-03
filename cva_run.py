"""
cva_run.py
==========
Object-oriented CVA pipeline for experimental vibration data.

Classes: CVAConfig, CVAWindowResult, CVARun

Changes from previous version
------------------------------
- CVAWindowResult now stores Jnow (rank x ncols_past+nin) — rectangular
  state projector matching MATLAB's J(1:rk,:)
- compute_one_step_error() — physical output one-step error per channel,
  matching MATLAB cva_sweep_error logic
- plot_iterated_prediction() — multi-step iterated prediction with optional
  reanchoring, matching MATLAB iterated prediction section

Author: Norman / Claude
Date:   2026-04-16
"""

from __future__ import annotations
import json
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.neighbors import NearestNeighbors

import cva_record
from cva_record import RecordSpec, default_spec, sniff_record

from cva_core import cvaonestep, delaybuild, reset_cva_state
from cva_plotting import show_if_enabled


# =============================================================================
# CVAConfig
# =============================================================================

@dataclass
class CVAConfig:
    lags:                   int   = 64
    tol:                    float = 5e-4
    neighbors:              int   = 960
    ranklock:               int   = 0
    eval_stride:            int   = 1
    fs:                     Optional[float] = None
    freq_min:               float = 0.0
    freq_max:               float = 0.0
    sig_threshold_fraction: float = 0.25
    sig_peak_mindist:       int   = 3     # minimum peak separation in histogram bins
                                          # bin width = (freq_max-freq_min)/200 Hz
                                          # increase to merge spurious nearby peaks


# =============================================================================
# level-of-detail helper
# =============================================================================

def _lod(t, x, max_points):
    """
    Reduce a long trace to about max_points plotted columns.

    Returns (t_blocks, mean, min, max) so the caller can draw a min/max band
    with the mean through it; min and max come back None when the trace is
    already short enough to plot sample for sample. Plain striding would
    drop the single-sample peak of an impact, which is exactly the feature
    being looked for.
    """
    n = len(t)
    if n <= max_points * 2:
        return t, x, None, None
    blk = int(np.ceil(n / max_points))
    m = n // blk * blk
    tb = t[:m].reshape(-1, blk)[:, 0]
    xb = x[:m].reshape(-1, blk)
    return tb, xb.mean(axis=1), xb.min(axis=1), xb.max(axis=1)


# =============================================================================
# CVAWindowResult
# =============================================================================

@dataclass
class CVAWindowResult:
    window_idx:  int
    sample_idx:  int
    time:        float
    rank:        int
    eigenvalues: np.ndarray
    Phi:         np.ndarray
    Jnow:        np.ndarray    # (rank, n_out*lags + n_in) — rectangular projector
    spast:       np.ndarray
    sfuture:     np.ndarray
    indpast:     np.ndarray


# =============================================================================
# CVARun
# =============================================================================

class CVARun:
    def __init__(self, cfg: CVAConfig):
        self.cfg        = cfg
        self.data_file: Optional[Path] = None
        self.time:    Optional[np.ndarray] = None
        self.inputd:  Optional[np.ndarray] = None
        self.datout:  Optional[np.ndarray] = None
        self.fs:      Optional[float]      = None
        self.n_out:   int = 0
        self.n_in:    int = 0
        self.spec:    Optional[RecordSpec] = None
        self.meta:    dict = {}          # how the signals were assembled
        self.in_names:  List[str] = []
        self.out_names: List[str] = []
        self.past_db:   Optional[np.ndarray] = None
        self.future_db: Optional[np.ndarray] = None
        self.n_win:     int = 0
        self.windows: List[CVAWindowResult] = []
        self.sig:     dict = {}

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------
    def load_record(self, spec: RecordSpec, cache: dict = None):
        """
        Load the signals described by a RecordSpec.

        This is the general entry point: it handles a combined file with or
        without a time column, and the separate '<base>in.dat' /
        '<base>out.dat' pairs, along with decimation and the analysis time
        window. See cva_record for what a spec holds.

        cache : optional {path string: array} of files already read.
        """
        time, inputd, datout, meta = cva_record.load_signals(spec, cache=cache)
        self.spec, self.meta = spec, meta
        self.data_file = Path(spec.input_path)
        self.time, self.inputd, self.datout = time, inputd, datout
        self.n_in  = self.inputd.shape[1]
        self.n_out = self.datout.shape[1]
        self.in_names  = meta['in_names']
        self.out_names = meta['out_names']
        self.fs = meta['fs']
        if self.cfg.freq_max == 0.0:
            self.cfg.freq_max = self.fs / 2.0
        N = len(self.time)
        print(f'  Samples : {N}  ({N/self.fs:.2f} s)')
        print(f'  Outputs : {self.n_out}   Inputs : {self.n_in}')
        print(f'  Fs      : {self.fs:.1f} Hz')
        return self

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------
    def load_data(self, filepath, col_time=0, col_force=1,
                  col_accel_start=2, n_in=1, raw=None,
                  t_start=None, t_end=None, decimate=1):
        """
        Read one combined record laid out as [time, input(s), responses].

        Kept for callers that describe the file by column index. The header
        rows and delimiter are detected, so this now opens the PicoScope
        exports as well as the plain numeric records. For separate
        input/output files, or a record with no time column, build a
        RecordSpec and call load_record().

        raw : optional (N, n_col) array already read from the file. Lets a
              caller that has parsed the file once (e.g. the GUI's PSD
              preview) re-slice it without a second read.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f'Not found: {p}')
        info = sniff_record(p)
        n_col = info.n_col if raw is None else int(np.asarray(raw).shape[1])
        spec = RecordSpec(
            input_path=p,
            time_col=col_time,
            in_cols=list(range(col_force, col_force + n_in)),
            out_cols=list(range(col_accel_start, n_col)),
            fs=self.cfg.fs,
            decimate=decimate, t_start=t_start, t_end=t_end)
        cache = {str(p): np.asarray(raw)} if raw is not None else None
        return self.load_record(spec, cache=cache)

    # ------------------------------------------------------------------
    # plot_spectral_density
    # ------------------------------------------------------------------
    def plot_spectral_density(self, nperseg=0, f_max=0.0, db_range=120.0,
                              save_path=None, dpi=150):
        """
        Welch power spectral density of the excitation and response channels.

        Needs load_data() only — no CVA results — so a record can be judged
        before committing to a multi-minute sweep: where the drive has energy,
        which response bands are excited, and whether any channel is dead or
        clipped.

        Parameters
        ----------
        nperseg   : Welch segment length in samples. 0 = largest power of two
                    up to 4096 that fits the record.
        f_max     : upper frequency limit of the plots. 0 = Nyquist.
        db_range  : dB shown below each panel's peak.

        Returns the Figure (three stacked panels: excitation PSD, response
        PSD overlay, response PSD per channel as a map).
        """
        if self.datout is None:
            raise RuntimeError('Call load_data() first')

        N = self.datout.shape[0]
        if not nperseg:
            nperseg = int(2 ** np.floor(np.log2(min(N, 4096))))
        nperseg = int(max(64, min(nperseg, N)))

        f, pxx_in  = welch(self.inputd, fs=self.fs, nperseg=nperseg, axis=0)
        _, pxx_out = welch(self.datout, fs=self.fs, nperseg=nperseg, axis=0)
        f_hi = f_max if f_max and f_max < f[-1] else f[-1]
        keep = f <= f_hi
        f = f[keep]
        pxx_in, pxx_out = pxx_in[keep, :], pxx_out[keep, :]

        def _db(x):
            return 10.0 * np.log10(np.clip(x, 1e-30, None))

        db_in, db_out = _db(pxx_in), _db(pxx_out)
        df = f[1] - f[0] if len(f) > 1 else 0.0

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                 layout='constrained')
        fig.suptitle(
            f'Spectral Density — {self.data_file.name if self.data_file else ""}\n'
            f'fs={self.fs:.1f} Hz   {N} samples ({N/self.fs:.2f} s)   '
            f'nperseg={nperseg}   df={df:.2f} Hz',
            fontsize=12, fontweight='bold')

        # ---- excitation ------------------------------------------------
        ax = axes[0]
        for i in range(self.n_in):
            ax.plot(f, db_in[:, i], lw=1.0, label=f'Input {i+1}')
        pk = db_in.max()
        ax.set_ylim(pk - db_range, pk + 5)
        ax.set_ylabel('PSD (dB)')
        ax.set_title(f'Excitation — {self.n_in} channel'
                     f'{"s" if self.n_in > 1 else ""}')
        if self.n_in <= 8:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

        # ---- response overlay ------------------------------------------
        ax = axes[1]
        for j in range(self.n_out):
            ax.plot(f, db_out[:, j], lw=0.6, alpha=0.45, color='steelblue')
        mean_db = _db(pxx_out.mean(axis=1))
        ax.plot(f, mean_db, lw=1.6, color='darkorange',
                label='Mean across channels')
        pk = db_out.max()
        ax.set_ylim(pk - db_range, pk + 5)
        ax.set_ylabel('PSD (dB)')
        ax.set_title(f'Response — {self.n_out} channels overlaid')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

        # ---- response channel map --------------------------------------
        ax = axes[2]
        pk = db_out.max()
        im = ax.imshow(db_out.T, aspect='auto', origin='lower',
                       extent=[f[0], f[-1], -0.5, self.n_out - 0.5],
                       cmap='viridis', vmin=pk - db_range, vmax=pk)
        # Spans all three panels so the shared frequency axis stays aligned.
        fig.colorbar(im, ax=list(axes), label='PSD (dB)')
        ax.set_ylabel('Response channel')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Response PSD per channel')

        # CVA analysis band, when it is narrower than the plotted axis
        lo, hi = self.cfg.freq_min, self.cfg.freq_max
        if hi and (lo > f[0] or hi < f[-1]):
            for a in axes[:2]:
                a.axvspan(lo, min(hi, f[-1]), color='k', alpha=0.05,
                          zorder=0)
                a.axvline(lo, color='k', ls=':', lw=0.8)
                a.axvline(min(hi, f[-1]), color='k', ls=':', lw=0.8)

        axes[0].set_xlim(f[0], f[-1])
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig

    # ------------------------------------------------------------------
    # plot_time_history
    # ------------------------------------------------------------------
    def plot_time_history(self, in_ch=0, out_ch=None, max_out=3,
                          window=None, max_points=3000,
                          save_path=None, dpi=150):
        """
        Excitation and response time histories, for choosing a window.

        Needs load_data()/load_record() only. Plots one excitation channel
        and a few responses rather than all of them — a 9-channel record at
        150 000 samples is unreadable drawn in full, and the point here is
        to see where the useful part of the record is.

        Long records are drawn at reduced resolution as a min/max band with
        the block mean through it, so a transient peak between two plotted
        samples still shows up instead of being skipped.

        Parameters
        ----------
        in_ch      : which excitation channel to show.
        out_ch     : response channels to show; None = the first max_out.
        max_out    : cap on response panels when out_ch is None.
        window     : (t_start, t_end) in seconds to shade as the proposed
                     analysis window. None = shade nothing.
        max_points : target number of plotted points per panel.

        Returns the Figure.
        """
        if self.datout is None:
            raise RuntimeError('Call load_data() or load_record() first')

        if out_ch is None:
            out_ch = list(range(min(max_out, self.n_out)))
        out_ch = [c for c in np.atleast_1d(out_ch).tolist()
                  if 0 <= c < self.n_out]
        in_ch = int(np.clip(in_ch, 0, self.n_in - 1))

        t = np.asarray(self.time, dtype=float)
        st = cva_record.classify(self.inputd, self.datout)

        def label(kind, idx):
            names = self.in_names if kind == 'in' else self.out_names
            units = self.meta.get(f'{kind}_units', []) if self.meta else []
            name = (names[idx] if idx < len(names)
                    else f'{"Input" if kind == "in" else "Response"} {idx}')
            unit = units[idx] if idx < len(units) else ''
            return f'{name} {unit}'.strip()

        rows = 1 + len(out_ch)
        fig, axes = plt.subplots(rows, 1, figsize=(12, 2.2 * rows + 1.2),
                                 sharex=True, layout='constrained')
        axes = np.atleast_1d(axes)

        src = Path(self.meta.get('input_path', str(self.data_file or ''))).name
        if self.meta.get('response_path'):
            src += ' + ' + Path(self.meta['response_path']).name
        extra = ''
        if self.meta.get('decimate', 1) > 1:
            extra += f'   decimated {self.meta["decimate"]}x'
        if self.meta.get('align_decimate', 1) > 1:
            extra += f'   input aligned {self.meta["align_decimate"]}x'
        # Three short lines rather than one long one: the single-line form
        # ran off both ends of the figure for a paired record.
        fig.suptitle(
            f'Time History — {src}\n'
            f'fs={self.fs:.1f} Hz   {len(t)} samples '
            f'({t[0]:.4f}-{t[-1]:.4f} s){extra}\n'
            f'{st}',
            fontsize=10, fontweight='bold')

        panels = [(axes[0], self.inputd[:, in_ch], label('in', in_ch),
                   'darkorange')]
        for ax, c in zip(axes[1:], out_ch):
            panels.append((ax, self.datout[:, c], label('out', c),
                           'steelblue'))

        for ax, y, lab, colour in panels:
            tb, mean, lo, hi = _lod(t, np.asarray(y, dtype=float), max_points)
            if lo is not None:
                ax.fill_between(tb, lo, hi, color=colour, alpha=0.45,
                                linewidth=0)
                ax.plot(tb, mean, color=colour, lw=0.6)
            else:
                ax.plot(tb, mean, color=colour, lw=0.7)
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)

        if window is not None:
            w0, w1 = float(window[0]), float(window[1])
            for ax in axes:
                ax.axvspan(w0, w1, color='seagreen', alpha=0.12, zorder=0)
                ax.axvline(w0, color='seagreen', ls='--', lw=1.0)
                ax.axvline(w1, color='seagreen', ls='--', lw=1.0)
            n_sel = int(((t >= w0) & (t <= w1)).sum())
            axes[0].set_title(
                f'window {w0:.4f}-{w1:.4f} s   ->   {n_sel} samples '
                f'({n_sel / self.fs:.3f} s)', fontsize=9)

        axes[-1].set_xlabel('Time (s)')
        axes[0].set_xlim(t[0], t[-1])
        if len(t) > max_points * 2:
            # Out of the axes, where it cannot sit on top of a trace.
            fig.text(0.995, 0.004,
                     f'drawn as min/max bands, '
                     f'{int(np.ceil(len(t) / max_points))} samples/point',
                     ha='right', va='bottom', fontsize=7, color='0.45')

        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig

    # ------------------------------------------------------------------
    # build_database
    # ------------------------------------------------------------------
    def build_database(self):
        if self.datout is None:
            raise RuntimeError('Call load_data() first')
        lags = self.cfg.lags
        print(f'\nBuilding database  (lags={lags}) ...')
        full = delaybuild(self.datout.T, self.n_out, lags)
        self.past_db   = full[:-1, :]
        self.future_db = full[1:,  :]
        self.n_win     = self.past_db.shape[0]
        print(f'  Windows : {self.n_win}')
        print(f'  past_db : {self.past_db.shape}')

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self, n_jobs=1, progress_cb=None, stop_cb=None):
        """
        Run CVA over all windows.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (default 1 = serial).
            -1 = use all available CPU cores.
            Requires ranklock > 0 OR runs window 0 serially first to
            determine rank, then parallelizes remaining windows.
        progress_cb : callable(done, total, elapsed, eta) or None
            Called as the sweep advances, on both the serial and the
            parallel path, roughly 100 times per run. Used by the GUI to
            drive its progress bar; a script can leave it None and read the
            printed percentages instead.

            It may be called from a joblib worker thread, so it must be
            cheap and thread-safe — set a value and return. Emitting a Qt
            signal from it will deadlock the GUI.
        stop_cb : callable() -> bool or None
            Polled once per window. Returning True abandons the sweep and
            raises InterruptedError, so a long run can be cancelled without
            waiting for the stage to finish.
        """
        if self.past_db is None:
            raise RuntimeError('Call build_database() first')
        cfg  = self.cfg
        lags = cfg.lags
        win_indices = np.arange(0, self.n_win, cfg.eval_stride)
        # Filter out windows where inow would exceed time array
        win_indices = [kw for kw in win_indices if kw + lags < len(self.time)]
        n_eval = len(win_indices)
        print(f'\nRunning CVA ...')
        print(f'  lags={lags}  tol={cfg.tol:.1e}  neighbors={cfg.neighbors}')
        print(f'  Windows to evaluate : {n_eval}  n_jobs={n_jobs}')
        self.windows = []
        reset_cva_state()
        t_start = time.time()
        print_every = max(1, n_eval // 10)
        last_report = [0.0]

        def _stop_requested():
            return stop_cb is not None and stop_cb()

        def _report(done):
            """One window finished. Prints every 10%, and calls progress_cb
            at most four times a second — often enough for a smooth bar,
            seldom enough to cost nothing on a fast sweep."""
            now = time.time()
            el  = now - t_start
            eta = el / done * (n_eval - done) if done else 0.0
            if done % print_every == 0 or done == n_eval:
                print(f'  {done/n_eval*100:5.1f}%  ({done}/{n_eval})'
                      f'  elapsed={el:.1f}s  eta={eta:.1f}s', flush=True)
            if progress_cb is not None and (now - last_report[0] >= 0.25
                                            or done == n_eval):
                last_report[0] = now
                progress_cb(done, n_eval, el, eta)

        # ----------------------------------------------------------
        # Determine ranklock from first window (serial)
        # ----------------------------------------------------------
        ranklock = cfg.ranklock
        kw0 = win_indices[0]
        inow0 = kw0 + lags
        try:
            r0 = cvaonestep(
                self.past_db, self.future_db, self.inputd,
                self.past_db[kw0:kw0+1, :],
                self.future_db[kw0:kw0+1, :],
                self.inputd[inow0:inow0+1, :],
                1,   # inow=1 triggers rank determination
                cfg.tol, cfg.neighbors, ranklock)
            if ranklock == 0:
                ranklock = int(r0['rank'])
                print(f'  Rank determined from window 0: {ranklock}')
            # Store window 0 result
            rnk0 = r0['rank']
            self.windows.append(CVAWindowResult(
                window_idx  = int(kw0),
                sample_idx  = int(inow0),
                time        = float(self.time[inow0]),
                rank        = rnk0,
                eigenvalues = np.linalg.eigvals(r0['Phi']),
                Phi         = r0['Phi'],
                Jnow        = r0['J'][:rnk0, :],
                spast       = r0['spast'],
                sfuture     = r0['sfuture'],
                indpast     = r0['indpast'],
            ))
        except Exception as e:
            warnings.warn(f'Window 0 failed: {e}')

        remaining = win_indices[1:]

        # ----------------------------------------------------------
        # Serial path
        # ----------------------------------------------------------
        if n_jobs in (0, 1) or len(remaining) == 0:   # 0 is meaningless to joblib
            for ki, kw in enumerate(remaining, start=1):
                if _stop_requested():
                    raise InterruptedError(
                        f'CVA sweep stopped after {len(self.windows)} '
                        f'of {n_eval} windows')
                inow = kw + lags
                try:
                    result = cvaonestep(
                        self.past_db, self.future_db, self.inputd,
                        self.past_db[kw:kw+1, :],
                        self.future_db[kw:kw+1, :],
                        self.inputd[inow:inow+1, :],
                        inow,
                        cfg.tol, cfg.neighbors, ranklock)
                    rnk  = result['rank']
                    self.windows.append(CVAWindowResult(
                        window_idx  = int(kw),
                        sample_idx  = int(inow),
                        time        = float(self.time[inow]),
                        rank        = rnk,
                        eigenvalues = np.linalg.eigvals(result['Phi']),
                        Phi         = result['Phi'],
                        Jnow        = result['J'][:rnk, :],
                        spast       = result['spast'],
                        sfuture     = result['sfuture'],
                        indpast     = result['indpast'],
                    ))
                except Exception as e:
                    warnings.warn(f'Window {kw} failed: {e}')

                _report(ki + 1)

        # ----------------------------------------------------------
        # Parallel path
        # ----------------------------------------------------------
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                warnings.warn('joblib not installed — falling back to serial. pip install joblib')
                return self.run(n_jobs=1, progress_cb=progress_cb,
                                stop_cb=stop_cb)

            # Capture arrays as local variables for pickling
            past_db   = self.past_db
            future_db = self.future_db
            inputd    = self.inputd
            time_vec  = self.time

            # Workers are threads, so a plain lock is enough to keep the
            # finished-window count and its reports serialised.
            tick_lock = threading.Lock()
            n_done    = [1]          # window 0 already ran above
            stopped   = [False]

            def _tick():
                with tick_lock:
                    n_done[0] += 1
                    _report(n_done[0])

            def _run_window(kw):
                if stopped[0] or _stop_requested():
                    # Returning early lets the queued windows drain in
                    # milliseconds instead of cancelling mid-factorisation.
                    stopped[0] = True
                    return None
                inow = kw + lags
                try:
                    result = cvaonestep(
                        past_db, future_db, inputd,
                        past_db[kw:kw+1, :],
                        future_db[kw:kw+1, :],
                        inputd[inow:inow+1, :],
                        inow,
                        cfg.tol, cfg.neighbors, ranklock)
                    rnk = result['rank']
                    return (kw, inow, float(time_vec[inow]), rnk,
                            np.linalg.eigvals(result['Phi']),
                            result['Phi'],
                            result['J'][:rnk, :],
                            result['spast'],
                            result['sfuture'],
                            result['indpast'])
                except Exception as e:
                    warnings.warn(f'Window {kw} failed: {e}')
                    return None
                finally:
                    _tick()

            print(f'  Launching parallel jobs (n_jobs={n_jobs}) ...')
            raw = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_run_window)(kw) for kw in remaining)

            # Collect results and sort by window index
            for res in raw:
                if res is None:
                    continue
                kw, inow, t, rnk, ev, Phi, Jnow, spast, sfuture, indpast = res
                self.windows.append(CVAWindowResult(
                    window_idx  = int(kw),
                    sample_idx  = int(inow),
                    time        = t,
                    rank        = rnk,
                    eigenvalues = ev,
                    Phi         = Phi,
                    Jnow        = Jnow,
                    spast       = spast,
                    sfuture     = sfuture,
                    indpast     = indpast,
                ))

            # Sort windows by index so downstream code stays consistent
            self.windows.sort(key=lambda w: w.window_idx)

            if stopped[0]:
                raise InterruptedError(
                    f'CVA sweep stopped after {len(self.windows)} '
                    f'of {n_eval} windows')

        print(f'\nDone. {len(self.windows)} windows in '
              f'{time.time()-t_start:.1f}s')

    # ------------------------------------------------------------------
    # compute_one_step_error
    # ------------------------------------------------------------------
    def compute_one_step_error(self, stride: int = 10,
                               verbose: bool = True) -> dict:
        """
        Compute physical output one-step prediction error per channel.

        Matches MATLAB cva_sweep_error logic:
          1. For each evaluated window, project the NEXT past window
             through Jnow to get s_next
          2. Fit C_obs from neighbor states and outputs
          3. y_hat = C_obs @ s_next
          4. Compare to actual output at inow

        Parameters
        ----------
        stride  : evaluate every Nth window (default 10)
        verbose : print summary statistics

        Returns
        -------
        dict with keys:
            errors_per_ch  : (n_pts, n_out) array of absolute errors
            rms_per_ch     : (n_out,) normalized RMS per channel
            rms_mean       : float  mean across channels
            times          : (n_pts,) evaluation times
        """
        if not self.windows:
            raise RuntimeError('Call run() first')

        lags   = self.cfg.lags
        n_out  = self.n_out
        datout = self.datout
        inputd = self.inputd

        errors_list = []
        times_list  = []

        for wi in range(0, len(self.windows), stride):
            wr  = self.windows[wi]
            kw  = wr.window_idx
            rk  = wr.rank
            if rk == 0:
                continue

            # Next window index and sample
            kw_next   = kw + 1
            inow_next = wr.sample_idx + 1
            if kw_next >= self.past_db.shape[0]:
                continue
            if inow_next >= len(datout):
                continue

            # Project next past+input through Jnow → s_next
            aug_next = np.concatenate([
                self.past_db[kw_next, :],
                inputd[inow_next, :]
            ])
            s_next = wr.Jnow @ aug_next    # (rank,)

            # Actual output at inow (same as MATLAB datout(idx_now+lags_past))
            y_true = datout[wr.sample_idx, :]   # (n_out,)

            # C_obs per channel from neighbor states and outputs
            indp  = wr.indpast
            nn_p1 = indp + lags - 1
            valid = (nn_p1 >= 0) & (nn_p1 < len(datout))
            if valid.sum() < 2:
                continue

            sp_use = wr.spast[valid, :]    # (n_valid, rank)
            err_ch = np.zeros(n_out)

            for ch in range(n_out):
                nb_out = datout[nn_p1[valid], ch]
                n_use  = min(sp_use.shape[0], len(nb_out))
                try:
                    C_obs, _, _, _ = np.linalg.lstsq(
                        sp_use[:n_use, :], nb_out[:n_use], rcond=None)
                    if len(C_obs) > rk:
                        C_obs = C_obs[:rk]
                    y_hat    = float(np.real(C_obs @ s_next))
                    err_ch[ch] = abs(y_hat - y_true[ch])
                except np.linalg.LinAlgError:
                    err_ch[ch] = np.nan

            errors_list.append(err_ch)
            times_list.append(wr.time)

        if not errors_list:
            return {}

        errors = np.array(errors_list)    # (n_pts, n_out)
        times  = np.array(times_list)

        sig_rms = np.std(datout, axis=0) + 1e-12
        rms_per_ch = (np.sqrt(np.nanmean(errors**2, axis=0)) / sig_rms) * 100

        result = dict(
            errors_per_ch = errors,
            rms_per_ch    = rms_per_ch,
            rms_mean      = float(np.mean(rms_per_ch)),
            times         = times,
        )

        if verbose:
            print(f'\nOne-step prediction error:')
            print(f'  Windows evaluated : {len(errors_list)} (stride={stride})')
            for ch in range(n_out):
                print(f'  Ch {ch+1}: norm RMS = {rms_per_ch[ch]:.3f}%')
            print(f'  Mean across channels: {result["rms_mean"]:.3f}%')

        return result

    # ------------------------------------------------------------------
    # plot_iterated_prediction
    # ------------------------------------------------------------------
    def plot_iterated_prediction(
        self,
        channel:      int  = 0,
        iter_K:       int  = 10,
        iter_stride:  int  = 100,
        reanchor:     bool = True,
        save_path     = None,
        dpi:          int  = 150,
    ):
        """
        Multi-step iterated prediction for one output channel.

        Matches MATLAB iterated prediction section:
          - At each starting window, fit C_obs from neighbors
          - Step 1: use true next past window
          - Steps 2+: shift window, insert previous predicted value
          - Optional reanchor: re-run neighbor search at each step
          - Plot predicted traces overlaid on true signal
          - Report RMS divergence per trial

        Parameters
        ----------
        channel     : output channel (0-based)
        iter_K      : prediction horizon (steps)
        iter_stride : evaluate every Nth window
        reanchor    : re-run neighbor search at each step (adaptive)
        save_path   : PNG save path
        dpi         : figure resolution
        """
        if not self.windows:
            raise RuntimeError('Call run() first')

        lags      = self.cfg.lags
        n_out     = self.n_out
        datout    = self.datout
        inputd    = self.inputd
        time_vec  = self.time
        neighbors = self.cfg.neighbors
        ch        = channel

        print(f'\n=== Iterated Multi-Step Prediction ===')
        print(f'  Channel  : {ch+1}')
        print(f'  Horizon  : {iter_K} steps  ({iter_K/self.fs:.4f} s)')
        print(f'  Stride   : every {iter_stride} windows')
        print(f'  Reanchor : {reanchor}')

        # Select evaluation windows
        eval_wins = list(range(0, len(self.windows), iter_stride))
        eval_wins = [wi for wi in eval_wins
                     if self.windows[wi].rank > 0]
        n_eval    = len(eval_wins)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time_vec, datout[:, ch], 'k-', lw=0.5,
                label='True', zorder=2)

        colors    = plt.cm.viridis(np.linspace(0, 0.9, n_eval))
        rms_divs  = []

        for pi_idx, wi in enumerate(eval_wins):
            wr     = self.windows[wi]
            kw     = wr.window_idx
            rk     = wr.rank
            Jnow   = wr.Jnow          # (rank, ncols)
            inow   = wr.sample_idx    # absolute sample index

            # --- Fit initial C_obs from neighbors ---
            indp   = wr.indpast
            nn_p1  = indp + lags - 1
            valid0 = (nn_p1 >= 0) & (nn_p1 < len(datout))
            if valid0.sum() < 2:
                continue

            sp0    = wr.spast[valid0, :]
            nb_out0 = datout[nn_p1[valid0], ch]
            n_use0  = min(sp0.shape[0], len(nb_out0))
            try:
                C_obs, _, _, _ = np.linalg.lstsq(
                    sp0[:n_use0, :], nb_out0[:n_use0], rcond=None)
                if len(C_obs) > rk:
                    C_obs = C_obs[:rk]
            except np.linalg.LinAlgError:
                continue

            y_iter = np.zeros(iter_K)
            t_iter = np.full(iter_K, np.nan)

            # Initial past and input windows
            past_iter = self.past_db[kw, :].copy()
            inp_iter  = inputd[inow, :].copy()

            for step in range(iter_K):
                if step == 0:
                    # Step 1: use true next window
                    kw_next = kw + 1
                    if kw_next >= self.past_db.shape[0]:
                        break
                    past_next = self.past_db[kw_next, :].copy()
                    inp_next  = inputd[min(inow+1, len(inputd)-1), :].copy()
                else:
                    # Steps 2+: shift window, insert predicted value
                    # Oldest-first ordering: drop first n_out elements,
                    # append new sample at end
                    past_next = np.concatenate([
                        past_iter[n_out:],
                        np.zeros(n_out)
                    ])
                    # Insert predicted value for this channel
                    past_next[-(n_out) + ch] = y_iter[step-1]
                    # Insert true data for other channels where available
                    for cc in range(n_out):
                        if cc != ch:
                            samp = inow + step
                            if samp < len(datout):
                                past_next[-(n_out) + cc] = datout[samp, cc]

                    # Advance input window
                    next_samp = inow + step
                    if next_samp < len(inputd):
                        new_u = inputd[next_samp, :]
                    else:
                        new_u = np.zeros(self.n_in)
                    inp_next = np.concatenate([inp_iter[self.n_in:], new_u])

                # --- Optional reanchor ---
                if reanchor and step > 0:
                    try:
                        nn_model = NearestNeighbors(
                            n_neighbors=min(neighbors+1,
                                            self.past_db.shape[0]-1),
                            algorithm='auto')
                        nn_model.fit(self.past_db)
                        _, idx_new = nn_model.kneighbors(
                            past_next.reshape(1, -1))
                        idx_new = idx_new[0, 1:]   # skip self

                        sp_new   = (Jnow @ np.column_stack([
                            self.past_db[idx_new, :],
                            inputd[idx_new, :]
                        ]).T).T                     # (n_new, rank)

                        nn_p1_new = idx_new + lags - 1
                        valid_nw  = (nn_p1_new >= 0) & (nn_p1_new < len(datout))
                        n_use2    = min(sp_new[valid_nw].shape[0],
                                        valid_nw.sum())
                        if n_use2 >= 2:
                            nb_new = datout[nn_p1_new[valid_nw], ch]
                            C_new, _, _, _ = np.linalg.lstsq(
                                sp_new[valid_nw][:n_use2, :],
                                nb_new[:n_use2], rcond=None)
                            if len(C_new) > rk:
                                C_new = C_new[:rk]
                            C_obs = C_new
                    except Exception:
                        pass   # keep previous C_obs

                # --- Project to state and predict output ---
                aug_next     = np.concatenate([past_next, inp_next])
                s_next       = Jnow @ aug_next
                y_iter[step] = float(np.real(C_obs @ s_next))

                # Time target
                target = inow + lags + step
                if target < len(time_vec):
                    t_iter[step] = time_vec[target]
                else:
                    y_iter = y_iter[:step]
                    t_iter = t_iter[:step]
                    break

                # Advance windows
                past_iter = past_next
                inp_iter  = inp_next

            # Trim valid steps
            valid_s = ~np.isnan(t_iter) & (t_iter > 0)
            t_plot  = t_iter[valid_s]
            y_plot  = y_iter[valid_s]
            if len(t_plot) == 0:
                continue

            # RMS divergence vs true signal
            t0 = t_plot[0]; t1 = t_plot[-1]
            true_seg = datout[(time_vec >= t0) & (time_vec <= t1), ch]
            n_comp   = min(len(y_plot), len(true_seg))
            if n_comp > 0:
                rms_d = (np.sqrt(np.mean((y_plot[:n_comp] -
                                          true_seg[:n_comp])**2)) /
                         (np.sqrt(np.mean(true_seg[:n_comp]**2)) + 1e-12)
                         * 100)
                rms_divs.append(rms_d)
            else:
                rms_d = np.nan

            ax.plot(t_plot, y_plot, '-', color=colors[pi_idx], lw=1.2,
                    label=f't={time_vec[inow]:.3f}s ({rms_d:.1f}%)',
                    zorder=3)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'Ch {ch+1}', fontsize=12)
        ax.set_title(
            f'Iterated Prediction Ch {ch+1}  '
            f'({iter_K} steps, reanchor={reanchor})',
            fontsize=13)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig

        if rms_divs:
            print(f'  Ch {ch+1}: median RMS divergence = '
                  f'{np.median(rms_divs):.1f}%')

    # ------------------------------------------------------------------
    # eigenvalue_significance
    # ------------------------------------------------------------------
    def eigenvalue_significance(self):
        if not self.windows:
            raise RuntimeError('Call run() first')
        fs     = self.fs
        f_min  = self.cfg.freq_min
        f_max  = self.cfg.freq_max
        thresh = self.cfg.sig_threshold_fraction
        print(f'\nComputing eigenvalue significance '
              f'({f_min:.0f}–{f_max:.0f} Hz) ...')

        all_freqs=[]; all_damp=[]; all_mag=[]; all_times=[]

        for wr in self.windows:
            ev   = wr.eigenvalues
            ev   = ev[np.imag(ev) >= 0]
            lam  = np.log(ev + 1e-300) * fs
            freqs= np.abs(np.imag(lam)) / (2*np.pi)
            damp = -np.real(lam) / (np.abs(lam) + 1e-12) * 100
            mag  = np.abs(ev)
            mask = (freqs>=f_min)&(freqs<=f_max)&(damp>=0)&(damp<50)
            all_freqs.extend(freqs[mask].tolist())
            all_damp.extend(damp[mask].tolist())
            all_mag.extend(mag[mask].tolist())
            all_times.extend([wr.time]*int(np.sum(mask)))

        all_freqs=np.array(all_freqs); all_damp=np.array(all_damp)
        all_mag=np.array(all_mag);     all_times=np.array(all_times)

        n_bins  = 200
        f_edges = np.linspace(f_min, f_max, n_bins+1)
        f_cent  = 0.5*(f_edges[:-1]+f_edges[1:])
        hist_mag,_ = np.histogram(all_freqs, bins=f_edges, weights=all_mag)
        tval    = thresh*hist_mag.max() if hist_mag.max()>0 else 0
        peak_idx= self._find_peaks(hist_mag, tval, self.cfg.sig_peak_mindist)

        print(f'  Eigenvalue observations : {len(all_freqs)}')
        print(f'  Peaks found             : {len(peak_idx)}')
        for pi in peak_idx:
            print(f'    {f_cent[pi]:.2f} Hz')

        self.sig = dict(all_freqs=all_freqs, all_damping=all_damp,
                        all_mag=all_mag, all_times=all_times,
                        f_centers=f_cent, hist_mag=hist_mag,
                        peak_freqs=f_cent[peak_idx],
                        peak_mags=hist_mag[peak_idx],
                        threshold=tval)
        return self.sig

    @staticmethod
    def _find_peaks(y, threshold, min_sep=3):
        peaks = []
        for i in range(1, len(y)-1):
            if y[i]>threshold and y[i]>=y[i-1] and y[i]>=y[i+1]:
                if not peaks or (i-peaks[-1])>=min_sep:
                    peaks.append(i)
                elif y[i]>y[peaks[-1]]:
                    peaks[-1]=i
        return np.array(peaks, dtype=int)

    # ------------------------------------------------------------------
    # plot_significance
    # ------------------------------------------------------------------
    def plot_significance(self, save_path=None, dpi=150, z_threshold=None):
        if not self.sig:
            self.eigenvalue_significance()
        sig   = self.sig
        f_min = self.cfg.freq_min
        f_max = self.cfg.freq_max

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Eigenvalue Significance Analysis', fontsize=13,
                     fontweight='bold')

        ax = axes[0,0]
        log_mag = np.log10(np.clip(sig['all_mag'], 1e-8, None))
        sc = ax.scatter(sig['all_times'], sig['all_freqs'],
                        c=log_mag, cmap='viridis', s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='log10 |z|')
        ax.set_xlim(self.time[0], self.time[-1])
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Eigenvalue Freq vs Time')
        ax.grid(True, alpha=0.3)

        ax = axes[0,1]
        ax.bar(sig['f_centers'], sig['hist_mag'],
               width=(sig['f_centers'][1]-sig['f_centers'][0]),
               color='steelblue', alpha=0.7, label='Cumul. magnitude')
        ax.axhline(sig['threshold'], color='k', ls='--', lw=1,
                   label=f'Threshold ({self.cfg.sig_threshold_fraction:.0%})')
        if len(sig['peak_freqs'])>0:
            ax.scatter(sig['peak_freqs'], sig['peak_mags'],
                       marker='v', color='red', s=80, zorder=5,
                       label='Peaks')
            for pf, pm in zip(sig['peak_freqs'], sig['peak_mags']):
                ax.annotate(f'{pf:.1f}', (pf, pm),
                            textcoords='offset points', xytext=(0,6),
                            ha='center', fontsize=7, color='red')
        ax.set_xlim(f_min, f_max)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Cumulative Magnitude')
        ax.set_title(f'Reconstruction Magnitude  '
                     f'({f_min:.0f}–{f_max:.0f} Hz)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1,0]
        mag_thresh = z_threshold if z_threshold is not None else np.median(sig['all_mag'])
        mask = sig['all_mag']>=mag_thresh
        sc2 = ax.scatter(sig['all_times'][mask], sig['all_freqs'][mask],
                         c=sig['all_mag'][mask], cmap='plasma',
                         s=2, alpha=0.6)
        plt.colorbar(sc2, ax=ax, label='|z|')
        ax.set_xlim(self.time[0], self.time[-1])
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Significant Eigenvalues  (|z|>{mag_thresh:.2f})')
        ax.grid(True, alpha=0.3)

        ax = axes[1,1]
        sc3 = ax.scatter(sig['all_freqs'], sig['all_damping'],
                         c=sig['all_mag'], cmap='plasma',
                         s=2, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(sc3, ax=ax, label='|z|')
        ax.set_xlim(f_min, f_max); ax.set_ylim(0, 50)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Damping (%)')
        ax.set_title('Damping vs Frequency')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig
        
    # ------------------------------------------------------------------
    # plot_frequency_deviation
    # ------------------------------------------------------------------
    def plot_frequency_deviation(self, band_half=20.0, min_mag=0.99, ma_window=20,
                                 save_path=None, dpi=150):
        """
        Plot frequency deviation from mean for each detected mode.
    
        For each peak frequency in sig['peak_freqs'], isolates eigenvalues
        within +/- band_half Hz, subtracts the band mean, and plots the
        residual vs time coloured by |z|.
    
        Reveals:
          - Amplitude-dependent softening (gradual drift with ringdown)
          - Inter-modal frequency modulation (oscillatory pattern)
    
        Parameters
        ----------
        band_half : float  — Hz window either side of each peak (default 20)
        min_mag   : float  — minimum |z| threshold (default: median |z|)
        save_path : path   — PNG save path (optional)
        dpi       : int    — figure resolution
        """
        if not self.sig:
            self.eigenvalue_significance()
    
        if min_mag is None:
            min_mag = float(np.median(self.sig['all_mag']))
    
        peak_freqs = self.sig['peak_freqs']
        n_peaks    = len(peak_freqs)
        if n_peaks == 0:
            print('No peaks found in significance data.')
            return
    
        fig, axes = plt.subplots(n_peaks, 1,
                                 figsize=(14, 3*n_peaks),
                                 sharex=True)
        if n_peaks == 1:
            axes = [axes]
    
        for ax, pf in zip(axes, peak_freqs):
            mask = ((np.abs(self.sig['all_freqs'] - pf) < band_half) &
                    (self.sig['all_mag'] >= min_mag))
            f_band = self.sig['all_freqs'][mask]
            t_band = self.sig['all_times'][mask]
            m_band = self.sig['all_mag'][mask]
    
            if len(f_band) == 0:
                ax.set_title(f'{pf:.1f} Hz  — no data in band')
                continue
    
            f_mean  = float(np.mean(f_band))
            f_resid = f_band - f_mean
    
            sort_idx = np.argsort(t_band)
            t_sorted = t_band[sort_idx]
            f_sorted = (f_band - float(np.mean(f_band)))[sort_idx]
            m_sorted = m_band[sort_idx]
            sc = ax.scatter(t_sorted, f_sorted, c=m_sorted,
                            cmap='plasma', s=3, alpha=0.6, zorder=2)
            plt.colorbar(sc, ax=ax, label='|z|')
            if ma_window > 1 and len(f_sorted) > ma_window:
                kernel = np.ones(ma_window) / ma_window
                ma_f   = np.convolve(f_sorted, kernel, mode='valid')
                half   = ma_window // 2
                ma_t   = t_sorted[half : half + len(ma_f)]
                ax.plot(ma_t, ma_f, color='cyan', lw=1.8,
                        alpha=0.9, zorder=3, label=f'MA ({ma_window} pts)')
                ax.legend(fontsize=8, loc='upper right')
            ax.axhline(0, color='k', lw=0.8, ls='--')
            ax.set_ylabel('Δf (Hz)')
            ax.set_title(f'{pf:.1f} Hz  (mean={f_mean:.2f} Hz)')
            ax.grid(True, alpha=0.3)
    
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Frequency Deviation from Mean (per mode)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
    
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig

    # ------------------------------------------------------------------
    # plot_reconstruction
    # ------------------------------------------------------------------
    def plot_reconstruction(self, channel=0, bands=None,
                            band_width=15.0, n_bands_max=8,
                            save_path=None, dpi=150):
        if not self.windows:
            raise RuntimeError('Call run() first')
        if self.past_db is None:
            raise RuntimeError('Call build_database() first')

        fs   = self.fs
        lags = self.cfg.lags

        if bands is not None:
            recon_bands = np.array(bands)
        elif self.sig and len(self.sig.get('peak_freqs',[]))>0:
            peaks = self.sig['peak_freqs'][:n_bands_max]
            recon_bands = np.column_stack([
                np.maximum(0, peaks-band_width),
                np.minimum(fs/2, peaks+band_width)])
        else:
            recon_bands = np.array([[0, fs/2]])

        n_bands       = len(recon_bands)
        original_resp = self.datout[:, channel]
        n_windows     = len(self.windows)

        print(f'\n=== Modal Reconstruction  (Ch {channel+1}) ===')
        all_recon = []

        for b in range(n_bands):
            flo, fhi = recon_bands[b]
            rebuilt     = np.full(n_windows, np.nan)
            rebuilt_uf  = np.full(n_windows, np.nan)
            recon_times = np.full(n_windows, np.nan)

            for j, wr in enumerate(self.windows):
                if wr.rank==0: continue
                idx_j = wr.window_idx
                if idx_j >= self.past_db.shape[0]: break
                target_idx = idx_j + lags
                if target_idx >= len(self.time): continue
                recon_times[j] = self.time[target_idx]
                aug_query = np.concatenate([
                self.past_db[wr.window_idx, :],
                self.inputd[wr.sample_idx, :]
                ])
                statenow = wr.Jnow @ aug_query

                ev_j, V_j = np.linalg.eig(wr.Phi)
                freq_j    = np.abs(np.angle(ev_j))*fs/(2*np.pi)
                in_band   = (freq_j>=flo)&(freq_j<=fhi)
                try:
                    Phi_filt = np.real(
                        V_j @ np.diag(ev_j*in_band) @ np.linalg.inv(V_j))
                except np.linalg.LinAlgError:
                    continue

                indp  = wr.indpast
                nn_p1 = indp + lags
                valid = (nn_p1>=0)&(nn_p1<len(original_resp))
                if valid.sum()<2: continue
                nb_sp  = wr.spast[valid,:]
                nb_out = original_resp[nn_p1[valid]]
                try:
                    result,_,_,_ = np.linalg.lstsq(nb_sp, nb_out, rcond=None)
                    C_obs = result
                except np.linalg.LinAlgError:
                    continue

                rebuilt[j]    = float(np.real(C_obs @ Phi_filt @ statenow))
                rebuilt_uf[j] = float(np.real(C_obs @ statenow))

            vm = ~np.isnan(rebuilt)&~np.isnan(recon_times)
            all_recon.append(dict(flo=flo, fhi=fhi,
                                  times=recon_times[vm],
                                  filtered=rebuilt[vm],
                                  unfiltered=rebuilt_uf[vm]))
            print(f'  Band {b+1} ({flo:.0f}-{fhi:.0f} Hz): {vm.sum()} samples')

        n_rows  = n_bands+1
        colors  = plt.cm.tab10(np.linspace(0, 0.9, n_bands))
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2+2.5*n_rows),
                                 sharex=True)
        if n_rows==1: axes=[axes]
        fig.suptitle(f'Modal Reconstruction — Ch {channel+1}'
                     f'  (lags={lags}, tol={self.cfg.tol:.0e})',
                     fontsize=12, fontweight='bold')

        axes[0].plot(self.time, original_resp, 'b', lw=0.5,
                     label='Original', alpha=0.8)
        if all_recon and len(all_recon[0]['times'])>0:
            axes[0].plot(all_recon[0]['times'],
                         all_recon[0]['unfiltered'],
                         'r', lw=1.0, label='Unfiltered recon', alpha=0.8)
        axes[0].set_ylabel('Response')
        axes[0].set_title('Original vs Unfiltered Reconstruction')
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        for b, r in enumerate(all_recon):
            ax = axes[b+1]
            ax.plot(self.time, original_resp,
                    color=[0.7,0.7,1.0], lw=0.4, alpha=0.7,
                    label='Original')
            if len(r['times'])>0:
                ax.plot(r['times'], r['filtered'],
                        color=colors[b], lw=1.2,
                        label=f"{r['flo']:.0f}–{r['fhi']:.0f} Hz")
            ax.set_ylabel('Response')
            ax.set_title(f"Band {b+1}: {r['flo']:.1f}–{r['fhi']:.1f} Hz")
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------
    def save(self, filepath):
        p = Path(filepath)
        print(f'\nSaving to {p} ...')
        with h5py.File(str(p), 'w') as f:
            g = f.create_group('config')
            g.attrs['lags']        = self.cfg.lags
            g.attrs['tol']         = self.cfg.tol
            g.attrs['neighbors']   = self.cfg.neighbors
            g.attrs['ranklock']    = self.cfg.ranklock
            g.attrs['eval_stride'] = self.cfg.eval_stride
            g.attrs['fs']          = self.fs or 0.0
            g.attrs['freq_min']    = self.cfg.freq_min
            g.attrs['freq_max']    = self.cfg.freq_max
            g.attrs['sig_peak_mindist'] = self.cfg.sig_peak_mindist
            g.attrs['data_file']   = (str(self.data_file)
                                      if self.data_file else '')
            # How the signals were assembled — which files, columns,
            # decimation and time window. JSON keeps it one attribute
            # whatever fields cva_record grows later.
            g.attrs['record_meta'] = json.dumps(self.meta)

            g = f.create_group('data')
            g.create_dataset('time',   data=self.time,   compression='gzip')
            g.create_dataset('inputd', data=self.inputd, compression='gzip')
            g.create_dataset('datout', data=self.datout, compression='gzip')

            g = f.create_group('database')
            g.create_dataset('past_db',   data=self.past_db,
                             compression='gzip')
            g.create_dataset('future_db', data=self.future_db,
                             compression='gzip')

            g = f.create_group('windows')
            g.attrs['n_windows'] = len(self.windows)
            for i, wr in enumerate(self.windows):
                wg = g.create_group(str(i))
                wg.attrs['window_idx'] = wr.window_idx
                wg.attrs['sample_idx'] = wr.sample_idx
                wg.attrs['time']       = wr.time
                wg.attrs['rank']       = wr.rank
                wg.create_dataset('eigenvalues_real',
                                  data=np.real(wr.eigenvalues))
                wg.create_dataset('eigenvalues_imag',
                                  data=np.imag(wr.eigenvalues))
                wg.create_dataset('Phi',     data=wr.Phi,
                                  compression='gzip')
                wg.create_dataset('Jnow',    data=wr.Jnow,
                                  compression='gzip')
                wg.create_dataset('spast',   data=wr.spast,
                                  compression='gzip')
                wg.create_dataset('sfuture', data=wr.sfuture,
                                  compression='gzip')
                wg.create_dataset('indpast', data=wr.indpast)

            if self.sig:
                g = f.create_group('significance')
                for k, v in self.sig.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v, compression='gzip')
                    else:
                        g.attrs[k] = v

        print(f'  Saved {len(self.windows)} windows.')

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, filepath):
        p = Path(filepath)
        print(f'Loading from {p} ...')
        with h5py.File(str(p), 'r') as f:
            ca  = f['config'].attrs
            cfg = CVAConfig(
                lags        = int(ca['lags']),
                tol         = float(ca['tol']),
                neighbors   = int(ca['neighbors']),
                ranklock    = int(ca['ranklock']),
                eval_stride = int(ca['eval_stride']),
                fs          = float(ca['fs']) or None,
                freq_min    = float(ca['freq_min']),
                freq_max    = float(ca['freq_max']),
                sig_peak_mindist = int(ca.get('sig_peak_mindist', 3)))
            run = cls(cfg)
            run.fs        = float(ca['fs'])
            run.data_file = (Path(ca['data_file'])
                             if ca['data_file'] else None)
            run.time   = f['data/time'][:]
            run.inputd = f['data/inputd'][:]
            run.datout = f['data/datout'][:]
            run.n_out  = run.datout.shape[1]
            run.n_in   = run.inputd.shape[1]
            try:
                run.meta = json.loads(ca.get('record_meta', '{}'))
            except (ValueError, TypeError):
                run.meta = {}
            run.in_names  = run.meta.get('in_names', [])
            run.out_names = run.meta.get('out_names', [])
            run.past_db   = f['database/past_db'][:]
            run.future_db = f['database/future_db'][:]
            run.n_win     = run.past_db.shape[0]

            wg = f['windows']
            run.windows = []
            for i in range(int(wg.attrs['n_windows'])):
                w  = wg[str(i)]
                ev = (w['eigenvalues_real'][:] +
                      1j*w['eigenvalues_imag'][:])
                # Jnow may be absent in older HDF5 files
                Jnow = (w['Jnow'][:] if 'Jnow' in w
                        else np.zeros((int(w.attrs['rank']), 1)))
                run.windows.append(CVAWindowResult(
                    window_idx  = int(w.attrs['window_idx']),
                    sample_idx  = int(w.attrs['sample_idx']),
                    time        = float(w.attrs['time']),
                    rank        = int(w.attrs['rank']),
                    eigenvalues = ev,
                    Phi         = w['Phi'][:],
                    Jnow        = Jnow,
                    spast       = w['spast'][:],
                    sfuture     = w['sfuture'][:],
                    indpast     = w['indpast'][:]))

            if 'significance' in f:
                sig = {}
                for k in f['significance'].keys():
                    sig[k] = f['significance'][k][:]
                for k, v in f['significance'].attrs.items():
                    sig[k] = v
                run.sig = sig

        print(f'  Loaded {len(run.windows)} windows.')
        return run

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------
    def __repr__(self):
        parts = []
        if self.datout is not None:
            parts.append(f'data={len(self.time)}')
        if self.past_db is not None:
            parts.append(f'db={self.n_win}')
        if self.windows:
            parts.append(f'results={len(self.windows)}')
        if self.sig:
            parts.append(f'sig={len(self.sig.get("peak_freqs",[]))} peaks')
        has_jnow = (self.windows and
                    self.windows[0].Jnow.shape[1] > 1)
        parts.append(f'Jnow={"yes" if has_jnow else "no"}')
        return (f'CVARun(lags={self.cfg.lags}, tol={self.cfg.tol:.0e}, '
                f'neighbors={self.cfg.neighbors}'
                f'  [{", ".join(parts)}])')