"""
cva_run.py
==========
Object-oriented CVA pipeline for experimental vibration data.

Classes
-------
CVAConfig       — all tuning parameters in one dataclass
CVAWindowResult — results from one CVA window (stored per window)
CVARun          — main pipeline class

Typical usage
-------------
    from cva_run import CVAConfig, CVARun

    cfg = CVAConfig(lags=64, tol=5e-4, neighbors=960)
    run = CVARun(cfg)
    run.load_data('s01d11.txt')
    run.build_database()
    run.run()
    run.eigenvalue_significance()
    run.plot_significance()
    run.save('s01d11_cva.h5')

    # Later session — reload without rerunning
    run2 = CVARun.load('s01d11_cva.h5')
    run2.plot_significance()

Data file format (whitespace- or comma-delimited text)
-------------------------------------------------------
    col 1       : time (s)
    col 2       : force input
    cols 3..N+2 : accelerometer outputs (N channels)

Dependencies
------------
    numpy, scipy, scikit-learn, h5py, matplotlib
    cva_core.py  (must be on Python path or in same directory)

Author: Norman / Claude
Date:   2026-04-12
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# Local module — must be in same directory or on sys.path
from cva_core import cvaonestep, delaybuild, reset_cva_state


# =============================================================================
# CVAConfig — all tuning parameters
# =============================================================================
@dataclass
class CVAConfig:
    """
    All CVA tuning parameters in one place.

    Parameters
    ----------
    lags        : number of time lags (past and future)
    tol         : SVD rank tolerance
    neighbors   : maximum number of nearest neighbors
    ranklock    : fixed rank (0 = auto at first window)
    eval_stride : stride between processed windows (1 = every window)
    fs          : sample rate Hz (auto-detected from data if None)
    freq_min    : min frequency for significance search (Hz)
    freq_max    : max frequency for significance search (Hz)
    sig_threshold_fraction : fraction of max magnitude for peak threshold
    """
    lags:                    int   = 64
    tol:                     float = 5e-4
    neighbors:               int   = 960
    ranklock:                int   = 0
    eval_stride:             int   = 1
    fs:                      Optional[float] = None
    freq_min:                float = 0.0
    freq_max:                float = 0.0     # 0 = Nyquist/2
    sig_threshold_fraction:  float = 0.05


# =============================================================================
# CVAWindowResult — per-window results
# =============================================================================
@dataclass
class CVAWindowResult:
    """Results from one CVA window evaluation."""
    window_idx:  int                        # index into past/future database
    sample_idx:  int                        # absolute sample index in data
    time:        float                      # time (s)
    rank:        int
    eigenvalues: np.ndarray                 # complex, shape (rank,)
    Phi:         np.ndarray                 # (rank, rank)
    spast:       np.ndarray                 # (neighbors, rank)
    sfuture:     np.ndarray                 # (neighbors, rank)
    indpast:     np.ndarray                 # neighbor indices


# =============================================================================
# CVARun — main pipeline
# =============================================================================
class CVARun:
    """
    Full CVA identification pipeline.

    Attributes (populated by methods)
    ----------------------------------
    cfg          : CVAConfig
    data_file    : Path
    time         : (N,) array
    inputd       : (N, nin) array
    datout       : (N, n_out) array
    fs           : float
    past_db      : (n_win, lags*n_out) array
    future_db    : (n_win, lags*n_out) array
    windows      : list of CVAWindowResult
    sig          : dict — eigenvalue significance results
    """

    def __init__(self, cfg: CVAConfig):
        self.cfg       = cfg
        self.data_file: Optional[Path] = None

        # Data arrays
        self.time:    Optional[np.ndarray] = None
        self.inputd:  Optional[np.ndarray] = None
        self.datout:  Optional[np.ndarray] = None
        self.fs:      Optional[float]      = None
        self.n_out:   int = 0
        self.n_in:    int = 0

        # CVA database
        self.past_db:   Optional[np.ndarray] = None
        self.future_db: Optional[np.ndarray] = None
        self.n_win:     int = 0

        # Results
        self.windows: List[CVAWindowResult] = []
        self.sig:     dict = {}

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------
    def load_data(self, filepath: str | Path,
                  col_time: int = 0,
                  col_force: int = 1,
                  col_accel_start: int = 2) -> None:
        """
        Load text data file.

        Parameters
        ----------
        filepath        : path to whitespace/comma delimited text file
        col_time        : column index for time (0-based, default 0)
        col_force       : column index for force input (default 1)
        col_accel_start : first accelerometer column (default 2)
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f'Data file not found: {p}')

        self.data_file = p
        print(f'Loading: {p}')

        raw = np.loadtxt(str(p))

        self.time   = raw[:, col_time]
        self.inputd = raw[:, col_force:col_force+1]   # (N, 1)
        self.datout = raw[:, col_accel_start:]         # (N, n_out)

        N         = len(self.time)
        self.n_out = self.datout.shape[1]
        self.n_in  = self.inputd.shape[1]

        # Sample rate — from config or auto-detected
        if self.cfg.fs is not None:
            self.fs = self.cfg.fs
        else:
            dt = np.median(np.diff(self.time))
            self.fs = 1.0 / dt

        # Default freq_max to Nyquist if not set
        if self.cfg.freq_max == 0.0:
            self.cfg.freq_max = self.fs / 2.0

        print(f'  Samples : {N}  ({N/self.fs:.2f} s)')
        print(f'  Outputs : {self.n_out}   Inputs : {self.n_in}')
        print(f'  Fs      : {self.fs:.1f} Hz')

    # ------------------------------------------------------------------
    # build_database
    # ------------------------------------------------------------------
    def build_database(self) -> None:
       lags = self.cfg.lags
       print(f'\nBuilding database  (lags={lags}) ...')

       # past:   rows 0..N-2*lags     — windows ending at sample t
       # future: rows 1..N-2*lags+1   — windows starting one step later
       full = delaybuild(self.datout.T, self.n_out, lags)

       self.past_db   = full[:-1, :]   # all but last row
       self.future_db = full[1:,  :]   # all but first row — shifted by 1
       self.n_win     = self.past_db.shape[0]

       print(f'  Windows : {self.n_win}')
       print(f'  past_db : {self.past_db.shape}')

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Run CVA over all (strided) windows.
        Populates self.windows list.
        """
        if self.past_db is None:
            raise RuntimeError('Call build_database() before run()')

        cfg       = self.cfg
        lags      = cfg.lags
        stride    = cfg.eval_stride
        neighbors = cfg.neighbors
        tol       = cfg.tol
        ranklock  = cfg.ranklock

        # Window indices to evaluate
        win_indices = np.arange(0, self.n_win, stride)
        n_eval      = len(win_indices)

        print(f'\nRunning CVA ...')
        print(f'  lags={lags}  tol={tol:.1e}  neighbors={neighbors}')
        print(f'  Windows to evaluate : {n_eval}')

        self.windows = []
        reset_cva_state()

        t_start = time.time()

        for ki, kw in enumerate(win_indices):
            inow = kw + lags     # absolute sample index (1-based to match MATLAB)

            if inow >= len(self.time):
                break

            qpast   = self.past_db[kw:kw+1, :]
            qfuture = self.future_db[kw:kw+1, :]
            qinput  = self.inputd[inow:inow+1, :]

            # First call triggers rank lock
            inow_matlab = 1 if ki == 0 else inow

            try:
                result = cvaonestep(
                    past        = self.past_db,
                    future      = self.future_db,
                    inputd      = self.inputd,
                    querypast   = qpast,
                    queryfuture = qfuture,
                    queryinput  = qinput,
                    inow        = inow_matlab,
                    tol         = tol,
                    neighbors   = neighbors,
                    ranklock    = ranklock,
                )

                ev = np.linalg.eigvals(result['Phi'])

                wr = CVAWindowResult(
                    window_idx  = int(kw),
                    sample_idx  = int(inow),
                    time        = float(self.time[inow]),
                    rank        = result['rank'],
                    eigenvalues = ev,
                    Phi         = result['Phi'],
                    spast       = result['spast'],
                    sfuture     = result['sfuture'],
                    indpast     = result['indpast'],
                )
                self.windows.append(wr)

            except Exception as e:
                warnings.warn(f'Window {kw} failed: {e}')

            # Progress every 10%
            if (ki + 1) % max(1, n_eval // 10) == 0:
                elapsed = time.time() - t_start
                pct     = (ki + 1) / n_eval * 100
                eta     = elapsed / (ki + 1) * (n_eval - ki - 1)
                print(f'  {pct:5.1f}%  ({ki+1}/{n_eval})  '
                      f'elapsed={elapsed:.1f}s  eta={eta:.1f}s')

        elapsed = time.time() - t_start
        print(f'\nDone. {len(self.windows)} windows in {elapsed:.1f}s')

    # ------------------------------------------------------------------
    # eigenvalue_significance
    # ------------------------------------------------------------------
    def eigenvalue_significance(self) -> dict:
        """
        Compute eigenvalue significance analysis.

        For each window's eigenvalues, converts to frequency/damping,
        weights by reconstruction magnitude (proxy for observability),
        and accumulates a frequency histogram.

        Populates self.sig and returns it.
        """
        if not self.windows:
            raise RuntimeError('Call run() before eigenvalue_significance()')

        fs      = self.fs
        f_min   = self.cfg.freq_min
        f_max   = self.cfg.freq_max
        thresh  = self.cfg.sig_threshold_fraction

        print(f'\nComputing eigenvalue significance ({f_min:.0f}–{f_max:.0f} Hz) ...')

        # Collect all significant eigenvalues across windows
        all_freqs   = []
        all_damping = []
        all_mag     = []
        all_times   = []

        for wr in self.windows:
            ev = wr.eigenvalues

            # Keep only upper half-plane (conjugate pairs, Im >= 0)
            ev = ev[np.imag(ev) >= 0]

            # Convert z-domain -> frequency and damping
            lam   = np.log(ev) * fs          # continuous-time eigenvalues
            freqs = np.abs(np.imag(lam)) / (2 * np.pi)
            damp  = -np.real(lam) / (np.abs(lam) + 1e-12) * 100  # percent

            # Magnitude of eigenvalue (proximity to unit circle = low damping)
            mag = np.abs(ev)

            # Frequency band filter
            mask = (freqs >= f_min) & (freqs <= f_max) & (damp >= 0) & (damp < 50)

            all_freqs.extend(freqs[mask].tolist())
            all_damping.extend(damp[mask].tolist())
            all_mag.extend(mag[mask].tolist())
            all_times.extend([wr.time] * int(np.sum(mask)))

        all_freqs   = np.array(all_freqs)
        all_damping = np.array(all_damping)
        all_mag     = np.array(all_mag)
        all_times   = np.array(all_times)

        # Frequency histogram weighted by magnitude
        n_bins  = 200
        f_edges = np.linspace(f_min, f_max, n_bins + 1)
        f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])

        hist_counts, _ = np.histogram(all_freqs, bins=f_edges)
        hist_mag,    _ = np.histogram(all_freqs, bins=f_edges, weights=all_mag)

        # Peak detection (simple local max above threshold)
        cumulative_mag = hist_mag
        threshold_val  = thresh * cumulative_mag.max()
        peak_indices   = self._find_peaks(cumulative_mag, threshold_val)
        peak_freqs     = f_centers[peak_indices]
        peak_mags      = cumulative_mag[peak_indices]

        print(f'  Total eigenvalue observations : {len(all_freqs)}')
        print(f'  Peaks found above threshold   : {len(peak_freqs)}')
        for pf, pm in sorted(zip(peak_freqs, peak_mags), key=lambda x: x[0]):
            print(f'    {pf:6.2f} Hz   (mag={pm:.2f})')

        self.sig = {
            'all_freqs':    all_freqs,
            'all_damping':  all_damping,
            'all_mag':      all_mag,
            'all_times':    all_times,
            'f_centers':    f_centers,
            'hist_counts':  hist_counts,
            'hist_mag':     hist_mag,
            'peak_freqs':   peak_freqs,
            'peak_mags':    peak_mags,
            'threshold':    threshold_val,
        }
        return self.sig

    @staticmethod
    def _find_peaks(y: np.ndarray, threshold: float,
                    min_sep: int = 3) -> np.ndarray:
        """Simple local maximum finder above threshold."""
        peaks = []
        n = len(y)
        for i in range(1, n - 1):
            if y[i] > threshold and y[i] >= y[i-1] and y[i] >= y[i+1]:
                # Enforce minimum separation
                if not peaks or (i - peaks[-1]) >= min_sep:
                    peaks.append(i)
                elif y[i] > y[peaks[-1]]:
                    peaks[-1] = i
        return np.array(peaks, dtype=int)

    # ------------------------------------------------------------------
    # plot_significance  (4-panel figure matching MATLAB output)
    # ------------------------------------------------------------------
    def plot_significance(self) -> None:
        """
        4-panel eigenvalue significance figure:
          TL: freq vs time (colored by log10 |eigenvalue|)
          TR: cumulative magnitude histogram with peak markers
          BL: significant eigenvalues only (|z| > threshold)
          BR: damping vs frequency scatter
        """
        if not self.sig:
            self.eigenvalue_significance()

        sig    = self.sig
        fs     = self.fs
        f_min  = self.cfg.freq_min
        f_max  = self.cfg.freq_max

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Eigenvalue Significance Analysis', fontsize=13,
                     fontweight='bold')

        # --- TL: freq vs time, color = log10(|z|) ---
        ax = axes[0, 0]
        log_mag = np.log10(np.clip(sig['all_mag'], 1e-8, None))
        sc = ax.scatter(sig['all_times'], sig['all_freqs'],
                        c=log_mag, cmap='viridis', s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='log10 |z|')
        ax.set_xlim(self.time[0], self.time[-1])
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Eigenvalue Freq vs Time  (color = log10 magnitude)')
        ax.grid(True, alpha=0.3)

        # --- TR: cumulative magnitude histogram + peaks ---
        ax = axes[0, 1]
        ax.bar(sig['f_centers'], sig['hist_mag'],
               width=(sig['f_centers'][1] - sig['f_centers'][0]),
               color='steelblue', alpha=0.7, label='Cumul. magnitude')
        ax.axhline(sig['threshold'], color='k', ls='--', lw=1,
                   label=f'Threshold ({self.cfg.sig_threshold_fraction:.0%})')
        ax.axhline(sig['hist_mag'].max() * 1.05, color='g', ls=':', lw=1)

        if len(sig['peak_freqs']) > 0:
            ax.scatter(sig['peak_freqs'], sig['peak_mags'],
                       marker='v', color='red', s=80, zorder=5,
                       label='Peaks')
            for pf, pm in zip(sig['peak_freqs'], sig['peak_mags']):
                ax.annotate(f'{pf:.1f}', (pf, pm),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=7, color='red')

        ax.set_xlim(f_min, f_max)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Cumulative Magnitude')
        ax.set_title(f'Reconstruction Magnitude by Frequency  '
                     f'(search: {f_min:.0f}–{f_max:.0f} Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- BL: significant eigenvalues (|z| above median) ---
        ax = axes[1, 0]
        mag_thresh = np.median(sig['all_mag'])
        mask_sig   = sig['all_mag'] >= mag_thresh
        sc2 = ax.scatter(sig['all_times'][mask_sig],
                         sig['all_freqs'][mask_sig],
                         c=sig['all_mag'][mask_sig],
                         cmap='plasma', s=2, alpha=0.6)
        plt.colorbar(sc2, ax=ax, label='|z|')
        ax.set_xlim(self.time[0], self.time[-1])
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Significant Eigenvalues  (|z| > {mag_thresh:.2f})')
        ax.grid(True, alpha=0.3)

        # --- BR: damping vs frequency scatter ---
        ax = axes[1, 1]
        sc3 = ax.scatter(sig['all_freqs'], sig['all_damping'],
                         c=sig['all_mag'], cmap='plasma',
                         s=2, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(sc3, ax=ax, label='|z|')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(0, 50)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Damping (%)')
        ax.set_title('Damping vs Frequency  (significant eigenvalues)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------------
    # plot_reconstruction
    # ------------------------------------------------------------------
    def plot_reconstruction(
        self,
        channel:    int = 0,
        bands:      Optional[list] = None,
        band_width: float = 5.0,
        n_bands_max: int  = 8,
    ) -> None:
        """
        Modal time history reconstruction per frequency band.

        For each frequency band, filters the Phi eigenvalues to only those
        in-band, then reconstructs the response using a locally fitted
        observation vector C_obs from neighbor states.

        Parameters
        ----------
        channel    : output channel to reconstruct (0-based, default 0)
        bands      : list of [flo, fhi] pairs in Hz.
                     None = auto-generate from significance peaks.
        band_width : half-width around each peak for auto bands (Hz)
        n_bands_max: max number of auto bands to plot
        """
        if not self.windows:
            raise RuntimeError('Call run() before plot_reconstruction()')
        if self.past_db is None:
            raise RuntimeError('Call build_database() before plot_reconstruction()')

        import numpy as np
        import matplotlib.pyplot as plt

        fs   = self.fs
        lags = self.cfg.lags
        dt   = 1.0 / fs

        # ------------------------------------------------------------------
        # Build frequency bands
        # ------------------------------------------------------------------
        if bands is not None:
            recon_bands = np.array(bands)
        elif self.sig and len(self.sig.get('peak_freqs', [])) > 0:
            peaks = self.sig['peak_freqs'][:n_bands_max]
            recon_bands = np.column_stack([
                np.maximum(0, peaks - band_width),
                np.minimum(fs / 2, peaks + band_width)
            ])
            print(f'Using {len(recon_bands)} auto-detected bands from significance peaks.')
        else:
            recon_bands = np.array([[0, fs / 2]])
            print('No bands detected — using full bandwidth.')

        n_bands      = len(recon_bands)
        original_resp = self.datout[:, channel]
        n_windows    = len(self.windows)

        print(f'\n=== Modal Reconstruction  (Ch {channel+1}) ===')
        for b in range(n_bands):
            print(f'  Band {b+1}: {recon_bands[b,0]:.1f} – {recon_bands[b,1]:.1f} Hz')

        # ------------------------------------------------------------------
        # Reconstruct per band
        # ------------------------------------------------------------------
        all_recon = []

        for b in range(n_bands):
            flo = recon_bands[b, 0]
            fhi = recon_bands[b, 1]

            rebuilt    = np.full(n_windows, np.nan)
            rebuilt_uf = np.full(n_windows, np.nan)
            recon_times = np.full(n_windows, np.nan)

            for j, wr in enumerate(self.windows):
                if wr.rank == 0:
                    continue

                rk    = wr.rank
                idx_j = wr.window_idx
                if idx_j >= self.past_db.shape[0]:
                    break

                # Time target — one step ahead of "now"
                target_idx = idx_j + lags
                if target_idx >= len(self.time):
                    continue
                recon_times[j] = self.time[target_idx]

                # --- State at current window ---
                qp_j = self.past_db[idx_j, :]     # (lags*n_out,)
                qi_j = self.inputd[idx_j, :]       # (n_in,)
                aug  = np.concatenate([qp_j, qi_j])

                J_j = wr.Phi   # NOTE: we use spast centroid instead — see below

                # Use mean of neighbor states as proxy for current state
                statenow = np.mean(wr.spast, axis=0)    # (rk,)

                # --- Band-filtered Phi ---
                ev_j, V_j = np.linalg.eig(wr.Phi)
                freq_j    = np.abs(np.angle(ev_j)) * fs / (2 * np.pi)
                in_band   = (freq_j >= flo) & (freq_j <= fhi)

                try:
                    Phi_filt = np.real(
                        V_j @ np.diag(ev_j * in_band) @ np.linalg.inv(V_j)
                    )
                except np.linalg.LinAlgError:
                    continue

                # --- C_obs from neighbor states and outputs ---
                indp   = wr.indpast
                nn_p1  = indp + lags       # neighbor output sample indices
                valid  = (nn_p1 >= 0) & (nn_p1 < len(original_resp))

                if valid.sum() < 2:
                    continue

                nb_sp  = wr.spast[valid, :]              # (n_valid, rk)
                nb_out = original_resp[nn_p1[valid]]     # (n_valid,)

                try:
                    result, _, _, _ = np.linalg.lstsq(nb_sp, nb_out, rcond=None)
                    C_obs = result    # (rk,)
                except np.linalg.LinAlgError:
                    continue

                rebuilt[j]    = float(np.real(C_obs @ Phi_filt @ statenow))
                rebuilt_uf[j] = float(np.real(C_obs @ statenow))

            valid_mask = ~np.isnan(rebuilt) & ~np.isnan(recon_times)
            all_recon.append({
                'flo':        flo,
                'fhi':        fhi,
                'times':      recon_times[valid_mask],
                'filtered':   rebuilt[valid_mask],
                'unfiltered': rebuilt_uf[valid_mask],
            })
            print(f'    Band {b+1}: {valid_mask.sum()} reconstructed samples')

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        n_rows  = n_bands + 1
        colors  = plt.cm.tab10(np.linspace(0, 0.9, n_bands))
        t_full  = self.time

        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(14, 2 + 2.5 * n_rows),
                                 sharex=True)
        if n_rows == 1:
            axes = [axes]

        fig.suptitle(
            f'Modal Reconstruction — Ch {channel+1}  '
            f'(lags={lags}, tol={self.cfg.tol:.0e})',
            fontsize=12, fontweight='bold')

        # Row 0: original vs unfiltered reconstruction
        ax = axes[0]
        ax.plot(t_full, original_resp, 'b', linewidth=0.5,
                label='Original', alpha=0.8)
        if all_recon:
            r = all_recon[0]
            if len(r['times']) > 0:
                ax.plot(r['times'], r['unfiltered'], 'r', linewidth=1.0,
                        label='Unfiltered recon', alpha=0.8)
        ax.set_ylabel('Response')
        ax.set_title('Original vs Unfiltered Reconstruction')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Rows 1..n_bands: per-band reconstruction
        for b, r in enumerate(all_recon):
            ax = axes[b + 1]
            ax.plot(t_full, original_resp,
                    color=[0.7, 0.7, 1.0], linewidth=0.4, alpha=0.7,
                    label='Original')
            if len(r['times']) > 0:
                ax.plot(r['times'], r['filtered'],
                        color=colors[b], linewidth=1.2,
                        label=f"{r['flo']:.0f}–{r['fhi']:.0f} Hz")
            ax.set_ylabel('Response')
            ax.set_title(f"Band {b+1}: {r['flo']:.1f}–{r['fhi']:.1f} Hz")
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    # ------------------------------------------------------------------
    # save  (HDF5)
    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        """
        Save all results to HDF5 file.

        Structure
        ---------
        /config/           — CVAConfig parameters
        /data/             — time, inputd, datout
        /database/         — past_db, future_db
        /windows/N/        — per-window results (Phi, eigenvalues, etc.)
        /significance/     — sig dict arrays
        """
        p = Path(filepath)
        print(f'\nSaving to {p} ...')

        with h5py.File(str(p), 'w') as f:

            # Config
            grp = f.create_group('config')
            grp.attrs['lags']       = self.cfg.lags
            grp.attrs['tol']        = self.cfg.tol
            grp.attrs['neighbors']  = self.cfg.neighbors
            grp.attrs['ranklock']   = self.cfg.ranklock
            grp.attrs['eval_stride']= self.cfg.eval_stride
            grp.attrs['fs']         = self.fs if self.fs else 0.0
            grp.attrs['freq_min']   = self.cfg.freq_min
            grp.attrs['freq_max']   = self.cfg.freq_max
            grp.attrs['data_file']  = str(self.data_file) if self.data_file else ''

            # Raw data
            grp = f.create_group('data')
            grp.create_dataset('time',   data=self.time,   compression='gzip')
            grp.create_dataset('inputd', data=self.inputd, compression='gzip')
            grp.create_dataset('datout', data=self.datout, compression='gzip')

            # Database
            grp = f.create_group('database')
            grp.create_dataset('past_db',   data=self.past_db,   compression='gzip')
            grp.create_dataset('future_db', data=self.future_db, compression='gzip')

            # Windows
            grp = f.create_group('windows')
            grp.attrs['n_windows'] = len(self.windows)
            for i, wr in enumerate(self.windows):
                wg = grp.create_group(str(i))
                wg.attrs['window_idx'] = wr.window_idx
                wg.attrs['sample_idx'] = wr.sample_idx
                wg.attrs['time']       = wr.time
                wg.attrs['rank']       = wr.rank
                wg.create_dataset('eigenvalues_real', data=np.real(wr.eigenvalues))
                wg.create_dataset('eigenvalues_imag', data=np.imag(wr.eigenvalues))
                wg.create_dataset('Phi',     data=wr.Phi,     compression='gzip')
                wg.create_dataset('spast',   data=wr.spast,   compression='gzip')
                wg.create_dataset('sfuture', data=wr.sfuture, compression='gzip')
                wg.create_dataset('indpast', data=wr.indpast)

            # Significance
            if self.sig:
                grp = f.create_group('significance')
                for key, val in self.sig.items():
                    if isinstance(val, np.ndarray):
                        grp.create_dataset(key, data=val, compression='gzip')
                    else:
                        grp.attrs[key] = val

        print(f'  Saved {len(self.windows)} windows.')

    # ------------------------------------------------------------------
    # load  (class method — reconstruct from HDF5)
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, filepath: str | Path) -> CVARun:
        """
        Reconstruct a CVARun from a previously saved HDF5 file.
        """
        p = Path(filepath)
        print(f'Loading from {p} ...')

        with h5py.File(str(p), 'r') as f:

            # Config
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
            )

            run = cls(cfg)
            run.fs        = float(ca['fs'])
            run.data_file = Path(ca['data_file']) if ca['data_file'] else None

            # Raw data
            run.time   = f['data/time'][:]
            run.inputd = f['data/inputd'][:]
            run.datout = f['data/datout'][:]
            run.n_out  = run.datout.shape[1]
            run.n_in   = run.inputd.shape[1]

            # Database
            run.past_db   = f['database/past_db'][:]
            run.future_db = f['database/future_db'][:]
            run.n_win     = run.past_db.shape[0]

            # Windows
            wgrp     = f['windows']
            n_win    = int(wgrp.attrs['n_windows'])
            run.windows = []
            for i in range(n_win):
                wg = wgrp[str(i)]
                ev = (wg['eigenvalues_real'][:] +
                      1j * wg['eigenvalues_imag'][:])
                wr = CVAWindowResult(
                    window_idx  = int(wg.attrs['window_idx']),
                    sample_idx  = int(wg.attrs['sample_idx']),
                    time        = float(wg.attrs['time']),
                    rank        = int(wg.attrs['rank']),
                    eigenvalues = ev,
                    Phi         = wg['Phi'][:],
                    spast       = wg['spast'][:],
                    sfuture     = wg['sfuture'][:],
                    indpast     = wg['indpast'][:],
                )
                run.windows.append(wr)

            # Significance
            if 'significance' in f:
                sig = {}
                for key in f['significance'].keys():
                    sig[key] = f['significance'][key][:]
                for key, val in f['significance'].attrs.items():
                    sig[key] = val
                run.sig = sig

        print(f'  Loaded {len(run.windows)} windows.')
        return run

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = []
        if self.datout is not None:
            status.append(f'data={len(self.time)} samples')
        if self.past_db is not None:
            status.append(f'db={self.n_win} windows')
        if self.windows:
            status.append(f'results={len(self.windows)} windows')
        if self.sig:
            status.append(f'sig={len(self.sig["peak_freqs"])} peaks')
        return (f'CVARun(lags={self.cfg.lags}, tol={self.cfg.tol:.0e}, '
                f'neighbors={self.cfg.neighbors}  [{", ".join(status)}])')


# =============================================================================
# Quick demo / self-test
# =============================================================================
if __name__ == '__main__':
    import tempfile, os

    print('=== CVARun self-test ===\n')

    # Synthetic data — 4 channels, 1 input, 3 s at 2048 Hz
    rng  = np.random.default_rng(42)
    fs   = 2048.0
    N    = int(3 * fs)
    t    = np.arange(N) / fs
    u    = rng.standard_normal(N)

    # Simple lightly-damped oscillator response
    y = np.zeros((N, 4))
    for ch in range(4):
        f0  = 50 + ch * 20            # Hz
        zeta = 0.02
        omega = 2 * np.pi * f0
        y[:, ch] = np.exp(-zeta * omega * t) * np.sin(omega * t)
    y += 0.05 * rng.standard_normal((N, 4))

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w')
    data_block = np.column_stack([t, u, y])
    np.savetxt(tmp.name, data_block, fmt='%.6f')
    tmp.close()

    # Run pipeline
    cfg = CVAConfig(lags=16, tol=1e-3, neighbors=100,
                    eval_stride=50, fs=fs,
                    freq_min=0, freq_max=200)
    run = CVARun(cfg)
    run.load_data(tmp.name)
    run.build_database()
    run.run()
    print(run)

    run.eigenvalue_significance()

    # Save and reload
    h5_path = tmp.name.replace('.txt', '.h5')
    run.save(h5_path)
    run2 = CVARun.load(h5_path)
    print(run2)

    # Cleanup
    os.unlink(tmp.name)
    os.unlink(h5_path)

    print('\nSelf-test passed.')
