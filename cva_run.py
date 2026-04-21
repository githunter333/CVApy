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
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from cva_core import cvaonestep, delaybuild, reset_cva_state


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
    sig_threshold_fraction: float = 0.05


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
        self.past_db:   Optional[np.ndarray] = None
        self.future_db: Optional[np.ndarray] = None
        self.n_win:     int = 0
        self.windows: List[CVAWindowResult] = []
        self.sig:     dict = {}

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------
    def load_data(self, filepath, col_time=0, col_force=1,
                  col_accel_start=2, n_in=1):
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f'Not found: {p}')
        self.data_file = p
        print(f'Loading: {p}')
        raw = np.loadtxt(str(p))
        self.time   = raw[:, col_time]
        self.inputd = raw[:, col_force:col_force+n_in]
        self.datout = raw[:, col_accel_start:]
        N = len(self.time)
        self.n_out = self.datout.shape[1]
        self.n_in  = self.inputd.shape[1]
        self.fs = (self.cfg.fs if self.cfg.fs
                   else 1.0 / np.median(np.diff(self.time)))
        if self.cfg.freq_max == 0.0:
            self.cfg.freq_max = self.fs / 2.0
        print(f'  Samples : {N}  ({N/self.fs:.2f} s)')
        print(f'  Outputs : {self.n_out}   Inputs : {self.n_in}')
        print(f'  Fs      : {self.fs:.1f} Hz')

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
    def run(self, n_jobs=1):
        """
        Run CVA over all windows.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (default 1 = serial).
            -1 = use all available CPU cores.
            Requires ranklock > 0 OR runs window 0 serially first to
            determine rank, then parallelizes remaining windows.
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
        if n_jobs == 1 or len(remaining) == 0:
            for ki, kw in enumerate(remaining, start=1):
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

                if (ki) % max(1, n_eval//10) == 0:
                    el  = time.time() - t_start
                    eta = el / ki * (n_eval - ki)
                    pct = ki / n_eval * 100
                    print(f'  {pct:5.1f}%  ({ki}/{n_eval})'
                          f'  elapsed={el:.1f}s  eta={eta:.1f}s', flush=True)

        # ----------------------------------------------------------
        # Parallel path
        # ----------------------------------------------------------
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                warnings.warn('joblib not installed — falling back to serial. pip install joblib')
                return self.run(n_jobs=1)

            # Capture arrays as local variables for pickling
            past_db   = self.past_db
            future_db = self.future_db
            inputd    = self.inputd
            time_vec  = self.time

            def _run_window(kw):
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
    ) -> None:
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
        plt.show()

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
        peak_idx= self._find_peaks(hist_mag, tval)

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
        plt.show()
        
    # ------------------------------------------------------------------
    # plot_frequency_deviation
    # ------------------------------------------------------------------
    def plot_frequency_deviation(self, band_half=20.0, min_mag=0.99
                                 ,
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
    
            sc = ax.scatter(t_band, f_resid, c=m_band,
                            cmap='plasma', s=3, alpha=0.6)
            plt.colorbar(sc, ax=ax, label='|z|')
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
        plt.show()

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
        plt.show()

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
            g.attrs['data_file']   = (str(self.data_file)
                                      if self.data_file else '')

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
                freq_max    = float(ca['freq_max']))
            run = cls(cfg)
            run.fs        = float(ca['fs'])
            run.data_file = (Path(ca['data_file'])
                             if ca['data_file'] else None)
            run.time   = f['data/time'][:]
            run.inputd = f['data/inputd'][:]
            run.datout = f['data/datout'][:]
            run.n_out  = run.datout.shape[1]
            run.n_in   = run.inputd.shape[1]
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