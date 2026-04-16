"""
cva_run.py
==========
Object-oriented CVA pipeline for experimental vibration data.

Classes: CVAConfig, CVARun

Author: Norman / Claude
Date:   2026-04-14
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

from cva_core import cvaonestep, delaybuild, reset_cva_state


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


@dataclass
class CVAWindowResult:
    window_idx:  int
    sample_idx:  int
    time:        float
    rank:        int
    eigenvalues: np.ndarray
    Phi:         np.ndarray
    spast:       np.ndarray
    sfuture:     np.ndarray
    indpast:     np.ndarray


class CVARun:
    def __init__(self, cfg: CVAConfig):
        self.cfg      = cfg
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
    def load_data(self, filepath, col_time=0, col_force=1,
                  col_accel_start=2):
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f'Not found: {p}')
        self.data_file = p
        print(f'Loading: {p}')
        raw = np.loadtxt(str(p))
        self.time   = raw[:, col_time]
        self.inputd = raw[:, col_force:col_force+1]
        self.datout = raw[:, col_accel_start:]
        N = len(self.time)
        self.n_out = self.datout.shape[1]
        self.n_in  = self.inputd.shape[1]
        self.fs = self.cfg.fs if self.cfg.fs else 1.0 / np.median(np.diff(self.time))
        if self.cfg.freq_max == 0.0:
            self.cfg.freq_max = self.fs / 2.0
        print(f'  Samples : {N}  ({N/self.fs:.2f} s)')
        print(f'  Outputs : {self.n_out}   Inputs : {self.n_in}')
        print(f'  Fs      : {self.fs:.1f} Hz')

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
    def run(self):
        if self.past_db is None:
            raise RuntimeError('Call build_database() first')
        cfg = self.cfg
        lags = cfg.lags
        win_indices = np.arange(0, self.n_win, cfg.eval_stride)
        n_eval = len(win_indices)
        print(f'\nRunning CVA ...')
        print(f'  lags={lags}  tol={cfg.tol:.1e}  neighbors={cfg.neighbors}')
        print(f'  Windows to evaluate : {n_eval}')
        self.windows = []
        reset_cva_state()
        t_start = time.time()

        for ki, kw in enumerate(win_indices):
            inow = kw + lags
            if inow >= len(self.time):
                break
            qpast   = self.past_db[kw:kw+1, :]
            qfuture = self.future_db[kw:kw+1, :]
            qinput  = self.inputd[inow:inow+1, :]
            inow_m  = 1 if ki == 0 else inow

            try:
                result = cvaonestep(
                    self.past_db, self.future_db, self.inputd,
                    qpast, qfuture, qinput, inow_m,
                    cfg.tol, cfg.neighbors, cfg.ranklock)
                ev = np.linalg.eigvals(result['Phi'])
                self.windows.append(CVAWindowResult(
                    window_idx  = int(kw),
                    sample_idx  = int(inow),
                    time        = float(self.time[inow]),
                    rank        = result['rank'],
                    eigenvalues = ev,
                    Phi         = result['Phi'],
                    spast       = result['spast'],
                    sfuture     = result['sfuture'],
                    indpast     = result['indpast'],
                ))
            except Exception as e:
                warnings.warn(f'Window {kw} failed: {e}')

            if (ki+1) % max(1, n_eval//10) == 0:
                el  = time.time() - t_start
                eta = el / (ki+1) * (n_eval - ki - 1)
                pct = (ki+1) / n_eval * 100
                print(f'  {pct:5.1f}%  ({ki+1}/{n_eval})'
                      f'  elapsed={el:.1f}s  eta={eta:.1f}s', flush=True)

        print(f'\nDone. {len(self.windows)} windows in {time.time()-t_start:.1f}s')

    # ------------------------------------------------------------------
    def eigenvalue_significance(self):
        if not self.windows:
            raise RuntimeError('Call run() first')
        fs    = self.fs
        f_min = self.cfg.freq_min
        f_max = self.cfg.freq_max
        thresh = self.cfg.sig_threshold_fraction
        print(f'\nComputing eigenvalue significance ({f_min:.0f}–{f_max:.0f} Hz) ...')

        all_freqs = []; all_damp = []; all_mag = []; all_times = []

        for wr in self.windows:
            ev = wr.eigenvalues
            ev = ev[np.imag(ev) >= 0]
            lam   = np.log(ev + 1e-300) * fs
            freqs = np.abs(np.imag(lam)) / (2 * np.pi)
            damp  = -np.real(lam) / (np.abs(lam) + 1e-12) * 100
            mag   = np.abs(ev)
            mask  = (freqs >= f_min) & (freqs <= f_max) & (damp >= 0) & (damp < 50)
            all_freqs.extend(freqs[mask].tolist())
            all_damp.extend(damp[mask].tolist())
            all_mag.extend(mag[mask].tolist())
            all_times.extend([wr.time] * int(np.sum(mask)))

        all_freqs = np.array(all_freqs); all_damp  = np.array(all_damp)
        all_mag   = np.array(all_mag);   all_times = np.array(all_times)

        n_bins   = 200
        f_edges  = np.linspace(f_min, f_max, n_bins+1)
        f_cent   = 0.5 * (f_edges[:-1] + f_edges[1:])
        hist_mag, _ = np.histogram(all_freqs, bins=f_edges, weights=all_mag)
        tval        = thresh * hist_mag.max() if hist_mag.max() > 0 else 0
        peak_idx    = self._find_peaks(hist_mag, tval)

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
            if y[i] > threshold and y[i] >= y[i-1] and y[i] >= y[i+1]:
                if not peaks or (i - peaks[-1]) >= min_sep:
                    peaks.append(i)
                elif y[i] > y[peaks[-1]]:
                    peaks[-1] = i
        return np.array(peaks, dtype=int)

    # ------------------------------------------------------------------
    def plot_significance(self, save_path=None, dpi=150):
        if not self.sig:
            self.eigenvalue_significance()
        sig   = self.sig
        f_min = self.cfg.freq_min
        f_max = self.cfg.freq_max

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Eigenvalue Significance Analysis', fontsize=13,
                     fontweight='bold')

        ax = axes[0, 0]
        log_mag = np.log10(np.clip(sig['all_mag'], 1e-8, None))
        sc = ax.scatter(sig['all_times'], sig['all_freqs'],
                        c=log_mag, cmap='viridis', s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='log10 |z|')
        ax.set_xlim(self.time[0], self.time[-1])
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Eigenvalue Freq vs Time  (color = log10 magnitude)')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.bar(sig['f_centers'], sig['hist_mag'],
               width=(sig['f_centers'][1]-sig['f_centers'][0]),
               color='steelblue', alpha=0.7, label='Cumul. magnitude')
        ax.axhline(sig['threshold'], color='k', ls='--', lw=1,
                   label=f'Threshold ({self.cfg.sig_threshold_fraction:.0%})')
        if sig['f_centers'][-1] > 0:
            ax.axhline(sig['hist_mag'].max()*1.05, color='g', ls=':', lw=1)
        if len(sig['peak_freqs']) > 0:
            ax.scatter(sig['peak_freqs'], sig['peak_mags'],
                       marker='v', color='red', s=80, zorder=5, label='Peaks')
            for pf, pm in zip(sig['peak_freqs'], sig['peak_mags']):
                ax.annotate(f'{pf:.1f}', (pf, pm),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=7, color='red')
        ax.set_xlim(f_min, f_max)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Cumulative Magnitude')
        ax.set_title(f'Reconstruction Magnitude  (search: {f_min:.0f}–{f_max:.0f} Hz)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        mag_thresh = np.median(sig['all_mag'])
        mask = sig['all_mag'] >= mag_thresh
        sc2 = ax.scatter(sig['all_times'][mask], sig['all_freqs'][mask],
                         c=sig['all_mag'][mask], cmap='plasma', s=2, alpha=0.6)
        plt.colorbar(sc2, ax=ax, label='|z|')
        ax.set_xlim(self.time[0], self.time[-1]); ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Significant Eigenvalues  (|z| > {mag_thresh:.2f})')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        sc3 = ax.scatter(sig['all_freqs'], sig['all_damping'],
                         c=sig['all_mag'], cmap='plasma',
                         s=2, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(sc3, ax=ax, label='|z|')
        ax.set_xlim(f_min, f_max); ax.set_ylim(0, 50)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Damping (%)')
        ax.set_title('Damping vs Frequency  (significant eigenvalues)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        plt.show()

    # ------------------------------------------------------------------
    def plot_reconstruction(self, channel=0, bands=None,
                            band_width=5.0, n_bands_max=8,
                            save_path=None, dpi=150):
        if not self.windows:
            raise RuntimeError('Call run() first')
        if self.past_db is None:
            raise RuntimeError('Call build_database() first')

        fs   = self.fs
        lags = self.cfg.lags

        if bands is not None:
            recon_bands = np.array(bands)
        elif self.sig and len(self.sig.get('peak_freqs', [])) > 0:
            peaks = self.sig['peak_freqs'][:n_bands_max]
            recon_bands = np.column_stack([
                np.maximum(0, peaks - band_width),
                np.minimum(fs/2, peaks + band_width)])
            print(f'Using {len(recon_bands)} auto-detected bands.')
        else:
            recon_bands = np.array([[0, fs/2]])

        n_bands       = len(recon_bands)
        original_resp = self.datout[:, channel]
        n_windows     = len(self.windows)

        print(f'\n=== Modal Reconstruction  (Ch {channel+1}) ===')
        all_recon = []

        for b in range(n_bands):
            flo, fhi = recon_bands[b]
            rebuilt    = np.full(n_windows, np.nan)
            rebuilt_uf = np.full(n_windows, np.nan)
            recon_times = np.full(n_windows, np.nan)

            for j, wr in enumerate(self.windows):
                if wr.rank == 0: continue
                idx_j = wr.window_idx
                if idx_j >= self.past_db.shape[0]: break
                target_idx = idx_j + lags
                if target_idx >= len(self.time): continue
                recon_times[j] = self.time[target_idx]
                statenow = np.mean(wr.spast, axis=0)

                ev_j, V_j = np.linalg.eig(wr.Phi)
                freq_j    = np.abs(np.angle(ev_j)) * fs / (2*np.pi)
                in_band   = (freq_j >= flo) & (freq_j <= fhi)
                try:
                    Phi_filt = np.real(V_j @ np.diag(ev_j * in_band) @ np.linalg.inv(V_j))
                except np.linalg.LinAlgError:
                    continue

                indp   = wr.indpast
                nn_p1  = indp + lags
                valid  = (nn_p1 >= 0) & (nn_p1 < len(original_resp))
                if valid.sum() < 2: continue
                nb_sp  = wr.spast[valid, :]
                nb_out = original_resp[nn_p1[valid]]
                try:
                    result, _, _, _ = np.linalg.lstsq(nb_sp, nb_out, rcond=None)
                    C_obs = result
                except np.linalg.LinAlgError:
                    continue

                rebuilt[j]    = float(np.real(C_obs @ Phi_filt @ statenow))
                rebuilt_uf[j] = float(np.real(C_obs @ statenow))

            vm = ~np.isnan(rebuilt) & ~np.isnan(recon_times)
            all_recon.append(dict(flo=flo, fhi=fhi,
                                  times=recon_times[vm],
                                  filtered=rebuilt[vm],
                                  unfiltered=rebuilt_uf[vm]))
            print(f'  Band {b+1} ({flo:.0f}-{fhi:.0f} Hz): {vm.sum()} samples')

        n_rows  = n_bands + 1
        colors  = plt.cm.tab10(np.linspace(0, 0.9, n_bands))
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2+2.5*n_rows),
                                 sharex=True)
        if n_rows == 1: axes = [axes]
        fig.suptitle(f'Modal Reconstruction — Ch {channel+1}'
                     f'  (lags={lags}, tol={self.cfg.tol:.0e})',
                     fontsize=12, fontweight='bold')

        axes[0].plot(self.time, original_resp, 'b', lw=0.5,
                     label='Original', alpha=0.8)
        if all_recon and len(all_recon[0]['times']) > 0:
            axes[0].plot(all_recon[0]['times'], all_recon[0]['unfiltered'],
                         'r', lw=1.0, label='Unfiltered recon', alpha=0.8)
        axes[0].set_ylabel('Response')
        axes[0].set_title('Original vs Unfiltered Reconstruction')
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        for b, r in enumerate(all_recon):
            ax = axes[b+1]
            ax.plot(self.time, original_resp,
                    color=[0.7, 0.7, 1.0], lw=0.4, alpha=0.7, label='Original')
            if len(r['times']) > 0:
                ax.plot(r['times'], r['filtered'], color=colors[b], lw=1.2,
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
            g.attrs['data_file']   = str(self.data_file) if self.data_file else ''

            g = f.create_group('data')
            g.create_dataset('time',   data=self.time,   compression='gzip')
            g.create_dataset('inputd', data=self.inputd, compression='gzip')
            g.create_dataset('datout', data=self.datout, compression='gzip')

            g = f.create_group('database')
            g.create_dataset('past_db',   data=self.past_db,   compression='gzip')
            g.create_dataset('future_db', data=self.future_db, compression='gzip')

            g = f.create_group('windows')
            g.attrs['n_windows'] = len(self.windows)
            for i, wr in enumerate(self.windows):
                wg = g.create_group(str(i))
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

            if self.sig:
                g = f.create_group('significance')
                for k, v in self.sig.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v, compression='gzip')
                    else:
                        g.attrs[k] = v

        print(f'  Saved {len(self.windows)} windows.')

    @classmethod
    def load(cls, filepath):
        p = Path(filepath)
        print(f'Loading from {p} ...')
        with h5py.File(str(p), 'r') as f:
            ca  = f['config'].attrs
            cfg = CVAConfig(lags=int(ca['lags']), tol=float(ca['tol']),
                            neighbors=int(ca['neighbors']),
                            ranklock=int(ca['ranklock']),
                            eval_stride=int(ca['eval_stride']),
                            fs=float(ca['fs']) or None,
                            freq_min=float(ca['freq_min']),
                            freq_max=float(ca['freq_max']))
            run = cls(cfg)
            run.fs        = float(ca['fs'])
            run.data_file = Path(ca['data_file']) if ca['data_file'] else None
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
                w = wg[str(i)]
                ev = w['eigenvalues_real'][:] + 1j*w['eigenvalues_imag'][:]
                run.windows.append(CVAWindowResult(
                    window_idx=int(w.attrs['window_idx']),
                    sample_idx=int(w.attrs['sample_idx']),
                    time=float(w.attrs['time']),
                    rank=int(w.attrs['rank']),
                    eigenvalues=ev,
                    Phi=w['Phi'][:], spast=w['spast'][:],
                    sfuture=w['sfuture'][:], indpast=w['indpast'][:]))
            if 'significance' in f:
                sig = {}
                for k in f['significance'].keys():
                    sig[k] = f['significance'][k][:]
                for k, v in f['significance'].attrs.items():
                    sig[k] = v
                run.sig = sig
        print(f'  Loaded {len(run.windows)} windows.')
        return run

    def __repr__(self):
        parts = []
        if self.datout is not None: parts.append(f'data={len(self.time)}')
        if self.past_db is not None: parts.append(f'db={self.n_win}')
        if self.windows: parts.append(f'results={len(self.windows)}')
        if self.sig: parts.append(f'sig={len(self.sig.get("peak_freqs",[]))} peaks')
        return (f'CVARun(lags={self.cfg.lags}, tol={self.cfg.tol:.0e}, '
                f'neighbors={self.cfg.neighbors}  [{", ".join(parts)}])')
