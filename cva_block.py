"""
cva_block.py
============
Time-block modal tracking: how frequencies, damping and mode shapes evolve
along a record.

Where CVARun builds each model from the windows whose *history* looks most
like the current one, wherever in the record they occur, this fits each model
from one contiguous block of time. The estimate at time t therefore depends
only on data near t, which is what "evolution along the record" needs.

Per block
---------
A short input/output model is fitted by least squares,

    y(k+1) = A1 y(k) + ... + Ap y(k-p+1) + B0 u(k+1) + ... + Bp u(k-p+1)

and its poles and mode shapes are read from the companion matrix of the A's.
The block supplies the long time span, so p stays small (4 by default);
u(k+1) is included because an accelerometer sees the drive instantly.

Across blocks
-------------
A reference set of modes comes from one fit to the whole record. Each block's
poles are matched to it by frequency and by mode-shape correlation (MAC), so
every mode becomes one continuous track and poles that are not repeatable
drop out. References that turn up in too few blocks are discarded.

What it reports
---------------
The *effective* modal properties over each block. A joint that changes state
within a cycle is averaged over the block; this shows how that average moves,
not the separate states.

    cfg = BlockConfig(lags=4, block_sec=0.5)
    trk = BlockTracker(cfg).fit(time, inputd, datout, fs)
    trk.summary()
    fig = trk.plot_tracks()

Classes: BlockConfig, BlockTracker
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from cva_plotting import show_if_enabled


@dataclass
class BlockConfig:
    lags:            int   = 4      # past samples of each response in the model
    block_sec:       float = 0.5    # block length, seconds
    overlap:         float = 0.5    # fraction of a block shared with the next
    freq_min:        float = 0.0    # Hz; modes outside the band are ignored
    freq_max:        float = 0.0    # Hz; 0 = Nyquist
    zeta_max:        float = 0.10   # poles more heavily damped are ignored
    freq_tol:        float = 0.05   # match window, fraction of mode frequency
    mac_min:         float = 0.70   # shape agreement needed for a match
    follow:          float = 0.30   # how fast a track's expected frequency
                                    # follows what it finds (0 = fixed)
    min_persistence: float = 0.50   # a mode must appear in this share of blocks
    max_jitter:      float = 0.005  # ... and must not jump about: block-to-block
                                    # jitter above this fraction of its
                                    # frequency marks a numerical pole. A steady
                                    # drift does not count as jitter.
    merge_tol:       float = 0.01   # reference poles closer than this fraction
                                    # in frequency, with matching shape, are one


# =============================================================================
# one model fit
# =============================================================================

def _mac(a: np.ndarray, b: np.ndarray) -> float:
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(abs(np.vdot(a, b)) ** 2 / den) if den > 0 else 0.0


def fit_poles(Y: np.ndarray, U: Optional[np.ndarray], lags: int, fs: float,
              start: int = 0, stop: Optional[int] = None):
    """
    Fit the short model to samples start..stop of (Y, U) and return its
    oscillatory poles.

    Y : (N, n_out) responses.   U : (N, n_in) inputs, or None for output-only.

    Returns (freq_hz, zeta, shapes): natural frequency and damping ratio of
    each pole with positive imaginary part, sorted by frequency, and the
    (n_out, n_poles) complex mode shapes in the units of Y.
    """
    N, n_out = Y.shape
    stop = N if stop is None else min(stop, N)
    k = np.arange(max(start, 0) + lags - 1, stop - 1)
    n_reg = n_out * lags + (0 if U is None else U.shape[1] * (lags + 1))
    if len(k) < 2 * n_reg:
        raise ValueError(
            f'block of {stop - start} samples is too short for {n_reg} '
            f'unknowns per channel; lengthen the block or reduce lags')
    cols = [Y[k - j] for j in range(lags)]
    if U is not None:
        cols += [U[k + 1 - j] for j in range(lags + 1)]
    theta = np.linalg.lstsq(np.hstack(cols), Y[k + 1], rcond=None)[0].T

    n = n_out * lags
    comp = np.zeros((n, n))
    comp[:n_out] = theta[:, :n]
    comp[n_out:, :-n_out] = np.eye(n - n_out)
    ev, vec = np.linalg.eig(comp)
    keep = ev.imag > 1e-9
    ev, vec = ev[keep], vec[:n_out, keep]
    lam = np.log(ev) * fs
    freq = np.abs(lam) / (2 * np.pi)
    zeta = -lam.real / np.abs(lam)
    order = np.argsort(freq)
    return freq[order], zeta[order], vec[:, order]


# =============================================================================
# BlockTracker
# =============================================================================

class BlockTracker:
    def __init__(self, cfg: Optional[BlockConfig] = None):
        self.cfg = cfg or BlockConfig()
        self.fs: float = 0.0
        self.n_out: int = 0
        self.times:  Optional[np.ndarray] = None   # (n_blocks,) block centres
        self.freq:   Optional[np.ndarray] = None   # (n_blocks, n_modes), NaN = not found
        self.zeta:   Optional[np.ndarray] = None
        self.mac:    Optional[np.ndarray] = None   # shape agreement with reference
        self.shapes: Optional[np.ndarray] = None   # (n_blocks, n_out, n_modes) complex
        self.level:  Optional[np.ndarray] = None   # (n_blocks, n_out) response rms
        self.in_level: Optional[np.ndarray] = None # (n_blocks, n_in) input rms
        self.ref_freq:   Optional[np.ndarray] = None
        self.ref_zeta:   Optional[np.ndarray] = None
        self.ref_shapes: Optional[np.ndarray] = None  # (n_out, n_modes)
        self.persistence: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_run(cls, run, cfg: Optional[BlockConfig] = None):
        """Track the record already loaded into a CVARun."""
        if run.datout is None:
            raise RuntimeError('Call load_data() or load_record() first')
        return cls(cfg).fit(run.time, run.inputd, run.datout, run.fs)

    # ------------------------------------------------------------------
    def fit(self, time, inputd, datout, fs):
        cfg = self.cfg
        Yp = np.asarray(datout, dtype=float)
        if Yp.ndim == 1:
            Yp = Yp[:, None]
        time = np.asarray(time, dtype=float)
        N, n_out = Yp.shape
        self.fs, self.n_out = float(fs), n_out

        # Unit-variance channels keep the fit from being ruled by whichever
        # channel happens to have the largest numbers; shapes are converted
        # back to the units of the record afterwards.
        y_scale = Yp.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        Y = Yp / y_scale
        U = Up = None
        if inputd is not None and np.size(inputd):
            Up = np.asarray(inputd, dtype=float)
            if Up.ndim == 1:
                Up = Up[:, None]
            u_scale = Up.std(axis=0)
            u_scale[u_scale == 0] = 1.0
            U = Up / u_scale

        f_hi = cfg.freq_max if cfg.freq_max > 0 else fs / 2.0

        def physical(f, z):
            return ((f >= cfg.freq_min) & (f <= f_hi)
                    & (z > 0) & (z < cfg.zeta_max))

        def unit(v):
            v = v * y_scale[:, None]
            return v / v[np.abs(v).argmax(axis=0), np.arange(v.shape[1])]

        # ---- reference modes from the whole record --------------------
        f, z, V = fit_poles(Y, U, cfg.lags, fs)
        ok = physical(f, z)
        f, z, V = f[ok], z[ok], unit(V[:, ok])
        keep = []
        for i in range(len(f)):          # a mode fitted twice is one mode
            if all(abs(f[i] - f[j]) > cfg.merge_tol * f[i]
                   or _mac(V[:, i], V[:, j]) < cfg.mac_min for j in keep):
                keep.append(i)
        ref_f, ref_z, ref_V = f[keep], z[keep], V[:, keep]
        n_ref = len(ref_f)
        if n_ref == 0:
            raise RuntimeError(
                'no oscillatory poles in the analysis band; check the '
                'frequency band, zeta_max and the number of lags')

        # ---- blocks ---------------------------------------------------
        W = int(round(cfg.block_sec * fs))
        step = max(1, int(round(W * (1.0 - cfg.overlap))))
        starts = np.arange(0, N - W + 1, step)
        if len(starts) == 0:
            raise ValueError(
                f'record of {N / fs:.2f} s is shorter than one block '
                f'({cfg.block_sec:.2f} s)')
        nb = len(starts)
        freq = np.full((nb, n_ref), np.nan)
        zeta = np.full((nb, n_ref), np.nan)
        mac = np.full((nb, n_ref), np.nan)
        shapes = np.full((nb, n_out, n_ref), np.nan + 0j)
        expect = ref_f.copy()

        for bi, s in enumerate(starts):
            f, z, V = fit_poles(Y, U, cfg.lags, fs, s, s + W)
            ok = physical(f, z)
            f, z, V = f[ok], z[ok], unit(V[:, ok])
            if len(f) == 0:
                continue
            near = (np.abs(f[None, :] - expect[:, None])
                    <= cfg.freq_tol * expect[:, None])
            if n_out > 1:
                score = np.array([[_mac(ref_V[:, j], V[:, c])
                                   for c in range(len(f))]
                                  for j in range(n_ref)])
                score[score < cfg.mac_min] = -1.0
            else:                         # one channel has no shape to compare
                score = 1.0 - np.abs(f[None, :] - expect[:, None]) / expect[:, None]
            score[~near] = -1.0
            while True:                   # best pairs first, each pole used once
                j, c = np.unravel_index(score.argmax(), score.shape)
                if score[j, c] < 0:
                    break
                freq[bi, j], zeta[bi, j] = f[c], z[c]
                mac[bi, j] = _mac(ref_V[:, j], V[:, c])
                shapes[bi, :, j] = V[:, c]
                expect[j] += cfg.follow * (f[c] - expect[j])
                score[j, :] = -1.0
                score[:, c] = -1.0

        # ---- keep the modes that are repeatable -----------------------
        pers = np.mean(~np.isnan(freq), axis=0)
        jitter = np.full(n_ref, np.inf)
        for j in range(n_ref):
            d = np.diff(freq[:, j])       # NaN wherever either block missed
            d = d[~np.isnan(d)]
            if len(d) >= 2:
                jitter[j] = d.std() / np.sqrt(2.0) / ref_f[j]
        good = (pers >= cfg.min_persistence) & (jitter <= cfg.max_jitter)
        if not good.any():
            raise RuntimeError(
                'no mode was found consistently from block to block; try a '
                'longer block or more lags')
        self.times = time[starts] + 0.5 * (time[starts + W - 1] - time[starts])
        self.freq, self.zeta, self.mac = freq[:, good], zeta[:, good], mac[:, good]
        self.shapes = shapes[:, :, good]
        self.ref_freq, self.ref_zeta = ref_f[good], ref_z[good]
        self.ref_shapes = ref_V[:, good]
        self.persistence = pers[good]
        self.level = np.array([Yp[s:s + W].std(axis=0) for s in starts])
        self.in_level = (None if Up is None else
                         np.array([Up[s:s + W].std(axis=0) for s in starts]))
        return self

    # ------------------------------------------------------------------
    @property
    def n_modes(self) -> int:
        return 0 if self.ref_freq is None else len(self.ref_freq)

    def trend(self):
        """Straight-line change of each mode's frequency over the tracked
        span, in Hz (end minus start)."""
        out = np.full(self.n_modes, np.nan)
        span = self.times[-1] - self.times[0]
        for j in range(self.n_modes):
            ok = ~np.isnan(self.freq[:, j])
            if ok.sum() >= 3 and span > 0:
                out[j] = np.polyfit(self.times[ok], self.freq[ok, j], 1)[0] * span
        return out

    def summary(self, verbose: bool = True) -> dict:
        """Per-mode mean, block-to-block scatter and trend."""
        if self.freq is None:
            raise RuntimeError('Call fit() first')
        res = dict(
            freq_mean=np.nanmean(self.freq, axis=0),
            freq_std=np.nanstd(self.freq, axis=0),
            zeta_mean=np.nanmean(self.zeta, axis=0),
            zeta_std=np.nanstd(self.zeta, axis=0),
            mac_mean=np.nanmean(self.mac, axis=0),
            persistence=self.persistence,
            trend_hz=self.trend())
        if verbose:
            cfg = self.cfg
            print(f'\nBlock tracking: {len(self.times)} blocks of '
                  f'{cfg.block_sec:.2f} s, lags={cfg.lags}, '
                  f'{self.n_modes} modes')
            print('  mode   freq Hz   scatter   trend Hz   zeta %  scatter'
                  '    MAC   found')
            for j in range(self.n_modes):
                print(f'  {j + 1:3d}  {res["freq_mean"][j]:9.2f}  '
                      f'{res["freq_std"][j]:7.2f}  {res["trend_hz"][j]:+9.2f}  '
                      f'{100 * res["zeta_mean"][j]:7.2f}  '
                      f'{100 * res["zeta_std"][j]:6.2f}  '
                      f'{res["mac_mean"][j]:6.3f}  '
                      f'{100 * res["persistence"][j]:5.0f}%')
        return res

    # ------------------------------------------------------------------
    def plot_tracks(self, save_path=None, dpi=150):
        """
        One row per mode: frequency, damping and shape agreement against
        time. The shaded band is the mode's own block-to-block scatter, the
        floor a change has to clear to be believed.

        Returns the Figure.
        """
        if self.freq is None:
            raise RuntimeError('Call fit() first')
        n = self.n_modes
        fig, axes = plt.subplots(n, 3, figsize=(14, 1.9 * n + 1.4),
                                 sharex=True, squeeze=False,
                                 layout='constrained')
        fig.suptitle(
            f'Modal tracking — blocks of {self.cfg.block_sec:.2f} s, '
            f'lags={self.cfg.lags}', fontsize=12, fontweight='bold')
        t = self.times
        for j in range(n):
            f, z, m = self.freq[:, j], 100 * self.zeta[:, j], self.mac[:, j]
            for ax, y, colour in ((axes[j, 0], f, 'steelblue'),
                                  (axes[j, 1], z, 'darkorange')):
                mu, sd = np.nanmean(y), np.nanstd(y)
                ax.axhspan(mu - sd, mu + sd, color=colour, alpha=0.12)
                ax.axhline(mu, color=colour, lw=0.8, ls='--')
                ax.plot(t, y, '.-', color=colour, ms=4, lw=0.8)
                ax.grid(True, alpha=0.3)
            axes[j, 0].set_ylabel(f'{np.nanmean(f):.1f} Hz', fontsize=9)
            axes[j, 2].plot(t, m, '.-', color='seagreen', ms=4, lw=0.8)
            axes[j, 2].set_ylim(min(0.9, np.nanmin(m) - 0.01), 1.002)
            axes[j, 2].grid(True, alpha=0.3)
        axes[0, 0].set_title('Frequency (Hz)')
        axes[0, 1].set_title('Damping (% critical)')
        axes[0, 2].set_title('Shape agreement with whole-record shape (MAC)')
        for ax in axes[-1]:
            ax.set_xlabel('Time (s)')
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')
        show_if_enabled()
        return fig

    def __repr__(self):
        if self.freq is None:
            return 'BlockTracker(not fitted)'
        return (f'BlockTracker(blocks={len(self.times)}, '
                f'modes={self.n_modes}, lags={self.cfg.lags}, '
                f'block={self.cfg.block_sec:.2f}s)')
