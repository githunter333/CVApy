"""
cva_level.py
============
Do the in-band modes change with response level?

Takes block-by-block mode tracks (cva_block.BlockTracker) and sets each
mode's frequency and damping against the response level in the same block.
A linear system gives flat lines. A hardening spring raises frequency with
level, free play lowers it at low level, and friction usually raises damping
with level.

Level varies within any random record because the response envelope wanders;
records taken at different test levels extend the range, so several records
of one system can be laid on the same axes.

    trk = track_in_band(t, u, y, fs, band_top=416.0)
    lv = level_dependence(trk, label='0 dB')
    lv.table()
    plot_level([lv_low, lv])          # first record sets the mode list
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from cva_plotting import show_if_enabled
from cva_spectra import CASE_COLORS


@dataclass
class LevelResult:
    label:   str
    level:   np.ndarray      # (nb,) response rms per block, rss over channels
    freq:    np.ndarray      # (nb, n_modes) Hz
    zeta:    np.ndarray      # (nb, n_modes)
    ref_freq: np.ndarray     # (n_modes,)
    f_slope: np.ndarray      # % change in frequency per doubling of level
    f_se:    np.ndarray
    z_slope: np.ndarray      # change in damping, % of critical, per doubling
    z_se:    np.ndarray
    f_mean:  np.ndarray
    z_mean:  np.ndarray

    def table(self, verbose: bool = True):
        rows = [dict(mode=j + 1, freq=self.f_mean[j], zeta=self.z_mean[j],
                     f_slope=self.f_slope[j], f_se=self.f_se[j],
                     z_slope=self.z_slope[j], z_se=self.z_se[j])
                for j in range(len(self.ref_freq))]
        if verbose:
            span = self.level.max() / self.level.min()
            print(f"\n{self.label}: modes against response level "
                  f"(block level spans a factor of {span:.1f})")
            print(f"{'mode':>5} {'freq Hz':>9} {'zeta %':>7} "
                  f"{'freq change / doubling':>24} "
                  f"{'zeta change / doubling':>24}")
            for r in rows:
                fs_ = abs(r['f_slope']) > 3 * r['f_se']
                zs_ = abs(r['z_slope']) > 3 * r['z_se']
                print(f"{r['mode']:>5} {r['freq']:9.2f} {100*r['zeta']:7.2f} "
                      f"{r['f_slope']:+10.3f}% ±{r['f_se']:6.3f} "
                      f"{'*' if fs_ else ' '}   "
                      f"{100*r['z_slope']:+10.3f} ±{100*r['z_se']:6.3f} "
                      f"{'*' if zs_ else ' '}")
            print("      * = more than three standard errors from zero")
        return rows


def _slope(x, y):
    """Least-squares slope of y on x with a standard error that treats
    half-overlapping blocks as half as many independent ones."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6 or np.ptp(x[ok]) == 0:
        return np.nan, np.nan
    x, y = x[ok], y[ok]
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    n_eff = max(3, len(x) // 2)
    return float(coef[0]), float(np.sqrt(r.var() / (n_eff * x.var())))


def track_in_band(time, inputd, datout, fs: float, band_top: float,
                  lags: int = 2, block_sec: float = 0.5, cfg=None):
    """Block-track the in-band modes with everything above the excitation
    band filtered out first.

    This matters. The fit weights high frequencies heavily, so response a
    nonlinearity puts above the band (rattling, harmonics) drags the in-band
    poles, and since that response changes with level it shows up as a
    false level dependence. On the six-mass dead-zone record, fitting the
    raw data gave frequency falling 0.06 to 0.46% per doubling and damping
    up to twice the linear values; filtered first, the modes match the
    linear system and are flat with level.

    The record is low-passed and decimated so the new Nyquist frequency is
    about 1.5 times band_top. Returns the fitted BlockTracker."""
    from scipy.signal import decimate
    from cva_block import BlockConfig, BlockTracker
    q = max(1, int(fs // (3.0 * band_top)))
    t = np.asarray(time, float)
    U = np.atleast_2d(np.asarray(inputd, float).T).T
    Y = np.atleast_2d(np.asarray(datout, float).T).T
    if q > 1:
        U = decimate(U, q, axis=0, zero_phase=True)
        Y = decimate(Y, q, axis=0, zero_phase=True)
        t = t[::q]
    edge = int(round(0.5 * fs / q))          # filter start-up at each end
    U, Y, t = U[edge:-edge], Y[edge:-edge], t[edge:-edge]
    if cfg is None:
        cfg = BlockConfig(lags=lags, block_sec=block_sec, freq_max=band_top)
    return BlockTracker(cfg).fit(t, U, Y, fs / q)


def level_dependence(trk, label: str = '') -> LevelResult:
    """Frequency and damping of each tracked mode against block response
    level. Level is the root-sum-square over channels of the block rms."""
    level = np.sqrt((trk.level ** 2).sum(axis=1))
    lg = np.log2(level)
    n = trk.freq.shape[1]
    f_mean = np.nanmean(trk.freq, axis=0)
    z_mean = np.nanmean(trk.zeta, axis=0)
    fs_, fe, zs_, ze = (np.full(n, np.nan) for _ in range(4))
    for j in range(n):
        a, b = _slope(lg, trk.freq[:, j])
        fs_[j], fe[j] = 100 * a / f_mean[j], 100 * b / f_mean[j]
        zs_[j], ze[j] = _slope(lg, trk.zeta[:, j])
    return LevelResult(label=label, level=level, freq=trk.freq.copy(),
                       zeta=trk.zeta.copy(), ref_freq=trk.ref_freq.copy(),
                       f_slope=fs_, f_se=fe, z_slope=zs_, z_se=ze,
                       f_mean=f_mean, z_mean=z_mean)


def plot_level(results: Sequence[LevelResult], match_tol: float = 0.12,
               f_max: float = 0.0, save_path=None, dpi: int = 130):
    """One row per mode of the first record: frequency (left) and damping
    (right) against block response level, every record in its own colour.
    Modes of the other records are matched to the first by frequency."""
    import matplotlib.pyplot as plt
    r0 = results[0]
    modes = [j for j in range(len(r0.ref_freq))
             if not f_max or r0.ref_freq[j] <= f_max]
    fig, axes = plt.subplots(len(modes), 2,
                             figsize=(14, 2.7 * len(modes) + 1.2),
                             squeeze=False)
    for row, j in enumerate(modes):
        axf, axz = axes[row]
        for i, r in enumerate(results):
            col = CASE_COLORS[i % len(CASE_COLORS)]
            d = np.abs(r.f_mean - r0.f_mean[j]) / r0.f_mean[j]
            if not np.isfinite(d).any() or np.nanmin(d) > match_tol:
                continue
            k = int(np.nanargmin(d))
            axf.semilogx(r.level, r.freq[:, k], '.', color=col, ms=4,
                         alpha=0.6,
                         label=f"{r.label}: {r.f_mean[k]:.2f} Hz, "
                               f"{r.f_slope[k]:+.2f}% per doubling")
            axz.semilogx(r.level, 100 * r.zeta[:, k], '.', color=col, ms=4,
                         alpha=0.6,
                         label=f"{r.label}: {100*r.z_mean[k]:.2f}%, "
                               f"{100*r.z_slope[k]:+.2f} per doubling")
        # never zoom in further than 1% in frequency or 0.2 points in
        # damping, so numerical scatter is not drawn as if it were a trend
        for ax, span in ((axf, 0.01 * r0.f_mean[j]), (axz, 0.2)):
            lo, hi = ax.get_ylim()
            if hi - lo < span:
                mid = 0.5 * (lo + hi)
                ax.set_ylim(mid - span / 2, mid + span / 2)
            ax.ticklabel_format(axis='y', useOffset=False)
        axf.set_ylabel('Hz')
        axz.set_ylabel('% of critical')
        axf.set_title(f"mode near {r0.f_mean[j]:.0f} Hz: frequency",
                      fontsize=10, loc='left')
        axz.set_title(f"mode near {r0.f_mean[j]:.0f} Hz: damping",
                      fontsize=10, loc='left')
        for ax in (axf, axz):
            ax.grid(True, which='both', alpha=0.25)
            ax.legend(fontsize=7.5, framealpha=0.9)
            if row == len(modes) - 1:
                ax.set_xlabel('response level in the block (rms, all '
                              'channels combined)')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.5, 0.005,
             "Each dot is one block of the record. A linear system gives "
             "flat lines; a slope means the mode changes with level.   "
             "First record listed is the baseline.",
             ha='center', va='bottom', fontsize=9, color='0.25')
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
