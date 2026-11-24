"""
cva_residual.py
===============
The non-coherent response as a time history.

`cva_spectra` says how much of each response is not linearly related to the
input. This module says *when*. It estimates the linear path from input to
each response (the frequency response function), passes the measured input
through it, and subtracts the result from the measured response:

    response  =  linear part  +  residual

The residual's spectrum is the non-coherent spectral density. Having it as
a time history lets us look at its envelope along the record, how impulsive
it is, how its level follows the level of the linear response, and how its
spectral peaks move.

    rr = noncoherent_history(inputd, datout, fs, label='dead zone')
    rr.table()
    fig = plot_residual(rr, channel=3)

The split inherits the limits of the coherence estimate. A resonance
narrower than the frequency resolution leaves part of its linear response
in the residual, so judge a record against a linear or low-level record of
the same system, or look at the residual above the excitation band, where
the linear part is zero by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, csd, fftconvolve, sosfiltfilt, spectrogram
from scipy.stats import kurtosis

from cva_plotting import show_if_enabled
from cva_spectra import excitation_band


@dataclass
class ResidualResult:
    label:  str
    fs:     float
    time:   np.ndarray           # (N,)
    total:  np.ndarray           # (N, n_out) measured response
    linear: np.ndarray           # (N, n_out) input passed through the FRF
    resid:  np.ndarray           # (N, n_out) total - linear
    band:   Tuple[float, float]  # excitation band, Hz
    names:  List[str] = field(default_factory=list)
    trim:   Tuple[int, int] = (0, 0)   # samples at each end left out of the
                                       # statistics (filter start-up and lead)

    def _core(self, x):
        a, b = self.trim
        return x[a:len(x) - b]

    # ---- pieces ------------------------------------------------------------
    def out_of_band(self, x: Optional[np.ndarray] = None):
        """High-pass a signal (default: the residual) just above the
        excitation band. Returns None if the band reaches Nyquist."""
        x = self.resid if x is None else x
        edge = 1.05 * self.band[1]
        if edge >= 0.45 * self.fs:
            return None
        sos = butter(8, edge, btype='high', fs=self.fs, output='sos')
        return sosfiltfilt(sos, x, axis=0)

    def envelopes(self, block_sec: float = 0.1, x: Optional[np.ndarray] = None):
        """Block rms of the linear part and of x (default: the residual).
        Returns block-centre times, rms_linear, rms_x."""
        x = self._core(self.resid if x is None else x)
        time = self._core(self.time)
        L = max(1, int(round(block_sec * self.fs)))
        nb = len(time) // L

        def rms(a):
            return np.sqrt((a[:nb * L].reshape(nb, L, -1) ** 2).mean(axis=1))
        t = time[:nb * L].reshape(nb, L).mean(axis=1)
        return t, rms(self._core(self.linear)), rms(x)

    # ---- numbers -----------------------------------------------------------
    @property
    def share(self):
        """Residual mean square over total mean square, per channel."""
        return ((self._core(self.resid) ** 2).mean(axis=0)
                / (self._core(self.total) ** 2).mean(axis=0))

    @property
    def has_out_of_band(self):
        """Per channel: is there enough residual above the excitation band
        to say anything about it (more than 0.1% of the response rms)?"""
        x = self.out_of_band()
        if x is None:
            return np.zeros(self.resid.shape[1], bool)
        return (self._core(x).std(axis=0)
                > 1e-3 * self._core(self.total).std(axis=0))

    @property
    def kurt(self):
        """Excess kurtosis of the residual: near 0 for a smooth, steady
        nonlinearity, large when it arrives in bursts (impacts, rattling)."""
        return kurtosis(self._core(self.resid), axis=0)

    @property
    def kurt_out_of_band(self):
        x = self.out_of_band()
        if x is None:
            return np.full(self.resid.shape[1], np.nan)
        return np.where(self.has_out_of_band,
                        kurtosis(self._core(x), axis=0), np.nan)

    def level_exponent(self, channel: int, out_of_band: bool = False,
                       block_sec: float = 0.25):
        """How the residual grows with response level within the record.

        Fits  residual rms ~ (linear rms)^p  over blocks. p near 1 means the
        residual simply scales with the response: a tension/compression
        stiffness change, or linear response left in the residual by limited
        resolution. p well above 1 means it grows faster than the response
        (p near 3 for a cubic spring). p below 1 means it matters most at
        low level (free play, dead zone).

        Returns p, its standard error, and the two block-rms arrays."""
        x = self.out_of_band() if out_of_band else self.resid
        if x is None or (out_of_band and not self.has_out_of_band[channel]):
            return np.nan, np.nan, None, None
        _, lin, res = self.envelopes(block_sec, x)
        a, b = lin[:, channel], res[:, channel]
        ok = (a > 0) & (b > 0)
        la, lb = np.log(a[ok]), np.log(b[ok])
        A = np.column_stack([la, np.ones_like(la)])
        coef, *_ = np.linalg.lstsq(A, lb, rcond=None)
        r = lb - A @ coef
        # blocks overlap in content through the ringing, so the standard
        # error is computed as if only every fourth block were independent
        n_eff = max(3, ok.sum() // 4)
        se = np.sqrt(r.var() / (n_eff * la.var()))
        return float(coef[0]), float(se), a, b

    def table(self, verbose: bool = True):
        rows = []
        ko = self.kurt_out_of_band
        for c in range(self.resid.shape[1]):
            p, se, *_ = self.level_exponent(c)
            po, seo, *_ = self.level_exponent(c, out_of_band=True)
            rows.append(dict(channel=self.names[c], share=self.share[c],
                             kurt=self.kurt[c], kurt_oob=ko[c],
                             exponent=p, exponent_se=se,
                             exponent_oob=po, exponent_oob_se=seo))
        if verbose:
            print(f"\n{self.label}: residual (non-coherent) time history")
            print(f"{'channel':>8} {'share':>7} {'kurt':>7} {'kurt oob':>9} "
                  f"{'exponent':>14} {'exponent oob':>14}")
            for r in rows:
                print(f"{r['channel']:>8} {100*r['share']:6.1f}% "
                      f"{r['kurt']:7.2f} {r['kurt_oob']:9.2f} "
                      f"{r['exponent']:7.2f} ±{r['exponent_se']:5.2f} "
                      f"{r['exponent_oob']:7.2f} ±{r['exponent_oob_se']:5.2f}")
        return rows


def noncoherent_history(inputd, datout, fs: float, label: str = '',
                        nperseg: int = 0,
                        names: Optional[Sequence[str]] = None,
                        skip_sec: float = 0.0) -> ResidualResult:
    """
    Split each response into the part predicted linearly from the input(s)
    and the residual.

    The frequency response is the H1 estimate from Welch cross-spectra with
    segments of nperseg samples (0 picks about 0.3 Hz resolution, capped at
    an eighth of the record). Its inverse transform is used as a filter
    whose memory is one segment long, mostly causal with a short
    non-causal lead.
    """
    U = np.atleast_2d(np.asarray(inputd, float).T).T
    Y = np.atleast_2d(np.asarray(datout, float).T).T
    k0 = int(round(skip_sec * fs))
    U, Y = U[k0:], Y[k0:]
    U = U - U.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    N, n_in = U.shape
    n_out = Y.shape[1]
    if nperseg <= 0:
        nperseg = int(2 ** np.round(np.log2(fs / 0.3)))
        nperseg = min(nperseg, int(2 ** np.floor(np.log2(N / 8.5))))
    nperseg = min(nperseg, N)
    kw = dict(fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window='hann')

    freq, _ = csd(U[:, 0], U[:, 0], **kw)
    F = len(freq)
    Suu = np.zeros((F, n_in, n_in), complex)
    for i in range(n_in):
        for j in range(i, n_in):
            _, s = csd(U[:, i], U[:, j], **kw)
            Suu[:, i, j] = s
            Suu[:, j, i] = np.conj(s)
    Suy = np.zeros((F, n_in, n_out), complex)
    for c in range(n_out):
        for i in range(n_in):
            _, s = csd(U[:, i], Y[:, c], **kw)
            Suy[:, i, c] = s
    s_in = np.stack([Suu[:, i, i].real for i in range(n_in)], axis=1)
    band = excitation_band(freq, s_in)

    # H[f, i, c]: response c per unit input i. Where the input has no energy
    # the estimate is meaningless, so it is set to zero there.
    scale = np.trace(Suu.real, axis1=1, axis2=2).max()
    reg = 1e-9 * scale * np.eye(n_in)
    H = np.zeros((F, n_in, n_out), complex)
    for k in range(F):
        H[k] = np.linalg.solve(Suu[k] + reg, Suy[k])
    H[(freq < band[0]) | (freq > band[1])] = 0.0

    lead = nperseg // 8                     # non-causal taps kept
    lin = np.zeros_like(Y)
    for i in range(n_in):
        for c in range(n_out):
            h = np.fft.irfft(H[:, i, c], n=nperseg)
            h = np.roll(h, lead)            # h[k] is now lag k - lead
            full = fftconvolve(U[:, i], h, mode='full')
            lin[:, c] += full[lead:lead + N]

    if names is None:
        names = [f"ch {c + 1}" for c in range(n_out)]
    time = (k0 + np.arange(N)) / fs
    trim = (min(nperseg // 4, N // 8), min(lead, N // 8))
    return ResidualResult(label=label, fs=fs, time=time, total=Y, linear=lin,
                          resid=Y - lin, band=band, names=list(names),
                          trim=trim)


def track_residual(rr: ResidualResult, lags: int = 8, block_sec: float = 0.5,
                   cfg=None):
    """Frequency, damping and shape of the residual's own spectral peaks,
    block by block along the record.

    The residual has no input by construction, so this is an output-only
    fit across all channels (see cva_block.BlockTracker). The peaks it
    finds are the "apparent modes" of the non-coherent response: true
    modes rung by the nonlinearity, or peaks the nonlinearity creates.
    Returns the fitted BlockTracker; call .summary() or .plot_tracks()."""
    from cva_block import BlockConfig, BlockTracker
    if cfg is None:
        cfg = BlockConfig(lags=lags, block_sec=block_sec, max_jitter=0.02)
    trk = BlockTracker(cfg)
    trk.fit(rr._core(rr.time), None, rr._core(rr.resid), rr.fs)
    return trk


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def plot_residual(rr: ResidualResult, channel: int = 0,
                  zoom: Optional[Tuple[float, float]] = None,
                  f_max: float = 0.0, save_path=None, dpi: int = 130):
    """One channel of one record, five views of the residual:

    1. a short stretch of the measured response, its linear part and the
       residual, so the split can be seen directly;
    2. block rms of the linear part and of the residual along the whole
       record: when the nonlinearity acts;
    3. spectrogram of the residual: how its peaks move;
    4. residual level against linear-response level, block by block, with
       the fitted exponent;
    5. kurtosis of the residual at every channel: steady or in bursts.
    """
    import matplotlib.pyplot as plt
    c = channel % rr.resid.shape[1]
    f_max = f_max or rr.fs / 2
    t = rr.time
    if zoom is None:
        mid = 0.5 * (t[0] + t[-1])
        zoom = (mid, mid + 0.25)
    z = (t >= zoom[0]) & (t <= zoom[1])

    fig = plt.figure(figsize=(15, 12.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.25, 1.15])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[z], rr.total[z, c], color='k', lw=1.4, label='measured response')
    ax.plot(t[z], rr.linear[z, c], color='tab:blue', lw=1.0,
            label='linear part (input through the frequency response)')
    ax.plot(t[z], rr.resid[z, c], color='tab:red', lw=1.0,
            label='residual = measured - linear (non-coherent)')
    ax.set_title(f"1. {rr.names[c]}: a {zoom[1]-zoom[0]:.2f} s stretch",
                 fontsize=10, loc='left')
    ax.set_xlabel('s')
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1, 1.0),
              ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    tb, lin, res = rr.envelopes(0.1)
    ax.semilogy(tb, lin[:, c], color='tab:blue', lw=1.0,
                label='linear part, rms in 0.1 s blocks')
    ax.semilogy(tb, res[:, c], color='tab:red', lw=1.0,
                label='residual, rms in 0.1 s blocks')
    oob = rr.out_of_band()
    if oob is not None and rr.has_out_of_band[c]:
        _, _, ro = rr.envelopes(0.1, oob)
        ax.semilogy(tb, ro[:, c], color='tab:orange', lw=0.9,
                    label=f'residual above {1.05*rr.band[1]:.0f} Hz '
                          f'(out of band)')
    ax.axvspan(*zoom, color='0.85', zorder=0)
    ax.set_title(f"2. {rr.names[c]}: level along the record (grey bar = the "
                 f"stretch above)", fontsize=10, loc='left')
    ax.set_xlabel('s')
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1, 1.0),
              ncol=3, frameon=False)
    ax.grid(True, which='major', alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    nseg = int(2 ** np.round(np.log2(rr.fs * 0.4)))
    f, ts, S = spectrogram(rr.resid[:, c], fs=rr.fs, nperseg=nseg,
                           noverlap=nseg * 3 // 4)
    m = f <= f_max
    SdB = 10 * np.log10(S[m] + 1e-300)
    top = np.percentile(SdB, 99.9)
    im = ax.pcolormesh(ts + t[0], f[m], SdB, vmin=top - 60, vmax=top,
                       shading='auto', cmap='magma')
    ax.axhline(rr.band[1], color='w', lw=0.8, ls='--')
    ax.text(t[0], rr.band[1], ' top of excitation band', color='w',
            fontsize=8, va='bottom')
    ax.set_title(f"3. {rr.names[c]}: spectrogram of the residual "
                 f"(dB, 60 dB range)", fontsize=10, loc='left')
    ax.set_ylabel('Hz')
    ax.set_xlabel('s')
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = fig.add_subplot(gs[3, 0])
    p, se, a, b = rr.level_exponent(c)
    ax.loglog(a, b, '.', color='tab:red', ms=4, alpha=0.6,
              label=f'whole residual: exponent {p:.2f} ± {se:.2f}')
    po, seo, ao, bo = rr.level_exponent(c, out_of_band=True)
    if ao is not None:
        ax.loglog(ao, bo, '.', color='tab:orange', ms=4, alpha=0.6,
                  label=f'out of band: exponent {po:.2f} ± {seo:.2f}')
    x = np.array([a.min(), a.max()])
    for k, ls in ((1, ':'), (3, '--')):
        y0 = np.exp(np.log(b).mean() + k * (np.log(x) - np.log(a).mean()))
        ax.loglog(x, y0, color='0.4', ls=ls, lw=1.0,
                  label=f'slope {k} for reference')
    ax.set_xlabel('linear-part rms in 0.25 s blocks')
    ax.set_ylabel('residual rms in the same block')
    ax.set_title(f"4. {rr.names[c]}: does the residual grow faster than "
                 f"the response?", fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25)

    ax = fig.add_subplot(gs[3, 1])
    n_out = rr.resid.shape[1]
    x = np.arange(n_out)
    ax.bar(x - 0.2, rr.kurt, 0.4, color='tab:red', label='whole residual')
    ko = rr.kurt_out_of_band
    if np.isfinite(ko).any():
        ax.bar(x + 0.2, ko, 0.4, color='tab:orange',
               label='residual out of band')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x, rr.names)
    ax.set_title("5. excess kurtosis of the residual: 0 = steady, "
                 "large = arrives in bursts", fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f"{rr.label}: non-coherent response in time, {rr.names[c]} "
                 f"(residual is {100*rr.share[c]:.1f}% of its mean square)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.text(0.5, 0.005,
             "linear part = what a linear system would give for this input."
             "   residual = everything else (nonlinearity, noise, and any "
             "resonance too sharp to resolve).   out of band = above the "
             "excitation band, where the linear part is zero.",
             ha='center', va='bottom', fontsize=9, color='0.25')
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
