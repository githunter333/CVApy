"""
cva_spectra.py
==============
First look at a record for signs of nonlinear behaviour, using nothing but
spectra and amplitude statistics.

For each response channel the spectral density is split into the part that
is linearly related to the input (coherent) and the part that is not
(non-coherent). A linear system with a clean measurement is almost entirely
coherent. Non-coherent energy, energy above the excitation band, and skewed
or heavy-tailed amplitude distributions are the consequences of nonlinearity
this module puts numbers on.

    res = analyze_record(inputd, datout, fs, label='dead zone')
    res.table()                  # one row per channel
    fig = plot_record(res)       # dashboard for one record
    fig = plot_comparison([lin, res])   # several records side by side

The coherence of a perfectly linear record is still below one near sharp
resonances (finite frequency resolution) and wherever the input has no
energy. Always compare against a linear or low-level record of the same
system; that record is the floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import csd, find_peaks
from scipy.stats import kurtosis, skew

from cva_plotting import show_if_enabled


@dataclass
class SpectraResult:
    label:   str
    fs:      float
    freq:    np.ndarray          # (F,)
    s_in:    np.ndarray          # (F, n_in) input spectral densities
    s_tot:   np.ndarray          # (F, n_out) total response
    coh:     np.ndarray          # (F, n_out) (multiple) coherence
    band:    Tuple[float, float]  # excitation band, Hz
    in_rms:  np.ndarray          # (n_in,)
    rms:     np.ndarray          # (n_out,)
    skew:    np.ndarray          # (n_out,)
    kurt:    np.ndarray          # (n_out,) excess kurtosis, 0 for Gaussian
    in_skew: np.ndarray = None
    in_kurt: np.ndarray = None
    skew_se: np.ndarray = None   # standard error from 8 blocks of the record
    kurt_se: np.ndarray = None
    nperseg: int = 0
    coarse_noncoherent: np.ndarray = None   # same fraction at half the
                                            # frequency resolution
    names:   List[str] = field(default_factory=list)

    # ---- derived spectra -------------------------------------------------
    @property
    def s_coh(self):
        return self.coh * self.s_tot

    @property
    def s_non(self):
        return (1.0 - self.coh) * self.s_tot

    def _power(self, s, lo, hi):
        m = (self.freq >= lo) & (self.freq < hi)
        return np.trapezoid(s[m], self.freq[m], axis=0)

    # ---- per-channel numbers ----------------------------------------------
    @property
    def frac_noncoherent(self):
        """Share of each channel's mean square that is non-coherent."""
        ny = self.fs / 2
        return self._power(self.s_non, 0, ny) / self._power(self.s_tot, 0, ny)

    @property
    def frac_out_of_band(self):
        """Share of each channel's mean square above the excitation band."""
        ny = self.fs / 2
        return (self._power(self.s_tot, self.band[1], ny)
                / self._power(self.s_tot, 0, ny))

    @property
    def frac_noncoherent_in_band(self):
        lo, hi = self.band
        return self._power(self.s_non, lo, hi) / self._power(self.s_tot, lo, hi)

    @property
    def resolution_limited(self):
        """True per channel where halving the frequency resolution raises
        the non-coherent share noticeably. The share is then set by
        resonances narrower than the resolution, not by the system, and
        should be read as an upper bound."""
        if self.coarse_noncoherent is None:
            return np.zeros(self.s_tot.shape[1], bool)
        fine = self.frac_noncoherent
        return ((self.coarse_noncoherent - fine > 0.02)
                & (self.coarse_noncoherent > 1.1 * fine))

    def noncoherent_peaks(self, channel: int, n: int = 6,
                          prominence_db: float = 6.0):
        """Frequencies of the strongest peaks in one channel's non-coherent
        spectrum, strongest first."""
        s = 10 * np.log10(self.s_non[:, channel] + 1e-300)
        k = max(3, int(round(len(s) / 400)))          # light smoothing
        s = np.convolve(s, np.ones(k) / k, mode='same')
        pk, _ = find_peaks(s, prominence=prominence_db)
        pk = pk[np.argsort(s[pk])[::-1][:n]]
        return self.freq[pk]

    def table(self, verbose: bool = True):
        rows = []
        for c in range(self.s_tot.shape[1]):
            rows.append(dict(channel=self.names[c], rms=self.rms[c],
                             skew=self.skew[c], kurt=self.kurt[c],
                             noncoherent=self.frac_noncoherent[c],
                             noncoherent_in_band=self.frac_noncoherent_in_band[c],
                             out_of_band=self.frac_out_of_band[c]))
        if verbose:
            print(f"\n{self.label}: excitation band "
                  f"{self.band[0]:.0f}-{self.band[1]:.0f} Hz, "
                  f"input rms {np.round(self.in_rms, 3)}, "
                  f"input skew {np.round(self.in_skew, 2)}, "
                  f"kurt {np.round(self.in_kurt, 2)}")
            print(f"{'channel':>8} {'rms':>9} {'skew':>13} {'kurt':>13} "
                  f"{'non-coh':>8} {'in-band':>8} {'out-band':>9}")
            lim = self.resolution_limited
            for c, r in enumerate(rows):
                print(f"{r['channel']:>8} {r['rms']:9.3f} "
                      f"{r['skew']:6.2f} ±{self.skew_se[c]:5.2f} "
                      f"{r['kurt']:6.2f} ±{self.kurt_se[c]:5.2f} "
                      f"{100*r['noncoherent']:7.1f}% "
                      f"{100*r['noncoherent_in_band']:7.1f}% "
                      f"{100*r['out_of_band']:8.1f}%"
                      f"{'  resolution-limited' if lim[c] else ''}")
        return rows


def excitation_band(freq, s_in, drop_db: float = 20.0, smooth_hz: float = 0.0):
    """Band over which the input holds its level: the unbroken span, around
    the strongest part of the input, where the summed input spectral density
    stays within drop_db of its typical in-band value.

    The span must be unbroken so that an isolated line far above the band
    (a harmonic of the structure showing up in the control signal, mains
    pickup) does not stretch it. The spectrum is averaged over about 1% of
    the frequency range first so one noisy bin does not cut it short."""
    s = s_in.sum(axis=1) if s_in.ndim == 2 else s_in
    s = s[1:]
    f = freq[1:]
    k = max(1, int(round((smooth_hz or 0.01 * f[-1]) / (f[1] - f[0]))))
    s = np.convolve(s, np.ones(k) / k, mode='same')
    ref = np.median(s[s > 0.1 * s.max()])
    ok = s > ref * 10 ** (-drop_db / 10)
    c = np.cumsum(s * ok)
    i = int(np.searchsorted(c, 0.5 * c[-1]))      # middle of the energy
    lo = hi = i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(s) - 1 and ok[hi + 1]:
        hi += 1
    return float(f[lo]), float(f[hi])


def _spectra(U, Y, kw):
    """Input spectra, response spectra and (multiple) coherence."""
    n_in, n_out = U.shape[1], Y.shape[1]
    freq, _ = csd(U[:, 0], U[:, 0], **kw)
    F = len(freq)
    Suu = np.zeros((F, n_in, n_in), complex)
    for i in range(n_in):
        for j in range(i, n_in):
            _, s = csd(U[:, i], U[:, j], **kw)
            Suu[:, i, j] = s
            Suu[:, j, i] = np.conj(s)
    Suy = np.zeros((F, n_in, n_out), complex)
    Syy = np.zeros((F, n_out))
    for c in range(n_out):
        _, s = csd(Y[:, c], Y[:, c], **kw)
        Syy[:, c] = s.real
        for i in range(n_in):
            _, s = csd(U[:, i], Y[:, c], **kw)
            Suy[:, i, c] = s

    # coherent power: Suy^H Suu^-1 Suy, per frequency and channel
    coh = np.zeros((F, n_out))
    scale = np.trace(Suu.real, axis1=1, axis2=2).max()
    reg = 1e-12 * scale * np.eye(n_in)
    for k in range(F):
        X = np.linalg.solve(Suu[k] + reg, Suy[k])
        coh[k] = np.einsum('ic,ic->c', np.conj(Suy[k]), X).real
    coh = np.clip(coh / np.maximum(Syy, 1e-300), 0.0, 1.0)

    s_in = np.stack([Suu[:, i, i].real for i in range(n_in)], axis=1)
    return freq, s_in, Syy, coh


def analyze_record(inputd, datout, fs: float, label: str = '',
                   nperseg: int = 0, names: Optional[Sequence[str]] = None,
                   skip_sec: float = 0.0) -> SpectraResult:
    """
    inputd : (N,) or (N, n_in) control / excitation channels.
    datout : (N,) or (N, n_out) responses.
    nperseg: Welch segment length; 0 picks one giving about 0.3 Hz
             resolution, capped so there are at least 16 segments.
    skip_sec: seconds dropped from the start (start-up transient).

    With more than one input the coherence is the multiple coherence: the
    share of the response explained by all inputs together.
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

    freq, s_in, Syy, coh = _spectra(U, Y, kw)
    kw2 = dict(kw, nperseg=nperseg // 2, noverlap=nperseg // 4)
    f2, _, Syy2, coh2 = _spectra(U, Y, kw2)
    coarse = (np.trapezoid((1 - coh2) * Syy2, f2, axis=0)
              / np.trapezoid(Syy2, f2, axis=0))

    nb = 8
    L = N // nb
    blocks = Y[:nb * L].reshape(nb, L, n_out)
    skew_se = skew(blocks, axis=1).std(axis=0, ddof=1) / np.sqrt(nb)
    kurt_se = kurtosis(blocks, axis=1).std(axis=0, ddof=1) / np.sqrt(nb)

    if names is None:
        names = [f"ch {c + 1}" for c in range(n_out)]
    return SpectraResult(label=label, fs=fs, freq=freq, s_in=s_in, s_tot=Syy,
                         coh=coh, band=excitation_band(freq, s_in),
                         in_rms=U.std(axis=0), rms=Y.std(axis=0),
                         skew=skew(Y, axis=0), kurt=kurtosis(Y, axis=0),
                         in_skew=skew(U, axis=0), in_kurt=kurtosis(U, axis=0),
                         skew_se=skew_se, kurt_se=kurt_se, nperseg=nperseg,
                         coarse_noncoherent=coarse, names=list(names))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
DEFINITIONS = ("total = all of the response.   coherent = the part linearly "
               "related to the input.   non-coherent = the remainder "
               "(nonlinearity, noise).   Grey span = excitation band; "
               "anything to its right is out of band.")
CASE_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
               'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']


def _shade_band(ax, band):
    ax.axvspan(band[0], band[1], color='0.92', zorder=0)


def _smooth(s, freq, hz):
    """Running mean of a spectral density over a span of hz."""
    k = int(round(hz / (freq[1] - freq[0])))
    if k < 2:
        return s
    return np.convolve(s, np.ones(k) / k, mode='same')


def _footer(fig, extra=''):
    fig.text(0.5, 0.005, DEFINITIONS + extra, ha='center', va='bottom',
             fontsize=9, color='0.25', wrap=True)


def plot_record(res: SpectraResult, channels: Optional[Sequence[int]] = None,
                f_max: float = 0.0, smooth_hz: float = 1.0,
                save_path=None, dpi: int = 130):
    """Dashboard for one record: total, coherent and non-coherent spectral
    density for each channel, then the per-channel fractions and amplitude
    statistics. Curves are averaged over smooth_hz for legibility."""
    import matplotlib.pyplot as plt
    n_out = res.s_tot.shape[1]
    ch = list(range(n_out)) if channels is None else list(channels)
    f_max = f_max or res.fs / 2
    ncol = 3
    nrow = int(np.ceil(len(ch) / ncol))
    fig = plt.figure(figsize=(14, 3.0 * nrow + 3.9))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [1.1])
    m = (res.freq > 0) & (res.freq <= f_max)
    f = res.freq
    for i, c in enumerate(ch):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        _shade_band(ax, res.band)
        ax.semilogy(f[m], _smooth(res.s_tot[:, c], f, smooth_hz)[m],
                    color='k', lw=1.6, label='total')
        ax.semilogy(f[m], _smooth(res.s_coh[:, c], f, smooth_hz)[m],
                    color='tab:blue', lw=1.1, label='coherent with input')
        ax.semilogy(f[m], _smooth(res.s_non[:, c], f, smooth_hz)[m],
                    color='tab:red', lw=1.1, label='non-coherent')
        top = res.s_tot[m, c].max()
        ax.set_ylim(top * 1e-7, top * 3)
        ax.set_xlim(0, f_max)
        lim = 'at most ' if res.resolution_limited[c] else ''
        ax.set_title(f"{res.names[c]}: {lim}"
                     f"{100*res.frac_noncoherent[c]:.1f}% non-coherent, "
                     f"{100*res.frac_out_of_band[c]:.1f}% out of band",
                     fontsize=9.5)
        ax.grid(True, which='major', alpha=0.3)
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
        ax.set_ylabel('spectral density')
        if i // ncol == nrow - 1:
            ax.set_xlabel('Hz')

    x = np.arange(len(ch))
    lab = [res.names[c] for c in ch]
    ax = fig.add_subplot(gs[nrow, 0])
    ax.bar(x - 0.2, 100 * res.frac_noncoherent[ch], 0.4, color='tab:red',
           label='non-coherent share')
    ax.bar(x + 0.2, 100 * res.frac_out_of_band[ch], 0.4, color='tab:gray',
           label='out-of-band share')
    ax.set_xticks(x, lab)
    ax.set_ylabel('% of mean square')
    ax.set_title('share of each channel\'s energy', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax = fig.add_subplot(gs[nrow, 1])
    ax.bar(x, res.skew[ch], 0.5, color='tab:green',
           yerr=2 * res.skew_se[ch], capsize=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x, lab)
    ax.set_title('skewness: 0 = symmetric amplitudes (bars ±2 s.e.)',
                 fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax = fig.add_subplot(gs[nrow, 2])
    ax.bar(x, res.kurt[ch], 0.5, color='tab:purple',
           yerr=2 * res.kurt_se[ch], capsize=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x, lab)
    ax.set_title('excess kurtosis: 0 = Gaussian (bars ±2 s.e.)',
                 fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    if len(ch) > 8:                      # many channels: slant the labels
        for a in fig.axes[-3:]:
            a.tick_params(axis='x', labelrotation=45)
    fig.suptitle(f"{res.label}   excitation band {res.band[0]:.0f}-"
                 f"{res.band[1]:.0f} Hz, input rms "
                 f"{', '.join(f'{v:.3g}' for v in res.in_rms)}", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig


def plot_comparison(results: Sequence[SpectraResult], channel: int = -1,
                    f_max: float = 0.0, smooth_hz: float = 1.0,
                    save_path=None, dpi: int = 130):
    """Several records of the same system (test levels, or a baseline
    against other cases). The first record is the baseline.

    Top: one panel per record for one channel, showing that record's total
    and non-coherent spectral density over the baseline's total in grey.
    Each is divided by its record's input mean square, so a linear system
    would overlay the baseline at any level. Bottom: per-channel shares and
    statistics, one bar colour per record."""
    import matplotlib.pyplot as plt
    n = len(results)
    r0 = results[0]
    n_out = r0.s_tot.shape[1]
    c = channel % n_out
    f_max = f_max or r0.fs / 2
    colors = [CASE_COLORS[i % len(CASE_COLORS)] for i in range(n)]
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(15, 3.4 * nrow + 4.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[nrow * 1.05, 1.0])
    gt = outer[0].subgridspec(nrow, ncol)
    gb = outer[1].subgridspec(1, 4)

    def norm(r, s):
        return _smooth(s, r.freq, smooth_hz) / (r.in_rms ** 2).sum()

    base = norm(r0, r0.s_tot[:, c])
    top = max(norm(r, r.s_tot[:, c]).max() for r in results)
    for i, (r, col) in enumerate(zip(results, colors)):
        ax = fig.add_subplot(gt[i // ncol, i % ncol])
        m = (r.freq > 0) & (r.freq <= f_max)
        m0 = (r0.freq > 0) & (r0.freq <= f_max)
        _shade_band(ax, r.band)
        if i > 0:
            ax.semilogy(r0.freq[m0], base[m0], color='0.55', lw=2.2,
                        label=f"{r0.label}: total (baseline)")
        ax.semilogy(r.freq[m], norm(r, r.s_tot[:, c])[m], color=col, lw=1.5,
                    label=f"{r.label}: total")
        ax.semilogy(r.freq[m], norm(r, r.s_non[:, c])[m], color='k', lw=0.9,
                    ls='--', label=f"{r.label}: non-coherent")
        ax.set_ylim(top * 1e-8, top * 3)
        ax.set_xlim(0, f_max)
        ax.set_title(f"{r.label}", fontsize=12, color=col, fontweight='bold')
        ax.grid(True, which='major', alpha=0.3)
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
        if i % ncol == 0:
            ax.set_ylabel(f"{r0.names[c]} spectral density\n/ input mean "
                          f"square")
        if i // ncol == nrow - 1:
            ax.set_xlabel('Hz')

    x = np.arange(n_out)
    w = 0.8 / n
    panels = [('non-coherent share, % of mean square',
               lambda r: 100 * r.frac_noncoherent),
              ('out-of-band share, % of mean square',
               lambda r: 100 * r.frac_out_of_band),
              ('skewness (0 = symmetric)', lambda r: r.skew),
              ('excess kurtosis (0 = Gaussian)', lambda r: r.kurt)]
    for j, (title, fn) in enumerate(panels):
        ax = fig.add_subplot(gb[0, j])
        for i, (r, col) in enumerate(zip(results, colors)):
            ax.bar(x + (i - (n - 1) / 2) * w, fn(r), w, color=col,
                   label=r.label)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x, [s.split()[-1] for s in r0.names])
        ax.set_xlabel(f"{r0.names[0].rsplit(' ', 1)[0]} number")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig, "   Panel titles and bars share one colour per record.")
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
