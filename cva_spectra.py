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
    from_spectra: bool = False   # built by analyze_spectra: no time history,
                                 # so no skewness or kurtosis
    oob_known: bool = True       # False when the spectra stop at the top of
                                 # the excitation band
    n_avg:   int = 0             # averages behind the spectra, if known

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
        if not self.oob_known:
            return np.full(self.s_tot.shape[1], np.nan)
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

    def notes(self):
        """Cautions that apply to a result built from spectra."""
        out = []
        if not self.from_spectra:
            return out
        out.append('rms values cover only the frequency range of the '
                   'spectra; skewness and kurtosis need time histories')
        if not self.oob_known:
            out.append(f'the spectra stop at {self.freq[-1]:.4g} Hz, the top '
                       f'of the excitation band, so energy above the band '
                       f'cannot be seen')
        n_in = self.s_in.shape[1]
        if self.n_avg and self.n_avg < 10 * n_in:
            out.append(f'{self.n_avg} averages for {n_in} inputs: the '
                       f'coherence reads high by about {n_in}/{self.n_avg}, '
                       f'so the non-coherent share reads low')
        elif not self.n_avg and n_in > 1:
            out.append(f'number of averages not given; with {n_in} inputs '
                       f'and few averages the non-coherent share reads low')
        return out

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
        if verbose and self.from_spectra:
            print(f"\n{self.label}: from spectra, {self.freq[0]:.4g} to "
                  f"{self.freq[-1]:.4g} Hz at "
                  f"{self.freq[1] - self.freq[0]:.4g} Hz; excitation band "
                  f"{self.band[0]:.0f}-{self.band[1]:.0f} Hz, "
                  f"input rms {np.round(self.in_rms, 3)}")
            for line in self.notes():
                print('  note: ' + line)
            print(f"{'channel':>8} {'rms':>9} {'non-coh':>8} {'in-band':>8} "
                  f"{'out-band':>9}")
            lim = self.resolution_limited
            for c, r in enumerate(rows):
                oob = (f"{100*r['out_of_band']:8.1f}%" if self.oob_known
                       else f"{'-':>9}")
                print(f"{r['channel']:>8} {r['rms']:9.3f} "
                      f"{100*r['noncoherent']:7.1f}% "
                      f"{100*r['noncoherent_in_band']:7.1f}% {oob}"
                      f"{'  resolution-limited' if lim[c] else ''}")
        elif verbose:
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


def _freq_first(a, F, what):
    """Move the frequency axis of a 3-D array to the front."""
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError(f'{what} must have three dimensions, got {a.shape}')
    if a.shape[0] == F and a.shape[2] != F:
        return a
    if a.shape[2] == F:
        return np.moveaxis(a, 2, 0)
    raise ValueError(f'{what} has shape {a.shape}; no axis matches the '
                     f'{F} frequency lines')


def _columns(a, F, what):
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] != F and a.shape[1] == F:
        a = a.T
    if a.shape[0] != F:
        raise ValueError(f'{what} has shape {a.shape}; expected {F} '
                         f'frequency lines')
    return a


def analyze_spectra(freq, s_resp=None, coherence=None, s_in=None,
                    c_resp=None, c_in=None, H=None, c_in_resp=None,
                    label: str = '', names: Optional[Sequence[str]] = None,
                    band: Optional[Tuple[float, float]] = None,
                    n_avg: int = 0) -> SpectraResult:
    """
    The coherent / non-coherent / out-of-band split from stored spectra,
    for a test where the time histories were not kept. The result works
    with table(), plot_record() and plot_comparison() like one from
    analyze_record(); skewness and kurtosis are not available.

    freq   : (F,) frequencies, Hz, evenly spaced.

    Response spectral density, one of
      s_resp : (F, n_out)
      c_resp : (n_out, n_out, F) or (F, n_out, n_out) cross-spectral
               matrix; its diagonal is used.

    Coherence with the inputs, one of
      coherence : (F, n_out); the multiple coherence if several inputs.
      H and c_in : frequency responses (n_out, n_in, F) or (F, n_out, n_in)
               and the input cross-spectral matrix (n_in, n_in, F) or
               (F, n_in, n_in), with c_in[i, j] = E[u_i conj(u_j)] so that
               the coherent part is H c_in H^H.
      c_in_resp and c_in : input-to-response cross-spectra (n_in, n_out, F)
               or (F, n_in, n_out), E[conj(u) y], and the input matrix.
    Frequency responses with input spectra but no response spectral density
    or coherence are not enough: they give only the coherent part.

    Input spectral density (for the excitation band and the input rms)
      s_in : (F, n_in), or the diagonal of c_in. If neither is given, give
             band.

    band   : (low, high) Hz excitation band; found from the input spectra
             if omitted.
    n_avg  : number of averages behind the spectra, if known. With few
             averages per input the coherence reads high.

    Out-of-band energy can be reported only if the spectra run past the top
    of the excitation band; otherwise it is left blank.
    """
    freq = np.asarray(freq, float).ravel()
    F = len(freq)
    if F < 8 or np.ptp(np.diff(freq)) > 1e-6 * freq[-1]:
        raise ValueError('freq must be evenly spaced, at least 8 lines')
    df = freq[1] - freq[0]

    if s_resp is not None:
        Syy = _columns(s_resp, F, 's_resp')
    elif c_resp is not None:
        Cy = _freq_first(c_resp, F, 'c_resp')
        Syy = np.real(np.einsum('kii->ki', Cy))
    else:
        raise ValueError('Give the response spectral density: s_resp or '
                         'c_resp')
    n_out = Syy.shape[1]

    Cu = None if c_in is None else _freq_first(c_in, F, 'c_in')
    Suy = None                     # (F, n_in, n_out), E[conj(u) y]
    if coherence is not None:
        coh = _columns(coherence, F, 'coherence')
    elif H is not None and Cu is not None:
        Hf = _freq_first(H, F, 'H')                     # (F, n_out, n_in)
        Suy = np.einsum('kji,koj->kio', Cu, Hf)        # E[conj(u) y]
        coh = np.real(np.einsum('koj,kjl,kol->ko', Hf, Cu, np.conj(Hf)))
        coh = coh / np.maximum(Syy, 1e-300)
    elif c_in_resp is not None and Cu is not None:
        Suy = _freq_first(c_in_resp, F, 'c_in_resp')
        coh = _coherent_power(np.conj(Cu), Suy) / np.maximum(Syy, 1e-300)
    else:
        raise ValueError('Give the coherence, or H with c_in, or c_in_resp '
                         'with c_in. Frequency responses alone give only '
                         'the coherent part.')
    if coh.shape != Syy.shape:
        raise ValueError(f'coherence {coh.shape} does not match the '
                         f'response spectra {Syy.shape}')
    strong = Syy > 1e-6 * Syy.max(axis=0)
    if np.mean(coh[strong] > 1.05) > 0.01:
        raise ValueError(
            'The coherent part exceeds the measured response at '
            f'{100 * np.mean(coh[strong] > 1.05):.0f}% of the lines. Check '
            'that the arrays belong together and follow the conventions in '
            'the docstring (c_in[i, j] = E[u_i conj(u_j)]).')
    coh = np.clip(coh, 0.0, 1.0)

    if s_in is not None:
        Sin = _columns(s_in, F, 's_in')
    elif Cu is not None:
        Sin = np.real(np.einsum('kii->ki', Cu))
    else:
        Sin = None
    if band is None:
        if Sin is None:
            raise ValueError('Give the input spectral density (s_in or '
                             'c_in) or the excitation band')
        band = excitation_band(freq, Sin)
    if Sin is None:
        Sin = np.full((F, 1), np.nan)

    # the same split at half the frequency resolution, where the cross
    # spectra are there to form it: pairs of lines averaged
    coarse = None
    if Suy is not None and Cu is not None:
        n2 = F // 2
        pair = lambda a: 0.5 * (a[0:2 * n2:2] + a[1:2 * n2:2])
        S2 = pair(Syy)
        c2 = np.clip(_coherent_power(pair(np.conj(Cu)), pair(Suy))
                     / np.maximum(S2, 1e-300), 0.0, 1.0)
        coarse = ((1 - c2) * S2).sum(axis=0) / S2.sum(axis=0)

    # out of band is visible only if the spectra run past the band
    oob_known = freq[-1] > band[1] + max(5 * df, 0.02 * band[1])
    if names is None:
        names = [f"ch {c + 1}" for c in range(n_out)]
    nan = np.full(n_out, np.nan)
    n_in = Sin.shape[1]
    return SpectraResult(
        label=label, fs=2.0 * (freq[-1] + df), freq=freq, s_in=Sin,
        s_tot=Syy, coh=coh, band=(float(band[0]), float(band[1])),
        in_rms=np.sqrt(np.trapezoid(Sin, freq, axis=0)),
        rms=np.sqrt(np.trapezoid(Syy, freq, axis=0)),
        skew=nan, kurt=nan.copy(),
        in_skew=np.full(n_in, np.nan), in_kurt=np.full(n_in, np.nan),
        skew_se=nan.copy(), kurt_se=nan.copy(), nperseg=0,
        coarse_noncoherent=coarse, names=[str(n) for n in names],
        from_spectra=True, oob_known=bool(oob_known), n_avg=int(n_avg))


def _coherent_power(Suu, Suy):
    """Suy^H Suu^-1 Suy per frequency and response channel."""
    n_in = Suu.shape[1]
    scale = np.trace(Suu.real, axis1=1, axis2=2).max()
    reg = 1e-12 * scale * np.eye(n_in)
    X = np.linalg.solve(Suu + reg, Suy)
    return np.einsum('kic,kic->kc', np.conj(Suy), X).real


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
                     + (f"{100*res.frac_out_of_band[c]:.1f}% out of band"
                        if res.oob_known else "out of band not covered"),
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
    ax.bar(x + 0.2, np.nan_to_num(100 * res.frac_out_of_band[ch]), 0.4,
           color='tab:gray',
           label='out-of-band share' if res.oob_known
           else 'out-of-band share: not covered by the spectra')
    ax.set_xticks(x, lab)
    ax.set_ylabel('% of mean square')
    ax.set_title('share of each channel\'s energy', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    if res.from_spectra:
        ax = fig.add_subplot(gs[nrow, 1])
        ax.bar(x, 100 * res.frac_noncoherent_in_band[ch], 0.5,
               color='tab:red')
        ax.set_xticks(x, lab)
        ax.set_ylabel('% of in-band mean square')
        ax.set_title('non-coherent share inside the excitation band',
                     fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax = fig.add_subplot(gs[nrow, 2])
        ax.axis('off')
        import textwrap
        ax.text(0.0, 0.95, 'From spectra only:\n\n'
                + '\n\n'.join(textwrap.fill('- ' + n, 52,
                                              subsequent_indent='  ')
                              for n in res.notes()),
                va='top', ha='left', fontsize=9, transform=ax.transAxes)
    else:
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
        for a in fig.axes[-3:-1] if res.from_spectra else fig.axes[-3:]:
            for t in a.get_xticklabels():    # end of each label at its bar
                t.set(rotation=45, ha='right', rotation_mode='anchor')
    fig.suptitle(f"{res.label}   excitation band {res.band[0]:.0f}-"
                 f"{res.band[1]:.0f} Hz, input rms "
                 f"{', '.join(f'{v:.3g}' for v in res.in_rms)}", fontsize=13,
                 y=1 - 0.12 / fig.get_figheight())
    # leave half an inch at the top for the heading
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 0.5 / fig.get_figheight()))
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
        # "mass 1", "mass 2", ... -> ticks 1, 2, ... under "mass number";
        # any other naming is shown in full
        heads = {s.rsplit(' ', 1)[0] for s in r0.names}
        if len(heads) == 1 and all(s.split()[-1].isdigit() and ' ' in s
                                   for s in r0.names):
            ax.set_xticks(x, [s.split()[-1] for s in r0.names])
            ax.set_xlabel(f"{heads.pop()} number")
        else:
            ax.set_xticks(x, r0.names, rotation=30 if len(r0.names) > 6
                          else 0, ha='right' if len(r0.names) > 6
                          else 'center')
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig, "   Panel titles and bars share one colour per record.")
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
