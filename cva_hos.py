"""
cva_hos.py
==========
Higher-order spectra: which frequencies are phase-locked to which.

A linear system driven by Gaussian noise keeps every frequency independent
of every other. A nonlinearity ties them together, and the way it does so
says what kind it is:

  quadratic coupling (bicoherence)
      energy at f1 and f2 locked to energy at f1 + f2. Comes from an
      asymmetric nonlinearity: stiffness that differs in tension and
      compression, a one-sided stop.

  cubic coupling (a slice of the tricoherence)
      energy at f1, f1 and f2 locked to energy at 2 f1 + f2 (the third
      harmonic when f2 = f1). Comes from a symmetric nonlinearity: a cubic
      spring, free play. A symmetric nonlinearity leaves the bicoherence at
      zero, which is why both are needed.

Both are normalised to lie between 0 (no locking) and 1 (complete).

The value expected from an uncoupled record is not zero. It depends on the
number of segments and, strongly, on how bursty the record is: if a few
segments carry most of the energy, a few terms dominate the average. The
floor is therefore worked out on the record itself, cell by cell: it is the
value the cell would average if every segment's contribution kept its size
but had a random phase, which is what "no locking" means.

Coupling is called present where a cell stands more than 30 times above
its own floor. With tens of thousands of cells on a map the best of them
reaches about 10 by chance (the linear six-mass records peak at 11), and
real coupling in those records stands at 60 to 120.

    hh = analyze_hos(datout, fs, label='80% tension loss', band_top=416.0)
    hh.table()
    fig = plot_hos(hh, channel=3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks, get_window

from cva_plotting import show_if_enabled
from cva_spectra import CASE_COLORS


@dataclass
class HosResult:
    label:  str
    fs:     float
    freq:   np.ndarray        # (K,) grid for f1 and f2
    bic:    np.ndarray        # (n_out, K, K) squared bicoherence, f1 x f2
    cub:    np.ndarray        # (n_out, K, K) squared cubic coupling
    bic_x:  np.ndarray        # (n_out, K, K) bic / that cell's floor
    cub_x:  np.ndarray        # (n_out, K, K)
    bic_floor: np.ndarray     # (n_out,) typical (median) cell floor
    cub_floor: np.ndarray
    psd:    np.ndarray        # (F, n_out)
    freq_psd: np.ndarray
    modes:  np.ndarray        # in-band mode frequencies used for the table
    band_top: float
    n_seg:  int
    names:  List[str] = field(default_factory=list)

    # ---- per channel -------------------------------------------------------
    def _peak(self, arr_x, arr, c):
        i, j = np.unravel_index(np.nanargmax(arr_x[c]), arr_x[c].shape)
        return i, j, arr[c, i, j], arr_x[c, i, j]

    @property
    def quad_index(self):
        """Squared bicoherence at each channel's most significant cell."""
        return np.array([self._peak(self.bic_x, self.bic, c)[2]
                         for c in range(len(self.names))])

    @property
    def cubic_index(self):
        return np.array([self._peak(self.cub_x, self.cub, c)[2]
                         for c in range(len(self.names))])

    @property
    def quad_excess(self):
        """How many times its own floor the most significant cell stands."""
        return np.nanmax(self.bic_x, axis=(1, 2))

    @property
    def cubic_excess(self):
        return np.nanmax(self.cub_x, axis=(1, 2))

    def source_type(self, margin: float = 30.0):
        """Per channel: 'asymmetric', 'symmetric' or 'none'.

        Quadratic coupling more than `margin` times its floor reads as
        asymmetric, whatever the cubic map shows: an asymmetric
        nonlinearity makes both, because sum frequencies combine again.
        Cubic coupling alone reads as symmetric."""
        out = []
        for c in range(len(self.names)):
            q = self.quad_excess[c] > margin
            k = self.cubic_excess[c] > margin
            out.append('asymmetric' if q else 'symmetric' if k else 'none')
        return out

    def strongest(self, c: int):
        """Most significant cell of each map at channel c:
        (f1, f2, target) for quadratic, then for cubic."""
        i, j, _, _ = self._peak(self.bic_x, self.bic, c)
        q = (self.freq[i], self.freq[j], self.freq[i] + self.freq[j])
        i, j, _, _ = self._peak(self.cub_x, self.cub, c)
        k = (self.freq[i], self.freq[j], 2 * self.freq[i] + self.freq[j])
        return q, k

    def _at(self, arr, arr_x, c, f1, f2):
        """Value and excess at the most significant of the 3 x 3 cells
        around (f1, f2); modes rarely sit exactly on a bin."""
        df = self.freq[1] - self.freq[0]
        i, j = int(round(f1 / df)), int(round(f2 / df))
        K = len(self.freq)
        best = (np.nan, np.nan)
        for a in (i - 1, i, i + 1):
            for b in (j - 1, j, j + 1):
                if 0 < a < K and 0 < b < K and np.isfinite(arr_x[c, a, b]):
                    if not best[1] > arr_x[c, a, b]:
                        best = (arr[c, a, b], arr_x[c, a, b])
        return best

    def pair_table(self, channel: int, verbose: bool = True):
        """Coupling at each pair of in-band modes, and whether the response
        spectrum has a peak at the frequency that coupling would feed."""
        c = channel
        df = self.freq_psd[1] - self.freq_psd[0]
        s = 10 * np.log10(self.psd[:, c] + 1e-300)
        pk, _ = find_peaks(s, prominence=4.0)
        peaks = self.freq_psd[pk]

        def near_peak(f):
            if f >= self.fs / 2 or len(peaks) == 0:
                return np.nan
            d = np.abs(peaks - f)
            return peaks[d.argmin()] if d.min() <= max(0.015 * f, 2 * df) else np.nan

        rows = []
        m = self.modes
        for a in range(len(m)):
            for b in range(a, len(m)):
                f1, f2 = m[a], m[b]
                v, x = self._at(self.bic, self.bic_x, c, f1, f2)
                rows.append(dict(kind='quadratic', f1=f1, f2=f2,
                                 target=f1 + f2, value=v, excess=x,
                                 peak=near_peak(f1 + f2)))
        for a in range(len(m)):
            for b in range(len(m)):
                f1, f2 = m[a], m[b]
                v, x = self._at(self.cub, self.cub_x, c, f1, f2)
                rows.append(dict(kind='cubic', f1=f1, f2=f2,
                                 target=2 * f1 + f2, value=v, excess=x,
                                 peak=near_peak(2 * f1 + f2)))
        rows = [r for r in rows if np.isfinite(r['value'])]
        if verbose:
            print(f"\n{self.label}, {self.names[c]}: coupling at the in-band "
                  f"mode pairs, most significant first")
            print(f"{'kind':>10} {'f1':>7} {'f2':>7} {'feeds':>8} "
                  f"{'coupling':>9} {'x floor':>8}  spectral peak there")
            for r in sorted(rows, key=lambda r: -r['excess'])[:10]:
                pk_ = (f"{r['peak']:.1f} Hz" if np.isfinite(r['peak'])
                       else 'none')
                print(f"{r['kind']:>10} {r['f1']:7.1f} {r['f2']:7.1f} "
                      f"{r['target']:8.1f} {r['value']:9.3f} "
                      f"{r['excess']:8.1f}  {pk_}")
        return rows

    def table(self, verbose: bool = True):
        kinds = self.source_type()
        rows = [dict(channel=self.names[c], quad=self.quad_index[c],
                     quad_floor=self.quad_excess[c],
                     cubic=self.cubic_index[c],
                     cubic_floor=self.cubic_excess[c], kind=kinds[c],
                     strongest=self.strongest(c))
                for c in range(len(self.names))]
        if verbose:
            print(f"\n{self.label}: frequency coupling "
                  f"({self.n_seg} segments, {self.freq[1]:.2f} Hz resolution)")
            print(f"{'channel':>8} {'quadratic':>10} {'x floor':>7} "
                  f"{'cubic':>8} {'x floor':>7}  {'reads as':<11} "
                  f"strongest cell")
            for r in rows:
                q, k = r['strongest']
                where = ''
                if r['kind'] == 'asymmetric':
                    where = f"{q[0]:.0f} + {q[1]:.0f} -> {q[2]:.0f} Hz"
                elif r['kind'] == 'symmetric':
                    where = f"2 x {k[0]:.0f} + {k[1]:.0f} -> {k[2]:.0f} Hz"
                print(f"{r['channel']:>8} {r['quad']:10.3f} "
                      f"{r['quad_floor']:7.1f} {r['cubic']:8.3f} "
                      f"{r['cubic_floor']:7.1f}  {r['kind']:<11} {where}")
        return rows


# ---------------------------------------------------------------------------
def _segments(y, nperseg, window):
    step = nperseg // 2
    n = (len(y) - nperseg) // step + 1
    idx = np.arange(nperseg)[None, :] + step * np.arange(n)[:, None]
    seg = y[idx]
    seg = seg - seg.mean(axis=1, keepdims=True)
    return np.fft.rfft(seg * window, axis=1)


def _maps(X, K, k_min):
    """Squared bicoherence and squared cubic coupling on a K x K grid of
    (f1, f2) bins, and each cell's floor: its expected value if every
    segment's term kept its size but had a random phase. Cells whose target
    frequency lies beyond Nyquist, or whose inputs are below k_min, are
    NaN. Returns bic, cub, bic_floor, cub_floor, all K x K."""
    F = X.shape[1]
    out = [np.full((K, K), np.nan) for _ in range(4)]
    bic, cub, bfl, cfl = out
    P = (np.abs(X) ** 2).mean(axis=0)
    X2 = X * X
    n = X.shape[0]

    def cell(prod, tgt):
        term = prod * np.conj(X[:, tgt])
        den = np.maximum((np.abs(prod) ** 2).mean(axis=0) * P[tgt], 1e-300)
        return (np.abs(term.mean(axis=0)) ** 2 / den,
                (np.abs(term) ** 2).mean(axis=0) / n / den)

    for i in range(k_min, K):
        j = np.arange(i, min(K, F - i))         # quadratic: (i, j) -> i + j
        if len(j):
            bic[i, j], bfl[i, j] = cell(X[:, [i]] * X[:, j], i + j)
            bic[j, i], bfl[j, i] = bic[i, j], bfl[i, j]
        j = np.arange(k_min, K)                 # cubic: (i, i, j) -> 2i + j
        j = j[2 * i + j < F]
        if len(j):
            cub[i, j], cfl[i, j] = cell(X2[:, [i]] * X[:, j], 2 * i + j)
    return bic, cub, bfl, cfl


def _mode_peaks(freq, psd, band_top, n_max=6):
    s = 10 * np.log10((psd / psd.max(axis=0)).sum(axis=1) + 1e-300)
    m = (freq > 0) & (freq <= band_top)
    idx = np.flatnonzero(m)
    pk, pr = find_peaks(s[idx], prominence=6.0)
    pk = idx[pk][np.argsort(pr['prominences'])[::-1][:n_max]]
    return np.sort(freq[pk])


def analyze_hos(datout, fs: float, label: str = '', band_top: float = 0.0,
                nperseg: int = 1024, f_max: float = 0.0,
                modes: Optional[Sequence[float]] = None,
                names: Optional[Sequence[str]] = None,
                skip_sec: float = 0.0) -> HosResult:
    """
    datout  : (N,) or (N, n_out) responses.
    band_top: top of the excitation band; in-band modes for the pair table
              are the spectral peaks below it (0 = Nyquist).
    nperseg : segment length; 1024 at 2500 Hz is 2.4 Hz resolution. Keep
              the resolution no finer than the width of the modes. A mode
              spread over several bins splits its coupling between them and
              reads weaker, most of all on the cubic map: for modes at 2%
              damping, 256 detects a cubic term that 1024 misses.
    f_max   : top of the f1 and f2 axes (0 = band_top, or Nyquist / 3).
    modes   : in-band mode frequencies, if already known.
    """
    Y = np.atleast_2d(np.asarray(datout, float).T).T
    Y = Y[int(round(skip_sec * fs)):]
    Y = Y - Y.mean(axis=0)
    N, n_out = Y.shape
    if names is None:
        names = [f"ch {c + 1}" for c in range(n_out)]
    band_top = band_top or fs / 2
    f_max = f_max or min(band_top * 1.05, fs / 6)
    freq_full = np.fft.rfftfreq(nperseg, 1 / fs)
    df = freq_full[1]
    K = int(min(len(freq_full) // 2, np.ceil(f_max / df) + 1))
    k_min = max(2, int(np.ceil(4.0 / df)))        # skip the lowest few Hz
    window = get_window('hann', nperseg)

    bic = np.zeros((n_out, K, K))
    cub = np.zeros((n_out, K, K))
    bic_x = np.zeros((n_out, K, K))
    cub_x = np.zeros((n_out, K, K))
    bfl = np.zeros(n_out)
    cfl = np.zeros(n_out)
    psd = np.zeros((len(freq_full), n_out))
    n_seg = 0
    for c in range(n_out):
        X = _segments(Y[:, c], nperseg, window)
        n_seg = len(X)
        psd[:, c] = (np.abs(X) ** 2).mean(axis=0)
        bic[c], cub[c], b0, c0 = _maps(X, K, k_min)
        bic_x[c], cub_x[c] = bic[c] / b0, cub[c] / c0
        bfl[c], cfl[c] = np.nanmedian(b0), np.nanmedian(c0)
    if modes is None:
        modes = _mode_peaks(freq_full, psd, band_top)
    return HosResult(label=label, fs=fs, freq=freq_full[:K], bic=bic, cub=cub,
                     bic_x=bic_x, cub_x=cub_x,
                     bic_floor=bfl, cub_floor=cfl, psd=psd,
                     freq_psd=freq_full, modes=np.asarray(modes, float),
                     band_top=band_top, n_seg=n_seg, names=list(names))


# ---------------------------------------------------------------------------
def plot_hos(hh: HosResult, channel: int = 0, save_path=None, dpi: int = 130):
    """One channel: the quadratic and cubic coupling maps with the in-band
    mode frequencies marked, and both indices at every channel against
    their floors."""
    import matplotlib.pyplot as plt
    c = channel % len(hh.names)
    f = hh.freq
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    panels = [(hh.bic[c], hh.quad_excess[c],
               '1. quadratic coupling: f1 and f2 locked to f1 + f2\n'
               '(asymmetric nonlinearity: tension differs from compression)'),
              (hh.cub[c], hh.cubic_excess[c],
               '2. cubic coupling: f1, f1 and f2 locked to 2 f1 + f2\n'
               '(symmetric nonlinearity: cubic spring, free play)')]
    for j, (arr, floor, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        vmax = max(0.2, np.nanmax(arr))
        im = ax.pcolormesh(f, f, arr.T, shading='auto', cmap='viridis',
                           vmin=0, vmax=vmax)
        for m in hh.modes:
            ax.axvline(m, color='w', lw=0.6, ls=':')
            ax.axhline(m, color='w', lw=0.6, ls=':')
        ax.set_xlabel('f1, Hz')
        ax.set_ylabel('f2, Hz')
        ax.set_title(f"{title}\nlargest {np.nanmax(arr):.2f}; most "
                     f"significant cell {floor:.0f} x its floor; dotted "
                     f"lines = in-band modes",
                     fontsize=10, loc='left')
        fig.colorbar(im, ax=ax, pad=0.01,
                     label='0 = independent, 1 = fully locked')

    x = np.arange(len(hh.names))
    for j, (val, name, col) in enumerate(
            [(hh.quad_excess, 'quadratic', 'tab:orange'),
             (hh.cubic_excess, 'cubic', 'tab:purple')]):
        ax = fig.add_subplot(gs[1, j])
        ax.bar(x, val, 0.55, color=col,
               label=f"{name} coupling, most significant cell / its floor")
        ax.axhline(30.0, color='k', lw=1.0, ls='--',
                   label='called present above this line')
        ax.set_xticks(x, hh.names)
        ax.set_yscale('log')
        ax.set_ylim(1.0, max(60.0, 1.5 * np.nanmax(val)))
        if len(hh.names) > 8:
            for t in ax.get_xticklabels():   # end of each label at its bar
                t.set(rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title(f"{j + 3}. {name} coupling at every channel",
                     fontsize=10, loc='left')
        ax.legend(fontsize=8, framealpha=0.9, loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)

    kinds = hh.source_type()
    fig.suptitle(f"{hh.label}: frequency coupling at {hh.names[c]} "
                 f"(reads as: {kinds[c]})", fontsize=13)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.text(0.5, 0.005,
             "A bright cell at (f1, f2) means those frequencies are "
             "phase-locked to their sum (left) or to 2 f1 + f2 (right).   "
             "A linear system gives maps at the floor everywhere.",
             ha='center', va='bottom', fontsize=9, color='0.25')
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig


def plot_hos_comparison(results: Sequence[HosResult], save_path=None,
                        dpi: int = 130):
    """Quadratic and cubic coupling at every channel, one colour per
    record, each bar drawn as a multiple of its own floor."""
    import matplotlib.pyplot as plt
    n = len(results)
    r0 = results[0]
    x = np.arange(len(r0.names))
    w = 0.8 / n
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, name, val, flo in (
            (axes[0], 'quadratic (asymmetric nonlinearity)',
             lambda r: r.quad_excess, lambda r: 1.0),
            (axes[1], 'cubic (symmetric nonlinearity)',
             lambda r: r.cubic_excess, lambda r: 1.0)):
        for i, r in enumerate(results):
            ax.bar(x + (i - (n - 1) / 2) * w, val(r) / flo(r), w,
                   color=CASE_COLORS[i % len(CASE_COLORS)], label=r.label)
        ax.axhline(30.0, color='k', lw=0.8, ls='--')
        ax.text(x[-1] + 0.45, 30.0, ' called\n present\n above', fontsize=7.5,
                va='center')
        ax.set_yscale('log')
        ax.set_xticks(x, r0.names)
        ax.set_ylabel('most significant cell / its floor')
        ax.set_title(f"{name}", fontsize=11, loc='left')
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, axis='y', which='both', alpha=0.25)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.008,
             "Bars: the most significant cell of each map, as a multiple of "
             "its own floor. Chance alone reaches about 10.\n"
             "Quadratic present = asymmetric (tension differs from "
             "compression). Cubic without quadratic = symmetric (cubic "
             "spring, free play).",
             ha='center', va='bottom', fontsize=9, color='0.25')
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
