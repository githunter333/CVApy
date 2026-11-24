"""
cva_bands.py
============
Response peaks above the excitation band, one band at a time.

A nonlinearity can ring modes the input never reaches (a dead zone does
this), or build peaks of its own. For each such peak this module gives

  * the band-limited time history at every channel, and its envelope;
  * frequency and apparent damping, block by block along the record;
  * which channels carry it;
  * how its level moves against the level of the in-band response.

    bb = analyze_bands(datout, fs, band_top=416.0, label='dead zone')
    bb.table()
    for k in range(len(bb.bands)):
        plot_band(bb, k)

Each band is brought down to zero frequency (complex demodulation), which
gives the envelope directly and a well-conditioned fit: within a block the
complex envelope of a single mode obeys z[n+1] = a z[n], and the pole a
carries the frequency and the decay rate. The pole is taken from the
envelope's autocorrelation at lags 1 and 2, which measurement noise does not
bias.

"Apparent damping" is the decay rate of the band's free ringing between
excitations. It equals the modal damping when the mode is rung by short,
irregular kicks, and overstates it if the forcing is itself narrow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from cva_plotting import show_if_enabled


@dataclass
class Band:
    f0:      float                # centre (spectral peak), Hz
    half:    float                # half-width, Hz
    signal:  np.ndarray           # (N, n_out) band-limited time history
    envelope: np.ndarray          # (N, n_out)
    times:   np.ndarray           # (nb,) block centres
    freq:    np.ndarray           # (nb,) Hz
    zeta:    np.ndarray           # (nb,) apparent damping ratio
    level:   np.ndarray           # (nb,) band rms, strongest channel
    freq_all: float               # whole-record values
    zeta_all: float
    zeta_halfpower: float         # cross-check from the spectral peak width
    rms:     np.ndarray           # (n_out,) band rms per channel
    share:   np.ndarray           # (n_out,) band mean square / channel total
    corr_inband: float            # correlation of log band level with log
                                  # in-band level, block by block

    @property
    def strongest(self) -> int:
        return int(np.argmax(self.rms))


@dataclass
class BandsResult:
    label:    str
    fs:       float
    time:     np.ndarray
    band_top: float
    inband:   np.ndarray          # (N, n_out) response below band_top
    inband_level: np.ndarray      # (nb,) rss over channels of block rms
    block_sec: float
    bands:    List[Band] = field(default_factory=list)
    names:    List[str] = field(default_factory=list)
    freq_psd: np.ndarray = None
    psd:      np.ndarray = None   # (F, n_out)

    def table(self, verbose: bool = True):
        rows = []
        for b in self.bands:
            ok = np.isfinite(b.freq)
            rows.append(dict(
                f0=b.f0, half=b.half, freq=b.freq_all,
                freq_scatter=np.nanstd(b.freq), zeta=b.zeta_all,
                zeta_median=np.nanmedian(b.zeta),
                zeta_scatter=np.nanstd(b.zeta),
                zeta_halfpower=b.zeta_halfpower,
                strongest=self.names[b.strongest],
                share=b.share[b.strongest], corr=b.corr_inband,
                found=ok.mean(),
                steady=bool(abs(b.zeta_all - np.nanmedian(b.zeta))
                            < 0.4 * b.zeta_all)))
        if verbose:
            print(f"\n{self.label}: response peaks above {self.band_top:.0f} Hz")
            if not rows:
                print("  none found")
            else:
                print(f"{'band Hz':>12} {'freq':>8} {'scatter':>8} "
                      f"{'zeta %':>7} {'median':>7} {'scatter':>8} "
                      f"{'half-pwr':>9} {'strongest':>10} {'share':>7} "
                      f"{'vs in-band':>11}")
            for r in rows:
                print(f"{r['f0']-r['half']:5.0f}-{r['f0']+r['half']:<5.0f} "
                      f"{r['freq']:8.2f} {r['freq_scatter']:8.2f} "
                      f"{100*r['zeta']:7.2f} {100*r['zeta_median']:7.2f} "
                      f"{100*r['zeta_scatter']:8.2f} "
                      f"{100*r['zeta_halfpower']:9.2f} {r['strongest']:>10} "
                      f"{100*r['share']:6.1f}% {r['corr']:11.2f}"
                      f"{'' if r['steady'] else '   not one steady mode'}")
        return rows


# ---------------------------------------------------------------------------
def find_bands(freq, psd, band_top: float, n_max: int = 4,
               prominence_db: float = 5.0, rel_half: float = 0.08,
               smooth_hz: float = 5.0, range_db: float = 30.0,
               min_share: float = 1e-4):
    """Peaks above band_top in the spectral density of the channel with the
    most out-of-band energy. Returns a list of (centre, half-width) in
    order of frequency, keeping the n_max strongest. Half-width is rel_half
    of the centre, narrowed so neighbouring bands do not overlap. Peaks more
    than range_db below the strongest are dropped, and nothing is returned
    if the out-of-band energy is under min_share of that channel's total."""
    m = (freq > 1.03 * band_top) & (freq < 0.97 * freq[-1])
    idx = np.flatnonzero(m)
    if len(idx) < 5:
        return []
    df = freq[1] - freq[0]
    oob = psd[idx].sum(axis=0)
    c = int(np.argmax(oob / psd.sum(axis=0)))
    if oob[c] < min_share * psd[:, c].sum():
        return []
    k = max(1, int(round(smooth_hz / df)))
    s = np.convolve(psd[:, c], np.ones(k) / k, mode='same')
    db = 10 * np.log10(s + 1e-300)
    pk, _ = find_peaks(db[idx], prominence=prominence_db,
                       distance=max(1, int(0.08 * band_top / df)))
    pk = idx[pk]
    if len(pk) == 0:                      # one broad hump: take its top
        pk = np.array([idx[np.argmax(db[idx])]])
    pk = pk[db[pk] > db[pk].max() - range_db]
    pk = pk[np.argsort(db[pk])[::-1][:n_max]]
    centres = np.sort(freq[pk])
    out = []
    for i, f0 in enumerate(centres):
        half = rel_half * f0
        if i > 0:
            half = min(half, 0.45 * (f0 - centres[i - 1]))
        if i < len(centres) - 1:
            half = min(half, 0.45 * (centres[i + 1] - f0))
        half = min(half, f0 - 1.02 * band_top)
        if half > 4 * df:
            out.append((float(f0), float(half)))
    return out


def _demodulate(Y, fs, f0, half):
    """Complex envelope of the band f0 +/- half at every channel."""
    t = np.arange(len(Y)) / fs
    carrier = np.exp(-2j * np.pi * f0 * t)[:, None]
    sos = butter(6, half, btype='low', fs=fs, output='sos')
    w = Y * carrier
    z = sosfiltfilt(sos, w.real, axis=0) + 1j * sosfiltfilt(sos, w.imag, axis=0)
    return 2.0 * z, carrier


def _pole(z, q):
    """Shared one-step pole of the decimated complex envelopes z[::q],
    from autocorrelation lags 1 and 2 summed over channels."""
    zd = z[::q]
    if len(zd) < 8:
        return np.nan
    r1 = np.sum(zd[1:] * np.conj(zd[:-1]))
    r2 = np.sum(zd[2:] * np.conj(zd[:-2]))
    if abs(r1) == 0:
        return np.nan
    return r2 / r1


def _freq_zeta(a, f0, fs_d):
    if not np.isfinite(a) or abs(a) <= 0:
        return np.nan, np.nan
    f = f0 + np.angle(a) * fs_d / (2 * np.pi)
    sigma = -np.log(abs(a)) * fs_d
    if f <= 0:
        return np.nan, np.nan
    return f, sigma / (2 * np.pi * f)


def _halfpower(freq, s, f0, half):
    """Damping ratio from the width of the spectral peak at half power."""
    m = (freq >= f0 - half) & (freq <= f0 + half)
    f, p = freq[m], s[m]
    if len(f) < 5:
        return np.nan
    i = int(np.argmax(p))
    above = p >= 0.5 * p[i]
    lo = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(f) - 1 and above[hi + 1]:
        hi += 1
    if lo == 0 or hi == len(f) - 1:
        return np.nan
    return (f[hi] - f[lo]) / (2 * f[i])


def analyze_bands(datout, fs: float, band_top: float, label: str = '',
                  bands: Optional[Sequence[Tuple[float, float]]] = None,
                  names: Optional[Sequence[str]] = None,
                  block_sec: float = 0.5, skip_sec: float = 0.0,
                  n_max: int = 4) -> BandsResult:
    """
    datout   : (N,) or (N, n_out) responses.
    band_top : top of the excitation band, Hz (SpectraResult.band[1]).
    bands    : optional list of (centre Hz, half-width Hz); found from the
               spectrum when omitted.
    block_sec: block length for the frequency and damping tracks.
    """
    Y = np.atleast_2d(np.asarray(datout, float).T).T
    k0 = int(round(skip_sec * fs))
    Y = Y[k0:]
    Y = Y - Y.mean(axis=0)
    N, n_out = Y.shape
    time = (k0 + np.arange(N)) / fs
    if names is None:
        names = [f"ch {c + 1}" for c in range(n_out)]

    nper = int(min(2 ** np.round(np.log2(fs / 0.6)), 2 ** np.floor(np.log2(N / 8))))
    fpsd, psd = welch(Y, fs=fs, nperseg=nper, axis=0)
    if bands is None:
        bands = find_bands(fpsd, psd, band_top, n_max=n_max)

    sos = butter(8, band_top, btype='low', fs=fs, output='sos')
    inband = sosfiltfilt(sos, Y, axis=0)

    W = int(round(block_sec * fs))
    step = max(1, W // 2)
    starts = np.arange(0, N - W + 1, step)
    centres = time[starts] + 0.5 * W / fs
    in_level = np.array([np.sqrt((inband[s:s + W] ** 2).mean()) for s in starts])
    total_ms = (Y ** 2).mean(axis=0)

    out = []
    for f0, half in bands:
        z, carrier = _demodulate(Y, fs, f0, half)
        sig = (z * np.conj(carrier)).real
        env = np.abs(z)
        # Envelope sampled at four times the half-width: on synthetic modes
        # of known damping this reads within about 4% of the truth, where
        # twice the half-width reads 10-25% high.
        q = max(1, int(round(fs / (4.0 * half))))
        fs_d = fs / q
        f_all, z_all = _freq_zeta(_pole(z, q), f0, fs_d)
        fr = np.full(len(starts), np.nan)
        ze = np.full(len(starts), np.nan)
        for i, s in enumerate(starts):
            fr[i], ze[i] = _freq_zeta(_pole(z[s:s + W], q), f0, fs_d)
        bad = ~np.isfinite(fr) | (np.abs(fr - f0) > half) | (ze <= 0)
        fr[bad] = np.nan
        ze[bad] = np.nan
        rms = sig.std(axis=0)
        c = int(np.argmax(rms))
        lev = np.array([sig[s:s + W, c].std() for s in starts])
        ok = (lev > 0) & (in_level > 0)
        corr = (np.corrcoef(np.log(lev[ok]), np.log(in_level[ok]))[0, 1]
                if ok.sum() > 3 else np.nan)
        out.append(Band(f0=f0, half=half, signal=sig, envelope=env,
                        times=centres, freq=fr, zeta=ze, level=lev,
                        freq_all=f_all, zeta_all=z_all,
                        zeta_halfpower=_halfpower(fpsd, psd[:, c], f0, half),
                        rms=rms, share=rms ** 2 / total_ms,
                        corr_inband=float(corr)))
    return BandsResult(label=label, fs=fs, time=time, band_top=band_top,
                       inband=inband, inband_level=in_level,
                       block_sec=block_sec, bands=out, names=list(names),
                       freq_psd=fpsd, psd=psd)


# ---------------------------------------------------------------------------
def plot_band(bb: BandsResult, k: int = 0,
              zoom: Optional[Tuple[float, float]] = None,
              save_path=None, dpi: int = 130):
    """One out-of-band peak on one page.

    1. where the band sits in the response spectrum;
    2. which channels carry the band;
    3. a short stretch of the band-limited time history at its strongest
       channel, with its envelope, over the in-band response;
    4. band level and in-band level along the whole record;
    5. frequency block by block;  6. apparent damping block by block.
    """
    import matplotlib.pyplot as plt
    b = bb.bands[k]
    c = b.strongest
    t = bb.time
    if zoom is None:
        mid = 0.5 * (t[0] + t[-1])
        zoom = (mid, mid + 0.25)
    z = (t >= zoom[0]) & (t <= zoom[1])
    lo, hi = b.f0 - b.half, b.f0 + b.half

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.1, 1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(bb.freq_psd, bb.psd[:, c], color='k', lw=1.0,
                label=f"{bb.names[c]} response spectral density")
    ax.axvspan(0, bb.band_top, color='0.92', zorder=0,
               label='excitation band')
    ax.axvspan(lo, hi, color='tab:orange', alpha=0.35,
               label=f"this band, {lo:.0f}-{hi:.0f} Hz")
    top = bb.psd[:, c].max()
    ax.set_ylim(top * 1e-8, top * 3)
    ax.set_xlim(0, bb.fs / 2)
    ax.set_xlabel('Hz')
    ax.set_title('1. where the band sits', fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(bb.names))
    ax.bar(x, b.rms, 0.6, color='tab:orange')
    for xi, s in zip(x, b.share):
        ax.text(xi, b.rms[xi], f"{100*s:.1f}%", ha='center', va='bottom',
                fontsize=8)
    ax.set_xticks(x, bb.names)
    ax.set_ylabel('band rms')
    ax.set_title('2. which channels carry it (label = share of that '
                 'channel\'s mean square)', fontsize=10, loc='left')
    ax.grid(True, axis='y', alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t[z], bb.inband[z, c], color='0.6', lw=1.2,
            label=f"in-band response (below {bb.band_top:.0f} Hz)")
    ax.plot(t[z], b.signal[z, c], color='tab:orange', lw=0.9,
            label=f"band {lo:.0f}-{hi:.0f} Hz time history")
    ax.plot(t[z], b.envelope[z, c], color='tab:red', lw=1.2,
            label='its envelope')
    ax.plot(t[z], -b.envelope[z, c], color='tab:red', lw=1.2)
    ax.set_xlabel('s')
    ax.set_title(f"3. {bb.names[c]}: a {zoom[1]-zoom[0]:.2f} s stretch",
                 fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1, 1.0),
              ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.semilogy(b.times, bb.inband_level, color='0.5', lw=1.2,
                label=f"in-band level, rms in {bb.block_sec:g} s blocks")
    ax.semilogy(b.times, b.level, color='tab:orange', lw=1.2,
                label=f"band level at {bb.names[c]}")
    ax.axvspan(*zoom, color='0.85', zorder=0)
    ax.set_xlabel('s')
    ax.set_title(f"4. level along the record: correlation of the two "
                 f"{b.corr_inband:+.2f} (grey bar = the stretch above)",
                 fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1, 1.0),
              ncol=2, frameon=False)
    ax.grid(True, which='major', alpha=0.3)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(b.times, b.freq, '.-', color='tab:blue', ms=4, lw=0.6,
            label=f"each {bb.block_sec:g} s block")
    ax.axhline(b.freq_all, color='k', lw=1.0, ls='--',
               label=f"whole record {b.freq_all:.1f} Hz")
    ax.set_ylim(lo, hi)
    ax.set_xlabel('s')
    ax.set_ylabel('Hz')
    ax.set_title(f"5. frequency: block-to-block scatter "
                 f"{np.nanstd(b.freq):.2f} Hz", fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(b.times, 100 * b.zeta, '.-', color='tab:green', ms=4, lw=0.6,
            label=f"each {bb.block_sec:g} s block")
    ax.axhline(100 * b.zeta_all, color='k', lw=1.0, ls='--',
               label=f"whole record {100*b.zeta_all:.2f}%")
    if np.isfinite(b.zeta_halfpower):
        ax.axhline(100 * b.zeta_halfpower, color='tab:purple', lw=1.0, ls=':',
                   label=f"from spectral peak width "
                         f"{100*b.zeta_halfpower:.2f}%")
    ax.set_ylim(0, max(1e-3, 100 * 2.5 * np.nanmedian(b.zeta)))
    ax.set_xlabel('s')
    ax.set_ylabel('% of critical')
    ax.set_title('6. apparent damping', fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{bb.label}: out-of-band peak near {b.f0:.0f} Hz "
                 f"({b.freq_all:.1f} Hz, {100*b.zeta_all:.2f}% damping)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.text(0.5, 0.005,
             "in-band = below the top of the excitation band.   band time "
             "history = the response filtered to this peak alone.   "
             "apparent damping = decay rate of the band's ringing; equals "
             "modal damping when the mode is rung by short kicks.",
             ha='center', va='bottom', fontsize=9, color='0.25')
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
