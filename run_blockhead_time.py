"""
run_blockhead_time.py
=====================
Does the spectral density change during the blockhead record?

Splits the 23 s record into 1 s blocks, and for every channel reports how the
power in three frequency ranges moves from the first third of the record to
the last third, and where the largest narrow-band change is.

    python run_blockhead_time.py

Pages and the table go to results/blockhead_time/.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import spectrogram, welch

from run_blockhead import DATA, OUT

RANGES = [(2, 2031, 'in band, 2-2031 Hz'),
          (2031, 3000, '2031-3000 Hz'),
          (3000, 6400, '3000-6400 Hz')]
NPERSEG = 1024            # 12.5 Hz


def load_all():
    m = sio.loadmat(DATA)
    fs = float(m['SampleRate'].squeeze())
    names = (['Inp X', 'Inp Y', 'Inp Z', 'Rot X', 'Rot Y', 'Rot Z']
             + [str(r[1][0]) for r in m['cinp_corners']]
             + [str(r[1][0]) for r in m['cinp_center']]
             + [str(r[1][0]) for r in m['crsp_full']])
    y = np.hstack([m['ainp_6dof'], m['ainp_corners'], m['ainp_center'],
                   m['arsp_full']])
    return y, fs, names


def band_history(y, fs, block_sec=1.0):
    """Power in each of RANGES for each block: (n_blocks, n_ch, 3), and the
    block centre times."""
    n = int(block_sec * fs)
    nb = len(y) // n
    out = np.zeros((nb, y.shape[1], len(RANGES)))
    for b in range(nb):
        f, p = welch(y[b * n:(b + 1) * n], fs, nperseg=NPERSEG, axis=0)
        for r, (lo, hi, _) in enumerate(RANGES):
            k = (f >= lo) & (f < hi)
            out[b, :, r] = p[k].sum(0) * (f[1] - f[0])
    return (np.arange(nb) + 0.5) * block_sec, out


def main():
    y, fs, names = load_all()
    t, bp = band_history(y, fs)
    nb = len(t)
    third = nb // 3
    first, last = bp[:third].mean(0), bp[-third:].mean(0)
    change = 10 * np.log10(last / first)                 # dB, last vs first
    spread = 10 * np.log10(bp.max(0) / bp.min(0))        # dB, max vs min block
    # narrow-band: PSD of the first and last third, 50 Hz smoothing
    n3 = len(y) // 3
    f, p1 = welch(y[:n3], fs, nperseg=NPERSEG, axis=0)
    _, p3 = welch(y[-n3:], fs, nperseg=NPERSEG, axis=0)
    ker = np.ones(4) / 4
    sm = lambda p: np.apply_along_axis(np.convolve, 0, p, ker, 'same')
    d = 10 * np.log10(sm(p3) / sm(p1))
    keep = (f >= 20) & (f <= 6000)
    lines = ['Change in power, last third of the record against the first '
             'third (dB), and the\nlargest difference between any two 1 s '
             'blocks (dB). Narrow-band: 50 Hz wide.\n',
             f"{'channel':>8} " + ' '.join(f"{'chg':>6} {'spread':>6}"
                                           for _ in RANGES)
             + '   largest narrow-band change',
             f"{'':>8} " + ' '.join(f"{lab[:13]:>13}" for *_, lab in RANGES)]
    for c, nm in enumerate(names):
        k = np.argmax(np.abs(d[keep, c]))
        lines.append(f"{nm:>8} " + ' '.join(
            f"{change[c, r]:+6.2f} {spread[c, r]:6.2f}"
            for r in range(len(RANGES)))
            + f"   {d[keep, c][k]:+5.2f} dB at {f[keep][k]:6.0f} Hz")
    text = '\n'.join(lines)
    out = OUT / 'blockhead_time'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'table.txt').write_text(text)
    print(text)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for nm in ('A17-X', 'A10-Y', 'A4-Y', 'A5-Z'):
        c = names.index(nm)
        fig, ax = plt.subplots(3, 1, figsize=(11, 11))
        fs_, ts_, S = spectrogram(y[:, c], fs, nperseg=NPERSEG,
                                  noverlap=NPERSEG // 2)
        # average to 0.5 s columns
        q = int(0.5 / (ts_[1] - ts_[0]))
        ncol = S.shape[1] // q
        S = S[:, :ncol * q].reshape(len(fs_), ncol, q).mean(2)
        im = ax[0].pcolormesh(np.arange(ncol + 1) * 0.5, fs_,
                              10 * np.log10(S[:-1]), shading='flat',
                              cmap='viridis')
        fig.colorbar(im, ax=ax[0], label='PSD, dB re 1 g²/Hz')
        ax[0].axhline(2031, color='w', ls='--', lw=1)
        ax[0].text(0.3, 2100, 'top of excitation band, 2031 Hz', color='w',
                   fontsize=8)
        ax[0].set(xlabel='time, s', ylabel='frequency, Hz',
                  title=f"{nm}: spectral density against time "
                        "(0.5 s columns, 12.5 Hz resolution)")
        for (lo, hi, lab), col in zip(RANGES, ('k', 'tab:red', 'tab:blue')):
            r = [x[2] for x in RANGES].index(lab)
            ax[1].plot(t, 10 * np.log10(bp[:, c, r] / bp[:, c, r].mean()),
                       'o-', color=col, ms=3,
                       label=f"{lab}: last third {change[c, r]:+.2f} dB "
                             "against first third")
        ax[1].axhline(0, color='grey', lw=0.5)
        ax[1].set(xlabel='time, s',
                  ylabel='power in 1 s block, dB re record mean',
                  title='Power in three frequency ranges, each relative to '
                        'its own mean over the record')
        ax[1].legend(fontsize=8)
        ax[2].semilogy(f, p1[:, c], 'tab:blue', lw=1,
                       label='first third of the record (0-7.7 s)')
        ax[2].semilogy(f, p3[:, c], 'tab:red', lw=1,
                       label='last third of the record (15.4-23 s)')
        ax[2].axvline(2031, color='grey', ls='--', lw=1,
                      label='top of excitation band')
        ax[2].set(xlabel='frequency, Hz', ylabel='PSD, g²/Hz',
                  title='Spectral density early and late', xlim=(0, 6400))
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"psd_time_{nm}.png", dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
