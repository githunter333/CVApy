"""
run_bridge.py
=============
Every nonlinearity analysis on the bench "bridge" random tests of 21 May 2025:
one record of the structure as built ("normal") and three with loose
quarters placed on it, each 10 s at 10,000 Hz from a PicoScope.

    Channel A  drive voltage                      (not used)
    Channel B  drive force, 50 mV/lb              -> the input
    Channel C  response accelerometer, 100 mV/g
    Channel D  accelerometer next to the drive, 100 mV/g

    python run_bridge.py

Only "loose 4" shows rattling (24.8% non-coherent and excess kurtosis 1.07
at D, against 3.3% and 0.06 for normal). "loose 2" and "loose 3" read like
the normal record and were probably run with nothing loose (N. Hunter,
24 Nov 2026).

Pages and tables for each record go to results/bridge/<record>/, and the
side-by-side comparisons to results/bridge/.
"""
from pathlib import Path

import numpy as np

from cva_report import full_report

DIR = Path('/Users/nhunterjr/Waveforms/Waveforms 2/test setup random responses')
RECORDS = {
    'normal': 'Bridge Random Test Normal 052125.txt',
    'loose 2': 'Bridge Random Test loose quarters 052125-0002.txt',
    'loose 3': 'Bridge Random Test loose quarters 052125-0003.txt',
    'loose 4': 'Bridge Random Test loose quarters 052125-0004.txt',
}
NAMES = ['C response', 'D at drive']
NPERSEG = 4096          # 2.4 Hz; about 48 averages in 10 s
# two response channels: 16 lags gives the model enough states for the
# eight modes in the excitation band
OUT = Path(__file__).parent / 'results' / 'bridge'


def load(name):
    """Force (lb) as a column, the two accelerations (g), and the sample
    rate, for one of RECORDS."""
    d = np.nan_to_num(np.genfromtxt(DIR / RECORDS[name], skip_header=3))
    fs = round(1.0 / (d[1, 0] - d[0, 0]))
    return d[:, 2:3] * 1000 / 50, d[:, 3:5] * 1000 / 100, float(fs)


def main():
    import matplotlib.pyplot as plt

    from cva_hos import plot_hos_comparison
    from cva_spectra import plot_comparison
    res = {}
    for name in RECORDS:
        u, y, fs = load(name)
        res[name] = full_report(u, y, fs, OUT / name.replace(' ', '_'),
                                label=name, names=NAMES, channel=0,
                                nperseg=NPERSEG, track_lags=16)
    spectra = [r['spectra'] for r in res.values() if 'spectra' in r]
    for c, nm in enumerate(NAMES):
        plot_comparison(spectra, channel=c,
                        save_path=OUT / f"compare_spectra_{nm[0]}.png")
    hos = [r['hos'] for r in res.values() if 'hos' in r]
    plot_hos_comparison(hos, save_path=OUT / 'compare_coupling.png')
    plt.close('all')
    return res


if __name__ == '__main__':
    main()
