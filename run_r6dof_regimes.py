"""
run_r6dof_regimes.py
====================
Step 5 on the simulated six-mass records: is there more than one state at
the same response level, or one state seen at different levels?

    python run_r6dof_regimes.py

Figures go to results/step5/.
"""
from pathlib import Path

import cva_plotting
import numpy as np

from cva_level import track_in_band
from cva_regimes import analyze_regimes, plot_regimes
from cva_residual import noncoherent_history
from run_r6dof_spectra import FS, NPERSEG, RECORDS, load

OUT = Path(__file__).parent / 'results' / 'step5'
BAND_TOP = 416.0


def main():
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"mass {i}" for i in range(1, 7)]
    res = {}
    for label, fname in RECORDS.items():
        u, y = load(fname)
        rr = noncoherent_history(u, y, FS, label=label, names=names,
                                 skip_sec=1.0, nperseg=NPERSEG)
        t = np.arange(len(u)) / FS
        trk = track_in_band(t[2500:], u[2500:], y[2500:], FS, BAND_TOP)
        rg = analyze_regimes(trk, rr, label=label)
        rg.summary()
        slug = label.replace(' ', '_').replace(',', '').replace('%', 'pct')
        plot_regimes(rg, save_path=OUT / f"regimes_{slug}.png")
        res[label] = rg
    return res


if __name__ == '__main__':
    main()
