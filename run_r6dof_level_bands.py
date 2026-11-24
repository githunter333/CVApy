"""
run_r6dof_level_bands.py
========================
Step 3 on the simulated six-mass records: in-band modes against response
level, and the response peaks above the excitation band (frequency,
damping, time history). See run_r6dof_spectra.py for the file layout.

    python run_r6dof_level_bands.py

Figures go to results/step3/.
"""
from pathlib import Path

import numpy as np

import cva_plotting
from cva_bands import analyze_bands, plot_band
from cva_level import level_dependence, plot_level, track_in_band
from run_r6dof_spectra import FS, RECORDS, load

OUT = Path(__file__).parent / 'results' / 'step3'
BAND_TOP = 416.0         # top of the excitation band, from step 1
SKIP = 2500              # first second: start-up from rest
# Peaks too small for the automatic finder but worth a page: the dead zone
# also rings the fifth linear mode (521.5 Hz), faintly.
EXTRA_BANDS = {'dead zone': [(521.5, 30.0)]}


def slug(label):
    return label.replace(' ', '_').replace(',', '').replace('%', 'pct')


def main():
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"mass {i}" for i in range(1, 7)]
    lev, bands = {}, {}
    for label, fname in RECORDS.items():
        u, y = load(fname)
        u, y = u[SKIP:], y[SKIP:]
        t = np.arange(len(u)) / FS + SKIP / FS
        try:
            trk = track_in_band(t, u, y, FS, BAND_TOP)
            lev[label] = level_dependence(trk, label=label)
            lev[label].table()
        except RuntimeError as e:
            print(f"\n{label}: in-band tracking failed: {e}")
        bb = analyze_bands(y, FS, BAND_TOP, label=label, names=names)
        if label in EXTRA_BANDS:
            found = [(b.f0, b.half) for b in bb.bands]
            bb = analyze_bands(y, FS, BAND_TOP, label=label, names=names,
                               bands=sorted(found + EXTRA_BANDS[label]))
        bb.table()
        bands[label] = bb
        for k, b in enumerate(bb.bands):
            plot_band(bb, k, zoom=(30.0, 30.25),
                      save_path=OUT / f"band_{slug(label)}_{b.f0:.0f}Hz.png")

    base = [lev[k] for k in ('linear', 'linear, 5x level') if k in lev]
    for label in RECORDS:
        if label.startswith('linear') or label not in lev:
            continue
        plot_level(base + [lev[label]],
                   save_path=OUT / f"level_{slug(label)}.png")
    plot_level(base, save_path=OUT / 'level_linear.png')
    return lev, bands


if __name__ == '__main__':
    main()
