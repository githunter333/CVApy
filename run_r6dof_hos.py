"""
run_r6dof_hos.py
================
Step 4 on the simulated six-mass records: quadratic and cubic frequency
coupling. See run_r6dof_spectra.py for the file layout.

    python run_r6dof_hos.py

Figures go to results/step4/.
"""
from pathlib import Path

import cva_plotting
from cva_hos import analyze_hos, plot_hos, plot_hos_comparison
from run_r6dof_spectra import FS, RECORDS, load

OUT = Path(__file__).parent / 'results' / 'step4'
BAND_TOP = 416.0
CHANNEL = 3              # mass 4, next to the nonlinear spring


def slug(label):
    return label.replace(' ', '_').replace(',', '').replace('%', 'pct')


def main():
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"mass {i}" for i in range(1, 7)]
    res = {}
    for label, fname in RECORDS.items():
        _, y = load(fname)
        hh = analyze_hos(y, FS, label=label, band_top=BAND_TOP, names=names,
                         skip_sec=1.0)
        hh.table()
        hh.pair_table(CHANNEL)
        plot_hos(hh, channel=CHANNEL,
                 save_path=OUT / f"coupling_{slug(label)}_mass4.png")
        res[label] = hh
    plot_hos_comparison([res[k] for k in ('linear', '10% tension loss',
                                          '80% tension loss', 'dead zone',
                                          'cubic')],
                        save_path=OUT / 'compare_types.png')
    return res


if __name__ == '__main__':
    main()
