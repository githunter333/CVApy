"""
run_r6dof_spectra.py
====================
Step 1 on the simulated six-mass records: coherent / non-coherent /
out-of-band spectra and amplitude statistics, linear baseline against each
nonlinear case.

Each r6dof output file holds, per row: six mass accelerations, then the base
acceleration actually applied (column 7), then two further columns not used
here. The sample rate is 2500 Hz. The matching `...in.dat` file is the
unscaled drive; two records (`linlong`, `cubic2ss`) were run at five times
that level, so the input is always taken from column 7.

    python run_r6dof_spectra.py

Figures go to results/step1/.
"""
from pathlib import Path

import numpy as np

import cva_plotting
from cva_spectra import analyze_record, plot_comparison, plot_record

DATA = Path('/Users/nhunterjr/Code/shared/Nonlinear Data/Experimental_Data/r6dof')
OUT = Path(__file__).parent / 'results' / 'step1'
FS = 2500.0
# The first mode has about 0.14% damping, a resonance 0.16 Hz wide, so the
# coherence needs much finer resolution than the default: 0.076 Hz here.
NPERSEG = 32768

# label -> output file
RECORDS = {
    'linear':            'r6doflinout2.dat',
    'linear, 5x level':  'r6doflinoutlong.dat',
    '10% tension loss':  'r6dofblin10pct2out.dat',
    '80% tension loss':  'r6dofblin80pct2out.dat',
    'dead zone':         'r6dofdeadzone2out.dat',
    'cubic':             'r6dofcubic2out.dat',
    'cubic s':           'r6dofcubic2sout.dat',
    'cubic ss, 5x level': 'r6dofcubic2ssout.dat',
}


def load(name):
    a = np.loadtxt(DATA / name)
    return a[:, 6], a[:, :6]


def main():
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"mass {i}" for i in range(1, 7)]
    res = {}
    for label, fname in RECORDS.items():
        u, y = load(fname)
        res[label] = analyze_record(u, y, FS, label=label, names=names,
                                    skip_sec=1.0, nperseg=NPERSEG)
        res[label].table()
        slug = label.replace(' ', '_').replace(',', '').replace('%', 'pct')
        plot_record(res[label], save_path=OUT / f"record_{slug}.png")
    plot_comparison([res[k] for k in ('linear', '10% tension loss',
                                      '80% tension loss', 'dead zone',
                                      'cubic')],
                    channel=5, save_path=OUT / 'compare_types_mass6.png')
    plot_comparison([res[k] for k in ('linear', 'linear, 5x level')],
                    channel=5, save_path=OUT / 'compare_linear_levels.png')
    plot_comparison([res[k] for k in ('linear', 'cubic', 'cubic s',
                                      'cubic ss, 5x level')],
                    channel=5, save_path=OUT / 'compare_cubic.png')
    for label in ('linear', '80% tension loss', 'dead zone', 'cubic'):
        pk = res[label].noncoherent_peaks(5)
        print(f"{label}: strongest non-coherent peaks at mass 6: "
              f"{np.round(np.sort(pk), 1)} Hz")
    return res


if __name__ == '__main__':
    main()
