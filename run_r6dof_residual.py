"""
run_r6dof_residual.py
=====================
Step 2 on the simulated six-mass records: the non-coherent response as a
time history. See run_r6dof_spectra.py for the file layout.

    python run_r6dof_residual.py

Figures go to results/step2/.
"""
from pathlib import Path

import cva_plotting
from cva_residual import noncoherent_history, plot_residual, track_residual
from run_r6dof_spectra import FS, NPERSEG, RECORDS, load

OUT = Path(__file__).parent / 'results' / 'step2'
CHANNEL = 3          # mass 4, next to the nonlinear spring


def main():
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"mass {i}" for i in range(1, 7)]
    res = {}
    for label, fname in RECORDS.items():
        u, y = load(fname)
        rr = noncoherent_history(u, y, FS, label=label, names=names,
                                 skip_sec=1.0, nperseg=NPERSEG)
        rr.table()
        slug = label.replace(' ', '_').replace(',', '').replace('%', 'pct')
        plot_residual(rr, channel=CHANNEL, zoom=(30.0, 30.25),
                      save_path=OUT / f"residual_{slug}_mass4.png")
        if not label.startswith('linear'):
            print(f"\n{label}: peaks of the residual, tracked without input")
            trk = track_residual(rr)
            trk.summary()
            trk.plot_tracks(save_path=OUT / f"residual_tracks_{slug}.png")
        res[label] = rr
    return res


if __name__ == '__main__':
    main()
