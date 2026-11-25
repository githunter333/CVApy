"""
run_r6dof_sweep.py
==================
The local CVA sweep on the simulated six-mass records, each nonlinear case
against the linear baseline. See run_r6dof_spectra.py for the file layout.

    python run_r6dof_sweep.py                 # every record
    python run_r6dof_sweep.py "dead zone"     # the baseline and one record

For each record the sweep is run with the three neighbour choices (shape,
time, level) and reports:

  * the model order the data supports, against the linear record's;
  * each tracked mode: how often it is found, its frequency scatter from
    window to window, and its change per doubling of response level;
  * each mode's shift from the linear record's, poles the linear record
    does not have, and at what level they appear;
  * one state or two.

Figures and tables.txt go to results/sweep/.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

import cva_plotting
from cva_phi_cluster import CVAPhiCluster
from cva_run import CVAConfig, CVARun
from run_r6dof_spectra import FS, RECORDS, load

OUT = Path(__file__).parent / 'results' / 'sweep'
SKIP = 2500                  # first second: start-up from rest
MODES = ('shape', 'time', 'level')
COLORS = {'shape': 'tab:orange', 'time': 'tab:blue', 'level': 'tab:green'}
LAGS, TOL, NEIGHBORS, STRIDE = 16, 5e-4, 960, 250
F_MAX = 1000.0
BASELINE = {'cubic ss, 5x level': 'linear, 5x level'}   # default: 'linear'


def slug(label):
    return label.replace(' ', '_').replace(',', '').replace('%', 'pct')


def sweep(label, mode):
    """One sweep. Returns dict with the run, the mode tracks, the response
    level per window, and the printed log."""
    u, y = load(RECORDS[label])
    u, y = u[SKIP:, None], y[SKIP:]
    run = CVARun(CVAConfig(lags=LAGS, tol=TOL, neighbors=NEIGHBORS,
                           eval_stride=STRIDE, fs=FS, neighbor_mode=mode))
    run.time = np.arange(len(y)) / FS + SKIP / FS
    run.inputd, run.datout, run.fs = u, y, FS
    run.n_out, run.n_in = y.shape[1], 1
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        run.build_database()
        run.run(n_jobs=4)
        cl = CVAPhiCluster(run, f_max=F_MAX)
        try:
            cl.fit()
        except ValueError as e:        # no pole repeats from window to window
            cl.ref_freq = np.zeros(0)
            cl.track_freq = cl.track_zeta = np.zeros((len(cl.use_idx), 0))
            cl.verdict = f'no modes tracked: {e}'
    lev = run.window_level()[[run.windows[w].window_idx for w in cl.use_idx]]
    return dict(run=run, cl=cl, lev=lev, log=log.getvalue())


def per_doubling(lev, f):
    """Percent change of a frequency track per doubling of level."""
    ok = np.isfinite(f)
    if ok.sum() < 10:
        return np.nan
    return float(100 * np.polyfit(np.log2(lev[ok]), f[ok], 1)[0]
                 / np.nanmedian(f))


def by_level(lev, found, n=5):
    """Share of windows where `found` is true, in n groups of equal size
    from quietest to loudest."""
    grp = np.clip(np.digitize(lev, np.quantile(lev, np.linspace(0, 1, n + 1))[1:-1]),
                  0, n - 1)
    return np.array([100 * found[grp == k].mean() for k in range(n)])


def table(label, res, base):
    """Print the tables for one record; returns the rows for the summary."""
    print(f"\n{'=' * 78}\n{label}"
          f"{'' if base is None else '   (baseline: ' + base[0] + ')'}")
    rows = {}
    for mode in MODES:
        r = res[mode]
        run, cl, lev = r['run'], r['cl'], r['lev']
        b = None if base is None else base[1][mode]['run']
        o = run.order_report(baseline=b, verbose=False)
        rows[mode] = o
        print(f"\n  neighbours by {mode}: order supported {o['order']:.0f} "
              f"({o['lo']:.0f} to {o['hi']:.0f})"
              + (f", linear baseline {o['baseline']:.0f}, "
                 f"excess {o['excess']:+.0f}" if b is not None else '')
              + f"; quietest third {o['quiet']:.0f}, loudest third "
                f"{o['loud']:.0f}")
        bref = None if b is None else base[1][mode]['cl'].ref_freq
        print(f"    {'mode Hz':>9} {'found':>6} {'scatter Hz':>11} "
              f"{'damping %':>10} {'per doubling':>13} "
              f"{'quiet fifth':>12} {'loud fifth':>11}")
        for j, f0 in enumerate(cl.ref_freq):
            f = cl.track_freq[:, j]
            share = by_level(lev, np.isfinite(f))
            note = ''
            if bref is not None and len(bref):
                near = bref[np.argmin(np.abs(bref - f0))]
                shift = 100 * (f0 / near - 1)
                if abs(shift) >= 10:
                    note = '   <- no such mode tracked in the linear record'
                elif abs(shift) >= 0.1:
                    note = f'   <- linear {near:.2f} Hz, {shift:+.1f}%'
            print(f"    {f0:9.2f} {100*np.isfinite(f).mean():5.0f}% "
                  f"{np.nanstd(f):11.3f} "
                  f"{100*np.nanmedian(cl.track_zeta[:, j]):10.3f} "
                  f"{per_doubling(lev, f):+12.3f}% "
                  f"{share[0]:11.0f}% {share[-1]:10.0f}%"
                  f"{note}")
        print(f"    -> {cl.verdict}")
    return rows


def plot_record(label, res, base, save_path):
    """Per record: each tracked mode against response level, and how often
    each mode is found from the quietest to the loudest windows."""
    import matplotlib.pyplot as plt
    cl0 = res['level']['cl']
    nm = len(cl0.ref_freq)
    if nm == 0:
        return
    ncol = min(nm, 3)
    nrow = int(np.ceil(nm / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 3.6 * nrow + 4.6))
    gs = fig.add_gridspec(nrow + 1, ncol,
                          height_ratios=[1] * nrow + [1.25])
    for j, f0 in enumerate(cl0.ref_freq):
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        for mode in MODES:
            cl, lev = res[mode]['cl'], res[mode]['lev']
            if len(cl.ref_freq) == 0:
                continue
            k = int(np.argmin(np.abs(cl.ref_freq - f0)))
            if abs(cl.ref_freq[k] - f0) > 0.03 * f0:
                continue
            f = cl.track_freq[:, k]
            ax.semilogx(lev / np.median(lev), f, '.', color=COLORS[mode],
                        ms=3, alpha=0.7,
                        label=f"by {mode}: ±{np.nanstd(f):.3g} Hz, "
                              f"{per_doubling(lev, f):+.2f}%/doubling")
        if base is not None:
            bref = base[1]['level']['cl'].ref_freq
            k = int(np.argmin(np.abs(bref - f0)))
            if abs(bref[k] - f0) < 0.03 * f0:
                ax.axhline(bref[k], color='k', lw=1.0,
                           label=f"linear record: {bref[k]:.2f} Hz")
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.set_title(f"mode near {f0:.1f} Hz", fontsize=10, loc='left')
        ax.set_xlabel('response level in the window / median')
        ax.set_ylabel('Hz')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

    ax = fig.add_subplot(gs[nrow, :])
    cl, lev = res['level']['cl'], res['level']['lev']
    cmap = plt.cm.viridis(np.linspace(0, 0.9, nm))
    for j, f0 in enumerate(cl.ref_freq):
        ax.plot(range(5), by_level(lev, np.isfinite(cl.track_freq[:, j])),
                'o-', color=cmap[j], label=f"{f0:.1f} Hz")
    ax.set_xticks(range(5), ['quietest fifth', '2', '3', '4',
                             'loudest fifth'])
    ax.set_ylim(0, 105)
    ax.set_xlabel('windows grouped by response level')
    ax.set_ylabel('% of windows with the pole')
    ax.set_title('How often each mode is found, by response level '
                 '(neighbours by level)', fontsize=10, loc='left')
    ax.legend(fontsize=8, ncol=min(nm, 6), title='mode')
    ax.grid(True, alpha=0.3)
    o = res['level']['run'].order_report(verbose=False)
    fig.suptitle(f"{label}: sweep with the input history in each fit. "
                 f"Order supported {o['order']:.0f}; "
                 f"{res['level']['cl'].verdict}", fontsize=11, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_orders(orders, save_path):
    """Model order supported by each record, for each neighbour choice."""
    import matplotlib.pyplot as plt
    labels = list(orders)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    w = 0.26
    for i, mode in enumerate(MODES):
        x = np.arange(len(labels)) + (i - 1) * w
        med = np.array([orders[k][mode]['order'] for k in labels])
        lo = np.array([orders[k][mode]['lo'] for k in labels])
        hi = np.array([orders[k][mode]['hi'] for k in labels])
        ax.bar(x, med, w, color=COLORS[mode],
               yerr=[med - lo, hi - med], capsize=2, ecolor='0.3',
               label=f"neighbours by {mode} (bar: median; whisker: "
                     f"10th to 90th percentile of windows)")
    if 'linear' in orders:
        ax.axhline(orders['linear']['level']['order'], color='k', ls='--',
                   lw=1.0, label='linear record')
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=9)
    ax.set_ylabel('model order the data supports')
    ax.set_title(f"Model order needed once the input is accounted for "
                 f"(lags {LAGS}, tol {TOL:g}, {NEIGHBORS} neighbours). "
                 f"More than the linear record = response the input does "
                 f"not explain.", fontsize=10, loc='left', wrap=True)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(only=None):
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [k for k in RECORDS
              if not only or k in only or k.startswith('linear')]
    res, orders, text = {}, {}, io.StringIO()
    for label in labels:
        print(f"sweeping {label} ...", flush=True)
        res[label] = {m: sweep(label, m) for m in MODES}
    for label in labels:
        bname = BASELINE.get(label, 'linear')
        base = (None if label.startswith('linear') or bname not in res
                else (bname, res[bname]))
        with contextlib.redirect_stdout(text):
            orders[label] = table(label, res[label], base)
        plot_record(label, res[label], base,
                    OUT / f"sweep_{slug(label)}.png")
    plot_orders(orders, OUT / 'sweep_orders.png')
    (OUT / 'tables.txt').write_text(text.getvalue())
    print(text.getvalue())
    return res, orders


if __name__ == '__main__':
    main(only=sys.argv[1:] or None)
