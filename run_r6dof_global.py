"""
run_r6dof_global.py
===================
Can the local models of a sweep be replaced by one model that depends on
the response level? (cva_global.GlobalModel)

    python run_r6dof_global.py                    # every record
    python run_r6dof_global.py cubic "dead zone"  # some of them

For each record the sweep is run with neighbours by time. A global model is
fitted to the windows of the first 40 s: each tracked mode's frequency,
damping, shape and presence as a function of the local response level, and
the model order likewise. It is then asked for the modes of every window in
the last 20 s from that window's level alone, and compared with

  * the modes fitted directly in those windows, and
  * a single linear model (one set of modes for all levels), fitted to the
    same first 40 s.

Tables go to results/global/tables.txt, one page per record to
results/global/global_<record>.png.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

import cva_plotting
from cva_global import GlobalModel, cluster_arrays
from run_r6dof_spectra import RECORDS
from run_r6dof_sweep import slug, sweep

OUT = Path(__file__).parent / 'results' / 'global'
T_SPLIT = 40.0


def evaluate(label):
    r = sweep(label, 'time')
    cl = r['cl']
    if len(cl.ref_freq) == 0:
        return None
    d = cluster_arrays(cl)
    train, test = d['time'] < T_SPLIT, d['time'] >= T_SPLIT
    gm = GlobalModel.from_cluster(cl, train, degree=1)
    lin = GlobalModel.from_cluster(cl, train, degree=0)
    return dict(cl=cl, d=d, train=train, test=test, gm=gm, lin=lin,
                s_gm=gm.score(cl, test), s_lin=lin.score(cl, test))


def table(label, e):
    d, test = e['d'], e['test']
    lev = np.exp(d['log_level'])
    print(f"\n{'=' * 78}\n{label}: fitted on {e['train'].sum()} windows "
          f"(first {T_SPLIT:.0f} s), tested on {test.sum()} (the rest); "
          f"level range x{lev.max() / lev.min():.1f}")
    print("  errors against the modes fitted directly in the test windows; "
          "'level' = global model, 'one' = single linear model")
    print(f"  {'mode Hz':>8} {'found':>6} | {'freq rms err, Hz':>18} | "
          f"{'damping rms err, %':>19} | {'shape MAC':>13} | "
          f"{'presence err':>13}")
    print(f"  {'':>8} {'':>6} | {'level':>8} {'one':>9} | {'level':>9} "
          f"{'one':>9} | {'level':>6} {'one':>6} | {'level':>6} {'one':>6}")
    for g, l in zip(e['s_gm'], e['s_lin']):
        print(f"  {g['mode']:8.1f} {100*g['found']:5.0f}% | "
              f"{g['freq_rms']:8.3f} {l['freq_rms']:9.3f} | "
              f"{100*g['zeta_rms']:9.3f} {100*l['zeta_rms']:9.3f} | "
              f"{g['mac']:6.3f} {l['mac']:6.3f} | "
              f"{g['presence_err']:6.2f} {l['presence_err']:6.2f}")
    print(f"  model order rms error: level {e['s_gm'][0]['order_rms']:.2f} "
          f"states, one {e['s_lin'][0]['order_rms']:.2f} "
          f"(order in the test windows: {np.percentile(d['order'][test], 10):.0f} "
          f"to {np.percentile(d['order'][test], 90):.0f})")


def plot(label, e, save_path, min_found=0.2):
    import matplotlib.pyplot as plt
    d, train, test, gm, lin = e['d'], e['train'], e['test'], e['gm'], e['lin']
    x = d['log_level']
    xr = x - np.median(x)                       # log(level / median)
    rel = np.exp(xr)
    grid = np.linspace(x.min(), x.max(), 60)
    grel = np.exp(grid - np.median(x))
    pg, pl = gm.predict(grid), lin.predict(grid)
    show = [j for j in range(len(gm.ref_freq))
            if np.isfinite(d['freq'][:, j]).mean() >= min_found]
    n = len(show) + 1
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.7 * n + 0.8),
                             squeeze=False)

    def curves(ax, yg, yl, unit):
        ax.plot(grel, yg, color='tab:red', lw=1.8,
                label='global model (depends on level)')
        ax.plot(grel, yl, color='k', lw=1.2, ls='--',
                label='single linear model')
        ax.set_xscale('log')
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3, which='both')

    def limits(v):
        lo, hi = np.nanpercentile(v, [2, 98])
        pad = 0.2 * (hi - lo) + 1e-9
        return lo - pad, hi + pad

    for row, j in enumerate(show):
        g, l = e['s_gm'][j], e['s_lin'][j]
        for col, (key, scale, unit) in enumerate(
                (('freq', 1.0, 'Hz'), ('zeta', 100.0, '% of critical'))):
            ax = axes[row, col]
            v = scale * d[key][:, j]
            ax.plot(rel[train], v[train], '.', color='0.65', ms=3,
                    label='windows used for the fit (first 40 s)')
            ax.plot(rel[test], v[test], '.', color='tab:blue', ms=4,
                    label='test windows (last 20 s)')
            curves(ax, scale * pg[key][:, j], scale * pl[key][:, j], unit)
            ax.set_ylim(*limits(v))
            what = 'frequency' if key == 'freq' else 'damping'
            err = ((g['freq_rms'], l['freq_rms'], 'Hz') if key == 'freq'
                   else (100 * g['zeta_rms'], 100 * l['zeta_rms'], '%'))
            ax.set_title(f"mode near {gm.ref_freq[j]:.1f} Hz: {what}. "
                         f"Test error {err[0]:.3g} {err[2]} (global), "
                         f"{err[1]:.3g} {err[2]} (single)",
                         fontsize=9, loc='left')
            if row == 0:
                ax.legend(fontsize=7, loc='best')
        ax = axes[row, 2]
        bins = np.quantile(x, np.linspace(0, 1, 9))
        mid, frac = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (x >= a) & (x <= b)
            mid.append(np.exp(x[m].mean() - np.median(x)))
            frac.append(100 * d['presence'][m, j].mean())
        ax.plot(mid, frac, 'o', color='tab:blue', ms=5,
                label='share of windows with this mode, by level')
        curves(ax, 100 * pg['presence'][:, j], 100 * pl['presence'][:, j],
               '% of windows')
        ax.set_ylim(-5, 105)
        ax.set_title(f"mode near {gm.ref_freq[j]:.1f} Hz: how often it is "
                     f"present", fontsize=9, loc='left')
        if row == 0:
            ax.legend(fontsize=7, loc='best')

    ax = axes[-1, 0]
    ax.plot(rel[train], d['order'][train], '.', color='0.65', ms=3)
    ax.plot(rel[test], d['order'][test], '.', color='tab:blue', ms=4)
    curves(ax, pg['order'], pl['order'], 'states')
    ax.set_title(f"model order. Test error "
                 f"{e['s_gm'][0]['order_rms']:.2g} states (global), "
                 f"{e['s_lin'][0]['order_rms']:.2g} (single)", fontsize=9,
                 loc='left')
    ax = axes[-1, 1]
    jj = np.arange(len(show))
    ax.bar(jj - 0.2, [e['s_gm'][j]['mac'] for j in show], 0.4,
           color='tab:red', label='global model')
    ax.bar(jj + 0.2, [e['s_lin'][j]['mac'] for j in show], 0.4, color='0.4',
           label='single linear model')
    ax.set_xticks(jj, [f"{gm.ref_freq[j]:.0f}" for j in show])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('mode, Hz')
    ax.set_ylabel('MAC')
    ax.set_title('mode shape: agreement with the shape fitted in each test '
                 'window (median MAC, 1 = same shape)', fontsize=9,
                 loc='left')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    axes[-1, 2].axis('off')
    for a in axes[-1, :1]:
        a.set_xlabel('response level in the window / median')
    for a in axes[len(show) - 1, :]:
        a.set_xlabel('response level in the window / median')
    fig.suptitle(f"{label}: one model that depends on level, against the "
                 f"modes fitted window by window", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(only=None):
    cva_plotting.set_show(False)
    OUT.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    res = {}
    for label in RECORDS:
        if only and label not in only:
            continue
        print(f"sweeping {label} ...", flush=True)
        e = evaluate(label)
        if e is None:
            continue
        res[label] = e
        with contextlib.redirect_stdout(text):
            table(label, e)
        plot(label, e, OUT / f"global_{slug(label)}.png")
    (OUT / 'tables.txt').write_text(text.getvalue())
    print(text.getvalue())
    return res


if __name__ == '__main__':
    main(only=sys.argv[1:] or None)
