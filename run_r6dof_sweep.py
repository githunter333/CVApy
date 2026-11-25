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
  * one state or two;
  * against time, one graph per mode: frequency and damping
    (sweep_time_<record>.png) and the mode's rebuilt acceleration
    (sweep_modal_<record>.png); level, order and every pole are on
    sweep_time_detail_<record>.png.

A mode is tracked when its pole is in PERSISTENCE (20%) of the windows.

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
PERSISTENCE = 0.2            # a mode is tracked if its pole is in this
                             # share of the windows (tracker default 0.5)
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
        cl = CVAPhiCluster(run, f_max=F_MAX,
                           min_persistence=PERSISTENCE)
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


def plot_time_detail(label, res, base, save_path, pole_mode='time'):
    """Per record, against time: response level, the order each window
    supports, every pole found, and each tracked mode's frequency and
    damping for the three neighbour choices."""
    import matplotlib.pyplot as plt
    # modes tracked with any neighbour choice, merged within 3%
    refs = []
    for mode in ('level', 'time', 'shape'):
        for f0 in res[mode]['cl'].ref_freq:
            if not any(abs(f0 - r) < 0.03 * r for r in refs):
                refs.append(float(f0))
    refs = np.sort(refs)
    nm = len(refs)
    fig = plt.figure(figsize=(15, 9.5 + 2.6 * nm))
    gs = fig.add_gridspec(3 + nm, 2,
                          height_ratios=[0.8, 0.8, 2.2] + [1] * nm)

    def times(r):
        return np.array([r['run'].windows[w].time for w in r['cl'].use_idx])

    ax_l = fig.add_subplot(gs[0, :])
    r = res[pole_mode]
    t = times(r)
    ax_l.semilogy(t, r['lev'] / np.median(r['lev']), color='0.25', lw=0.9,
                  label='response level in each window / median')
    ax_l.set_ylabel('level / median')
    ax_l.set_title('Response level', fontsize=10, loc='left')
    ax_l.legend(fontsize=8, loc='upper right')
    ax_l.grid(True, alpha=0.3, which='both')

    ax_o = fig.add_subplot(gs[1, :], sharex=ax_l)
    for mode in MODES:
        rr = res[mode]
        o = np.array([rr['run'].windows[w].rank_auto
                      for w in rr['cl'].use_idx])
        ax_o.plot(times(rr), o, '-', color=COLORS[mode], lw=0.8, alpha=0.85,
                  label=f"neighbours by {mode}: median {np.median(o):.0f}")
    if base is not None:
        ob = base[1]['level']['run'].order_report(verbose=False)['order']
        ax_o.axhline(ob, color='k', lw=1.0,
                     label=f"{base[0]} record: {ob:.0f}")
    ax_o.set_ylabel('states')
    ax_o.set_title('Model order each window supports', fontsize=10,
                   loc='left')
    ax_o.legend(fontsize=8, ncol=4, loc='upper right')
    ax_o.grid(True, alpha=0.3)

    ax_p = fig.add_subplot(gs[2, :], sharex=ax_l)
    cl = r['cl']
    tt, ff, zz = [], [], []
    for ti, wi in zip(t, cl.use_idx):
        _, fr, ze = cl._window_poles(wi)
        tt.append(np.full(len(fr), ti)); ff.append(fr); zz.append(ze)
    tt, ff, zz = map(np.concatenate, (tt, ff, zz))
    sc = ax_p.scatter(tt, ff, c=100 * zz, s=5, cmap='viridis_r',
                      vmin=0, vmax=10, label='one pole in one window')
    fig.colorbar(sc, ax=ax_p, pad=0.01, label='damping, % of critical')
    for j, f0 in enumerate(refs):
        ax_p.axhline(f0, color='tab:red', lw=0.6, alpha=0.6,
                     label='tracked modes (panels below)' if j == 0 else None)
    if base is not None:
        for j, f0 in enumerate(base[1]['level']['cl'].ref_freq):
            ax_p.axhline(f0, color='k', lw=0.6, ls='--', alpha=0.6,
                         label=f"modes tracked in the {base[0]} record"
                         if j == 0 else None)
    ax_p.set_ylim(0, F_MAX)
    ax_p.set_ylabel('Hz')
    ax_p.set_title(f'Every pole found, window by window '
                   f'(neighbours by {pole_mode})', fontsize=10, loc='left')
    ax_p.legend(fontsize=8, ncol=3, loc='upper right')
    ax_p.grid(True, alpha=0.3)

    for j, f0 in enumerate(refs):
        ax_f = fig.add_subplot(gs[3 + j, 0], sharex=ax_l)
        ax_z = fig.add_subplot(gs[3 + j, 1], sharex=ax_l)
        for mode in MODES:
            rr = res[mode]
            c = rr['cl']
            if len(c.ref_freq) == 0:
                continue
            k = int(np.argmin(np.abs(c.ref_freq - f0)))
            if abs(c.ref_freq[k] - f0) > 0.03 * f0:
                continue
            f, z = c.track_freq[:, k], 100 * c.track_zeta[:, k]
            found = 100 * np.isfinite(f).mean()
            ax_f.plot(times(rr), f, '.', color=COLORS[mode], ms=3,
                      alpha=0.7,
                      label=f"by {mode}: {np.nanmedian(f):.2f} "
                            f"± {np.nanstd(f):.3g} Hz, in {found:.0f}%")
            ax_z.plot(times(rr), z, '.', color=COLORS[mode], ms=3,
                      alpha=0.7,
                      label=f"by {mode}: {np.nanmedian(z):.2f} "
                            f"± {np.nanstd(z):.2g} %")
        if base is not None:
            bc = base[1]['level']['cl']
            k = int(np.argmin(np.abs(bc.ref_freq - f0)))
            if abs(bc.ref_freq[k] - f0) < 0.03 * f0:
                ax_f.axhline(bc.ref_freq[k], color='k', lw=1.0,
                             label=f"{base[0]} record: "
                                   f"{bc.ref_freq[k]:.2f} Hz")
                zb = 100 * np.nanmedian(bc.track_zeta[:, k])
                ax_z.axhline(zb, color='k', lw=1.0,
                             label=f"{base[0]} record: {zb:.2f} %")
        ax_f.ticklabel_format(axis='y', useOffset=False)
        ax_f.set_title(f"mode near {f0:.1f} Hz: frequency", fontsize=10,
                       loc='left')
        ax_z.set_title(f"mode near {f0:.1f} Hz: damping", fontsize=10,
                       loc='left')
        ax_f.set_ylabel('Hz')
        ax_z.set_ylabel('% of critical')
        for a in (ax_f, ax_z):
            a.legend(fontsize=7, ncol=2, loc='upper right')
            a.grid(True, alpha=0.3)
            if j == nm - 1:
                a.set_xlabel('time, s')
    if nm == 0:
        ax_p.set_xlabel('time, s')
    fig.suptitle(f"{label}: modes against time, sweep with the input "
                 f"history in each fit", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_time(label, res, base, save_path, mode='time'):
    """Per record, one graph for each tracked mode: frequency (left axis,
    Hz) and damping (right axis, % of critical) against time in seconds."""
    import matplotlib.pyplot as plt
    r = res[mode]
    cl = r['cl']
    nm = len(cl.ref_freq)
    if nm == 0:
        return
    t = np.array([r['run'].windows[w].time for w in cl.use_idx])
    bc = None if base is None else base[1][mode]['cl']
    fig, axes = plt.subplots(nm, 1, figsize=(13, 2.7 * nm + 1.0),
                             sharex=True, squeeze=False)

    def limits(v, ref=None):
        lo, hi = np.nanpercentile(v, [1, 99])
        if ref is not None:
            lo, hi = min(lo, ref), max(hi, ref)
        pad = 0.15 * (hi - lo) + 1e-9 + 1e-4 * abs(hi)
        return lo - pad, hi + pad

    for j, ax in enumerate(axes[:, 0]):
        f, z = cl.track_freq[:, j], 100 * cl.track_zeta[:, j]
        f0 = cl.ref_freq[j]
        fb = zb = None
        if bc is not None and len(bc.ref_freq):
            k = int(np.argmin(np.abs(bc.ref_freq - f0)))
            if abs(bc.ref_freq[k] - f0) < 0.03 * f0:
                fb = bc.ref_freq[k]
                zb = 100 * np.nanmedian(bc.track_zeta[:, k])
        cf, cz = 'tab:blue', 'tab:red'
        hf, = ax.plot(t, f, '.-', color=cf, ms=3, lw=0.6,
                      label=f"frequency: {np.nanmedian(f):.2f} "
                            f"± {np.nanstd(f):.3g} Hz")
        ax.set_ylim(*limits(f, fb))
        ax.set_ylabel('frequency, Hz', color=cf)
        ax.tick_params(axis='y', colors=cf)
        ax.ticklabel_format(axis='y', useOffset=False)
        az = ax.twinx()
        hz, = az.plot(t, z, '.-', color=cz, ms=3, lw=0.6, alpha=0.8,
                      label=f"damping: {np.nanmedian(z):.2f} "
                            f"± {np.nanstd(z):.2g} %")
        lo, hi = limits(z, zb)
        az.set_ylim(min(lo, 0), hi)
        az.set_ylabel('damping, % of critical', color=cz)
        az.tick_params(axis='y', colors=cz)
        hs = [hf, hz]
        if fb is not None:
            hs.append(ax.axhline(fb, color=cf, ls='--', lw=1.0,
                                 label=f"{base[0]} record frequency: "
                                       f"{fb:.2f} Hz"))
            hs.append(az.axhline(zb, color=cz, ls='--', lw=1.0,
                                 label=f"{base[0]} record damping: "
                                       f"{zb:.2f} %"))
        ax.legend(handles=hs, fontsize=8, ncol=4, loc='upper right')
        ax.set_title(f"mode near {f0:.1f} Hz, found in "
                     f"{100 * np.isfinite(f).mean():.0f}% of windows "
                     f"(gaps: not found)", fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('time, s')
    fig.suptitle(f"{label}: frequency and damping of each mode against "
                 f"time (neighbours by {mode})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_modal(label, res, save_path, mode='time'):
    """Per record, one graph for each tracked mode: the mode's rebuilt
    acceleration against time, at the mass where the mode is largest."""
    import matplotlib.pyplot as plt
    cl = res[mode]['cl']
    nm = len(cl.ref_freq)
    if nm == 0:
        return None
    h = cl.modal_history()
    t, m, y = h['time'], h['modal'], h['measured']
    ok = np.isfinite(h['fitted'][:, 0])
    total = np.nansum(m, axis=1)
    share = 100 * (1 - ((y - total)[ok] ** 2).sum() / (y[ok] ** 2).sum())
    fig, axes = plt.subplots(nm, 1, figsize=(13, 2.5 * nm + 1.0),
                             sharex=True, squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        rms = np.sqrt(np.nanmean(m[:, j] ** 2, axis=0))
        c = int(np.argmax(rms))
        found = 100 * np.isfinite(m[:, j, c]).mean()
        ax.plot(t, m[:, j, c], color='tab:blue', lw=0.3, rasterized=True,
                label=f"rebuilt acceleration of this mode at mass {c + 1}: "
                      f"rms {rms[c]:.3g} "
                      f"(measured at mass {c + 1}: rms {y[:, c].std():.3g})")
        ax.set_ylabel('acceleration')
        ax.set_title(f"mode near {cl.ref_freq[j]:.1f} Hz at mass {c + 1}, "
                     f"where it is largest; found {found:.0f}% of the time "
                     f"(gaps: not found)", fontsize=10, loc='left')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('time, s')
    fig.suptitle(f"{label}: rebuilt modal acceleration against time "
                 f"(neighbours by {mode}). The tracked modes together "
                 f"give {share:.1f}% of the measured response",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return share


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
        plot_time(label, res[label], base,
                  OUT / f"sweep_time_{slug(label)}.png")
        plot_time_detail(label, res[label], base,
                         OUT / f"sweep_time_detail_{slug(label)}.png")
        plot_modal(label, res[label], OUT / f"sweep_modal_{slug(label)}.png")
    plot_orders(orders, OUT / 'sweep_orders.png')
    (OUT / 'tables.txt').write_text(text.getvalue())
    print(text.getvalue())
    return res, orders


if __name__ == '__main__':
    main(only=sys.argv[1:] or None)
