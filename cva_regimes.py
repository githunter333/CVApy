"""
cva_regimes.py
==============
Does the system switch between distinct states, or does it only look that
way because the response level wanders?

Under random excitation the response envelope rises and falls over many
cycles. A nonlinearity that depends on level (free play, a cubic spring)
then behaves differently in the loud and quiet stretches, and the record
shows "regions of different character" lasting a second or more. That is
one state of the system seen at different levels, not two attractors.

Coexisting attractors mean something stronger: at the SAME response level
the system can be in either of two states, and it stays in one for a while
before changing. The test follows from that.

  1. Take block-by-block features of the system: the frequency of each
     tracked in-band mode (cva_level.track_in_band) and, if supplied, the
     level of the non-coherent response (cva_residual).
  2. Fit each feature as a smooth function of response level and remove
     it. How much that explains is the share of the "regime" behaviour
     that is simply level.
  3. Look at what is left. Two states show as two separate clusters, with a
     gap between them, a difference large enough to matter, and dwell in
     each lasting a second or more. One state shows as one cluster.

    trk = track_in_band(time, inputd, datout, fs, band_top)
    rr = noncoherent_history(inputd, datout, fs)        # optional
    rg = analyze_regimes(trk, rr, label='0 dB')
    rg.summary()
    fig = plot_regimes(rg)

Mode frequencies carry the test because they are measured far more
precisely than any level: on a linear record they repeat to a few parts
per million from block to block. A requirement that the two clusters differ
by at least 0.1% keeps that numerical scatter from ever being read as two
states. A linear record, and a nonlinear record with one attractor, must
both come out as one state; a simulated joint that changes state on its own
must come out as two. All three are in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from cva_plotting import show_if_enabled


@dataclass
class Feature:
    name:   str
    kind:   str              # 'frequency' or 'level'
    value:  np.ndarray       # (nb,) Hz, or log10 level
    fit:    np.ndarray       # predicted from response level
    rest:   np.ndarray       # value - fit (relative for frequency)
    r2_level: float          # share of its variation explained by level
    slope:  float            # change per doubling of level (% for frequency,
                             # exponent for level)
    separation: float        # Ashman's D of a two-cluster fit to `rest`
    gap:    float            # density in the valley / lower peak (1 = none)
    distance: float          # cluster centres apart: fraction of frequency,
                             # or log10 units for level
    minor_weight: float
    dwell_sec: float         # mean run length of the raw cluster labels
    two:    bool
    labels: np.ndarray       # (nb,) raw cluster labels, 1 = higher cluster


@dataclass
class RegimeResult:
    label:    str
    block_sec: float
    times:    np.ndarray
    x:        np.ndarray           # log10 response level per block
    features: List[Feature] = field(default_factory=list)
    two_states: bool = False
    by:       Optional[int] = None  # index of the deciding feature
    labels:   np.ndarray = None     # (nb,) 0/1
    state_names: tuple = ('quiet', 'loud')
    freq:     np.ndarray = None     # (nb, n_modes) for the per-state table
    zeta:     np.ndarray = None
    env_corr_sec: float = 0.0
    verdict:  str = ''

    def state_table(self):
        """Per state: share of time, mean level, and each mode's mean
        frequency and damping."""
        rows = []
        for s in (0, 1):
            m = self.labels == s
            rows.append(dict(state=self.state_names[s], share=float(m.mean()),
                             level=float(10 ** self.x[m].mean()) if m.any() else np.nan,
                             freq=np.nanmean(self.freq[m], axis=0) if m.any() else None,
                             zeta=np.nanmean(self.zeta[m], axis=0) if m.any() else None))
        return rows

    def summary(self, verbose: bool = True):
        if verbose:
            print(f"\n{self.label}: regimes ({self.block_sec:g} s blocks, "
                  f"level stays correlated for {self.env_corr_sec:.1f} s)")
            print(f"  {'feature':<22} {'level explains':>14} "
                  f"{'per doubling':>13} {'separation':>11} {'gap':>5} "
                  f"{'apart':>9} {'dwell s':>8}")
            for f in self.features:
                unit = '%' if f.kind == 'frequency' else ' (exponent)'
                apart = (f"{100*f.distance:.3f}%" if f.kind == 'frequency'
                         else f"x{10**f.distance:.1f}")
                print(f"  {f.name:<22} {100*f.r2_level:13.0f}% "
                      f"{f.slope:+11.3f}{unit[:1]:1} {f.separation:11.2f} "
                      f"{f.gap:5.2f} {apart:>9} {f.dwell_sec:8.2f}"
                      f"{'   <- two clusters' if f.two else ''}")
            print(f"  -> {self.verdict}")
            for r in self.state_table():
                if r['freq'] is None:
                    continue
                fz = ', '.join(f"{a:.2f} Hz/{100*b:.2f}%"
                               for a, b in zip(r['freq'], r['zeta']))
                print(f"     {r['state']:<18} {100*r['share']:3.0f}% of the "
                      f"time, level {r['level']:.3g}: {fz}")
        return dict(two_states=self.two_states, verdict=self.verdict)


# ---------------------------------------------------------------------------
def _corr_time(v, dt):
    """Area under the autocorrelation up to its first zero crossing."""
    v = v[np.isfinite(v)]
    v = v - v.mean()
    if len(v) < 4 or v.std() == 0:
        return 0.0
    n = len(v)
    ac = np.correlate(v, v, mode='full')[n - 1:] / (v.var() * n)
    neg = np.flatnonzero(ac <= 0)
    stop = neg[0] if len(neg) else n
    return float(dt * (0.5 + ac[1:stop].sum()))


def _two_clusters(v):
    """Two-Gaussian fit to a 1-D sample: separation (Ashman's D), centre
    distance, weight of the smaller cluster, labels (1 = higher)."""
    from sklearn.mixture import GaussianMixture
    V = v.reshape(-1, 1)
    scale = V.std() or 1.0
    g2 = GaussianMixture(2, n_init=5, random_state=0).fit(V / scale)
    mu = g2.means_.ravel() * scale
    sd = np.sqrt(g2.covariances_.ravel()) * scale
    D = np.sqrt(2.0) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
    lab = g2.predict(V / scale)
    if mu[0] > mu[1]:
        lab = 1 - lab
    return float(D), float(abs(mu[0] - mu[1])), float(g2.weights_.min()), lab


def _gap(v):
    """Depth of the valley between the two highest peaks of the density of
    v, as a fraction of the lower peak. 1.0 when there is only one peak."""
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    if v.std() == 0:
        return 1.0
    g = np.linspace(v.min(), v.max(), 256)
    d = gaussian_kde(v)(g)
    pk, _ = find_peaks(d, prominence=0.02 * d.max())
    if len(pk) < 2:
        return 1.0
    a, b = sorted(pk[np.argsort(d[pk])[::-1][:2]])
    return float(d[a:b].min() / min(d[a], d[b]))


def _mean_dwell(lab, dt):
    changes = np.flatnonzero(np.diff(lab) != 0)
    return float(dt * len(lab) / (len(changes) + 1))


def _min_dwell(lab, n_min):
    """Merge runs shorter than n_min blocks into the preceding state."""
    lab = lab.copy()
    i, n = 0, len(lab)
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if j - i < n_min and i > 0:
            lab[i:j] = lab[i - 1]
        i = j
    return lab


def _feature(name, kind, x, v, dt, min_dwell_sec, min_distance):
    ok = np.isfinite(v) & np.isfinite(x)
    fit = np.full(len(v), np.nan)
    rest = np.full(len(v), np.nan)
    coef = np.polyfit(x[ok], v[ok], 2)
    fit[ok] = np.polyval(coef, x[ok])
    ref = np.nanmedian(v) if kind == 'frequency' else 1.0
    rest[ok] = (v[ok] - fit[ok]) / ref
    var = np.var(v[ok])
    r2 = 1.0 - np.var(v[ok] - fit[ok]) / var if var > 0 else 0.0
    slope = float(np.polyval(np.polyder(coef), np.median(x[ok])) * np.log10(2))
    slope = 100 * slope / ref if kind == 'frequency' else slope / np.log10(2)
    D, dist, w, lab_ok = _two_clusters(rest[ok])
    gap = _gap(rest[ok])
    lab = np.zeros(len(v), int)
    lab[ok] = lab_ok
    for i in np.flatnonzero(~ok):                 # carry the last state over
        lab[i] = lab[i - 1] if i else 0
    dwell = _mean_dwell(lab, dt)
    two = bool(D >= 3.0 and gap <= 0.5 and w >= 0.15
               and dist >= min_distance and dwell >= min_dwell_sec)
    return Feature(name=name, kind=kind, value=v, fit=fit, rest=rest,
                   r2_level=float(max(r2, 0.0)), slope=slope, separation=D,
                   gap=gap, distance=dist, minor_weight=w, dwell_sec=dwell,
                   two=two, labels=lab)


def analyze_regimes(trk, rr=None, label: str = '',
                    min_dwell_sec: float = 1.0,
                    min_freq_change: float = 1e-3,
                    min_level_factor: float = 2.0) -> RegimeResult:
    """
    trk : BlockTracker from cva_level.track_in_band (in-band mode tracks).
    rr  : optional ResidualResult; adds the non-coherent level as a feature
          (above the excitation band if there is energy there).

    Two states are declared when any feature's left-over part forms two
    clusters at least three widths apart, with a clear gap between them,
    each holding 15% of the blocks or more, differing by min_freq_change
    (as a fraction of frequency) or min_level_factor, and with a mean dwell
    of min_dwell_sec or more.
    """
    t = trk.times
    dt = float(np.median(np.diff(t))) if len(t) > 1 else trk.cfg.block_sec
    x = np.log10(np.sqrt((trk.level ** 2).sum(axis=1)))
    feats = []
    for j in range(trk.freq.shape[1]):
        feats.append(_feature(f"mode {trk.ref_freq[j]:.0f} Hz frequency",
                              'frequency', x, trk.freq[:, j], dt,
                              min_dwell_sec, min_freq_change))
    if rr is not None:
        oob = rr.out_of_band()
        use_oob = oob is not None and rr.has_out_of_band.any()
        tb, _, non = rr.envelopes(0.25, oob if use_oob else None)
        y = np.log10(np.sqrt((non ** 2).sum(axis=1)) + 1e-300)
        k = max(1, int(round(dt / 0.25)))
        y = np.convolve(y, np.ones(k) / k, mode='same')
        feats.append(_feature('non-coherent level' + (' (out of band)' if use_oob else ''),
                              'level', x, np.interp(t, tb, y), dt,
                              min_dwell_sec, np.log10(min_level_factor)))

    n_min = max(1, int(round(min_dwell_sec / dt)))
    deciding = [i for i, f in enumerate(feats) if f.two]
    if deciding:
        by = max(deciding, key=lambda i: feats[i].separation)
        f = feats[by]
        labels = _min_dwell(f.labels, n_min)
        apart = (f"{100*f.distance:.2f}%" if f.kind == 'frequency'
                 else f"a factor of {10**f.distance:.1f}")
        verdict = (f"two states at the same response level: {f.name} sits "
                   f"in two clusters {apart} apart, mean dwell "
                   f"{f.dwell_sec:.1f} s; consistent with switching between "
                   f"attractors")
        names = ('state A', 'state B')
    else:
        by = None
        labels = (x > np.median(x)).astype(int)
        lead = max(feats, key=lambda f: f.r2_level)
        if lead.r2_level >= 0.5:
            verdict = (f"one state; {lead.name} follows the response level "
                       f"({100*lead.r2_level:.0f}% explained), so loud and "
                       f"quiet stretches of the record look different")
        else:
            verdict = ("one state; nothing beyond ordinary level "
                       "fluctuation")
        names = ('quiet half of blocks', 'loud half of blocks')
    return RegimeResult(label=label, block_sec=trk.cfg.block_sec, times=t,
                        x=x, features=feats, two_states=bool(deciding),
                        by=by, labels=labels, state_names=names,
                        freq=trk.freq, zeta=trk.zeta,
                        env_corr_sec=_corr_time(x, dt), verdict=verdict)


# ---------------------------------------------------------------------------
def plot_regimes(rg: RegimeResult, feature: Optional[int] = None,
                 save_path=None, dpi: int = 130):
    """For one feature (the deciding one, or the one level explains best):
       1. the feature against response level, block by block;
       2. what is left after removing the level effect: one cluster or two;
       3. level and the feature along the record, shaded by state.
       4. every feature's separation, so none is hidden."""
    import matplotlib.pyplot as plt
    if feature is None:
        feature = (rg.by if rg.by is not None else
                   int(np.argmax([f.r2_level for f in rg.features])))
    f = rg.features[feature]
    c0, c1 = 'tab:blue', 'tab:red'
    unit = 'Hz' if f.kind == 'frequency' else 'log10'
    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 0.75])

    ax = fig.add_subplot(gs[0, 0])
    for s, col in ((0, c0), (1, c1)):
        m = rg.labels == s
        ax.plot(rg.x[m], f.value[m], '.', color=col, ms=5, alpha=0.7,
                label=f"{rg.state_names[s]} ({100*m.mean():.0f}% of blocks)")
    o = np.argsort(rg.x)
    ax.plot(rg.x[o], f.fit[o], 'k-', lw=1.6,
            label=f"what level alone predicts "
                  f"({100*f.r2_level:.0f}% of the variation)")
    ax.set_xlabel('log10 response level in the block')
    ax.set_ylabel(f"{f.name}, {unit}")
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.set_title(f"1. {f.name} against response level, one dot per block",
                 fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    r = f.rest[np.isfinite(f.rest)]
    scale = 100.0 if f.kind == 'frequency' else 1.0
    bins = np.linspace(r.min(), r.max(), 40) * scale
    if rg.two_states:
        for s, col in ((0, c0), (1, c1)):
            ax.hist(scale * f.rest[rg.labels == s], bins=bins, color=col,
                    alpha=0.7, label=rg.state_names[s])
    else:
        ax.hist(scale * r, bins=bins, color='0.5', label='all blocks')
    ax.set_xlabel(f"{f.name} minus what level predicts "
                  f"({'% of frequency' if f.kind == 'frequency' else 'log10'})")
    ax.set_ylabel('blocks')
    ax.set_title(f"2. left over after removing the level effect: "
                 f"separation {f.separation:.1f} (needs 3), gap "
                 f"{f.gap:.2f} (needs 0.5 or less)", fontsize=10, loc='left')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(rg.times, rg.x, color='0.35', lw=1.1,
            label='response level, log10 (left axis)')
    ax.set_ylabel('log10 level')
    ax2 = ax.twinx()
    ax2.plot(rg.times, f.value, color='tab:orange', lw=1.1,
             label=f"{f.name} (right axis)")
    ax2.set_ylabel(unit)
    ax2.ticklabel_format(axis='y', useOffset=False)
    ax.fill_between(rg.times, 0, 1, where=rg.labels == 1, color=c1,
                    alpha=0.12, step='mid', transform=ax.get_xaxis_transform(),
                    label=f"shaded: {rg.state_names[1]}")
    ax.set_xlabel('s')
    ax.set_title(f"3. along the record (the level itself stays correlated "
                 f"for {rg.env_corr_sec:.1f} s)", fontsize=10, loc='left')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right',
              bbox_to_anchor=(1, 1.0), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    k = np.arange(len(rg.features))
    ax.bar(k, [g.separation for g in rg.features], 0.55,
           color=['tab:red' if g.two else '0.6' for g in rg.features])
    ax.axhline(3.0, color='k', lw=1.0, ls='--')
    ax.set_xticks(k, [g.name.replace(' frequency', '') for g in rg.features],
                  fontsize=8)
    ax.set_ylabel('cluster separation')
    ax.set_title('4. every feature tested (red = two clusters that pass '
                 'all conditions; dashed line = separation needed)',
                 fontsize=10, loc='left')
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f"{rg.label}: {rg.verdict}", fontsize=11.5, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(0.5, 0.005,
             "Two attractors would show as two separate clouds at the same "
             "response level in panel 1 and two humps in panel 2.   One "
             "cloud along a curve means one state seen at different levels.",
             ha='center', va='bottom', fontsize=9, color='0.25')
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    show_if_enabled()
    return fig
