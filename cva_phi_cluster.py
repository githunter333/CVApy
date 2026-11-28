"""
cva_phi_cluster.py
==================
Group the windows of a CVARun into dynamical regimes.

Each window's Phi is expressed in that window's own state basis, so the
entries of two Phi matrices cannot be compared directly: the same dynamics
in two bases give two different matrices. Clustering the entries therefore
splits a linear record into several "regimes" that are only differences of
basis and estimation noise.

The regimes are found here from quantities that do not depend on the basis:
the frequencies of the modes that persist from window to window.

  1. Reference modes: frequencies at which most windows have a pole.
  2. Each window's poles are matched to them, giving one frequency (and
     damping) track per mode.
  3. Each track is tested for two separate clusters after removing its
     smooth dependence on response level (the test of cva_regimes). The
     answer may be ONE: a linear record, or a record whose modes simply
     follow the response level, is one state.

fit(n_clusters=k) with k > 0 still forces k groups (k-means on the tracks).

The attribute names of the earlier Phi-entry version are kept so files and
callers still work: Phi_vec now holds the frequency tracks as percent
deviation from each mode's median, and Phi_pca their principal components.

Author: Norman / Claude
Date:   2026-04-14, regime test 2026-11-24
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from cva_plotting import show_if_enabled

from cva_run import CVARun


def _runs(flag):
    """Lengths of the unbroken runs of True in a boolean series."""
    b = np.concatenate([[0], np.asarray(flag, int), [0]])
    edges = np.flatnonzero(np.diff(b))
    return edges[1::2] - edges[::2]


def content_decision(order, presence, log_level=None, min_frac=0.15,
                     run_factor=2.5, valley=0.5):
    """
    One condition or two, from what the local models contain.

    order    : (n,) number of states each window supports.
    presence : (n, n_modes) bool, whether each tracked mode has a pole in
               each window.
    log_level: (n,) log response level, used only to name the conditions.

    Two things a linear system keeps fixed under random excitation are used:
    the number of states and the set of modes present. (How the response is
    shared between the modes is not used; that moves with the excitation
    even in a linear system.)

    Two conditions are reported when the order is two-valued: two groups of
    at least min_frac of the windows each, centres apart by at least 2
    states and 25%, with few windows in between (the count midway is at
    most `valley` of the smaller group's peak).

    Modes that come and go in stretches are found as well: present in
    min_frac to 1 - min_frac of the windows, in runs at least run_factor
    times as long as chance would give. They help decide which condition a
    window is in, and are named in the verdict, but on their own (with no
    change of order) they are reported as a note, not as a second
    condition; a single weak mode drifting in and out of its track is too
    little to go on.

    Returns dict: n_conditions (1 or 2), labels (n,) 0/1, order_two_valued,
    sustained (indices of the modes that come and go), conditions (list of
    dicts: name, share, order, level, present), level_corr, verdict.
    """
    order = np.asarray(order, float)
    P = np.asarray(presence, bool).reshape(len(order), -1)
    n = len(order)
    out = dict(n_conditions=1, labels=np.zeros(n, int),
               order_two_valued=False, sustained=[], conditions=[],
               level_corr=np.nan, verdict='one condition')
    if n < 20:
        out['verdict'] = 'one condition (too few windows to test)'
        return out

    # --- order: two groups with a gap between them ---
    lo, hi = np.percentile(order, [10, 90])
    cut = None
    if hi - lo >= max(2.0, 0.25 * max(lo, 1.0)):
        vals = np.unique(order)
        best = None
        for c in 0.5 * (vals[:-1] + vals[1:]):          # every split point
            a, b = order[order < c], order[order > c]
            if min(len(a), len(b)) < min_frac * n:
                continue
            score = len(a) * a.var() + len(b) * b.var()
            if best is None or score < best[0]:
                best = (score, c, a, b)
        if best is not None:
            _, c, a, b = best
            ca, cb = np.median(a), np.median(b)
            width = max(1.0, (cb - ca) / 6.0)
            count = lambda x: np.sum(np.abs(order - x) <= width)
            mid = count(0.5 * (ca + cb))
            if (cb - ca >= max(2.0, 0.25 * max(ca, 1.0))
                    and mid <= valley * min(count(ca), count(cb))):
                cut = c
    out['order_two_valued'] = cut is not None

    # --- modes that come and go in stretches ---
    sustained = []
    for j in range(P.shape[1]):
        f = P[:, j].mean()
        if not (min_frac <= f <= 1 - min_frac):
            continue
        on, off = _runs(P[:, j]), _runs(~P[:, j])
        if (on.mean() >= run_factor / (1 - f)
                and off.mean() >= run_factor / f):
            sustained.append(j)
    out['sustained'] = sustained

    if cut is None:
        notes = []
        if hi - lo >= max(2.0, 0.25 * max(lo, 1.0)):
            notes.append(f'the order varies between {lo:.0f} and {hi:.0f} '
                         f'without two distinct groups')
        if sustained:
            notes.append(f"{len(sustained)} mode"
                         f"{'s come' if len(sustained) > 1 else ' comes'} "
                         f"and go{'' if len(sustained) > 1 else 'es'} in "
                         f"stretches")
        if notes:
            out['verdict'] = 'one condition; ' + '; '.join(notes)
        return out

    # --- label the windows ---
    cols = []
    if cut is not None:
        cols.append((order > cut).astype(float))
    for j in sustained:
        cols.append(P[:, j].astype(float))
    Z = np.column_stack(cols)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    if Z.shape[1] == 1:
        lab = (Z[:, 0] > 0).astype(int)
    else:
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        lab = ((Z @ vt[0]) > 0).astype(int)
    if min(lab.mean(), 1 - lab.mean()) < min_frac:
        return out
    if log_level is not None:
        ll = np.asarray(log_level, float)
        if ll[lab == 1].mean() < ll[lab == 0].mean():
            lab = 1 - lab                         # 0 = quieter, 1 = louder
        out['level_corr'] = float(np.corrcoef(lab, ll)[0, 1])
    by_level = abs(out['level_corr']) >= 0.3
    names = ('quieter', 'louder') if by_level else ('A', 'B')
    for k in (0, 1):
        m = lab == k
        out['conditions'].append(dict(
            name=names[k], share=float(m.mean()),
            order=float(np.median(order[m])),
            level=(float(np.exp(np.mean(ll[m]))) if log_level is not None
                   else np.nan),
            present=P[m].mean(axis=0)))
    out['n_conditions'], out['labels'] = 2, lab
    c0, c1 = out['conditions']
    out['verdict'] = (
        f"two conditions: {c0['name']} ({100*c0['share']:.0f}% of the time, "
        f"order {c0['order']:.0f}) and {c1['name']} "
        f"({100*c1['share']:.0f}%, order {c1['order']:.0f})"
        + (', set by the response level' if by_level else ''))
    return out


def _get_physical(mean_ev, dt, fs, min_im_frac=0.01, f_min=None, f_max=None):
    ce      = np.log(mean_ev + 1e-300) / dt
    omega_d = np.abs(np.imag(ce))
    sigma   = -np.real(ce)
    omega_n = np.sqrt(omega_d**2 + sigma**2)
    freq    = omega_n / (2*np.pi)
    zeta    = sigma / (omega_n + 1e-12)
    mag     = np.abs(mean_ev)
    good = ((np.abs(np.imag(mean_ev)) > min_im_frac * mag) &
            (freq > 0.5) & (freq < fs/2) & (mag <= 1.0) & (np.imag(mean_ev) > 0))
    if f_min is not None:
        good &= (freq >= f_min)
    if f_max is not None:
        good &= (freq <= f_max)
    return freq[good], zeta[good], mag[good]


class CVAPhiCluster:
    def __init__(self, run, k_max=8, n_pca_var=95.0,
                 kmeans_rep=20, kmeans_iter=500, min_im_frac=0.01,
                 f_min=None, f_max=None, zeta_max=0.2, match_tol=0.03,
                 min_persistence=0.5, min_freq_change=1e-3,
                 min_dwell_sec=0.0):
        """
        zeta_max        : poles more heavily damped are not tracked.
        match_tol       : a window's pole belongs to a mode if within this
                          fraction of the mode's frequency.
        min_persistence : a mode must be found in this share of the windows.
        min_freq_change : two clusters must differ by at least this fraction
                          of the frequency to count as two states.
        min_dwell_sec   : and each must last this long on average (0 = not
                          required; the sweep's windows are fitted from
                          neighbours all over the record, so dwell means
                          less here than for time blocks).
        k_max           : kept for callers; not used by the regime test.
        """
        self.run         = run
        self.k_max       = k_max
        self.n_pca_var   = n_pca_var
        self.kmeans_rep  = kmeans_rep
        self.kmeans_iter = kmeans_iter
        self.min_im_frac = min_im_frac
        self.f_min       = f_min   # Optional lower frequency bound (Hz)
        self.f_max       = f_max   # Optional upper frequency bound (Hz)
        self.zeta_max        = zeta_max
        self.match_tol       = match_tol
        self.min_persistence = min_persistence
        self.min_freq_change = min_freq_change
        self.min_dwell_sec   = min_dwell_sec

        self.rk_mode:       int   = 0
        self.use_idx:       np.ndarray = np.array([], dtype=int)
        self.Phi_vec:       Optional[np.ndarray] = None
        self.Phi_pca:       Optional[np.ndarray] = None
        self.pca_model:     Optional[PCA]         = None
        self.inertia:       Optional[np.ndarray] = None
        self.cluster_ids:   Optional[np.ndarray] = None
        self.centroids_pca: Optional[np.ndarray] = None
        self.n_clusters:    int = 0
        self.regime_A:      int = 0
        self.regime_B:      int = 1
        self.minority:      List[int] = []
        self.centroid_Phis:   List[np.ndarray] = []
        self.cluster_mean_ev: List[np.ndarray] = []
        self.cluster_times:   Optional[np.ndarray] = None
        self._cum_var:  Optional[np.ndarray] = None
        self._n_pca:    int = 0
        self._k_elbow_d2: int = 0
        self._k_elbow_nd: int = 0

        # mode tracks and the regime test
        self.ref_freq:   Optional[np.ndarray] = None   # (n_modes,) Hz
        self.track_freq: Optional[np.ndarray] = None   # (n_use, n_modes), NaN = not found
        self.track_zeta: Optional[np.ndarray] = None
        self.track_pole: Optional[np.ndarray] = None   # complex
        self.content:    Optional[dict] = None   # content_conditions()
        self.level:      Optional[np.ndarray] = None   # (n_use,) log10 response rms
        self.sep_D:      Optional[np.ndarray] = None   # (n_modes,) cluster separation
        self.sep_gap:    Optional[np.ndarray] = None
        self.sep_dist:   Optional[np.ndarray] = None   # fraction of frequency
        self.sep_r2:     Optional[np.ndarray] = None   # share explained by level
        self.sep_two:    Optional[np.ndarray] = None   # bool
        self.deciding:   int = -1
        self.verdict:    str = ''
        self._auto_labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, n_clusters=0):
        self._select_windows()
        self._track_modes()
        self._features()
        self._separation_test()
        if n_clusters > 0:
            self.n_clusters = n_clusters
            print(f'Using override n_clusters = {n_clusters}')
        else:
            print(f'Using tested n_clusters = {self.n_clusters}')
        self._final_clustering(forced=n_clusters > 0)
        self._eigenvalue_analysis()
        self._frequency_shift_report()
        self.content_conditions()

    # ------------------------------------------------------------------
    # conditions: what the local models contain, window by window
    # ------------------------------------------------------------------
    def content_conditions(self, wide_tol: Optional[float] = None,
                           verbose: bool = True, **kw):
        """One condition or two, from the number of states each window
        supports and the set of modes present in it (content_decision).

        A mode counts as present when the window has a pole within wide_tol
        of its frequency (default three times match_tol), so a mode that
        has only shifted with level is not counted as gone.
        Keyword arguments go to content_decision. The result is kept in
        self.content."""
        run = self.run
        tol = wide_tol or 3.0 * self.match_tol
        wins = [run.windows[w] for w in self.use_idx]
        order = np.array([w.rank_auto or w.rank for w in wins], float)
        P = np.zeros((len(wins), len(self.ref_freq)), bool)
        for ii, wi in enumerate(self.use_idx):
            fr = self._window_poles(wi)[1]
            if len(fr):
                P[ii] = (np.abs(fr[:, None] - self.ref_freq[None, :])
                         < tol * self.ref_freq[None, :]).any(axis=0)
        lev = None
        if run.past_db is not None and run.datout is not None:
            lev = np.log(run.window_level()[[w.window_idx for w in wins]]
                         + 1e-300)
        res = content_decision(order, P, lev, **kw)
        res.update(time=np.array([w.time for w in wins]), order=order,
                   presence=P, log_level=lev)
        self.content = res
        if verbose:
            print(f"\nConditions (states and modes in each window): "
                  f"{res['verdict']}")
            if res['n_conditions'] == 2:
                head = ''.join(f'{f:8.0f}' for f in self.ref_freq)
                print(f"  {'condition':>10} {'time':>6} {'order':>6} "
                      f"{'level':>8}   % of windows with each mode (Hz)")
                print(f"  {'':>10} {'':>6} {'':>6} {'':>8} {head}")
                for c in res['conditions']:
                    row = ''.join(f'{100*v:7.0f}%' for v in c['present'])
                    print(f"  {c['name']:>10} {100*c['share']:5.0f}% "
                          f"{c['order']:6.0f} {c['level']:8.3g} {row}")
                if res['sustained']:
                    print('  modes that come and go in stretches: '
                          + ', '.join(f'{self.ref_freq[j]:.0f} Hz'
                                      for j in res['sustained']))
        return res

    def plot_conditions(self, save_path=None, dpi=150):
        """Response level, the order each window supports, and which modes
        are present, against time, coloured by condition."""
        res = getattr(self, 'content', None) or self.content_conditions(
            verbose=False)
        t, order, P, lab = (res['time'], res['order'], res['presence'],
                            res['labels'])
        two = res['n_conditions'] == 2
        colors = np.array(['tab:blue', 'tab:red'])
        names = ([c['name'] for c in res['conditions']] if two
                 else ['one condition'])
        nm = P.shape[1]
        fig, ax = plt.subplots(
            3, 1, figsize=(14, 7.5 + 0.28 * nm), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 1, 0.6 + 0.22 * nm]))
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0

        def shade(a):
            if not two:
                return
            for k in (0, 1):
                a.fill_between(t, 0, 1, where=lab == k, step='mid',
                               color=colors[k], alpha=0.13, lw=0,
                               transform=a.get_xaxis_transform(),
                               label=f"{names[k]} condition "
                                     f"({100*np.mean(lab == k):.0f}% of the "
                                     f"time)")

        if res['log_level'] is not None:
            lev = np.exp(res['log_level'])
            ax[0].semilogy(t, lev / np.median(lev), color='0.2', lw=1.0,
                           label='response level in each window / median')
        shade(ax[0])
        ax[0].set_ylabel('level / median')
        ax[0].set_title('Response level', loc='left', fontsize=10)
        ax[0].legend(fontsize=8, ncol=3, loc='upper right')
        ax[0].grid(True, alpha=0.3, which='both')

        ax[1].step(t, order, where='mid', color='0.2', lw=0.9,
                   label='states each window supports')
        shade(ax[1])
        if two:
            for k, c in enumerate(res['conditions']):
                ax[1].axhline(c['order'], color=colors[k], ls='--', lw=1.0,
                              label=f"{c['name']}: median {c['order']:.0f}")
        ax[1].set_ylabel('states')
        ax[1].set_title('Model order', loc='left', fontsize=10)
        handles, labels = ax[1].get_legend_handles_labels()
        keep = [i for i, l in enumerate(labels) if 'condition (' not in l]
        ax[1].legend([handles[i] for i in keep], [labels[i] for i in keep],
                     fontsize=8, ncol=3, loc='upper right')
        ax[1].grid(True, alpha=0.3)

        for j in range(nm):
            for k in ((0, 1) if two else (0,)):
                m = P[:, j] & (lab == k)
                ax[2].scatter(t[m], np.full(m.sum(), j), marker='|', s=90,
                              color=colors[k] if two else '0.2', lw=1.2)
        ticks = []
        for j, f0 in enumerate(self.ref_freq):
            mark = ' *' if j in res['sustained'] else ''
            ticks.append(f"{f0:.1f} Hz ({100*P[:, j].mean():.0f}%){mark}")
        ax[2].set_yticks(range(nm), ticks, fontsize=8)
        ax[2].set_ylim(-0.7, nm - 0.3)
        ax[2].set_xlabel('time, s')
        ax[2].set_title('Modes present in each window (a mark = the window '
                        'has that pole; % of windows in brackets'
                        + ('; * = comes and goes in stretches)'
                           if res['sustained'] else ')'),
                        loc='left', fontsize=10)
        ax[2].grid(True, axis='x', alpha=0.3)
        ax[2].set_xlim(t[0] - dt, t[-1] + dt)
        fig.suptitle('Conditions from model content: ' + res['verdict'],
                     fontsize=12, wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def _select_windows(self):
        ranks = np.array([w.rank for w in self.run.windows])
        valid = ranks[ranks > 0]
        if len(valid) == 0:
            raise ValueError('No valid windows')
        counts = np.bincount(valid)
        self.rk_mode = int(np.argmax(counts))
        self.use_idx = np.where(ranks == self.rk_mode)[0]
        n_use = len(self.use_idx)
        print(f'\n========================================')
        print(f'   CVA Regime Analysis (mode tracks)')
        print(f'========================================\n')
        print(f'Modal rank      : {self.rk_mode}')
        print(f'Windows at rank : {n_use} / {len(self.run.windows)}'
              f'  ({100*n_use/len(self.run.windows):.1f}%)')

    # ------------------------------------------------------------------
    def _window_poles(self, wi):
        """Oscillatory poles of one window: (pole, freq Hz, zeta)."""
        fs = self.run.fs
        ev = np.asarray(self.run.windows[wi].eigenvalues)
        mag = np.abs(ev)
        ev = ev[(np.imag(ev) > self.min_im_frac * mag) & (mag <= 1.0)
                & (mag > 0)]
        lam = np.log(ev) * fs
        freq = np.abs(lam) / (2 * np.pi)
        zeta = -np.real(lam) / (np.abs(lam) + 1e-300)
        good = (freq > 0.5) & (freq < fs / 2) & (zeta < self.zeta_max)
        if self.f_min is not None:
            good &= freq >= self.f_min
        if self.f_max is not None:
            good &= freq <= self.f_max
        return ev[good], freq[good], zeta[good]

    def _track_modes(self):
        """Reference modes and one frequency/damping track per mode."""
        n_use = len(self.use_idx)
        poles = [self._window_poles(wi) for wi in self.use_idx]
        allf = np.concatenate([p[1] for p in poles]) if poles else np.array([])
        if len(allf) == 0:
            raise ValueError('No oscillatory poles in the windows; check '
                             'the frequency band and zeta_max')
        tol = self.match_tol

        # density of pole frequencies on a log axis, bins of tol/8
        lf = np.log(allf)
        step = np.log1p(tol) / 8.0
        edges = np.arange(lf.min() - 4 * step, lf.max() + 5 * step, step)
        hist = np.histogram(lf, bins=edges)[0].astype(float)
        dens = np.convolve(hist, np.ones(5) / 5.0, mode='same')
        cent = np.exp(0.5 * (edges[:-1] + edges[1:]))
        refs = []
        for i in np.argsort(dens)[::-1]:
            if dens[i] <= 0:
                break
            f0 = cent[i]
            if any(abs(f0 - r) < 2 * tol * r for r in refs):
                continue
            hits = sum(bool(np.any(np.abs(p[1] - f0) < tol * f0))
                       for p in poles)
            if hits >= self.min_persistence * n_use:
                refs.append(f0)
        if not refs:
            raise ValueError(
                f'No mode is present in {100*self.min_persistence:.0f}% of '
                f'the windows; the poles do not repeat from window to window')
        refs = np.sort(np.array(refs))

        def match(refs):
            F = np.full((n_use, len(refs)), np.nan)
            Z = np.full_like(F, np.nan)
            P = np.full(F.shape, np.nan + 0j, dtype=complex)
            for ii, (ev, fr, ze) in enumerate(poles):
                if len(fr) == 0:
                    continue
                for j, r in enumerate(refs):
                    d = np.abs(fr - r)
                    k = int(np.argmin(d))
                    if d[k] < tol * r:
                        F[ii, j], Z[ii, j], P[ii, j] = fr[k], ze[k], ev[k]
            return F, Z, P

        F, Z, P = match(refs)
        refs = np.nanmedian(F, axis=0)          # centre on what was found
        F, Z, P = match(refs)
        self.ref_freq, self.track_freq = np.nanmedian(F, axis=0), F
        self.track_zeta, self.track_pole = Z, P

        # response level at each window: rms over its past window
        run = self.run; lags = run.cfg.lags
        lev = np.zeros(n_use)
        for ii, wi in enumerate(self.use_idx):
            k0 = run.windows[wi].window_idx
            seg = run.datout[k0:k0 + lags, :]
            lev[ii] = np.sqrt(np.mean(seg ** 2))
        self.level = np.log10(lev + 1e-300)

        print(f'\nModes present in at least '
              f'{100*self.min_persistence:.0f}% of windows: {len(refs)}')
        print(f"  {'mode':>10} {'found':>6} {'frequency Hz':>20} "
              f"{'damping %':>16}")
        for j in range(len(refs)):
            ok = np.isfinite(F[:, j])
            print(f'  {self.ref_freq[j]:8.2f} Hz {100*ok.mean():5.0f}% '
                  f'{np.nanmedian(F[:, j]):10.2f} +- {np.nanstd(F[:, j]):<7.3f} '
                  f'{100*np.nanmedian(Z[:, j]):7.3f} +- '
                  f'{100*np.nanstd(Z[:, j]):<6.3f}')

    def modal_history(self):
        """Each tracked mode's part of the measured response, sample by
        sample.

        A window's model is applied to the samples nearest that window in
        time. There the state is x = Jnow @ past, the response at the newest
        sample of the past block is fitted as y = C x over the window's
        neighbours (with a term D u for the input over the length of the
        past block, which the state does not carry), and with Phi = V diag(ev) inv(V) the state splits into
        modal coordinates q = inv(V) x. Mode j (a conjugate pair) then
        contributes 2 Re(C v_j q_j) to the response; this does not depend on
        how the states or eigenvectors are scaled.

        Returns dict:
          time   (N,)               time of each sample
          modal  (N, n_modes, n_out) contribution of each tracked mode at
                                    each channel, NaN where the mode is not
                                    found in that window
          direct (N, n_out)         D u, the response to the input over the
                                    last `lags` samples; small for lightly
                                    damped modes, larger when a mode's decay
                                    time is comparable to the block length
          fitted (N, n_out)         C x + D u, every state included
          measured (N, n_out)       the response at the same samples
        """
        if self.track_pole is None:
            raise RuntimeError('Call fit() first')
        run = self.run
        lags, n_out = run.cfg.lags, run.n_out
        nc = n_out * lags
        N = run.past_db.shape[0]
        nm = len(self.ref_freq)
        k0 = np.array([run.windows[w].window_idx for w in self.use_idx])
        order = np.argsort(k0)
        edges = np.concatenate([[0], (k0[order][:-1] + k0[order][1:]) // 2,
                                [N]])
        modal = np.full((N, nm, n_out), np.nan)
        fitted = np.full((N, n_out), np.nan)
        direct = np.zeros((N, n_out))
        U = None
        if getattr(run, 'inputd', None) is not None:
            # input history ending at each sample: u(n), u(n-1), ...
            u = np.asarray(run.inputd, float).reshape(len(run.datout), -1)
            m = max(1, min(lags, run.cfg.neighbors // (4 * u.shape[1])))
            U = np.zeros((len(u), m * u.shape[1]))
            for k in range(m):
                U[k:, k * u.shape[1]:(k + 1) * u.shape[1]] = u[:len(u) - k]
            U = U / (U.std(axis=0) + 1e-300)
        for a, ii in enumerate(order):
            wr = run.windows[self.use_idx[ii]]
            lo, hi = int(edges[a]), int(edges[a + 1])
            if hi <= lo or wr.rank == 0:
                continue
            newest = np.asarray(wr.indpast) + lags - 1
            ok = (newest >= 0) & (newest < run.datout.shape[0])
            if ok.sum() <= wr.rank:
                continue
            A = wr.spast[ok]
            if U is not None:
                A = np.hstack([A, U[newest[ok]]])
            CD = np.linalg.lstsq(A, run.datout[newest[ok]], rcond=None)[0].T
            C = CD[:, :wr.rank]                             # (n_out, rank)
            X = run.past_db[lo:hi] @ wr.Jnow[:, :nc].T      # (n, rank)
            fitted[lo:hi] = X @ C.T
            if U is not None:
                direct[lo:hi] = U[lo + lags - 1:hi + lags - 1] @ CD[:, wr.rank:].T
                fitted[lo:hi] += direct[lo:hi]
            ev, V = np.linalg.eig(wr.Phi)
            try:
                Q = X @ np.linalg.inv(V).T                  # (n, rank)
            except np.linalg.LinAlgError:
                continue
            for j in range(nm):
                pole = self.track_pole[ii, j]
                if not np.isfinite(pole):
                    continue
                g = int(np.argmin(np.abs(ev - pole)))
                modal[lo:hi, j, :] = 2.0 * np.real(
                    np.outer(Q[:, g], C @ V[:, g]))
        rows = np.arange(N) + lags - 1
        t = (np.asarray(run.time)[rows] if getattr(run, 'time', None)
             is not None else rows / run.fs)
        return dict(time=t, modal=modal, fitted=fitted, direct=direct,
                    measured=run.datout[rows])

    def _features(self):
        """Phi_vec: percent deviation of each mode's frequency from its
        median (gaps filled with 0). Phi_pca: its principal components."""
        V = 100.0 * (self.track_freq / self.ref_freq - 1.0)
        V = np.where(np.isfinite(V), V, 0.0)
        self.Phi_vec = V
        sd = V.std(axis=0); sd[sd == 0] = 1.0
        pca = PCA()
        score = pca.fit_transform(V / sd)
        if score.shape[1] < 2:
            score = np.column_stack([score, np.zeros(len(score))])
        cum = np.cumsum(pca.explained_variance_ratio_ * 100)
        n_pca = max(2, min(int(np.searchsorted(cum, self.n_pca_var)) + 1,
                           score.shape[1]))
        self.pca_model, self.Phi_pca = pca, score[:, :n_pca]
        self._cum_var, self._n_pca = cum, n_pca

    def _separation_test(self):
        """One state or two, mode by mode (see cva_regimes)."""
        from cva_regimes import _feature
        t = np.array([self.run.windows[wi].time for wi in self.use_idx])
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / self.run.fs
        nm = len(self.ref_freq)
        self.sep_D = np.zeros(nm); self.sep_gap = np.ones(nm)
        self.sep_dist = np.zeros(nm); self.sep_r2 = np.zeros(nm)
        self.sep_two = np.zeros(nm, bool)
        labels = [None] * nm
        print(f'\nRegime test per mode (after removing the trend with level):')
        print(f"  {'mode':>10} {'level explains':>15} {'separation':>11} "
              f"{'gap':>5} {'apart':>8}")
        for j in range(nm):
            v = self.track_freq[:, j]
            if np.isfinite(v).sum() < 20:
                continue
            try:
                ft = _feature(f'{self.ref_freq[j]:.0f} Hz', 'frequency',
                              self.level, v, dt, self.min_dwell_sec,
                              self.min_freq_change)
            except Exception as e:
                print(f'  {self.ref_freq[j]:8.2f} Hz  test failed: {e}')
                continue
            self.sep_D[j], self.sep_gap[j] = ft.separation, ft.gap
            self.sep_dist[j], self.sep_r2[j] = ft.distance, ft.r2_level
            self.sep_two[j], labels[j] = ft.two, ft.labels
            print(f'  {self.ref_freq[j]:8.2f} Hz {100*ft.r2_level:14.0f}% '
                  f'{ft.separation:11.2f} {ft.gap:5.2f} '
                  f'{100*ft.distance:7.3f}%'
                  f"{'   <- two clusters' if ft.two else ''}")
        print('  (two clusters need separation >= 3, gap <= 0.5, each with '
              f'15% of windows, and {100*self.min_freq_change:.2g}% apart)')
        two = np.flatnonzero(self.sep_two)
        if len(two):
            j = int(two[np.argmax(self.sep_D[two])])
            self.deciding, self._auto_labels = j, labels[j]
            self.n_clusters = 2
            self.verdict = (f'two states: the {self.ref_freq[j]:.1f} Hz mode '
                            f'sits in two clusters '
                            f'{100*self.sep_dist[j]:.2f}% apart at the same '
                            f'response level')
        else:
            self.deciding, self._auto_labels = -1, None
            self.n_clusters = 1
            lead = int(np.argmax(self.sep_r2))
            if self.sep_r2[lead] >= 0.5:
                self.verdict = (f'one state; the {self.ref_freq[lead]:.1f} Hz '
                                f'mode follows the response level '
                                f'({100*self.sep_r2[lead]:.0f}% explained)')
            else:
                self.verdict = 'one state; no mode separates into two clusters'
        print(f'  -> {self.verdict}')
        # kept for files and plots written for the earlier version
        self.inertia = np.zeros(0)
        self._k_elbow_d2 = self._k_elbow_nd = self.n_clusters

    def _final_clustering(self, forced=False):
        k = self.n_clusters
        n_use = len(self.use_idx)
        if forced and k > 1:
            print(f'\nRunning k-means on the mode tracks (k={k}) ...')
            km = KMeans(n_clusters=k, n_init=self.kmeans_rep,
                        max_iter=self.kmeans_iter, random_state=42)
            self.cluster_ids = km.fit(self.Phi_pca).labels_
        elif k == 2 and self._auto_labels is not None:
            self.cluster_ids = np.asarray(self._auto_labels, int)
        else:
            self.cluster_ids = np.zeros(n_use, int)
        self.centroids_pca = np.array(
            [self.Phi_pca[self.cluster_ids == ci].mean(axis=0)
             if np.any(self.cluster_ids == ci)
             else np.zeros(self.Phi_pca.shape[1]) for ci in range(k)])
        self.centroid_Phis = []
        sizes = np.bincount(self.cluster_ids, minlength=k)
        print('\nCluster sizes:')
        for ci in range(k):
            print(f'  Cluster {ci} : {sizes[ci]} windows  ({100*sizes[ci]/len(self.cluster_ids):.1f}%)')
        sorted_idx    = np.argsort(sizes)[::-1]
        self.regime_A = int(sorted_idx[0])
        # with one state B is the same cluster as A, so callers that index
        # by regime_B still work
        self.regime_B = int(sorted_idx[1]) if k > 1 else self.regime_A
        self.minority = [int(x) for x in sorted_idx[2:]]
        print(f'\nRegime A : Cluster {self.regime_A}  ({sizes[self.regime_A]} windows)')
        if k > 1:
            print(f'Regime B : Cluster {self.regime_B}  ({sizes[self.regime_B]} windows)')
        if self.minority:
            print(f'Minority : {self.minority}  ({sum(sizes[m] for m in self.minority)} windows)')
        self.cluster_times = np.full(len(self.run.windows), np.nan)
        for ii, wi in enumerate(self.use_idx):
            self.cluster_times[wi] = self.cluster_ids[ii]

    def _cluster_tracks(self, k):
        """Mean pole, frequency and damping of each mode within cluster k.
        Returned per reference mode, NaN where the cluster never has it."""
        m = self.cluster_ids == k
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                pole = (np.nanmean(np.real(self.track_pole[m]), axis=0)
                        + 1j * np.nanmean(np.imag(self.track_pole[m]), axis=0))
                freq = np.nanmean(self.track_freq[m], axis=0)
                zeta = np.nanmean(self.track_zeta[m], axis=0)
        return pole, freq, zeta

    def _eigenvalue_analysis(self):
        """cluster_mean_ev: for each cluster, the mean pole of each tracked
        mode. Poles are averaged mode by mode, so modes are never mixed."""
        self.cluster_mean_ev = []
        print('\nMean of each tracked mode within each cluster:')
        for k in range(self.n_clusters):
            pole, freq, zeta = self._cluster_tracks(k)
            ok = np.isfinite(freq)
            self.cluster_mean_ev.append(pole[ok])
            tag = ' [minority]' if k in self.minority else ''
            print(f'\n  Cluster {k}{tag}  '
                  f'({int((self.cluster_ids == k).sum())} windows):')
            for j in np.flatnonzero(ok):
                print(f'    {freq[j]:.2f} Hz   zeta={zeta[j]:.4f}   '
                      f'|z|={abs(pole[j]):.4f}')

    def _frequency_shift_report(self):
        if self.n_clusters < 2:
            print('\nOne state: no frequency shift between regimes to report.')
            return
        _, fA, zA = self._cluster_tracks(self.regime_A)
        _, fB, zB = self._cluster_tracks(self.regime_B)
        print(f'\nFrequency shift  Cluster {self.regime_A} vs Cluster {self.regime_B}:')
        for j in range(len(self.ref_freq)):
            if not (np.isfinite(fA[j]) and np.isfinite(fB[j])):
                continue
            pct = 100*(fA[j]-fB[j])/(fB[j]+1e-12)
            print(f'  {j+1}  {fA[j]:.2f} Hz zeta={zA[j]:.4f}'
                  f'  {fB[j]:.2f} Hz zeta={zB[j]:.4f}  {pct:+.2f}%')

    # ------------------------------------------------------------------
    # Plot methods — all accept save_path=None, dpi=150
    # ------------------------------------------------------------------
    def _savefig(self, fig, save_path, dpi):
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')

    def plot_all(self, save_dir=None, dpi=150):
        def sp(name):
            return Path(save_dir)/f'{name}.png' if save_dir else None
        self.plot_pca_variance(save_path=sp('clusters_pca_variance'), dpi=dpi)
        self.plot_elbow(save_path=sp('clusters_elbow'), dpi=dpi)
        self.plot_regime_timeseries(save_path=sp('clusters_timeseries'), dpi=dpi)
        self.plot_eigenvalues_zplane(save_path=sp('clusters_zplane'), dpi=dpi)
        self.plot_cluster_amplitude(save_path=sp('clusters_amplitude'), dpi=dpi)
        self.plot_pca_scatter(save_path=sp('clusters_pca_scatter'), dpi=dpi)
        if self.track_pole is not None:
            self.plot_modal_levels(save_path=sp('clusters_modal_levels'),
                                   dpi=dpi)
            self.plot_conditions(save_path=sp('clusters_conditions'),
                                 dpi=dpi)

    def modal_levels(self, win_sec: float = 0.1, history: dict = None):
        """Level of each tracked mode against time: rms of the mode's
        rebuilt acceleration (modal_history) over a sliding win_sec.

        Returns dict: time (M,), level (M, n_modes, n_out), measured
        (M, n_out) the same rms of the measured response; NaN where the mode
        was found in less than half of the window."""
        h = history or self.modal_history()
        fs = self.run.fs
        k = max(2, int(round(win_sec * fs)))
        step = max(1, k // 4)
        box = np.ones(k) / k

        def rms(x):
            good = np.isfinite(x)
            p = np.convolve(np.where(good, x, 0.0) ** 2, box, mode='same')
            w = np.convolve(good.astype(float), box, mode='same')
            out = np.sqrt(p / np.maximum(w, 1e-12))
            out[w < 0.5] = np.nan
            return out[::step]

        m, y = h['modal'], h['measured']
        t = h['time'][::step]
        level = np.stack([np.stack([rms(m[:, j, c])
                                    for c in range(m.shape[2])], axis=1)
                          for j in range(m.shape[1])], axis=1)
        measured = np.stack([rms(y[:, c]) for c in range(y.shape[1])],
                            axis=1)
        return dict(time=t, level=level, measured=measured)

    def plot_modal_levels(self, channel: Optional[int] = None,
                          win_sec: float = 0.0,
                          band_top: Optional[float] = None,
                          t_lim: Optional[Tuple[float, float]] = None,
                          names: Optional[List[str]] = None,
                          save_path=None, dpi=150):
        """Which part of the response is present when: for one response
        channel, the measured signal, its spectrogram, and the level of each
        tracked mode against time on one graph.

        channel  : response channel (0-based). None picks the channel where
                   the modes that come and go (found in under 90% of the
                   windows) are the largest share of the response.
        win_sec  : averaging time for the levels; 0 picks 0.1 s or 1/300 of
                   the time shown, whichever is longer.
        band_top : top of the excitation band in Hz, if known; modes above
                   it are drawn dashed and labelled.
        t_lim    : (start, end) seconds to show; the whole record if None.
        """
        from scipy.signal import spectrogram
        run = self.run
        fs = run.fs
        h = self.modal_history()
        t, y = h['time'], h['measured']
        if win_sec <= 0:
            span = (t_lim[1] - t_lim[0]) if t_lim else (t[-1] - t[0])
            win_sec = max(0.1, span / 300.0)
        lv = self.modal_levels(win_sec, history=h)
        nm = len(self.ref_freq)
        n_out = y.shape[1]
        if names is None:
            names = list(getattr(run, 'out_names', None) or [])
        if len(names) != n_out:
            names = [f'response {c + 1}' for c in range(n_out)]
        ms = np.nanmean(np.nan_to_num(h['modal']) ** 2, axis=0)  # (nm, n_out)
        if channel is None:
            # the modes that come and go, or failing that all but the
            # strongest one
            found = np.isfinite(self.track_freq).mean(axis=0)
            pick = found < 0.9
            if not pick.any():
                pick = np.arange(nm) != np.argmax(ms.sum(axis=1))
            channel = int(np.argmax(ms[pick].sum(axis=0)
                                    / (y.var(axis=0) + 1e-300)))
        channel = int(np.clip(channel, 0, n_out - 1))
        t0, t1 = t_lim if t_lim else (t[0], t[-1])

        fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1.3, 1.6]))
        w = (t >= t0) & (t <= t1)
        ax[0].plot(t[w], y[w, channel], color='0.2', lw=0.4,
                   label=f'measured response, {names[channel]}')
        ax[0].set_ylabel('response')
        ax[0].set_title('Measured response', loc='left', fontsize=10)
        ax[0].legend(fontsize=8, loc='upper right')
        ax[0].grid(True, alpha=0.3)

        nseg = int(2 ** np.round(np.log2(max(64, 0.1 * fs))))
        nseg = min(nseg, len(y) // 8)
        f, ts, Sxx = spectrogram(y[:, channel], fs=fs, nperseg=nseg,
                                 noverlap=nseg - max(1, nseg // 8))
        ts = ts + t[0]
        f_top = self.f_max or fs / 2
        ws, wf = (ts >= t0) & (ts <= t1), f <= f_top
        db = 10 * np.log10(Sxx[np.ix_(wf, ws)] + 1e-300)
        ax[1].pcolormesh(ts[ws], f[wf], db, shading='auto', cmap='viridis',
                         vmin=db.max() - 60, vmax=db.max(), rasterized=True)
        cmap = plt.cm.turbo(np.linspace(0.05, 0.95, max(nm, 2)))
        for j, f0 in enumerate(self.ref_freq):
            ax[1].plot([t1], [f0], marker='<', color=cmap[j], ms=9,
                       clip_on=False, zorder=5)
        if band_top:
            ax[1].axhline(band_top, color='w', ls='--', lw=1.0,
                          label=f'top of the excitation band, '
                                f'{band_top:.0f} Hz')
            ax[1].legend(fontsize=8, loc='upper right')
        ax[1].set_ylim(0, f_top)
        ax[1].set_ylabel('Hz')
        ax[1].set_title('Spectrogram of the measured response (60 dB '
                        'range); arrows at the right mark the tracked modes',
                        loc='left', fontsize=10)

        tl = lv['time']
        wl = (tl >= t0) & (tl <= t1)
        ax[2].semilogy(tl[wl], lv['measured'][wl, channel], color='k',
                       lw=1.6, label='measured response, all of it')
        for j, f0 in enumerate(self.ref_freq):
            above = bool(band_top) and f0 > band_top
            found = 100 * np.isfinite(lv['level'][:, j, channel]).mean()
            ax[2].semilogy(tl[wl], lv['level'][wl, j, channel],
                           color=cmap[j], lw=1.1,
                           ls='--' if above else '-',
                           label=f'{f0:.1f} Hz mode'
                                 + (', above the band' if above else '')
                                 + f' (found {found:.0f}% of the time)')
        good = lv['level'][wl][:, :, channel]
        if np.isfinite(good).any():
            lo = np.nanpercentile(good, 2)
            ax[2].set_ylim(max(lo / 3, 1e-6 * np.nanmax(lv['measured'])),
                           2 * np.nanmax(lv['measured'][wl, channel]))
        ax[2].set_ylabel(f'rms over {win_sec:.2g} s')
        ax[2].set_xlabel('time, s')
        ax[2].set_title('Level of each mode against time, from the rebuilt '
                        'modal response (gaps: mode not found)', loc='left',
                        fontsize=10)
        ax[2].legend(fontsize=8, ncol=3, loc='upper center',
                     bbox_to_anchor=(0.5, -0.16))
        ax[2].grid(True, alpha=0.3, which='both')
        ax[2].set_xlim(t0, t1)
        fig.suptitle(f'Modal content against time at {names[channel]}',
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def plot_pca_variance(self, save_path=None, dpi=150):
        if self._cum_var is None:
            raise RuntimeError('No PCA variance curve available — call fit() first, '
                               'or reload from a file saved by this version.')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(self._cum_var)+1), self._cum_var, 'b.-', ms=6, lw=1.2)
        ax.axvline(self._n_pca, color='r', ls='--', label=f'{self._n_pca} components')
        ax.axhline(self.n_pca_var, color='k', ls='--', label=f'{self.n_pca_var:.0f}%')
        ax.set_xlabel('PCA Component'); ax.set_ylabel('Cumulative Variance (%)')
        ax.set_title('Mode-frequency tracks, PCA — Cumulative Variance')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def plot_elbow(self, save_path=None, dpi=150):
        """Cluster separation of every tracked mode (the regime test).
        For a file saved by the Phi-entry version, its k-means elbow."""
        if self.sep_D is None:
            return self._plot_elbow_legacy(save_path, dpi)
        k = np.arange(len(self.ref_freq))
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax = axes[0]
        ax.bar(k, self.sep_D, 0.55,
               color=['tab:red' if t else '0.6' for t in self.sep_two])
        ax.axhline(3.0, color='k', ls='--', lw=1.0,
                   label='separation needed for two states (3)')
        ax.set_ylabel('cluster separation')
        ax.set_title('Regime test per mode (red = two clusters that pass '
                     'every condition)', fontsize=10, loc='left')
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)
        ax = axes[1]
        ax.bar(k, 100 * np.nanstd(self.track_freq, axis=0) / self.ref_freq,
               0.55, color='steelblue',
               label='window-to-window scatter of the frequency')
        ax.axhline(100 * self.min_freq_change, color='k', ls='--', lw=1.0,
                   label=f'smallest difference counted '
                         f'({100*self.min_freq_change:.2g}%)')
        ax.set_yscale('log')
        ax.set_ylabel('% of frequency')
        ax.set_xticks(k, [f'{f:.0f} Hz' for f in self.ref_freq], fontsize=8)
        ax.set_xlabel('tracked mode')
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3, which='both')
        fig.suptitle(f'Regimes: {self.verdict}', fontsize=11, wrap=True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def _plot_elbow_legacy(self, save_path=None, dpi=150):
        if self.inertia is None or len(self.inertia) == 0:
            raise RuntimeError('No regime test available — call fit() first.')
        # k_max may disagree with a reloaded inertia array; the data wins.
        k_arr = np.arange(1, len(self.inertia)+1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_arr, self.inertia, 'b.-', ms=10, lw=1.5, label='Inertia')
        ax.plot(self._k_elbow_d2, self.inertia[self._k_elbow_d2-1], 'rv',
                ms=12, label=f'2nd-diff  k={self._k_elbow_d2}')
        ax.plot(self._k_elbow_nd, self.inertia[self._k_elbow_nd-1], 'g^',
                ms=12, label=f'Normalized  k={self._k_elbow_nd}')
        ax.set_xlabel('k'); ax.set_ylabel('Inertia')
        ax.set_title('Elbow Plot'); ax.legend(); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def plot_regime_timeseries(self, save_path=None, dpi=150):
        windows = self.run.windows
        k    = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        fig.suptitle('Regime Analysis' + (f': {self.verdict}' if self.verdict else ''),
                     fontsize=12, fontweight='bold', wrap=True)
        axes[0].plot(self.run.time, self.run.datout[:, 0], 'k-', lw=0.5)
        axes[0].set_ylabel('Response Ch 1'); axes[0].set_title('Response Signal')
        axes[0].grid(True)
        axes[1].set_title('Cluster Assignment over Time')
        for ci in range(k):
            mask = self.cluster_ids == ci
            t_k  = np.array([windows[wi].time for wi in self.use_idx[mask]])
            axes[1].scatter(t_k, np.full(len(t_k), ci), s=8, color=cmap[ci],
                            label=f'Cluster {ci}  (n={mask.sum()})')
        axes[1].set_yticks(range(k))
        axes[1].set_yticklabels([f'Cluster {ci}' for ci in range(k)])
        axes[1].legend(loc='upper right', fontsize=7); axes[1].grid(True)
        axes[2].set_title('Cluster Detail')
        for ci in range(k):
            mask = self.cluster_ids == ci
            t_k  = np.array([windows[wi].time for wi in self.use_idx[mask]])
            axes[2].plot(t_k, np.full(len(t_k), ci), '.', color=cmap[ci], ms=3)
        axes[2].set_yticks(range(k))
        axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Cluster')
        axes[2].grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def plot_eigenvalues_zplane(self, save_path=None, dpi=150):
        dt = 1.0/self.run.fs; fs = self.run.fs
        k  = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))
        fig, ax = plt.subplots(figsize=(8, 6))
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, label='Unit circle')
        for ci in range(k):
            mean_ev = self.cluster_mean_ev[ci]
            mag     = np.abs(mean_ev)
            ce      = np.log(mean_ev+1e-300) / dt
            freq    = np.abs(np.imag(ce)) / (2*np.pi)
            good    = ((np.abs(np.imag(mean_ev)) > self.min_im_frac*mag) &
                       (freq > 0.5) & (freq < fs/2) & (mag <= 1.0) & (np.imag(mean_ev) > 0))
            if self.f_min is not None:
                good &= (freq >= self.f_min)
            if self.f_max is not None:
                good &= (freq <= self.f_max)
            is_min  = ci in self.minority
            ax.scatter(np.real(mean_ev[good]), np.imag(mean_ev[good]),
                       s=80 if not is_min else 60, color=cmap[ci],
                       marker='o' if not is_min else 'x', linewidths=2,
                       label=f'Cluster {ci}' + (' (minority)' if is_min else ''), zorder=3)
        ax.set_xlabel('Real'); ax.set_ylabel('Imaginary')
        ax.set_title('Mean pole of each tracked mode, per cluster — z-plane')
        ax.legend(loc='best'); ax.grid(True); ax.set_aspect('equal')
        lim = 1.3; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def plot_cluster_amplitude(self, save_path=None, dpi=150):
        k    = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))
        win_half = 20; windows = self.run.windows
        amp = np.zeros(len(self.use_idx))
        for ii, wi in enumerate(self.use_idx):
            idx_w = windows[wi].sample_idx
            lo = max(0, idx_w-win_half); hi = min(self.run.datout.shape[0], idx_w+win_half)
            amp[ii] = np.sqrt(np.mean(self.run.datout[lo:hi, 0]**2))
        fig, ax = plt.subplots(figsize=(8, 5))
        for ci in range(k):
            mask = self.cluster_ids == ci
            ax.hist(amp[mask], bins=25, color=cmap[ci], alpha=0.55,
                    label=f'Cluster {ci}')
        ax.set_xlabel('Local RMS Amplitude  (Ch 1)'); ax.set_ylabel('Count')
        ax.set_title('Cluster Assignment vs Local Response Amplitude')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    def plot_pca_scatter(self, save_path=None, dpi=150):
        k    = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))
        fig, ax = plt.subplots(figsize=(8, 6))
        for ci in range(k):
            mask = self.cluster_ids == ci
            ax.scatter(self.Phi_pca[mask, 0], self.Phi_pca[mask, 1],
                       s=10, color=cmap[ci], alpha=0.6,
                       label=f'Cluster {ci}  (n={mask.sum()})')
        for ci in range(k):
            ax.plot(self.centroids_pca[ci, 0], self.centroids_pca[ci, 1],
                    'k+', ms=16, lw=2.5,
                    label='_centroid' if ci > 0 else 'Centroid')
        ax.set_xlabel('PCA Component 1'); ax.set_ylabel('PCA Component 2')
        ax.set_title('Mode-frequency tracks in PCA space  (+ = centroid)')
        ax.legend(loc='best', fontsize=8); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); show_if_enabled()
        return fig

    # ------------------------------------------------------------------
    def save(self, filepath):
        p = Path(filepath)
        print(f'\nSaving clusters to {p} ...')
        with h5py.File(str(p), 'w') as f:
            f.attrs['n_clusters'] = self.n_clusters
            f.attrs['rk_mode']    = self.rk_mode
            f.attrs['regime_A']   = self.regime_A
            f.attrs['regime_B']   = self.regime_B
            f.attrs['minority']   = self.minority if self.minority else [-1]
            # fit settings + derived values the plots need after a reload
            f.attrs['k_max']       = self.k_max
            f.attrs['n_pca_var']   = self.n_pca_var
            f.attrs['kmeans_rep']  = self.kmeans_rep
            f.attrs['kmeans_iter'] = self.kmeans_iter
            f.attrs['min_im_frac'] = self.min_im_frac
            f.attrs['f_min']       = -1.0 if self.f_min is None else float(self.f_min)
            f.attrs['f_max']       = -1.0 if self.f_max is None else float(self.f_max)
            f.attrs['n_pca']       = self._n_pca
            f.attrs['k_elbow_d2']  = self._k_elbow_d2
            f.attrs['k_elbow_nd']  = self._k_elbow_nd
            if self._cum_var is not None:
                f.create_dataset('cum_var', data=self._cum_var)
            f.create_dataset('use_idx',       data=self.use_idx)
            f.create_dataset('cluster_ids',   data=self.cluster_ids)
            f.create_dataset('centroids_pca', data=self.centroids_pca)
            f.create_dataset('Phi_pca',       data=self.Phi_pca, compression='gzip')
            f.create_dataset('Phi_vec',       data=self.Phi_vec, compression='gzip')
            f.create_dataset('cluster_times', data=self.cluster_times)
            f.create_dataset('inertia',       data=self.inertia)
            if self.ref_freq is not None:
                f.attrs['verdict']  = self.verdict
                f.attrs['deciding'] = self.deciding
                f.attrs['min_freq_change'] = self.min_freq_change
                g = f.create_group('tracks')
                g.create_dataset('ref_freq',   data=self.ref_freq)
                g.create_dataset('track_freq', data=self.track_freq)
                g.create_dataset('track_zeta', data=self.track_zeta)
                g.create_dataset('track_pole_real', data=np.real(self.track_pole))
                g.create_dataset('track_pole_imag', data=np.imag(self.track_pole))
                g.create_dataset('level',    data=self.level)
                g.create_dataset('sep_D',    data=self.sep_D)
                g.create_dataset('sep_gap',  data=self.sep_gap)
                g.create_dataset('sep_dist', data=self.sep_dist)
                g.create_dataset('sep_r2',   data=self.sep_r2)
                g.create_dataset('sep_two',  data=self.sep_two.astype(np.int8))
            g = f.create_group('centroid_Phis')
            for ci, cp in enumerate(self.centroid_Phis):
                g.create_dataset(str(ci), data=cp)
            g = f.create_group('cluster_mean_ev')
            for ci, ev in enumerate(self.cluster_mean_ev):
                g.create_dataset(f'{ci}_real', data=np.real(ev))
                g.create_dataset(f'{ci}_imag', data=np.imag(ev))
        print(f'  Saved {self.n_clusters} clusters.')

    @classmethod
    def load(cls, filepath, run):
        p = Path(filepath)
        print(f'Loading clusters from {p} ...')
        with h5py.File(str(p), 'r') as f:
            a = f.attrs
            fmin = float(a.get('f_min', -1.0))
            fmax = float(a.get('f_max', -1.0))
            obj = cls(run,
                      k_max       = int(a.get('k_max', 8)),
                      n_pca_var   = float(a.get('n_pca_var', 95.0)),
                      kmeans_rep  = int(a.get('kmeans_rep', 20)),
                      kmeans_iter = int(a.get('kmeans_iter', 500)),
                      min_im_frac = float(a.get('min_im_frac', 0.01)),
                      f_min       = None if fmin < 0 else fmin,
                      f_max       = None if fmax < 0 else fmax)
            obj._n_pca        = int(a.get('n_pca', 0))
            obj._k_elbow_d2   = int(a.get('k_elbow_d2', 0))
            obj._k_elbow_nd   = int(a.get('k_elbow_nd', 0))
            if 'cum_var' in f:
                obj._cum_var  = f['cum_var'][:]
            obj.n_clusters    = int(f.attrs['n_clusters'])
            obj.rk_mode       = int(f.attrs['rk_mode'])
            obj.regime_A      = int(f.attrs['regime_A'])
            obj.regime_B      = int(f.attrs['regime_B'])
            obj.minority      = [int(x) for x in f.attrs['minority'] if x >= 0]
            obj.use_idx       = f['use_idx'][:]
            obj.cluster_ids   = f['cluster_ids'][:]
            obj.centroids_pca = f['centroids_pca'][:]
            obj.Phi_pca       = f['Phi_pca'][:]
            obj.Phi_vec       = f['Phi_vec'][:]
            obj.cluster_times = f['cluster_times'][:]
            obj.inertia       = f['inertia'][:]
            k = obj.n_clusters
            obj.centroid_Phis = [f['centroid_Phis'][str(ci)][:] for ci in range(k)
                                 if str(ci) in f['centroid_Phis']]
            if 'tracks' in f:
                g = f['tracks']
                obj.verdict  = str(a.get('verdict', ''))
                obj.deciding = int(a.get('deciding', -1))
                obj.min_freq_change = float(a.get('min_freq_change', 1e-3))
                obj.ref_freq   = g['ref_freq'][:]
                obj.track_freq = g['track_freq'][:]
                obj.track_zeta = g['track_zeta'][:]
                obj.track_pole = (g['track_pole_real'][:]
                                  + 1j * g['track_pole_imag'][:])
                obj.level    = g['level'][:]
                obj.sep_D    = g['sep_D'][:]
                obj.sep_gap  = g['sep_gap'][:]
                obj.sep_dist = g['sep_dist'][:]
                obj.sep_r2   = g['sep_r2'][:]
                obj.sep_two  = g['sep_two'][:].astype(bool)
            obj.cluster_mean_ev = [
                f['cluster_mean_ev'][f'{ci}_real'][:] +
                1j*f['cluster_mean_ev'][f'{ci}_imag'][:]
                for ci in range(k)]
        print(f'  Loaded {obj.n_clusters} clusters.')
        return obj

    def __repr__(self):
        if self.cluster_ids is None:
            return 'CVAPhiCluster(not fitted)'
        return (f'CVAPhiCluster(k={self.n_clusters}, rk={self.rk_mode}, '
                f'n_windows={len(self.use_idx)}, '
                f'regime_A={self.regime_A}, regime_B={self.regime_B})')
