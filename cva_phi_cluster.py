"""
cva_phi_cluster.py
==================
Cluster Phi matrices from CVARun to identify dynamical regimes.

Author: Norman / Claude
Date:   2026-04-14
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cva_run import CVARun


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
                 f_min=None, f_max=None):
        self.run         = run
        self.k_max       = k_max
        self.n_pca_var   = n_pca_var
        self.kmeans_rep  = kmeans_rep
        self.kmeans_iter = kmeans_iter
        self.min_im_frac = min_im_frac
        self.f_min       = f_min   # Optional lower frequency bound (Hz)
        self.f_max       = f_max   # Optional upper frequency bound (Hz)

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

    # ------------------------------------------------------------------
    def fit(self, n_clusters=0):
        self._select_windows()
        self._vectorize_phi()
        self._pca()
        self._elbow_sweep()
        if n_clusters > 0:
            self.n_clusters = n_clusters
            print(f'Using override n_clusters = {n_clusters}')
        else:
            print(f'Using auto-detected n_clusters = {self.n_clusters}')
        self._final_clustering()
        self._build_centroid_phis()
        self._eigenvalue_analysis()
        self._frequency_shift_report()

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
        print(f'   CVA Phi Matrix Clustering')
        print(f'========================================\n')
        print(f'Modal rank      : {self.rk_mode}')
        print(f'Windows at rank : {n_use} / {len(self.run.windows)}'
              f'  ({100*n_use/len(self.run.windows):.1f}%)')

    def _vectorize_phi(self):
        rk    = self.rk_mode
        n_use = len(self.use_idx)
        Phi_vec = np.zeros((n_use, rk*rk))
        for ii, wi in enumerate(self.use_idx):
            Phi_i = self.run.windows[wi].Phi.copy()
            rk_i = Phi_i.shape[0]
            if rk_i > rk:   Phi_i = Phi_i[:rk, :rk]
            elif rk_i < rk:
                pad = np.zeros((rk, rk)); pad[:rk_i, :rk_i] = Phi_i; Phi_i = pad
            for col in range(rk):
                ref = np.argmax(np.abs(Phi_i[:, col]))
                if Phi_i[ref, col] < 0:
                    Phi_i[:, col] = -Phi_i[:, col]
            Phi_vec[ii, :] = Phi_i.ravel()
        self.Phi_vec = Phi_vec
        print(f'\nVectorized Phi matrices : {Phi_vec.shape}')

    def _pca(self):
        print('Running PCA ...')
        pca   = PCA()
        score = pca.fit_transform(self.Phi_vec)
        expl  = pca.explained_variance_ratio_ * 100
        cum   = np.cumsum(expl)
        n_pca = int(np.searchsorted(cum, self.n_pca_var)) + 1
        n_pca = min(n_pca, score.shape[1])
        self.pca_model = pca
        self.Phi_pca   = score[:, :n_pca]
        self._cum_var  = cum
        self._n_pca    = n_pca
        print(f'PCA components kept : {n_pca}  ({cum[n_pca-1]:.1f}% variance)')

    def _elbow_sweep(self):
        print(f'\nElbow sweep k=1..{self.k_max} ...')
        inertia = np.zeros(self.k_max)
        for k in range(1, self.k_max+1):
            km = KMeans(n_clusters=k, n_init=self.kmeans_rep,
                        max_iter=self.kmeans_iter, random_state=42)
            km.fit(self.Phi_pca)
            inertia[k-1] = km.inertia_
            print(f'  k={k}  inertia={inertia[k-1]:.4g}')
        self.inertia = inertia
        d2 = np.diff(np.diff(inertia))
        self._k_elbow_d2 = int(np.argmax(np.abs(d2))) + 2
        inorm = (inertia - inertia[-1]) / (inertia[0] - inertia[-1] + 1e-12)
        knorm = np.arange(self.k_max) / (self.k_max - 1)
        self._k_elbow_nd = int(np.argmax(np.abs(inorm - (1-knorm)))) + 1
        self.n_clusters  = self._k_elbow_nd
        print(f'Auto elbow (2nd diff)  : k={self._k_elbow_d2}')
        print(f'Auto elbow (normalized): k={self._k_elbow_nd}')

    def _final_clustering(self):
        k = self.n_clusters
        print(f'\nRunning final k-means (k={k}, {self.kmeans_rep} replicates) ...')
        km = KMeans(n_clusters=k, n_init=self.kmeans_rep,
                    max_iter=self.kmeans_iter, random_state=42)
        km.fit(self.Phi_pca)
        self.cluster_ids   = km.labels_
        self.centroids_pca = km.cluster_centers_
        sizes = np.bincount(self.cluster_ids, minlength=k)
        print('\nCluster sizes:')
        for ci in range(k):
            print(f'  Cluster {ci} : {sizes[ci]} windows  ({100*sizes[ci]/len(self.cluster_ids):.1f}%)')
        sorted_idx    = np.argsort(sizes)[::-1]
        self.regime_A = int(sorted_idx[0])
        self.regime_B = int(sorted_idx[1])
        self.minority = [int(x) for x in sorted_idx[2:]]
        print(f'\nRegime A : Cluster {self.regime_A}  ({sizes[self.regime_A]} windows)')
        print(f'Regime B : Cluster {self.regime_B}  ({sizes[self.regime_B]} windows)')
        if self.minority:
            print(f'Minority : {self.minority}  ({sum(sizes[m] for m in self.minority)} windows)')
        self.cluster_times = np.full(len(self.run.windows), np.nan)
        for ii, wi in enumerate(self.use_idx):
            self.cluster_times[wi] = self.cluster_ids[ii]

    def _build_centroid_phis(self):
        rk = self.rk_mode
        Phi_mean = self.Phi_vec.mean(axis=0)
        self.centroid_Phis = []
        for k in range(self.n_clusters):
            cf = (self.centroids_pca[k, :] @
                  self.pca_model.components_[:self._n_pca, :]) + Phi_mean
            self.centroid_Phis.append(cf.reshape(rk, rk))

    def _eigenvalue_analysis(self):
        rk = self.rk_mode; dt = 1.0/self.run.fs; fs = self.run.fs
        self.cluster_mean_ev = []
        print('\nDirect eigenvalue averaging across cluster windows:')
        for k in range(self.n_clusters):
            win_idx = self.use_idx[self.cluster_ids == k]
            n_win   = len(win_idx)
            all_ev  = np.zeros((n_win, rk), dtype=complex)
            for wi_idx, wi in enumerate(win_idx):
                ev_i  = np.linalg.eigvals(self.run.windows[wi].Phi)
                isort = np.argsort(np.imag(ev_i))
                all_ev[wi_idx, :] = ev_i[isort]
            mean_ev = (np.mean(np.real(all_ev), 0) + 1j*np.mean(np.imag(all_ev), 0))
            self.cluster_mean_ev.append(mean_ev)
            mag  = np.abs(mean_ev)
            ce   = np.log(mean_ev+1e-300) / dt
            freq = np.abs(np.imag(ce)) / (2*np.pi)
            zeta = -np.real(ce) / (np.sqrt(np.imag(ce)**2+np.real(ce)**2)+1e-12)
            good = ((np.abs(np.imag(mean_ev)) > self.min_im_frac*mag) &
                    (freq > 0.5) & (freq < fs/2) & (mag <= 1.0) & (np.imag(mean_ev) > 0))
            if self.f_min is not None:
                good &= (freq >= self.f_min)
            if self.f_max is not None:
                good &= (freq <= self.f_max)
            tag = ' [minority]' if k in self.minority else ''
            print(f'\n  Cluster {k}{tag}  ({n_win} windows):')
            for ei in np.where(good)[0]:
                print(f'    {freq[ei]:.2f} Hz   zeta={zeta[ei]:.4f}   |z|={mag[ei]:.4f}')

    def _frequency_shift_report(self):
        dt = 1.0/self.run.fs; fs = self.run.fs
        fA,zA,mA = _get_physical(self.cluster_mean_ev[self.regime_A], dt, fs,
                                  self.min_im_frac, self.f_min, self.f_max)
        fB,zB,mB = _get_physical(self.cluster_mean_ev[self.regime_B], dt, fs,
                                  self.min_im_frac, self.f_min, self.f_max)
        fA,fB = fA[np.argsort(fA)], fB[np.argsort(fB)]
        zA,zB = zA[np.argsort(fA)], zB[np.argsort(fB)]
        mA,mB = mA[np.argsort(fA)], mB[np.argsort(fB)]
        n = min(len(fA), len(fB))
        print(f'\nFrequency shift  Cluster {self.regime_A} vs Cluster {self.regime_B}:')
        for mi in range(n):
            pct = 100*(fA[mi]-fB[mi])/(fB[mi]+1e-12)
            print(f'  {mi+1}  {fA[mi]:.2f} Hz zeta={zA[mi]:.4f}'
                  f'  {fB[mi]:.2f} Hz zeta={zB[mi]:.4f}  {pct:+.1f}%')

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

    def plot_pca_variance(self, save_path=None, dpi=150):
        if self._cum_var is None:
            raise RuntimeError('No PCA variance curve available — call fit() first, '
                               'or reload from a file saved by this version.')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(self._cum_var)+1), self._cum_var, 'b.-', ms=6, lw=1.2)
        ax.axvline(self._n_pca, color='r', ls='--', label=f'{self._n_pca} components')
        ax.axhline(self.n_pca_var, color='k', ls='--', label=f'{self.n_pca_var:.0f}%')
        ax.set_xlabel('PCA Component'); ax.set_ylabel('Cumulative Variance (%)')
        ax.set_title('Phi Matrix PCA — Cumulative Variance')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

    def plot_elbow(self, save_path=None, dpi=150):
        if self.inertia is None:
            raise RuntimeError('No elbow sweep available — call fit() first, or '
                               'reload from a file saved by this version.')
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
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

    def plot_regime_timeseries(self, save_path=None, dpi=150):
        windows = self.run.windows
        k    = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        fig.suptitle('Phi Matrix Regime Analysis', fontsize=13, fontweight='bold')
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
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

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
        ax.set_title('Mean Eigenvalues per Cluster — z-plane  (direct averaging)')
        ax.legend(loc='best'); ax.grid(True); ax.set_aspect('equal')
        lim = 1.3; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

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
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

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
        ax.set_title('Phi Matrices in PCA Space  (+ = centroid)')
        ax.legend(loc='best', fontsize=8); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

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
            obj.centroid_Phis = [f['centroid_Phis'][str(ci)][:] for ci in range(k)]
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
