"""
cva_phi_cluster.py
==================
Python translation of CVAPhiClusterm.m

Clusters local Phi matrices from a CVARun to identify global regime
structure and build representative centroid models per regime.

Usage
-----
    from cva_phi_cluster import CVAPhiCluster
    from cva_run import CVARun

    run = CVARun.load('s01d11_cva.h5')
    clust = CVAPhiCluster(run)
    clust.fit()          # PCA + elbow + k-means
    clust.plot_all()     # all 6 figures
    clust.save('s01d11_clusters.h5')

    # Or step by step:
    clust.fit(n_clusters=3)    # override auto-detected k

Key design notes (matching MATLAB version)
------------------------------------------
- Phi matrices are vectorized (full matrix, sign-normalized per column)
- PCA reduces to n_pca_var % explained variance before clustering
- Elbow auto-detection uses normalized distance method (k_elbow_nd)
- Eigenvalue averaging: eig() on each individual Phi window, sort by
  imaginary part, then average real/imag separately — NOT eig(mean(Phi))
- Two largest clusters = regime A and B

Dependencies
------------
    numpy, scipy, scikit-learn, matplotlib, h5py
    cva_run.py / cva_core.py on same path

Author: Norman / Claude
Date:   2026-04-12
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Local import
from cva_run import CVARun


# =============================================================================
# helpers
# =============================================================================

def _get_physical(mean_ev: np.ndarray, dt: float, fs: float,
                  min_im_frac: float = 0.01
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract physical modal parameters from mean eigenvalues.

    Returns (freqs_Hz, zetas, mags) for eigenvalues that are:
      - complex  (|Im| > min_im_frac * |ev|)
      - stable   (|ev| <= 1)
      - positive imaginary half
      - physical frequency (0.5 Hz .. fs/2)
    """
    ce      = np.log(mean_ev + 1e-300) / dt
    omega_d = np.abs(np.imag(ce))
    sigma   = -np.real(ce)
    omega_n = np.sqrt(omega_d**2 + sigma**2)
    freq    = omega_n / (2 * np.pi)
    zeta    = sigma / (omega_n + 1e-12)
    mag     = np.abs(mean_ev)

    good = (
        (np.abs(np.imag(mean_ev)) > min_im_frac * mag) &
        (freq > 0.5) & (freq < fs / 2) &
        (mag <= 1.0) &
        (np.imag(mean_ev) > 0)
    )

    return freq[good], zeta[good], mag[good]


# =============================================================================
# CVAPhiCluster
# =============================================================================

class CVAPhiCluster:
    """
    Cluster Phi matrices from a CVARun to identify dynamical regimes.

    Parameters
    ----------
    run          : CVARun object with completed windows
    k_max        : maximum clusters to sweep in elbow analysis
    n_pca_var    : % variance to retain in PCA (default 95)
    kmeans_rep   : k-means replicates for stability (default 20)
    kmeans_iter  : max iterations per replicate (default 500)
    min_im_frac  : min |Im(ev)| / |ev| to count as complex (default 0.01)

    After fit()
    -----------
    cluster_ids   : (n_use,)  cluster label per window (0-based)
    n_clusters    : int       final number of clusters
    regime_A      : int       largest cluster index
    regime_B      : int       second largest cluster index
    minority      : list      remaining cluster indices
    centroid_Phis : list of (rk, rk) arrays — centroid Phi per cluster
    cluster_mean_ev : list of (rk,) complex arrays — mean eigenvalues per cluster
    cluster_times : (n_use,) float — time of each clustered window
    """

    def __init__(
        self,
        run:          CVARun,
        k_max:        int   = 8,
        n_pca_var:    float = 95.0,
        kmeans_rep:   int   = 20,
        kmeans_iter:  int   = 500,
        min_im_frac:  float = 0.01,
    ):
        self.run         = run
        self.k_max       = k_max
        self.n_pca_var   = n_pca_var
        self.kmeans_rep  = kmeans_rep
        self.kmeans_iter = kmeans_iter
        self.min_im_frac = min_im_frac

        # Results populated by fit()
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
        self.centroid_Phis:    List[np.ndarray] = []
        self.cluster_mean_ev:  List[np.ndarray] = []
        self.cluster_times:    Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, n_clusters: int = 0) -> None:
        """
        Full pipeline: vectorize → PCA → elbow → k-means → eigenvalue analysis.

        Parameters
        ----------
        n_clusters : override auto-detected k (0 = auto)
        """
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

    # ------------------------------------------------------------------
    # _select_windows — keep only modal-rank windows
    # ------------------------------------------------------------------
    def _select_windows(self) -> None:
        windows = self.run.windows
        ranks   = np.array([w.rank for w in windows])

        # Modal rank = most common non-zero rank
        valid  = ranks[ranks > 0]
        if len(valid) == 0:
            raise ValueError('No valid windows found in CVARun')

        counts      = np.bincount(valid)
        self.rk_mode = int(np.argmax(counts))
        self.use_idx = np.where(ranks == self.rk_mode)[0]
        n_use        = len(self.use_idx)

        print(f'\n========================================')
        print(f'   CVA Phi Matrix Clustering')
        print(f'========================================\n')
        print(f'Modal rank      : {self.rk_mode}')
        print(f'Windows at rank : {n_use} / {len(windows)}'
              f'  ({100*n_use/len(windows):.1f}%)')

        if n_use < self.k_max * 10:
            print('WARNING: few windows at modal rank — clustering may be unreliable.')

    # ------------------------------------------------------------------
    # _vectorize_phi — flatten with sign normalization
    # ------------------------------------------------------------------
    def _vectorize_phi(self) -> None:
        rk    = self.rk_mode
        n_use = len(self.use_idx)
        Phi_vec = np.zeros((n_use, rk * rk))

        for ii, wi in enumerate(self.use_idx):
            Phi_i = self.run.windows[wi].Phi.copy()

            # Pad or trim to rk_mode
            rk_i = Phi_i.shape[0]
            if rk_i > rk:
                Phi_i = Phi_i[:rk, :rk]
            elif rk_i < rk:
                Phi_pad = np.zeros((rk, rk))
                Phi_pad[:rk_i, :rk_i] = Phi_i
                Phi_i = Phi_pad

            # Sign normalization: max-magnitude element of each column positive
            for col in range(rk):
                ref = np.argmax(np.abs(Phi_i[:, col]))
                if Phi_i[ref, col] < 0:
                    Phi_i[:, col] = -Phi_i[:, col]

            Phi_vec[ii, :] = Phi_i.ravel()

        self.Phi_vec = Phi_vec
        print(f'\nVectorized Phi matrices : {Phi_vec.shape}')

    # ------------------------------------------------------------------
    # _pca
    # ------------------------------------------------------------------
    def _pca(self) -> None:
        print(f'Running PCA on Phi vectors ...')

        pca   = PCA()
        score = pca.fit_transform(self.Phi_vec)

        explained    = pca.explained_variance_ratio_ * 100
        cum_var      = np.cumsum(explained)
        n_pca_idx    = np.searchsorted(cum_var, self.n_pca_var)
        n_pca        = int(n_pca_idx) + 1
        n_pca        = min(n_pca, score.shape[1])

        self.pca_model = pca
        self.Phi_pca   = score[:, :n_pca]
        self._cum_var  = cum_var
        self._n_pca    = n_pca

        print(f'PCA components kept : {n_pca}  ({cum_var[n_pca-1]:.1f}% variance)')

    # ------------------------------------------------------------------
    # _elbow_sweep
    # ------------------------------------------------------------------
    def _elbow_sweep(self) -> None:
        print(f'\nElbow sweep k=1..{self.k_max} ...')
        inertia = np.zeros(self.k_max)

        for k in range(1, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=self.kmeans_rep,
                        max_iter=self.kmeans_iter, random_state=42)
            km.fit(self.Phi_pca)
            inertia[k-1] = km.inertia_
            print(f'  k={k}  inertia={inertia[k-1]:.4g}')

        self.inertia = inertia

        # Method 1: second difference
        d1 = np.diff(inertia)
        d2 = np.diff(d1)
        k_elbow_d2 = int(np.argmax(np.abs(d2))) + 2   # +2 because double diff

        # Method 2: normalized distance from diagonal
        inertia_norm = (inertia - inertia[-1]) / (inertia[0] - inertia[-1] + 1e-12)
        k_norm       = np.arange(self.k_max) / (self.k_max - 1)
        dist_diag    = np.abs(inertia_norm - (1 - k_norm))
        k_elbow_nd   = int(np.argmax(dist_diag)) + 1   # +1 for 1-based k

        print(f'\nAuto elbow (2nd difference) : k={k_elbow_d2}')
        print(f'Auto elbow (normalized)     : k={k_elbow_nd}')

        self._k_elbow_d2 = k_elbow_d2
        self._k_elbow_nd = k_elbow_nd
        self.n_clusters  = k_elbow_nd   # default — can be overridden in fit()
        print(f'Using n_clusters = {self.n_clusters}  (normalized method)')

    # ------------------------------------------------------------------
    # _final_clustering
    # ------------------------------------------------------------------
    def _final_clustering(self) -> None:
        k = self.n_clusters
        print(f'\nRunning final k-means (k={k}, {self.kmeans_rep} replicates) ...')

        km = KMeans(n_clusters=k, n_init=self.kmeans_rep,
                    max_iter=self.kmeans_iter, random_state=42)
        km.fit(self.Phi_pca)

        self.cluster_ids   = km.labels_           # 0-based
        self.centroids_pca = km.cluster_centers_

        print('\nCluster sizes:')
        cluster_sizes = np.bincount(self.cluster_ids, minlength=k)
        for ci in range(k):
            pct = 100 * cluster_sizes[ci] / len(self.cluster_ids)
            print(f'  Cluster {ci} : {cluster_sizes[ci]} windows  ({pct:.1f}%)')

        # Regime A and B = two largest clusters
        sorted_idx    = np.argsort(cluster_sizes)[::-1]
        self.regime_A = int(sorted_idx[0])
        self.regime_B = int(sorted_idx[1])
        self.minority = [int(x) for x in sorted_idx[2:]]

        print(f'\nRegime A : Cluster {self.regime_A}  ({cluster_sizes[self.regime_A]} windows)')
        print(f'Regime B : Cluster {self.regime_B}  ({cluster_sizes[self.regime_B]} windows)')
        if self.minority:
            min_count = sum(cluster_sizes[m] for m in self.minority)
            print(f'Minority : Clusters {self.minority}  ({min_count} windows) — excluded')

        # Record cluster time labels
        windows = self.run.windows
        self.cluster_times = np.full(len(windows), np.nan)
        for ii, wi in enumerate(self.use_idx):
            self.cluster_times[wi] = self.cluster_ids[ii]

    # ------------------------------------------------------------------
    # _build_centroid_phis
    # ------------------------------------------------------------------
    def _build_centroid_phis(self) -> None:
        rk       = self.rk_mode
        Phi_mean = self.Phi_vec.mean(axis=0)
        pca      = self.pca_model
        n_pca    = self._n_pca

        self.centroid_Phis = []
        for k in range(self.n_clusters):
            # Reconstruct from PCA centroid
            centroid_full = (self.centroids_pca[k, :] @
                             pca.components_[:n_pca, :]) + Phi_mean
            self.centroid_Phis.append(centroid_full.reshape(rk, rk))

    # ------------------------------------------------------------------
    # _eigenvalue_analysis
    # ------------------------------------------------------------------
    def _eigenvalue_analysis(self) -> None:
        """
        Compute eig() on each individual Phi window, sort by Im part,
        then average within each cluster — matches MATLAB approach.
        """
        rk      = self.rk_mode
        dt      = 1.0 / self.run.fs
        fs      = self.run.fs
        windows = self.run.windows

        self.cluster_mean_ev = []

        print('\nDirect eigenvalue averaging across cluster windows:')

        for k in range(self.n_clusters):
            win_idx = self.use_idx[self.cluster_ids == k]
            n_win   = len(win_idx)

            all_ev = np.zeros((n_win, rk), dtype=complex)
            for wi_idx, wi in enumerate(win_idx):
                Phi_i = windows[wi].Phi
                ev_i  = np.linalg.eigvals(Phi_i)
                # Sort by imaginary part for consistent pairing
                isort = np.argsort(np.imag(ev_i))
                all_ev[wi_idx, :] = ev_i[isort]

            # Average real and imaginary parts separately
            mean_ev = (np.mean(np.real(all_ev), axis=0) +
                       1j * np.mean(np.imag(all_ev), axis=0))
            self.cluster_mean_ev.append(mean_ev)

            # Physical modal parameters
            mag     = np.abs(mean_ev)
            ce      = np.log(mean_ev + 1e-300) / dt
            omega_d = np.abs(np.imag(ce))
            sigma   = -np.real(ce)
            omega_n = np.sqrt(omega_d**2 + sigma**2)
            freq    = omega_n / (2 * np.pi)
            zeta    = sigma / (omega_n + 1e-12)

            good = (
                (np.abs(np.imag(mean_ev)) > self.min_im_frac * mag) &
                (freq > 0.5) & (freq < fs / 2) &
                (mag <= 1.0) &
                (np.imag(mean_ev) > 0)
            )

            tag = ' [minority]' if k in self.minority else ''
            print(f'\n  Cluster {k}{tag}  ({n_win} windows):')
            if not np.any(good):
                print('    (no physical complex pairs)')
            for ei in np.where(good)[0]:
                print(f'    {freq[ei]:.2f} Hz   zeta={zeta[ei]:.4f}'
                      f'   |z|={mag[ei]:.4f}')

    # ------------------------------------------------------------------
    # _frequency_shift_report
    # ------------------------------------------------------------------
    def _frequency_shift_report(self) -> None:
        dt = 1.0 / self.run.fs
        fs = self.run.fs

        fA, zA, mA = _get_physical(self.cluster_mean_ev[self.regime_A],
                                   dt, fs, self.min_im_frac)
        fB, zB, mB = _get_physical(self.cluster_mean_ev[self.regime_B],
                                   dt, fs, self.min_im_frac)

        isA = np.argsort(fA); fA, zA, mA = fA[isA], zA[isA], mA[isA]
        isB = np.argsort(fB); fB, zB, mB = fB[isB], zB[isB], mB[isB]
        n_show = min(len(fA), len(fB))

        print(f'\nFrequency shift  Cluster {self.regime_A} vs Cluster {self.regime_B}:')
        if n_show == 0:
            print('  No matching complex pairs found.')
            return

        print(f'  {"Mode":<4}  {"RegA":^38}  {"RegB":^38}  Shift')
        for mi in range(n_show):
            pct = 100 * (fB[mi] - fA[mi]) / (fA[mi] + 1e-12)
            print(f'  {mi+1:<4}  {fA[mi]:.2f} Hz zeta={zA[mi]:.4f} |z|={mA[mi]:.4f}'
                  f'  {fB[mi]:.2f} Hz zeta={zB[mi]:.4f} |z|={mB[mi]:.4f}'
                  f'  {fB[mi]-fA[mi]:+.2f} Hz ({pct:+.1f}%)')

    # ------------------------------------------------------------------
    # plot_all  — 6 figures matching MATLAB
    # ------------------------------------------------------------------
    def plot_all(self) -> None:
        """Produce all 6 figures."""
        self.plot_pca_variance()
        self.plot_elbow()
        self.plot_regime_timeseries()
        self.plot_eigenvalues_zplane()
        self.plot_cluster_amplitude()
        self.plot_pca_scatter()

    def plot_pca_variance(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(self._cum_var)+1), self._cum_var,
                'b.-', markersize=6, linewidth=1.2)
        ax.axvline(self._n_pca, color='r', ls='--',
                   label=f'{self._n_pca} components')
        ax.axhline(self.n_pca_var, color='k', ls='--',
                   label=f'{self.n_pca_var:.0f}%')
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Cumulative Variance (%)')
        ax.set_title('Phi Matrix PCA — Cumulative Variance')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()

    def plot_elbow(self) -> None:
        k_arr = np.arange(1, self.k_max + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_arr, self.inertia, 'b.-', markersize=10, linewidth=1.5,
                label='Inertia')
        ax.plot(self._k_elbow_d2, self.inertia[self._k_elbow_d2-1],
                'rv', markersize=12,
                label=f'2nd-diff  k={self._k_elbow_d2}')
        ax.plot(self._k_elbow_nd, self.inertia[self._k_elbow_nd-1],
                'g^', markersize=12,
                label=f'Normalized  k={self._k_elbow_nd}')
        ax.set_xlabel('Number of Clusters  k')
        ax.set_ylabel('Total Within-Cluster Inertia')
        ax.set_title('Elbow Plot — two auto-detection methods')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()

    def plot_regime_timeseries(self) -> None:
        windows   = self.run.windows
        all_times = np.array([w.time for w in windows])
        k         = self.n_clusters
        cmap      = plt.cm.tab10(np.linspace(0, 0.9, k))

        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        fig.suptitle('Phi Matrix Regime Analysis', fontsize=13, fontweight='bold')

        # Row 1: response channel 0
        axes[0].plot(self.run.time, self.run.datout[:, 0],
                     'k-', linewidth=0.5)
        axes[0].set_ylabel('Response Ch 1')
        axes[0].set_title('Response Signal')
        axes[0].grid(True)

        # Row 2: scatter per cluster
        axes[1].set_title('Cluster Assignment over Time')
        for ci in range(k):
            mask = self.cluster_ids == ci
            t_k  = np.array([windows[wi].time for wi in self.use_idx[mask]])
            axes[1].scatter(t_k, np.full(len(t_k), ci),
                            s=8, color=cmap[ci],
                            label=f'Cluster {ci}  (n={mask.sum()})')
        axes[1].set_yticks(range(k))
        axes[1].set_yticklabels([f'Cluster {ci}' for ci in range(k)])
        axes[1].legend(loc='upper right', fontsize=7)
        axes[1].grid(True)

        # Row 3: dot plot detail
        axes[2].set_title('Cluster Detail')
        for ci in range(k):
            mask = self.cluster_ids == ci
            t_k  = np.array([windows[wi].time for wi in self.use_idx[mask]])
            axes[2].plot(t_k, np.full(len(t_k), ci), '.',
                         color=cmap[ci], markersize=3)
        axes[2].set_yticks(range(k))
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Cluster')
        axes[2].grid(True)

        plt.tight_layout(); plt.show()

    def plot_eigenvalues_zplane(self) -> None:
        dt  = 1.0 / self.run.fs
        fs  = self.run.fs
        k   = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))

        fig, ax = plt.subplots(figsize=(8, 6))
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8,
                label='Unit circle')

        for ci in range(k):
            mean_ev = self.cluster_mean_ev[ci]
            mag     = np.abs(mean_ev)
            ce      = np.log(mean_ev + 1e-300) / dt
            omega_d = np.abs(np.imag(ce))
            sigma   = -np.real(ce)
            omega_n = np.sqrt(omega_d**2 + sigma**2)
            freq    = omega_n / (2 * np.pi)

            good = (
                (np.abs(np.imag(mean_ev)) > self.min_im_frac * mag) &
                (freq > 0.5) & (freq < fs / 2) &
                (mag <= 1.0) & (np.imag(mean_ev) > 0)
            )

            is_minority = ci in self.minority
            label = f'Cluster {ci}' + (' (minority)' if is_minority else '')
            marker = 'x' if is_minority else 'o'
            size   = 60 if is_minority else 80

            ax.scatter(np.real(mean_ev[good]), np.imag(mean_ev[good]),
                       s=size, color=cmap[ci], marker=marker,
                       linewidths=2 if is_minority else 0,
                       label=label, zorder=3)

        ax.set_xlabel('Real'); ax.set_ylabel('Imaginary')
        ax.set_title('Mean Eigenvalues per Cluster — z-plane  (direct averaging)')
        ax.legend(loc='best'); ax.grid(True); ax.set_aspect('equal')
        lim = 1.3; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        plt.tight_layout(); plt.show()

    def plot_cluster_amplitude(self) -> None:
        k    = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))
        win_half = 20
        windows  = self.run.windows

        amp = np.zeros(len(self.use_idx))
        for ii, wi in enumerate(self.use_idx):
            idx_w = windows[wi].sample_idx
            lo = max(0, idx_w - win_half)
            hi = min(self.run.datout.shape[0], idx_w + win_half)
            amp[ii] = np.sqrt(np.mean(self.run.datout[lo:hi, 0]**2))

        fig, ax = plt.subplots(figsize=(8, 5))
        for ci in range(k):
            mask = self.cluster_ids == ci
            ax.hist(amp[mask], bins=25, color=cmap[ci], alpha=0.55,
                    label=f'Cluster {ci}')
        ax.set_xlabel('Local RMS Amplitude  (Ch 1)')
        ax.set_ylabel('Count')
        ax.set_title('Cluster Assignment vs Local Response Amplitude')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()

    def plot_pca_scatter(self) -> None:
        k    = self.n_clusters
        cmap = plt.cm.tab10(np.linspace(0, 0.9, k))

        fig, ax = plt.subplots(figsize=(8, 6))
        for ci in range(k):
            mask = self.cluster_ids == ci
            ax.scatter(self.Phi_pca[mask, 0], self.Phi_pca[mask, 1],
                       s=10, color=cmap[ci], alpha=0.6,
                       label=f'Cluster {ci}  (n={mask.sum()})')
        # Centroids
        for ci in range(k):
            ax.plot(self.centroids_pca[ci, 0], self.centroids_pca[ci, 1],
                    'k+', markersize=16, linewidth=2.5,
                    label='_centroid' if ci > 0 else 'Centroid')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('Phi Matrices in PCA Space  (+ = centroid)')
        ax.legend(loc='best', fontsize=8); ax.grid(True)
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        """Save clustering results to HDF5."""
        p = Path(filepath)
        print(f'\nSaving clusters to {p} ...')

        with h5py.File(str(p), 'w') as f:
            f.attrs['n_clusters'] = self.n_clusters
            f.attrs['rk_mode']    = self.rk_mode
            f.attrs['regime_A']   = self.regime_A
            f.attrs['regime_B']   = self.regime_B
            f.attrs['minority']   = self.minority if self.minority else [-1]

            f.create_dataset('use_idx',       data=self.use_idx)
            f.create_dataset('cluster_ids',   data=self.cluster_ids)
            f.create_dataset('centroids_pca', data=self.centroids_pca)
            f.create_dataset('Phi_pca',       data=self.Phi_pca,
                             compression='gzip')
            f.create_dataset('Phi_vec',       data=self.Phi_vec,
                             compression='gzip')
            f.create_dataset('cluster_times', data=self.cluster_times)
            f.create_dataset('inertia',       data=self.inertia)

            grp = f.create_group('centroid_Phis')
            for ci, cp in enumerate(self.centroid_Phis):
                grp.create_dataset(str(ci), data=cp)

            grp = f.create_group('cluster_mean_ev')
            for ci, ev in enumerate(self.cluster_mean_ev):
                grp.create_dataset(f'{ci}_real', data=np.real(ev))
                grp.create_dataset(f'{ci}_imag', data=np.imag(ev))

        print(f'  Saved {self.n_clusters} clusters.')

    @classmethod
    def load(cls, filepath: str | Path, run: CVARun) -> CVAPhiCluster:
        """Reconstruct from HDF5. Requires the original CVARun object."""
        p = Path(filepath)
        print(f'Loading clusters from {p} ...')

        obj = cls(run)
        with h5py.File(str(p), 'r') as f:
            obj.n_clusters    = int(f.attrs['n_clusters'])
            obj.rk_mode       = int(f.attrs['rk_mode'])
            obj.regime_A      = int(f.attrs['regime_A'])
            obj.regime_B      = int(f.attrs['regime_B'])
            minority_raw      = f.attrs['minority']
            obj.minority      = [int(x) for x in minority_raw if x >= 0]

            obj.use_idx       = f['use_idx'][:]
            obj.cluster_ids   = f['cluster_ids'][:]
            obj.centroids_pca = f['centroids_pca'][:]
            obj.Phi_pca       = f['Phi_pca'][:]
            obj.Phi_vec       = f['Phi_vec'][:]
            obj.cluster_times = f['cluster_times'][:]
            obj.inertia       = f['inertia'][:]

            k = obj.n_clusters
            obj.centroid_Phis = [f['centroid_Phis'][str(ci)][:]
                                  for ci in range(k)]
            obj.cluster_mean_ev = [
                f['cluster_mean_ev'][f'{ci}_real'][:] +
                1j * f['cluster_mean_ev'][f'{ci}_imag'][:]
                for ci in range(k)
            ]

        print(f'  Loaded {obj.n_clusters} clusters.')
        return obj

    def __repr__(self) -> str:
        if self.cluster_ids is None:
            return f'CVAPhiCluster(not fitted)'
        return (f'CVAPhiCluster(k={self.n_clusters}, '
                f'rk={self.rk_mode}, '
                f'n_windows={len(self.use_idx)}, '
                f'regime_A={self.regime_A}, regime_B={self.regime_B})')
