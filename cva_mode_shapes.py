"""
cva_mode_shapes.py
==================
Python translation of CVAModeShapes_1.m

Extracts physical mode shapes from CVA state-space results by projecting
state-space eigenvectors through a locally fitted output matrix C_local.

Method
------
For each CVA window:
  1. Compute C_local by regressing neighbor outputs onto neighbor states
     C_local = (spast \\ datout[neighbor_times])^T   (n_out x rank)
  2. Compute eigensystem of Phi: [V, D] = eig(Phi)
  3. Project each physical eigenvector: phi_physical = C_local @ v
  4. Filter to physical complex modes (frequency, damping, stability)
  5. Normalize to unit magnitude at largest-magnitude DOF

Then group poles by frequency band and average shapes within each cluster.

Usage
-----
    from cva_mode_shapes import CVAModeShapes

    ms = CVAModeShapes(run, clust)
    ms.extract()
    ms.plot_regime_comparison()
    ms.plot_per_cluster()
    ms.plot_magnitude_overlay()
    ms.save('s16d240_modes.h5')

Dependencies
------------
    numpy, scipy, matplotlib, h5py
    cva_run.py, cva_phi_cluster.py on same path

Author: Norman / Claude
Date:   2026-04-13
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

from cva_phi_cluster import CVAPhiCluster
from cva_run import CVARun


# =============================================================================
# ClusterModes — results for one cluster
# =============================================================================
@dataclass
class ClusterModes:
    """Modal parameters and shapes for one cluster."""
    cluster_idx: int
    is_minority: bool
    freq:        np.ndarray          # (n_modes,)  Hz
    zeta:        np.ndarray          # (n_modes,)  damping ratio
    shapes:      np.ndarray          # (n_out, n_modes) complex


# =============================================================================
# CVAModeShapes
# =============================================================================
class CVAModeShapes:
    """
    Extract physical mode shapes from CVA results.

    Parameters
    ----------
    run          : CVARun   — completed CVA run
    clust        : CVAPhiCluster — completed clustering
    min_im_frac  : min |Im(ev)| / |ev| to count as complex (default 0.1)
    f_min        : minimum physical frequency Hz (default 0.5)
    zeta_max     : maximum damping ratio (default 0.5)
    n_modes_max  : max modes to plot per cluster (default 6)
    dof_labels   : list of channel label strings (None = auto)
    """

    def __init__(
        self,
        run:         CVARun,
        clust:       CVAPhiCluster,
        min_im_frac: float = 0.1,
        f_min:       float = 0.5,
        zeta_max:    float = 0.5,
        n_modes_max: int   = 6,
        dof_labels:  Optional[List[str]] = None,
    ):
        self.run         = run
        self.clust       = clust
        self.min_im_frac = min_im_frac
        self.f_min       = f_min
        self.zeta_max    = zeta_max
        self.n_modes_max = n_modes_max

        n_out = run.n_out
        self.dof_labels = dof_labels or [f'Ch {i+1}' for i in range(n_out)]
        self.dof_pos    = np.arange(1, n_out + 1)

        # Populated by extract()
        self.cluster_modes: List[Optional[ClusterModes]] = []

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------
    def extract(self) -> None:
        """
        Main extraction loop — mirrors MATLAB CVAModeShapes_1.m.
        Populates self.cluster_modes list.
        """
        run   = self.run
        clust = self.clust

        dt      = 1.0 / run.fs
        fs      = run.fs
        n_out   = run.n_out
        lags    = run.cfg.lags
        windows = run.windows

        print('\n========================================')
        print('   CVA Physical Mode Shapes')
        print('========================================\n')
        print('Extracting physical mode shapes per cluster...\n')

        n_clusters   = clust.n_clusters
        self.cluster_modes = []

        for k in range(n_clusters):
            is_minority = k in clust.minority
            tag         = ' [minority]' if is_minority else ''

            # Indices into run.windows for this cluster
            win_mask = clust.cluster_ids == k
            win_idx  = clust.use_idx[win_mask]
            n_win    = len(win_idx)

            all_poles  = []
            all_shapes = []

            for wi in win_idx:
                wr    = windows[wi]
                rk    = wr.rank
                Phi_i = wr.Phi
                sp    = wr.spast      # (neighbors, rk)
                indp  = wr.indpast    # (neighbors,) indices into past_db

                # --- Build C_local ---
                # neighbor output times = indpast + lags - 1
                # (indpast are 0-based indices into past_db; each past_db row
                #  corresponds to data sample indpast + lags - 1)
                nn_p1 = indp + lags - 1    # absolute sample indices (0-based)
                valid = (nn_p1 >= 0) & (nn_p1 < run.datout.shape[0])

                if valid.sum() < rk + 1:
                    continue

                nb_out = run.datout[nn_p1[valid], :]   # (n_valid, n_out)
                sp_use = sp[valid, :]                   # (n_valid, rk)

                # C_local = (sp_use \ nb_out)^T  →  lstsq
                # sp_use @ C_local^T ≈ nb_out
                # C_local^T = lstsq(sp_use, nb_out) → (rk, n_out)
                # C_local   = result^T              → (n_out, rk)
                try:
                    result, _, _, _ = np.linalg.lstsq(sp_use, nb_out,
                                                       rcond=None)
                    C_local = result.T    # (n_out, rk)
                except np.linalg.LinAlgError:
                    continue

                # --- Eigensystem ---
                ev_i, V_i = np.linalg.eig(Phi_i)
                # Note: numpy eig returns (eigenvalues, eigenvectors as columns)

                ce_i      = np.log(ev_i + 1e-300) / dt
                omega_d_i = np.abs(np.imag(ce_i))
                sigma_i   = -np.real(ce_i)
                omega_n_i = np.sqrt(omega_d_i**2 + sigma_i**2)
                freq_i    = omega_n_i / (2 * np.pi)
                zeta_i    = sigma_i / (omega_n_i + 1e-12)
                mag_i     = np.abs(ev_i)

                good_i = (
                    (np.abs(np.imag(ev_i)) > self.min_im_frac * mag_i) &
                    (freq_i > self.f_min) & (freq_i < fs / 2) &
                    (mag_i <= 1.01) &
                    (np.imag(ev_i) > 0) &
                    (zeta_i > 0) & (zeta_i < self.zeta_max)
                )

                idx_good = np.where(good_i)[0]
                if len(idx_good) == 0:
                    continue

                # --- Project each eigenvector through C_local ---
                for gi in idx_good:
                    v_state  = V_i[:, gi]           # (rk,) complex
                    phi_phys = C_local @ v_state     # (n_out,) complex

                    # Normalize to unit magnitude at largest-magnitude DOF
                    ref = np.argmax(np.abs(phi_phys))
                    if np.abs(phi_phys[ref]) < 1e-12:
                        continue
                    phi_phys = phi_phys / phi_phys[ref]

                    all_poles.append(ev_i[gi])
                    all_shapes.append(phi_phys)

            # ----------------------------------------------------------
            if len(all_poles) == 0:
                print(f'  Cluster {k}{tag}: no physical modes found')
                self.cluster_modes.append(None)
                continue

            all_poles  = np.array(all_poles)            # (N,) complex
            all_shapes = np.array(all_shapes).T         # (n_out, N) complex

            # --- Group poles by frequency, average shapes ---
            ce_all   = np.log(all_poles + 1e-300) / dt
            od_all   = np.abs(np.imag(ce_all))
            sig_all  = -np.real(ce_all)
            on_all   = np.sqrt(od_all**2 + sig_all**2)
            freq_all = on_all / (2 * np.pi)
            zeta_all = sig_all / (on_all + 1e-12)

            isort       = np.argsort(freq_all)
            freq_sort   = freq_all[isort]
            shapes_sort = all_shapes[:, isort]
            zeta_sort   = zeta_all[isort]

            # Bin into frequency groups
            used       = np.zeros(len(freq_sort), dtype=bool)
            mode_freq  = []
            mode_zeta  = []
            mode_shape = []

            for mi in range(len(freq_sort)):
                if used[mi]:
                    continue
                fc  = freq_sort[mi]
                bw = max(3.0, 0.08 * fc)
                grp = (np.abs(freq_sort - fc) < bw) & ~used
                used[grp] = True

                mean_shape = np.mean(shapes_sort[:, grp], axis=1)
                ref = np.argmax(np.abs(mean_shape))
                if np.abs(mean_shape[ref]) < 1e-12:
                    continue
                mean_shape = mean_shape / mean_shape[ref]

                mode_freq.append(float(np.mean(freq_sort[grp])))
                mode_zeta.append(float(np.mean(zeta_sort[grp])))
                mode_shape.append(mean_shape)

            mode_freq  = np.array(mode_freq)
            mode_zeta  = np.array(mode_zeta)
            mode_shape = np.column_stack(mode_shape) if mode_shape else \
                         np.zeros((n_out, 0), dtype=complex)

            cm = ClusterModes(
                cluster_idx = k,
                is_minority = is_minority,
                freq        = mode_freq,
                zeta        = mode_zeta,
                shapes      = mode_shape,
            )
            self.cluster_modes.append(cm)

            print(f'  Cluster {k}{tag} — {len(mode_freq)} modes:')
            for mi, (f, z) in enumerate(zip(mode_freq, mode_zeta)):
                print(f'    Mode {mi+1}: {f:.2f} Hz   zeta={z:.4f}')
            print()

    # ------------------------------------------------------------------
    # plot_regime_comparison  (Figure 1)
    # ------------------------------------------------------------------
    def plot_regime_comparison(self) -> None:
        """Imaginary part of mode shapes — regime A vs B side by side."""
        clust = self.clust
        mA    = self.cluster_modes[clust.regime_A]
        mB    = self.cluster_modes[clust.regime_B]

        if mA is None or mB is None:
            print('Cannot compare regimes — one or both have no modes.')
            return

        n_plot = min(mA.shapes.shape[1], mB.shapes.shape[1], self.n_modes_max)
        ncols  = 3
        nrows  = int(np.ceil(n_plot / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, 2 + 2.5 * nrows),
                                 squeeze=False)
        fig.suptitle(
            f'Physical Mode Shapes — Regime A (Cluster {clust.regime_A})'
            f' vs B (Cluster {clust.regime_B})',
            fontsize=12, fontweight='bold')

        for mi in range(n_plot):
            ax     = axes[mi // ncols][mi % ncols]
            phi_A  = mA.shapes[:, mi]
            phi_B  = mB.shapes[:, mi]
            pct    = 100 * (mB.freq[mi] - mA.freq[mi]) / (mA.freq[mi] + 1e-12)

            ax.plot(self.dof_pos, np.imag(phi_A), 'b.-',
                    markersize=12, linewidth=2,
                    label=f'RegA  {mA.freq[mi]:.2f} Hz')
            ax.plot(self.dof_pos, np.imag(phi_B), 'r.-',
                    markersize=12, linewidth=2,
                    label=f'RegB  {mB.freq[mi]:.2f} Hz')
            ax.axhline(0, color='k', ls='--', linewidth=0.8)
            ax.set_xticks(self.dof_pos)
            ax.set_xticklabels(self.dof_labels)
            ax.set_ylabel('Imag (deflection)')
            ax.set_title(
                f'Mode {mi+1}\nA: {mA.freq[mi]:.2f} Hz  '
                f'B: {mB.freq[mi]:.2f} Hz  Δ={pct:+.1f}%')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True)

        # Hide unused subplots
        for mi in range(n_plot, nrows * ncols):
            axes[mi // ncols][mi % ncols].set_visible(False)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # plot_per_cluster  (Figure 2)
    # ------------------------------------------------------------------
    def plot_per_cluster(self) -> None:
        """Real, imaginary, and magnitude for each cluster."""
        for k, cm in enumerate(self.cluster_modes):
            if cm is None:
                continue

            n_plot = min(cm.shapes.shape[1], self.n_modes_max)
            ncols  = 3
            nrows  = int(np.ceil(n_plot / ncols))
            tag    = ' [minority]' if cm.is_minority else ''

            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(12, 2 + 2.5 * nrows),
                                     squeeze=False)
            fig.suptitle(
                f'Cluster {k}{tag} — Physical Mode Shapes  (Real / Imag / Mag)',
                fontsize=12, fontweight='bold')

            for mi in range(n_plot):
                ax  = axes[mi // ncols][mi % ncols]
                phi = cm.shapes[:, mi]

                ax.plot(self.dof_pos, np.real(phi), 'b.-',
                        markersize=12, linewidth=1.8, label='Real')
                ax.plot(self.dof_pos, np.imag(phi), 'r.-',
                        markersize=12, linewidth=1.8, label='Imag')
                ax.plot(self.dof_pos, np.abs(phi),  'k--',
                        markersize=8,  linewidth=1.2, label='Magnitude')
                ax.axhline(0, color='k', ls=':', linewidth=0.8)
                ax.set_xticks(self.dof_pos)
                ax.set_xticklabels(self.dof_labels)
                ax.set_ylabel('Amplitude (normalized)')
                ax.set_title(
                    f'Mode {mi+1}:  {cm.freq[mi]:.2f} Hz'
                    f'   ζ={cm.zeta[mi]:.4f}')
                ax.legend(fontsize=7, loc='best')
                ax.grid(True)

            for mi in range(n_plot, nrows * ncols):
                axes[mi // ncols][mi % ncols].set_visible(False)

            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # plot_magnitude_overlay  (Figure 3)
    # ------------------------------------------------------------------
    def plot_magnitude_overlay(self) -> None:
        """Magnitude of mode shapes — all non-minority clusters overlaid."""
        clust       = self.clust
        non_min     = [k for k in range(clust.n_clusters)
                       if k not in clust.minority]

        if len(non_min) < 2:
            print('Need at least 2 non-minority clusters for overlay.')
            return

        n_modes_common = min(
            self.cluster_modes[k].shapes.shape[1]
            for k in non_min
            if self.cluster_modes[k] is not None
        )
        n_plot = min(n_modes_common, self.n_modes_max)
        ncols  = 3
        nrows  = int(np.ceil(n_plot / ncols))
        cmap   = plt.cm.tab10(np.linspace(0, 0.9, clust.n_clusters))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(12, 2 + 2.5 * nrows),
                                 squeeze=False)
        fig.suptitle('Mode Shape Magnitudes — All Non-Minority Clusters Overlaid',
                     fontsize=12, fontweight='bold')

        for mi in range(n_plot):
            ax = axes[mi // ncols][mi % ncols]
            for k in non_min:
                cm = self.cluster_modes[k]
                if cm is None or mi >= cm.shapes.shape[1]:
                    continue
                phi = cm.shapes[:, mi]
                ax.plot(self.dof_pos, np.abs(phi), '.-',
                        color=cmap[k], markersize=12, linewidth=1.8,
                        label=f'Cluster {k}  {cm.freq[mi]:.2f} Hz')
            ax.set_xticks(self.dof_pos)
            ax.set_xticklabels(self.dof_labels)
            ax.set_ylabel('|Magnitude| (normalized)')
            ax.set_title(f'Mode {mi+1}')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True)

        for mi in range(n_plot, nrows * ncols):
            axes[mi // ncols][mi % ncols].set_visible(False)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # plot_all
    # ------------------------------------------------------------------
    def plot_all(self) -> None:
        """Produce all three figures."""
        self.plot_regime_comparison()
        self.plot_per_cluster()
        self.plot_magnitude_overlay()

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        """Save mode shape results to HDF5."""
        p = Path(filepath)
        print(f'\nSaving mode shapes to {p} ...')

        with h5py.File(str(p), 'w') as f:
            f.attrs['n_clusters']  = len(self.cluster_modes)
            f.attrs['min_im_frac'] = self.min_im_frac
            f.attrs['f_min']       = self.f_min
            f.attrs['zeta_max']    = self.zeta_max

            grp = f.create_group('cluster_modes')
            for k, cm in enumerate(self.cluster_modes):
                if cm is None:
                    grp.create_group(str(k)).attrs['empty'] = True
                    continue
                cg = grp.create_group(str(k))
                cg.attrs['cluster_idx'] = cm.cluster_idx
                cg.attrs['is_minority'] = cm.is_minority
                cg.create_dataset('freq',         data=cm.freq)
                cg.create_dataset('zeta',         data=cm.zeta)
                cg.create_dataset('shapes_real',  data=np.real(cm.shapes),
                                  compression='gzip')
                cg.create_dataset('shapes_imag',  data=np.imag(cm.shapes),
                                  compression='gzip')

        print(f'  Saved {len(self.cluster_modes)} clusters.')

    @classmethod
    def load(cls, filepath: str | Path,
             run: CVARun, clust: CVAPhiCluster) -> CVAModeShapes:
        """Reconstruct from HDF5."""
        p = Path(filepath)
        print(f'Loading mode shapes from {p} ...')

        obj = cls(run, clust)
        with h5py.File(str(p), 'r') as f:
            obj.min_im_frac = float(f.attrs['min_im_frac'])
            obj.f_min       = float(f.attrs['f_min'])
            obj.zeta_max    = float(f.attrs['zeta_max'])

            grp = f['cluster_modes']
            n_k = int(f.attrs['n_clusters'])
            obj.cluster_modes = []
            for k in range(n_k):
                cg = grp[str(k)]
                if cg.attrs.get('empty', False):
                    obj.cluster_modes.append(None)
                    continue
                shapes = (cg['shapes_real'][:] +
                          1j * cg['shapes_imag'][:])
                obj.cluster_modes.append(ClusterModes(
                    cluster_idx = int(cg.attrs['cluster_idx']),
                    is_minority = bool(cg.attrs['is_minority']),
                    freq        = cg['freq'][:],
                    zeta        = cg['zeta'][:],
                    shapes      = shapes,
                ))

        print(f'  Loaded {len(obj.cluster_modes)} clusters.')
        return obj

    def __repr__(self) -> str:
        if not self.cluster_modes:
            return 'CVAModeShapes(not extracted)'
        n_modes = [cm.shapes.shape[1] if cm else 0
                   for cm in self.cluster_modes]
        return (f'CVAModeShapes(k={len(self.cluster_modes)}, '
                f'modes_per_cluster={n_modes})')
