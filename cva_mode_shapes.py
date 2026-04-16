"""
cva_mode_shapes.py
==================
Physical mode shape extraction from CVA results.

Author: Norman / Claude
Date:   2026-04-14
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

from cva_phi_cluster import CVAPhiCluster
from cva_run import CVARun


@dataclass
class ClusterModes:
    cluster_idx: int
    is_minority: bool
    freq:   np.ndarray
    zeta:   np.ndarray
    shapes: np.ndarray   # (n_out, n_modes) complex


class CVAModeShapes:
    def __init__(self, run, clust, min_im_frac=0.1, f_min=0.5,
                 zeta_max=0.5, n_modes_max=6, dof_labels=None):
        self.run         = run
        self.clust       = clust
        self.min_im_frac = min_im_frac
        self.f_min       = f_min
        self.zeta_max    = zeta_max
        self.n_modes_max = n_modes_max
        n_out = run.n_out
        self.dof_labels = dof_labels or [f'Ch {i+1}' for i in range(n_out)]
        self.dof_pos    = np.arange(1, n_out+1)
        self.cluster_modes: List[Optional[ClusterModes]] = []

    # ------------------------------------------------------------------
    def extract(self):
        run   = self.run; clust = self.clust
        dt    = 1.0/run.fs; fs = run.fs
        n_out = run.n_out; lags = run.cfg.lags
        windows = run.windows

        print('\n========================================')
        print('   CVA Physical Mode Shapes')
        print('========================================\n')

        self.cluster_modes = []

        for k in range(clust.n_clusters):
            is_minority = k in clust.minority
            tag = ' [minority]' if is_minority else ''
            win_idx = clust.use_idx[clust.cluster_ids == k]

            all_poles = []; all_shapes = []

            for wi in win_idx:
                wr   = windows[wi]; rk = wr.rank
                indp = wr.indpast
                nn_p1 = indp + lags
                valid = (nn_p1 >= 0) & (nn_p1 < run.datout.shape[0])
                if valid.sum() < rk+1: continue

                nb_out = run.datout[nn_p1[valid], :]
                sp_use = wr.spast[valid, :]
                try:
                    result, _, _, _ = np.linalg.lstsq(sp_use, nb_out, rcond=None)
                    C_local = result.T
                except np.linalg.LinAlgError:
                    continue

                ev_i, V_i = np.linalg.eig(wr.Phi)
                ce_i      = np.log(ev_i+1e-300) / dt
                omega_d_i = np.abs(np.imag(ce_i))
                sigma_i   = -np.real(ce_i)
                omega_n_i = np.sqrt(omega_d_i**2 + sigma_i**2)
                freq_i    = omega_n_i / (2*np.pi)
                zeta_i    = sigma_i / (omega_n_i+1e-12)
                mag_i     = np.abs(ev_i)
                good_i = ((np.abs(np.imag(ev_i)) > self.min_im_frac*mag_i) &
                           (freq_i > self.f_min) & (freq_i < fs/2) &
                           (mag_i <= 1.01) & (np.imag(ev_i) > 0) &
                           (zeta_i > 0) & (zeta_i < self.zeta_max))

                for gi in np.where(good_i)[0]:
                    phi_phys = C_local @ V_i[:, gi]
                    ref = np.argmax(np.abs(phi_phys))
                    if np.abs(phi_phys[ref]) < 1e-12: continue
                    phi_phys = phi_phys / phi_phys[ref]
                    all_poles.append(ev_i[gi])
                    all_shapes.append(phi_phys)

            if len(all_poles) == 0:
                print(f'  Cluster {k}{tag}: no physical modes found')
                self.cluster_modes.append(None); continue

            all_poles  = np.array(all_poles)
            all_shapes = np.array(all_shapes).T
            ce_all   = np.log(all_poles+1e-300) / dt
            freq_all = np.sqrt(np.imag(ce_all)**2 + np.real(ce_all)**2) / (2*np.pi)
            zeta_all = -np.real(ce_all) / (np.abs(ce_all)+1e-12)

            isort       = np.argsort(freq_all)
            freq_sort   = freq_all[isort]
            shapes_sort = all_shapes[:, isort]
            zeta_sort   = zeta_all[isort]

            used = np.zeros(len(freq_sort), dtype=bool)
            mode_freq = []; mode_zeta = []; mode_shape = []

            for mi in range(len(freq_sort)):
                if used[mi]: continue
                fc  = freq_sort[mi]
                bw  = max(3.0, 0.08*fc)
                grp = (np.abs(freq_sort-fc) < bw) & ~used
                used[grp] = True
                ms_mean = np.mean(shapes_sort[:, grp], axis=1)
                ref = np.argmax(np.abs(ms_mean))
                if np.abs(ms_mean[ref]) < 1e-12: continue
                ms_mean = ms_mean / ms_mean[ref]
                mode_freq.append(float(np.mean(freq_sort[grp])))
                mode_zeta.append(float(np.mean(zeta_sort[grp])))
                mode_shape.append(ms_mean)

            if not mode_shape:
                self.cluster_modes.append(None); continue

            mode_freq  = np.array(mode_freq)
            mode_zeta  = np.array(mode_zeta)
            mode_shape = np.column_stack(mode_shape)

            cm = ClusterModes(cluster_idx=k, is_minority=is_minority,
                              freq=mode_freq, zeta=mode_zeta, shapes=mode_shape)
            self.cluster_modes.append(cm)

            print(f'  Cluster {k}{tag} — {len(mode_freq)} modes:')
            for mi, (f, z) in enumerate(zip(mode_freq, mode_zeta)):
                print(f'    Mode {mi+1}: {f:.2f} Hz   zeta={z:.4f}')
            print()

    # ------------------------------------------------------------------
    def _savefig(self, fig, save_path, dpi):
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}')

    def _subplot_grid(self, n_plot, figsize_w=14, h_per_row=2.5):
        ncols = 3; nrows = int(np.ceil(n_plot/ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figsize_w, 2+h_per_row*nrows),
                                 squeeze=False)
        for mi in range(n_plot, nrows*ncols):
            axes[mi//ncols][mi%ncols].set_visible(False)
        return fig, axes, ncols

    def plot_regime_comparison(self, save_path=None, dpi=150):
        clust = self.clust
        mA    = self.cluster_modes[clust.regime_A]
        mB    = self.cluster_modes[clust.regime_B]
        if mA is None or mB is None:
            print('Cannot compare — one or both regimes have no modes.'); return
        n_plot = min(mA.shapes.shape[1], mB.shapes.shape[1], self.n_modes_max)
        fig, axes, ncols = self._subplot_grid(n_plot)
        fig.suptitle(f'Physical Mode Shapes — Regime A (Cluster {clust.regime_A})'
                     f' vs B (Cluster {clust.regime_B})',
                     fontsize=12, fontweight='bold')
        for mi in range(n_plot):
            ax    = axes[mi//ncols][mi%ncols]
            phi_A = mA.shapes[:, mi]; phi_B = mB.shapes[:, mi]
            pct   = 100*(mB.freq[mi]-mA.freq[mi])/(mA.freq[mi]+1e-12)
            ax.plot(self.dof_pos, np.imag(phi_A), 'b.-', ms=12, lw=2,
                    label=f'RegA  {mA.freq[mi]:.2f} Hz')
            ax.plot(self.dof_pos, np.imag(phi_B), 'r.-', ms=12, lw=2,
                    label=f'RegB  {mB.freq[mi]:.2f} Hz')
            ax.axhline(0, color='k', ls='--', lw=0.8)
            ax.set_xticks(self.dof_pos); ax.set_xticklabels(self.dof_labels)
            ax.set_ylabel('Imag (deflection)')
            ax.set_title(f'Mode {mi+1}\nA: {mA.freq[mi]:.2f} Hz  '
                         f'B: {mB.freq[mi]:.2f} Hz  Δ={pct:+.1f}%')
            ax.legend(fontsize=7, loc='best'); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

    def plot_per_cluster(self, save_path=None, dpi=150):
        for k, cm in enumerate(self.cluster_modes):
            if cm is None: continue
            n_plot = min(cm.shapes.shape[1], self.n_modes_max)
            tag    = ' [minority]' if cm.is_minority else ''
            fig, axes, ncols = self._subplot_grid(n_plot)
            fig.suptitle(f'Cluster {k}{tag} — Physical Mode Shapes  (Real / Imag / Mag)',
                         fontsize=12, fontweight='bold')
            for mi in range(n_plot):
                ax  = axes[mi//ncols][mi%ncols]
                phi = cm.shapes[:, mi]
                ax.plot(self.dof_pos, np.real(phi), 'b.-', ms=12, lw=1.8, label='Real')
                ax.plot(self.dof_pos, np.imag(phi), 'r.-', ms=12, lw=1.8, label='Imag')
                ax.plot(self.dof_pos, np.abs(phi),  'k--', ms=8,  lw=1.2, label='Magnitude')
                ax.axhline(0, color='k', ls=':', lw=0.8)
                ax.set_xticks(self.dof_pos); ax.set_xticklabels(self.dof_labels)
                ax.set_ylabel('Amplitude (normalized)')
                ax.set_title(f'Mode {mi+1}:  {cm.freq[mi]:.2f} Hz   ζ={cm.zeta[mi]:.4f}')
                ax.legend(fontsize=7, loc='best'); ax.grid(True)
            sp = (Path(str(save_path).replace('.png', f'_cluster{k}.png'))
                  if save_path else None)
            plt.tight_layout(); self._savefig(fig, sp, dpi); plt.show()

    def plot_magnitude_overlay(self, save_path=None, dpi=150):
        clust   = self.clust
        non_min = [k for k in range(clust.n_clusters) if k not in clust.minority]
        if len(non_min) < 2:
            print('Need at least 2 non-minority clusters.'); return
        n_modes = min(cm.shapes.shape[1]
                      for k in non_min
                      if (cm := self.cluster_modes[k]) is not None)
        n_plot  = min(n_modes, self.n_modes_max)
        cmap    = plt.cm.tab10(np.linspace(0, 0.9, clust.n_clusters))
        fig, axes, ncols = self._subplot_grid(n_plot)
        fig.suptitle('Mode Shape Magnitudes — All Non-Minority Clusters Overlaid',
                     fontsize=12, fontweight='bold')
        for mi in range(n_plot):
            ax = axes[mi//ncols][mi%ncols]
            for k in non_min:
                cm = self.cluster_modes[k]
                if cm is None or mi >= cm.shapes.shape[1]: continue
                ax.plot(self.dof_pos, np.abs(cm.shapes[:, mi]), '.-',
                        color=cmap[k], ms=12, lw=1.8,
                        label=f'Cluster {k}  {cm.freq[mi]:.2f} Hz')
            ax.set_xticks(self.dof_pos); ax.set_xticklabels(self.dof_labels)
            ax.set_ylabel('|Magnitude| (normalized)')
            ax.set_title(f'Mode {mi+1}')
            ax.legend(fontsize=7, loc='best'); ax.grid(True)
        plt.tight_layout(); self._savefig(fig, save_path, dpi); plt.show()

    def plot_all(self, save_dir=None, dpi=150):
        def sp(name):
            return Path(save_dir)/f'{name}.png' if save_dir else None
        self.plot_regime_comparison(save_path=sp('modes_regime_comparison'), dpi=dpi)
        self.plot_per_cluster(save_path=sp('modes_per_cluster'), dpi=dpi)
        self.plot_magnitude_overlay(save_path=sp('modes_magnitude_overlay'), dpi=dpi)

    # ------------------------------------------------------------------
    def save(self, filepath):
        p = Path(filepath)
        print(f'\nSaving mode shapes to {p} ...')
        with h5py.File(str(p), 'w') as f:
            f.attrs['n_clusters']  = len(self.cluster_modes)
            f.attrs['min_im_frac'] = self.min_im_frac
            f.attrs['f_min']       = self.f_min
            f.attrs['zeta_max']    = self.zeta_max
            g = f.create_group('cluster_modes')
            for k, cm in enumerate(self.cluster_modes):
                if cm is None:
                    g.create_group(str(k)).attrs['empty'] = True; continue
                cg = g.create_group(str(k))
                cg.attrs['cluster_idx'] = cm.cluster_idx
                cg.attrs['is_minority'] = cm.is_minority
                cg.create_dataset('freq', data=cm.freq)
                cg.create_dataset('zeta', data=cm.zeta)
                cg.create_dataset('shapes_real', data=np.real(cm.shapes), compression='gzip')
                cg.create_dataset('shapes_imag', data=np.imag(cm.shapes), compression='gzip')
        print(f'  Saved {len(self.cluster_modes)} clusters.')

    @classmethod
    def load(cls, filepath, run, clust):
        p = Path(filepath)
        print(f'Loading mode shapes from {p} ...')
        obj = cls(run, clust)
        with h5py.File(str(p), 'r') as f:
            obj.min_im_frac = float(f.attrs['min_im_frac'])
            obj.f_min       = float(f.attrs['f_min'])
            obj.zeta_max    = float(f.attrs['zeta_max'])
            g = f['cluster_modes']; n_k = int(f.attrs['n_clusters'])
            obj.cluster_modes = []
            for k in range(n_k):
                cg = g[str(k)]
                if cg.attrs.get('empty', False):
                    obj.cluster_modes.append(None); continue
                shapes = cg['shapes_real'][:] + 1j*cg['shapes_imag'][:]
                obj.cluster_modes.append(ClusterModes(
                    cluster_idx=int(cg.attrs['cluster_idx']),
                    is_minority=bool(cg.attrs['is_minority']),
                    freq=cg['freq'][:], zeta=cg['zeta'][:], shapes=shapes))
        print(f'  Loaded {len(obj.cluster_modes)} clusters.')
        return obj

    def __repr__(self):
        if not self.cluster_modes:
            return 'CVAModeShapes(not extracted)'
        n = [cm.shapes.shape[1] if cm else 0 for cm in self.cluster_modes]
        return f'CVAModeShapes(k={len(self.cluster_modes)}, modes_per_cluster={n})'
