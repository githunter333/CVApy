#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:51:59 2026

@author: nhunterjr
"""

"""
cva_diagnostics.py
==================
Diagnostic and visualization functions for CVA pipeline results.

All functions accept CVARun, CVAPhiCluster, and CVAModeShapes objects
as explicit arguments — no reliance on workspace namespace.

Functions
---------
plot_nonlinearity_localization  — rel displacement vs frequency shift scatter
plot_frequency_shift_summary    — bar chart of frequency shifts per mode
plot_pole_trajectories          — z-plane pole positions across multiple runs
compare_significance            — overlay significance plots for two runs
print_modal_summary             — tabular summary of modes per cluster

Usage
-----
    from cva_diagnostics import (plot_nonlinearity_localization,
                                  plot_frequency_shift_summary,
                                  print_modal_summary)

    plot_nonlinearity_localization(ms, clust, nl_ch_pair=(2, 3))
    plot_frequency_shift_summary(ms, clust)
    print_modal_summary(ms, clust)

Author: Norman / Claude
Date:   2026-04-13
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Local imports — only used for type hints, not required at runtime
try:
    from cva_run         import CVARun
    from cva_phi_cluster import CVAPhiCluster
    from cva_mode_shapes import CVAModeShapes
except ImportError:
    CVARun = CVAPhiCluster = CVAModeShapes = object


# =============================================================================
# plot_nonlinearity_localization
# =============================================================================
def plot_nonlinearity_localization(
    ms:          CVAModeShapes,
    clust:       CVAPhiCluster,
    nl_ch_pair:  Tuple[int, int] = (2, 3),
    freq_max:    float = 0.0,
    title_suffix: str = '',
) -> None:
    """
    Scatter plot of Ch_i - Ch_j relative displacement vs frequency shift.

    Tests whether modes with large relative displacement across the
    nonlinear element show larger frequency shifts between regimes.

    Parameters
    ----------
    ms          : CVAModeShapes object (after extract())
    clust       : CVAPhiCluster object (after fit())
    nl_ch_pair  : (i, j) 0-based channel indices of the nonlinear element
                  default (2, 3) = between Ch3 and Ch4
    freq_max    : upper frequency limit for plot (0 = auto)
    title_suffix: appended to plot title e.g. 'r6dofcubic2'
    """
    rA = clust.regime_A
    rB = clust.regime_B
    cmA = ms.cluster_modes[rA]
    cmB = ms.cluster_modes[rB]

    if cmA is None or cmB is None:
        print('Cannot plot — one or both regimes have no modes.')
        return

    ci, cj    = nl_ch_pair
    n_common  = min(cmA.shapes.shape[1], cmB.shapes.shape[1])

    rel_disp   = np.zeros(n_common)
    freq_shift = np.zeros(n_common)
    freqs      = np.zeros(n_common)

    for mi in range(n_common):
        phi_A = cmA.shapes[:, mi]
        rel_disp[mi]   = abs(abs(phi_A[ci]) - abs(phi_A[cj]))
        freq_shift[mi] = 100 * (cmA.freq[mi] - cmB.freq[mi]) / \
                         (cmB.freq[mi] + 1e-12)
        freqs[mi] = cmA.freq[mi]

    # Filter by freq_max
    if freq_max > 0:
        mask = freqs <= freq_max
        rel_disp   = rel_disp[mask]
        freq_shift = freq_shift[mask]
        freqs      = freqs[mask]

    title = f'CVA Nonlinearity Localization — Ch{ci+1}-Ch{cj+1}'
    if title_suffix:
        title += f'  ({title_suffix})'

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Left: rel_disp vs freq_shift, colored by frequency
    sc = axes[0].scatter(rel_disp, freq_shift, c=freqs,
                         cmap='viridis', s=50, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=axes[0], label='Frequency (Hz)')
    axes[0].axhline(0, color='k', ls='--', lw=0.8)
    axes[0].set_xlabel(f'Ch{ci+1}-Ch{cj+1} Relative Displacement (normalized)')
    axes[0].set_ylabel('Frequency Shift A-B (%)')
    axes[0].set_title('Nonlinearity Localization Test\nRel. disp vs freq shift')
    axes[0].grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(rel_disp) > 2:
        corr = np.corrcoef(rel_disp, freq_shift)[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}',
                     transform=axes[0].transAxes,
                     fontsize=10, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: freq vs freq_shift, colored by rel_disp
    sc2 = axes[1].scatter(freqs, freq_shift, c=rel_disp,
                          cmap='plasma', s=50, alpha=0.8, zorder=3)
    plt.colorbar(sc2, ax=axes[1], label=f'Ch{ci+1}-Ch{cj+1} rel disp')
    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Frequency Shift A-B (%)')
    axes[1].set_title('Frequency Shift vs Mode Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# plot_frequency_shift_summary
# =============================================================================
def plot_frequency_shift_summary(
    ms:          CVAModeShapes,
    clust:       CVAPhiCluster,
    freq_max:    float = 0.0,
    title_suffix: str = '',
) -> None:
    """
    Horizontal bar chart showing frequency shift (%) per identified mode.

    Parameters
    ----------
    ms          : CVAModeShapes object
    clust       : CVAPhiCluster object
    freq_max    : upper frequency limit (0 = auto)
    title_suffix: appended to plot title
    """
    rA  = clust.regime_A
    rB  = clust.regime_B
    cmA = ms.cluster_modes[rA]
    cmB = ms.cluster_modes[rB]

    if cmA is None or cmB is None:
        print('Cannot plot — one or both regimes have no modes.')
        return

    n_common  = min(cmA.shapes.shape[1], cmB.shapes.shape[1])
    shifts    = np.array([100 * (cmA.freq[mi] - cmB.freq[mi]) /
                          (cmB.freq[mi] + 1e-12) for mi in range(n_common)])
    freqs     = cmA.freq[:n_common]
    labels    = [f'{f:.1f} Hz' for f in freqs]

    if freq_max > 0:
        mask   = freqs <= freq_max
        shifts = shifts[mask]
        labels = [l for l, m in zip(labels, mask) if m]

    colors = ['steelblue' if s >= 0 else 'salmon' for s in shifts]

    fig, ax = plt.subplots(figsize=(7, max(4, len(shifts) * 0.3)))
    y_pos   = np.arange(len(shifts))
    ax.barh(y_pos, shifts, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Frequency Shift A-B (%)')
    ax.set_title(
        f'Frequency Shift per Mode\n'
        f'Regime A (Cluster {rA}) vs B (Cluster {rB})'
        + (f'  ({title_suffix})' if title_suffix else ''),
        fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# print_modal_summary
# =============================================================================
def print_modal_summary(
    ms:    CVAModeShapes,
    clust: CVAPhiCluster,
    freq_max: float = 0.0,
) -> None:
    """
    Print a tabular summary of identified modes for each cluster.

    Parameters
    ----------
    ms       : CVAModeShapes object
    clust    : CVAPhiCluster object
    freq_max : upper frequency limit (0 = show all)
    """
    rA  = clust.regime_A
    rB  = clust.regime_B
    cmA = ms.cluster_modes[rA]
    cmB = ms.cluster_modes[rB]

    if cmA is None or cmB is None:
        print('One or both regimes have no modes.')
        return

    n_common = min(cmA.shapes.shape[1], cmB.shapes.shape[1])

    print(f'\n{"="*72}')
    print(f'  Modal Summary — Regime A (Cluster {rA}) vs Regime B (Cluster {rB})')
    print(f'{"="*72}')
    print(f'  {"Mode":>4}  {"RegA (Hz)":>10}  {"ζA":>7}  '
          f'{"RegB (Hz)":>10}  {"ζB":>7}  {"Δf (%)":>8}')
    print(f'  {"-"*66}')

    for mi in range(n_common):
        fA  = cmA.freq[mi]
        fB  = cmB.freq[mi]
        zA  = cmA.zeta[mi]
        zB  = cmB.zeta[mi]
        pct = 100 * (fA - fB) / (fB + 1e-12)

        if freq_max > 0 and fA > freq_max:
            break

        marker = ' <--' if abs(pct) > 5 else ''
        print(f'  {mi+1:>4}  {fA:>10.2f}  {zA:>7.4f}  '
              f'{fB:>10.2f}  {zB:>7.4f}  {pct:>+8.2f}%{marker}')

    print(f'{"="*72}\n')


# =============================================================================
# compare_significance
# =============================================================================
def compare_significance(
    run1:        CVARun,
    run2:        CVARun,
    label1:      str = 'Run 1',
    label2:      str = 'Run 2',
    freq_max:    float = 0.0,
) -> None:
    """
    Overlay significance histogram for two CVARuns on the same axes.
    Useful for comparing linear vs nonlinear states or two amplitude levels.

    Parameters
    ----------
    run1, run2  : CVARun objects with completed eigenvalue_significance()
    label1/2    : legend labels
    freq_max    : upper frequency limit (0 = auto)
    """
    for run in (run1, run2):
        if not run.sig:
            run.eigenvalue_significance()

    f_max = freq_max if freq_max > 0 else max(
        run1.cfg.freq_max, run2.cfg.freq_max)

    fig, ax = plt.subplots(figsize=(12, 5))

    for run, label, color, alpha in [
        (run1, label1, 'steelblue', 0.6),
        (run2, label2, 'darkorange', 0.5),
    ]:
        sig = run.sig
        ax.bar(sig['f_centers'], sig['hist_mag'],
               width=(sig['f_centers'][1] - sig['f_centers'][0]),
               color=color, alpha=alpha, label=label)
        if len(sig['peak_freqs']) > 0:
            ax.scatter(sig['peak_freqs'], sig['peak_mags'],
                       marker='v', color=color, s=60, zorder=5)

    ax.set_xlim(0, f_max)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Cumulative Magnitude')
    ax.set_title(f'Significance Comparison: {label1} vs {label2}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# plot_pole_trajectories
# =============================================================================
def plot_pole_trajectories(
    runs:   List[CVARun],
    clusts: List[CVAPhiCluster],
    labels: Optional[List[str]] = None,
    regime: str = 'A',
) -> None:
    """
    Overlay z-plane mean eigenvalues from multiple runs to show how
    poles evolve across amplitude levels or nonlinearity types.

    Parameters
    ----------
    runs   : list of CVARun objects
    clusts : list of corresponding CVAPhiCluster objects
    labels : legend labels for each run (None = auto)
    regime : 'A' or 'B' — which regime to plot from each run
    """
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(runs))]

    cmap   = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    fig, ax = plt.subplots(figsize=(8, 8))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, label='Unit circle')

    for run, clust, label, color in zip(runs, clusts, labels, cmap):
        k   = clust.regime_A if regime == 'A' else clust.regime_B
        dt  = 1.0 / run.fs
        fs  = run.fs

        # Get mean eigenvalues for this cluster
        mean_ev = clust.cluster_mean_ev[k]
        mag     = np.abs(mean_ev)
        ce      = np.log(mean_ev + 1e-300) / dt
        freq    = np.abs(np.imag(ce)) / (2 * np.pi)

        good = (
            (np.abs(np.imag(mean_ev)) > 0.01 * mag) &
            (freq > 0.5) & (freq < fs / 2) &
            (mag <= 1.0) & (np.imag(mean_ev) > 0)
        )

        ax.scatter(np.real(mean_ev[good]), np.imag(mean_ev[good]),
                   color=color, s=60, alpha=0.8, label=label, zorder=3)

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Pole Trajectories — Regime {regime}\n'
                 f'({len(runs)} runs overlaid)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    lim = 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout()
    plt.show()