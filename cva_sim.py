"""
cva_sim.py
==========
Simulated test records with known answers.

A six-mass chain (ground - m1 - m2 - ... - m6), random force on mass 1,
acceleration responses at all six masses. The spring between masses 4 and 5
can be made softer in tension than in compression, and that softness can be
made to grow along the record.

These records exist so an estimator can be scored against the truth: the
modal frequencies and shapes of the compression state, the tension state,
and the blend of the two that a time-averaged fit should report.

    rec = simulate_chain(duration=10.0, softness=0.10)
    rec.time, rec.inputd, rec.datout, rec.fs
    rec.f_comp, rec.f_tens          # the two linear states, Hz
    rec.effective(t)                # blended reference at time t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
from scipy.linalg import eigh, expm

N_DOF = 6
JOINT = (3, 4)          # zero-based masses either side of the joint (4 and 5)


def chain_matrices(softness: float = 0.0, f_scale: float = 415.0):
    """Stiffness matrix of the chain with the joint softened by `softness`.

    f_scale sets the spring rate through k = (2 pi f_scale)^2 for unit
    masses; 415 Hz puts the six modes between 100 and 806 Hz.
    """
    k0 = (2 * np.pi * f_scale) ** 2
    ks = np.full(N_DOF, k0)             # ks[i] joins mass i-1 (or ground) to i
    ks[JOINT[1]] *= (1.0 - softness)
    K = np.zeros((N_DOF, N_DOF))
    for i in range(N_DOF):
        K[i, i] += ks[i]
        if i > 0:
            K[i - 1, i - 1] += ks[i]
            K[i, i - 1] -= ks[i]
            K[i - 1, i] -= ks[i]
    return K


def modes_of(K: np.ndarray):
    """Undamped natural frequencies (Hz) and mode shapes for unit masses."""
    w2, V = eigh(K)
    return np.sqrt(w2) / (2 * np.pi), V


@dataclass
class SimRecord:
    time:    np.ndarray
    inputd:  np.ndarray         # (N, 1) force on mass 1
    datout:  np.ndarray         # (N, 6) accelerations
    fs:      float
    tension: np.ndarray         # (N,) bool, joint in tension at each sample
    softness_of_t: Callable[[float], float] = field(repr=False, default=None)
    f_comp:  np.ndarray = None  # compression-state frequencies, Hz
    zeta:    np.ndarray = None  # compression-state damping ratios

    def state_modes(self, t: float, tension: bool):
        """Frequencies and shapes of one linear state at time t."""
        s = self.softness_of_t(t) if tension else 0.0
        return modes_of(chain_matrices(s))

    def effective(self, t: float):
        """Reference for a time-averaged fit: the chain with the joint at the
        mean of its two stiffnesses. An approximation, good to a fraction of
        the tension/compression split."""
        return modes_of(chain_matrices(0.5 * self.softness_of_t(t)))

    @property
    def f_tens(self):
        return self.state_modes(self.time[-1], True)[0]


def simulate_chain(duration: float = 10.0, fs: float = 2500.0,
                   softness: Union[float, Callable[[float], float]] = 0.0,
                   noise: float = 0.0, zeta_ref: float = 0.01,
                   f_ref: float = 300.0, substeps: int = 20,
                   seed: int = 0, n_levels: int = 26) -> SimRecord:
    """
    Simulate the chain under a random force held constant over each sample.

    softness : fraction by which the joint is softer in tension. A number,
               or a function of time in seconds for a joint that changes
               along the record (evaluated on n_levels stepped values).
    noise    : measurement noise added to each response, as a fraction of
               that channel's rms.
    zeta_ref, f_ref : stiffness-proportional damping giving zeta_ref at
               f_ref Hz, so damping rises in proportion to frequency.

    Each sample is advanced in `substeps` exact linear steps, with the joint
    state re-read at every substep.
    """
    soft_fn = softness if callable(softness) else (lambda t, s=softness: s)
    N = int(round(duration * fs))
    h = 1.0 / fs / substeps
    beta = 2.0 * zeta_ref / (2 * np.pi * f_ref)
    b = np.zeros(N_DOF)
    b[0] = 1.0

    def discrete(soft):
        K = chain_matrices(soft)
        A = np.zeros((2 * N_DOF + 1, 2 * N_DOF + 1))
        A[:N_DOF, N_DOF:2 * N_DOF] = np.eye(N_DOF)
        A[N_DOF:2 * N_DOF, :N_DOF] = -K
        A[N_DOF:2 * N_DOF, N_DOF:2 * N_DOF] = -beta * K
        A[N_DOF:2 * N_DOF, -1] = b
        E = expm(A * h)                       # exact step for held input
        return E[:2 * N_DOF, :2 * N_DOF], E[:2 * N_DOF, -1], K

    t_all = np.arange(N) / fs
    soft_all = np.array([soft_fn(t) for t in t_all])
    lo, hi = float(soft_all.min()), float(soft_all.max())
    levels = np.array([lo]) if hi == lo else np.linspace(lo, hi, n_levels)
    level_of = np.abs(soft_all[:, None] - levels[None, :]).argmin(axis=1)
    comp = discrete(0.0)
    tens = [discrete(s) for s in levels]

    rng = np.random.default_rng(seed)
    u = rng.standard_normal(N)
    x = np.zeros(2 * N_DOF)
    acc = np.zeros((N, N_DOF))
    tension = np.zeros(N, dtype=bool)
    ja, jb = JOINT
    for i in range(N):
        Ad_t, Bd_t, K_t = tens[level_of[i]]
        in_t = x[jb] - x[ja] > 0
        tension[i] = in_t
        K = K_t if in_t else comp[2]
        acc[i] = b * u[i] - K @ x[:N_DOF] - beta * (K @ x[N_DOF:])
        for _ in range(substeps):
            if x[jb] - x[ja] > 0:
                x = Ad_t @ x + Bd_t * u[i]
            else:
                x = comp[0] @ x + comp[1] * u[i]

    if noise:
        acc = acc + noise * acc.std(axis=0) * rng.standard_normal(acc.shape)

    f_comp, _ = modes_of(chain_matrices(0.0))
    return SimRecord(time=t_all, inputd=u[:, None], datout=acc, fs=fs,
                     tension=tension, softness_of_t=soft_fn,
                     f_comp=f_comp, zeta=zeta_ref * f_comp / f_ref)
