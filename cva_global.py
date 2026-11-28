"""
cva_global.py
=============
One model in place of hundreds of local ones.

The sweep fits a local linear model in every window. Those models do not
vary freely: in the records studied so far they follow the response level.
GlobalModel stores, for each tracked mode, how its frequency, damping,
mode shape and presence depend on the local response level, and how the
model order does. Given the level in a window it returns that window's
modes without fitting anything there.

    gm = GlobalModel.from_cluster(clust, train=time < 40)   # fit
    pred = gm.predict(log_level)                            # anywhere
    gm.score(clust, test=time >= 40)                        # how good

Only quantities that do not depend on how the states are scaled are stored:
poles, shapes (unit length, largest component real) and presence.

`degree` 0 gives the single linear model (one set of modes for every
level), which is the baseline to beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _robust_poly(x, y, degree):
    """Polynomial fit with points more than 3 scaled MADs out dropped,
    twice. Returns coefficients (highest power first) and the scatter of
    the kept points about the fit."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.array([np.nan]), np.nan
    d = int(min(degree, max(0, len(np.unique(np.round(x, 6))) - 1,
                            ) if len(x) > 2 else 0))
    if len(x) < 4 * (d + 1):
        d = 0
    keep = np.ones(len(x), bool)
    for _ in range(3):
        c = np.polyfit(x[keep], y[keep], d)
        r = y - np.polyval(c, x)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= 3 * mad + 1e-12 * (1 + np.abs(y).max())
        if new.sum() < 4 * (d + 1) or (new == keep).all():
            break
        keep = new
    return c, float(np.std(y[keep] - np.polyval(c, x[keep])))


def _logistic(x, p, degree):
    """Presence probability against x. Returns (a, b) of
    1 / (1 + exp(-(a + b x)))."""
    p = np.asarray(p, float)
    f = p.mean()
    eps = 0.5 / max(len(p), 1)
    a0 = float(np.log((f + eps) / (1 - f + eps)))
    if degree == 0 or f < 0.02 or f > 0.98 or np.std(x) == 0:
        return a0, 0.0
    a, b = a0, 0.0
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(50):                      # Newton steps, lightly damped
        q = 1 / (1 + np.exp(-(a + b * x)))
        W = q * (1 - q) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (p - q))
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def mac(a, b):
    """Modal assurance criterion between complex shapes (last axis)."""
    num = np.abs(np.sum(np.conj(a) * b, axis=-1)) ** 2
    den = (np.sum(np.abs(a) ** 2, axis=-1) * np.sum(np.abs(b) ** 2, axis=-1))
    return num / np.maximum(den, 1e-300)


@dataclass
class GlobalModel:
    ref_freq: np.ndarray                 # (n_modes,) Hz, for labels
    degree:   int = 1
    freq_c:   List[np.ndarray] = field(default_factory=list)
    zeta_c:   List[np.ndarray] = field(default_factory=list)
    pres_c:   List[tuple] = field(default_factory=list)
    shape_c:  List[Optional[np.ndarray]] = field(default_factory=list)
    order_c:  np.ndarray = None
    x_range:  tuple = (np.nan, np.nan)   # log levels the fit saw
    freq_sd:  np.ndarray = None          # scatter of the windows about the fit

    # ---- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, log_level, freq, zeta, presence, shapes=None, order=None,
            ref_freq=None, degree: int = 1):
        """
        log_level : (n,) log response level of each window.
        freq, zeta: (n, n_modes) tracked frequency (Hz) and damping ratio,
                    NaN where the mode is not found.
        presence  : (n, n_modes) bool.
        shapes    : (n, n_modes, n_out) complex, optional.
        order     : (n,) states each window supports, optional.
        degree    : polynomial degree in log level; 0 = one linear model.
        """
        x = np.asarray(log_level, float)
        freq, zeta = np.asarray(freq, float), np.asarray(zeta, float)
        nm = freq.shape[1]
        if ref_freq is None:
            ref_freq = np.nanmedian(freq, axis=0)
        gm = cls(ref_freq=np.asarray(ref_freq, float), degree=degree,
                 x_range=(float(np.min(x)), float(np.max(x))))
        sd = []
        for j in range(nm):
            c, s = _robust_poly(x, freq[:, j], degree)
            gm.freq_c.append(c); sd.append(s)
            gm.zeta_c.append(_robust_poly(x, zeta[:, j], degree)[0])
            gm.pres_c.append(_logistic(x, np.asarray(presence)[:, j], degree))
            if shapes is None:
                gm.shape_c.append(None)
                continue
            S = np.asarray(shapes)[:, j, :]
            ok = np.isfinite(S[:, 0])
            if ok.sum() < 4:
                gm.shape_c.append(None)
                continue
            # one sign convention for the mode: the channel that is largest
            # on average is made real and positive in every window
            k = int(np.argmax(np.abs(S[ok]).mean(axis=0)))
            S = S * np.exp(-1j * np.angle(S[:, k:k + 1]))
            d = degree if ok.sum() >= 4 * (degree + 1) else 0
            c = np.stack([np.polyfit(x[ok], S[ok, m].real, d)
                          + 1j * np.polyfit(x[ok], S[ok, m].imag, d)
                          for m in range(S.shape[1])], axis=1)
            gm.shape_c.append(c)                       # (d + 1, n_out)
        gm.freq_sd = np.array(sd)
        if order is not None:
            gm.order_c = _robust_poly(x, np.asarray(order, float), degree)[0]
        return gm

    @classmethod
    def from_cluster(cls, clust, train=None, degree: int = 1):
        """Fit from a fitted CVAPhiCluster. train: bool mask over the
        cluster's windows (default all)."""
        d = cluster_arrays(clust)
        m = np.ones(len(d['log_level']), bool) if train is None else train
        return cls.fit(d['log_level'][m], d['freq'][m], d['zeta'][m],
                       d['presence'][m], d['shapes'][m], d['order'][m],
                       ref_freq=clust.ref_freq, degree=degree)

    # ---- prediction ------------------------------------------------------
    def predict(self, log_level):
        """Modes at the given log level(s). Levels outside the range the
        fit saw are held at the nearest end of it. Returns dict: freq,
        zeta, presence (probability), shapes, order."""
        x = np.clip(np.atleast_1d(np.asarray(log_level, float)),
                    *self.x_range)
        nm = len(self.ref_freq)
        out = dict(
            freq=np.column_stack([np.polyval(c, x) for c in self.freq_c]),
            zeta=np.column_stack([np.polyval(c, x) for c in self.zeta_c]),
            presence=np.column_stack(
                [1 / (1 + np.exp(-(a + b * x))) for a, b in self.pres_c]),
            order=(np.polyval(self.order_c, x) if self.order_c is not None
                   else None), shapes=None)
        if any(c is not None for c in self.shape_c):
            n_out = next(c for c in self.shape_c if c is not None).shape[1]
            S = np.full((len(x), nm, n_out), np.nan + 0j)
            for j, c in enumerate(self.shape_c):
                if c is None:
                    continue
                v = np.stack([np.polyval(c[:, m], x)
                              for m in range(n_out)], axis=1)
                S[:, j] = v / np.linalg.norm(v, axis=1, keepdims=True)
            out['shapes'] = S
        return out

    # ---- scoring ---------------------------------------------------------
    def score(self, clust, test=None):
        """Compare predictions from level alone with the directly fitted
        windows. Returns one dict per mode: found, freq_rms (Hz),
        zeta_rms, mac (median), presence_err (mean |p - present|), and for
        the order 'order_rms' in the first entry."""
        d = cluster_arrays(clust)
        m = np.ones(len(d['log_level']), bool) if test is None else test
        p = self.predict(d['log_level'][m])
        rows = []
        for j in range(len(self.ref_freq)):
            f, z = d['freq'][m, j], d['zeta'][m, j]
            ok = np.isfinite(f)
            row = dict(mode=float(self.ref_freq[j]), found=float(ok.mean()),
                       freq_rms=np.nan, zeta_rms=np.nan, mac=np.nan,
                       presence_err=float(np.mean(np.abs(
                           p['presence'][:, j] - d['presence'][m, j]))))
            if ok.sum() >= 3:
                row['freq_rms'] = float(np.sqrt(np.mean(
                    (p['freq'][ok, j] - f[ok]) ** 2)))
                row['zeta_rms'] = float(np.sqrt(np.mean(
                    (p['zeta'][ok, j] - z[ok]) ** 2)))
                if p['shapes'] is not None:
                    row['mac'] = float(np.nanmedian(mac(
                        p['shapes'][ok, j], d['shapes'][m][ok, j])))
            rows.append(row)
        if rows and p['order'] is not None:
            rows[0]['order_rms'] = float(np.sqrt(np.mean(
                (p['order'] - d['order'][m]) ** 2)))
        return rows


def cluster_arrays(clust):
    """The per-window quantities a GlobalModel is fitted to, from a fitted
    CVAPhiCluster: time, log_level, freq, zeta, presence, shapes, order."""
    if getattr(clust, '_global_arrays', None) is None:
        c = clust.content or clust.content_conditions(verbose=False)
        clust._global_arrays = dict(
            time=c['time'], log_level=c['log_level'], order=c['order'],
            presence=c['presence'], freq=clust.track_freq,
            zeta=clust.track_zeta, shapes=clust.window_shapes())
    return clust._global_arrays
