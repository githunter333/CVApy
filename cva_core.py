"""
cva_core.py
===========
Python translation of MATLAB CVA functions:
  - delaybuild      : build time-delayed (Hankel) matrix
  - reduce_matrix   : SVD-based dimensionality reduction / whitening
  - cvaonestep      : one-step local CVA system identification

Dependencies: numpy, scipy, scikit-learn

Author: Norman / Claude
Date:   2026-04-12
"""

import numpy as np
from scipy.linalg import svd as scipy_svd, lstsq
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Module-level state (replaces MATLAB persistent variables)
# =============================================================================
_cva_state = {
    'ranklock': None,
    'Jref':     None,
}

def reset_cva_state():
    """Call before starting a new CVA run (replaces inow==1 reset)."""
    _cva_state['ranklock'] = None
    _cva_state['Jref']     = None


# =============================================================================
# delaybuild
# =============================================================================
def delaybuild(data: np.ndarray, nch: int, lags: int) -> np.ndarray:
    """
    Build time-delayed (Hankel) matrix from time series data.

    Parameters
    ----------
    data : array
        [nch x nsamples] or [nsamples x nch] — function auto-detects.
        For single channel, a 1-D array or row/column vector is accepted.
    nch  : int
        Number of channels.
    lags : int
        Number of time lags.

    Returns
    -------
    delayed : ndarray, shape (nsamples - lags + 1, nch * lags)
        Hankel matrix, oldest-first, lag-major column ordering:
        col = lag * nch + ch  (0-indexed)

    Examples
    --------
    Single channel, data = [1,2,3,4,5], lags=3:
        delayed = [[1,2,3],
                   [2,3,4],
                   [3,4,5]]

    Two channels, lags=2:
        col order: [ch0_lag0, ch1_lag0, ch0_lag1, ch1_lag1]
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))

    if nch > 1:
        # Ensure shape is (nch, nsamples)
        if data.shape[0] != nch:
            data = data.T
        nsamples = data.shape[1]
        nrows    = nsamples - lags + 1
        delayed  = np.zeros((nrows, nch * lags))

        for ch in range(nch):
            for lag in range(lags):
                col_idx = lag * nch + ch          # oldest-first, lag-major
                delayed[:, col_idx] = data[ch, lag : lag + nrows]
    else:
        # Single channel
        d = data.ravel()
        nsamples = len(d)
        nrows    = nsamples - lags + 1
        delayed  = np.zeros((nrows, lags))

        for lag in range(lags):
            delayed[:, lag] = d[lag : lag + nrows]

    return delayed


# =============================================================================
# reduce_matrix
# =============================================================================
def reduce_matrix(M: np.ndarray, tol: float):
    """
    SVD-based dimensionality reduction and whitening.

    Parameters
    ----------
    M   : ndarray, square covariance-like matrix
    tol : float
        Relative tolerance for rank determination (fraction of largest SV).

    Returns
    -------
    Vrr  : ndarray, shape (maxdim, ncols)
        Whitening transformation: diag(1/sqrt(sigma)) @ V.T
    rank : int
        Effective rank of M.
    """
    U, s, Vt = scipy_svd(M, full_matrices=False)

    if len(s) > 0 and s[0] > 0:
        normalized = s / s[0]
        rank = int(np.sum(normalized > tol))
    else:
        rank = 0

    maxdim = max(M.shape)
    diag_elements = np.zeros(maxdim)
    for i in range(maxdim):
        if i < rank and i < len(s):
            diag_elements[i] = 1.0 / np.sqrt(s[i])

    # Vrr = diag(1/sqrt(sigma)) @ V.T   (same as MATLAB diag(...) * V')
    # Vt is already V.T from scipy
    # We need to pad Vt if maxdim > len(s)
    n_sv = Vt.shape[0]
    ncols = Vt.shape[1]
    Vrr_full = np.zeros((maxdim, ncols))
    Vrr_full[:n_sv, :] = Vt
    Vrr = diag_elements[:, np.newaxis] * Vrr_full

    return Vrr, rank


# =============================================================================
# cvaonestep
# =============================================================================
def cvaonestep(
    past:         np.ndarray,
    future:       np.ndarray,
    inputd:       np.ndarray,
    querypast:    np.ndarray,
    queryfuture:  np.ndarray,
    queryinput:   np.ndarray,
    inow:         int,
    tol:          float,
    neighbors:    int,
    ranklock:     int = 0,
) -> dict:
    """
    One-step local Canonical Variate Analysis for system identification.

    Parameters
    ----------
    past        : (nsamples, nout*lags_past)   — past output history database
    future      : (nsamples, nout*lags_future) — future output database
    inputd      : (nsamples, nin)              — input database
    querypast   : (1, nout*lags_past)          — query past vector
    queryfuture : (1, nout*lags_future)        — query future (unused currently)
    queryinput  : (1, nin)                     — query input (unused currently)
    inow        : int                          — current sample index (1-based,
                                                 matches MATLAB convention)
    tol         : float                        — SVD rank tolerance
    neighbors   : int                          — number of nearest neighbors
    ranklock    : int                          — 0 = auto rank; >0 = locked rank

    Returns
    -------
    dict with keys:
        Phi      : (rank, rank)          state transition matrix
        Phi1     : (rank, rank)          state transition with input
        J        : (n_Ug, n_Vpr)        canonical variate projection
        L        : (rank, *)             future projection matrix
        rank     : int                   system order
        spast    : (neighbors, rank)     state estimates from past
        sfuture  : (neighbors, rank)     state estimates from future
        indpast  : (neighbors,)          neighbor indices into past/future
    """
    global _cva_state

    past       = np.atleast_2d(np.asarray(past,    dtype=float))
    future     = np.atleast_2d(np.asarray(future,  dtype=float))
    inputd     = np.atleast_2d(np.asarray(inputd,  dtype=float))
    querypast  = np.atleast_2d(np.asarray(querypast, dtype=float))

    # Ensure inputd is (nsamples, nin)
    if inputd.shape[0] == 1 and inputd.shape[1] > 1:
        inputd = inputd.T

    # Reset persistent state on first call
    if inow == 1:
        reset_cva_state()

    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if querypast.shape[0] != 1:
        raise ValueError(f'querypast must be a row vector, got shape {querypast.shape}')
    if past.shape[1] != querypast.shape[1]:
        raise ValueError(
            f'past ({past.shape[1]} cols) and querypast ({querypast.shape[1]} cols) '
            f'must have same number of columns')
    if neighbors >= past.shape[0]:
        raise ValueError(
            f'neighbors ({neighbors}) must be less than available samples ({past.shape[0]})')

    # ------------------------------------------------------------------
    # 2. Find nearest neighbors
    # ------------------------------------------------------------------
    nn_model = NearestNeighbors(n_neighbors=neighbors + 1,
                                algorithm='auto', metric='euclidean')
    nn_model.fit(past)
    _, idx = nn_model.kneighbors(querypast)   # idx: (1, neighbors+1)
    idx = idx[0]                               # flatten to 1D

    # Skip first match (self) — same as MATLAB idx(2:end)
    indpast = idx[1:]    # shape: (neighbors,)

    # ------------------------------------------------------------------
    # 3. Extract neighborhood data
    # ------------------------------------------------------------------
    pneighborsw = past[indpast, :]      # (neighbors, nout*lags)
    fneighborsw = future[indpast, :]    # (neighbors, nout*lags)
    ineighbors  = inputd[indpast, :]    # (neighbors, nin)

    # Future input (next sample) — guard against index out of bounds
    indpast_p1  = np.minimum(indpast + 1, inputd.shape[0] - 1)
    ineighborsf = inputd[indpast_p1, :]  # (neighbors, nin)

    # Uniform weighting (matches MATLAB ones())
    # neighborswindow = np.ones(neighbors)  # placeholder for future weighting

    # ------------------------------------------------------------------
    # 4. Append input to past/future blocks
    # ------------------------------------------------------------------
    pneighbors = np.hstack([pneighborsw, ineighbors])   # (neighbors, nout*lags+nin)
    fneighbors = np.hstack([fneighborsw, ineighborsf])  # (neighbors, nout*lags+nin)

    # ------------------------------------------------------------------
    # 5. Covariance matrices
    # ------------------------------------------------------------------
    Spp = pneighbors.T @ pneighbors
    Sff = fneighbors.T @ fneighbors
    Spf = pneighbors.T @ fneighbors

    # Tikhonov regularization
    reg_p = 1e-8 * np.trace(Spp) / Spp.shape[0]
    reg_f = 1e-8 * np.trace(Sff) / Sff.shape[0]
    Spp += reg_p * np.eye(Spp.shape[0])
    Sff += reg_f * np.eye(Sff.shape[0])

    # ------------------------------------------------------------------
    # 6. Dimensionality reduction via SVD whitening
    # ------------------------------------------------------------------
    Vpr, rankpast   = reduce_matrix(Spp, tol)
    Vfr, rankfuture = reduce_matrix(Sff, tol)

    # ------------------------------------------------------------------
    # 7. Canonical correlation matrix and SVD
    # ------------------------------------------------------------------
    Gpf = Vpr @ Spf @ Vfr.T

    Ug, sg, Vgt = scipy_svd(Gpf, full_matrices=False)
    # Note: scipy returns Vgt = V.T already

    if len(sg) > 0 and sg[0] > 0:
        norm_sg = sg / sg[0]
        grank   = int(np.sum(norm_sg > tol))
    else:
        grank = 0

    # ------------------------------------------------------------------
    # 8. Determine system order
    # ------------------------------------------------------------------
    if inow == 1:
        _cva_state['ranklock'] = min(rankpast, rankfuture, grank)

    if ranklock and ranklock > 0:
        rnk = int(ranklock)
    else:
        rnk = _cva_state['ranklock'] if _cva_state['ranklock'] is not None else \
              min(rankpast, rankfuture, grank)

    # ------------------------------------------------------------------
    # 9. Build projection matrix J
    # ------------------------------------------------------------------
    n_Ug  = Ug.shape[0]
    n_Vpr = Vpr.shape[0]

    Intdiag = np.zeros((n_Ug, n_Vpr))
    if rnk > 0:
        r = min(rnk, n_Ug, n_Vpr)
        Intdiag[:r, :r] = np.eye(r)

    J = (Ug @ Intdiag).T @ Vpr     # same as MATLAB (Ug*Intdiag)' * Vpr

    if inow == 1:
        _cva_state['Jref'] = J.copy()

    # ------------------------------------------------------------------
    # 10. Truncate J to rank rows
    # ------------------------------------------------------------------
    Jnew = J[:rnk, :]   # (rank, ncols_past+nin)

    # ------------------------------------------------------------------
    # 11. Future projection matrix L
    # ------------------------------------------------------------------
    # MATLAB: L = (Vfr' * Vgt);  L = L(1:rank,:)
    # scipy Vgt is already V.T, so Vfr.T @ Vgt.T = Vfr.T @ V
    # But MATLAB Vgt from [U,W,V]=svd(Gpf) is the V matrix, so Vgt in MATLAB
    # corresponds to Vgt.T in scipy (scipy returns Vh = V.T).
    L_full = Vfr.T @ Vgt.T          # matches MATLAB Vfr' * Vgt
    L      = L_full[:rnk, :]        # (rank, ncols_future+nin)

    # ------------------------------------------------------------------
    # 12. State estimates
    # ------------------------------------------------------------------
    spast_n   = (Jnew @ pneighbors.T).T    # (neighbors, rank)
    sfuture_n = (Jnew @ fneighbors.T).T   # (neighbors, rank)

    if spast_n.shape[1] != rnk:
        raise ValueError(f'spast wrong dim: expected {rnk}, got {spast_n.shape[1]}')
    if spast_n.shape[0] != sfuture_n.shape[0]:
        raise ValueError('spast and sfuture row count mismatch')

    # ------------------------------------------------------------------
    # 13. Identify state-space matrices via regularized least squares
    # ------------------------------------------------------------------
    if spast_n.shape[0] < 2:
        raise ValueError('Not enough state samples for regression')

    # Autonomous: x(k+1) = Phi * x(k)
    reg = 1e-6 * np.trace(spast_n.T @ spast_n) / spast_n.shape[1]
    A   = spast_n.T @ spast_n + reg * np.eye(spast_n.shape[1])
    B   = spast_n.T @ sfuture_n
    try:
        Phi = np.linalg.solve(A, B).T    # matches MATLAB (A\B)'
    except np.linalg.LinAlgError:
        Phi = (np.linalg.pinv(spast_n) @ sfuture_n).T

    # Validate / trim Phi
    if Phi.shape != (rnk, rnk):
        if Phi.size >= rnk * rnk:
            Phi = Phi[:rnk, :rnk]
        else:
            raise ValueError('Cannot construct valid Phi matrix')

    if np.any(np.isnan(Phi)) or np.any(np.isinf(Phi)):
        raise ValueError('Invalid Phi matrix (NaN or Inf)')

    # With input: x(k+1) = Phi1 * [x(k); u(k)]
    try:
        pastninput = np.hstack([spast_n, ineighborsf[:, -1:]])
        reg1 = 1e-6 * np.trace(pastninput.T @ pastninput) / pastninput.shape[1]
        A1   = pastninput.T @ pastninput + reg1 * np.eye(pastninput.shape[1])
        B1   = pastninput.T @ sfuture_n
        Phig = np.linalg.solve(A1, B1).T
        Phi1 = Phig[:, :rnk]
    except Exception:
        Phi1 = Phi.copy()

    # ------------------------------------------------------------------
    # Return results as dict
    # ------------------------------------------------------------------
    return {
        'Phi':     Phi,
        'Phi1':    Phi1,
        'J':       J,
        'L':       L,
        'rank':    rnk,
        'spast':   spast_n,
        'sfuture': sfuture_n,
        'indpast': indpast,
    }


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == '__main__':
    print('Testing cva_core.py ...')
    rng = np.random.default_rng(42)

    # Synthetic data: 4 channels, 1 input, 500 samples, 16 lags
    N      = 2000
    n_out  = 4
    n_in   = 1
    lags   = 16

    datout = rng.standard_normal((N, n_out))
    inputd = rng.standard_normal((N, n_in))

    # Build past/future via delaybuild
    past_db   = delaybuild(datout.T, n_out, lags)    # (N-lags+1, n_out*lags)
    future_db = delaybuild(datout.T, n_out, lags)

    n_win  = past_db.shape[0]
    kw     = 100
    inow   = kw + lags - 1

    result = cvaonestep(
        past     = past_db,
        future   = future_db,
        inputd   = inputd,
        querypast   = past_db[kw:kw+1, :],
        queryfuture = future_db[kw:kw+1, :],
        queryinput  = inputd[inow:inow+1, :],
        inow     = 1,          # trigger rank lock
        tol      = 1e-4,
        neighbors = 200,
        ranklock  = 0,
    )

    print(f"  rank    = {result['rank']}")
    print(f"  Phi     = {result['Phi'].shape}")
    print(f"  spast   = {result['spast'].shape}")
    print(f"  sfuture = {result['sfuture'].shape}")
    print(f"  max|z|  = {max(abs(np.linalg.eigvals(result['Phi']))):.4f}")
    print('Self-test passed.')
