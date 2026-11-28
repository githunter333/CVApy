"""A model that depends on level must recover a known dependence, and must
not invent one where there is none."""
import numpy as np

from cva_global import GlobalModel, mac


def _made_up(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, n)                       # log level
    freq = np.column_stack([100 + 0 * x, 200 + 4.0 * x]) \
        + 0.05 * rng.standard_normal((n, 2))
    zeta = np.column_stack([0.01 + 0 * x, 0.02 + 0.004 * x]) \
        + 1e-4 * rng.standard_normal((n, 2))
    present = np.column_stack([np.ones(n, bool),
                               rng.random(n) < 1 / (1 + np.exp(-4 * x))])
    freq[~present] = np.nan
    zeta[~present] = np.nan
    base = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, -0.6]])
    tilt = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    shapes = base[None] + x[:, None, None] * tilt[None] + 0j
    shapes = shapes / np.linalg.norm(shapes, axis=2, keepdims=True)
    shapes[~present] = np.nan
    order = np.round(8 + 4 * x)
    return x, freq, zeta, present, shapes, order


def test_global_model_recovers_a_level_dependence():
    x, freq, zeta, present, shapes, order = _made_up()
    tr, te = slice(0, 400), slice(400, None)
    gm = GlobalModel.fit(x[tr], freq[tr], zeta[tr], present[tr], shapes[tr],
                         order[tr], degree=1)
    one = GlobalModel.fit(x[tr], freq[tr], zeta[tr], present[tr],
                          shapes[tr], order[tr], degree=0)
    assert abs(gm.freq_c[1][0] - 4.0) < 0.1            # slope found
    assert abs(gm.freq_c[0][0]) < 0.05                 # none invented
    p, q = gm.predict(x[te]), one.predict(x[te])
    ok = present[te, 1]
    err = lambda a: np.sqrt(np.nanmean((a[ok, 1] - freq[te][ok, 1]) ** 2))
    assert err(p['freq']) < 0.1 < 1.0 < err(q['freq'])
    assert np.nanmedian(mac(p['shapes'][ok, 1], shapes[te][ok, 1])) > 0.999
    assert (np.nanmedian(mac(q['shapes'][ok, 1], shapes[te][ok, 1]))
            < np.nanmedian(mac(p['shapes'][ok, 1], shapes[te][ok, 1])))
    # presence follows level in the level model only
    assert np.corrcoef(p['presence'][:, 1], x[te])[0, 1] > 0.9
    assert np.ptp(q['presence'][:, 1]) < 1e-9
    assert np.sqrt(np.mean((p['order'] - order[te]) ** 2)) < 0.5
    # outside the levels seen, the prediction is held at the end value
    far = gm.predict([x[tr].max() + 5.0])['freq'][0, 1]
    assert abs(far - gm.predict([x[tr].max()])['freq'][0, 1]) < 1e-9
