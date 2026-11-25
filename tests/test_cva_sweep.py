"""The local sweep (CVARun) and its regime grouping (CVAPhiCluster), checked
on simulated chains whose frequencies, damping and state are known."""
import contextlib
import io

import numpy as np
import pytest

import cva_plotting
from cva_mode_shapes import CVAModeShapes
from cva_phi_cluster import CVAPhiCluster
from cva_run import CVAConfig, CVARun
from cva_sim import simulate_chain

cva_plotting.set_show(False)


def _sweep(rec, stride=100, **kw):
    cfg = CVAConfig(lags=16, tol=5e-4, neighbors=960, eval_stride=stride,
                    fs=rec.fs, **kw)
    run = CVARun(cfg)
    run.time, run.inputd, run.datout, run.fs = (rec.time, rec.inputd,
                                                rec.datout, rec.fs)
    run.n_out, run.n_in = rec.datout.shape[1], rec.inputd.shape[1]
    with contextlib.redirect_stdout(io.StringIO()):
        run.build_database()
        run.run(n_jobs=1)
    return run


def _cluster(run, **kw):
    cl = CVAPhiCluster(run, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        cl.fit()
    return cl


@pytest.fixture(scope='module')
def linear():
    return simulate_chain(duration=10.0, softness=0.0, noise=0.01, seed=1)


@pytest.fixture(scope='module')
def linear_run(linear):
    return _sweep(linear)


def test_input_history_recovers_the_modes(linear, linear_run):
    """With the input in the fit the order is twice the number of modes and
    every window gives the true frequencies and damping."""
    assert {w.rank for w in linear_run.windows} == {12}
    cl = _cluster(linear_run)
    assert len(cl.ref_freq) == 6
    assert np.allclose(cl.ref_freq, linear.f_comp, rtol=1e-3)
    assert np.all(np.nanstd(cl.track_freq, axis=0) < 0.2)          # Hz
    assert np.allclose(np.nanmedian(cl.track_zeta, axis=0), linear.zeta,
                       atol=5e-4)


def test_output_only_fit_is_still_available(linear):
    """use_input=False is the original fit: it runs, and scatters far more."""
    run = _sweep(linear, use_input=False)
    cl = _cluster(run)
    k = int(np.argmin(np.abs(cl.ref_freq - linear.f_comp[2])))
    assert np.nanstd(cl.track_freq[:, k]) > 1.0                     # Hz


@pytest.mark.parametrize('mode', ['shape', 'time', 'level'])
def test_linear_record_is_one_state(linear, linear_run, mode):
    run = linear_run if mode == 'shape' else _sweep(linear,
                                                    neighbor_mode=mode)
    cl = _cluster(run)
    assert cl.n_clusters == 1
    assert cl.regime_A == cl.regime_B == 0
    assert cl.verdict.startswith('one state')


def test_neighbour_choices_pick_what_they_say(linear):
    run = _sweep(linear, stride=2000, neighbor_mode='time')
    w = run.windows[len(run.windows) // 2]
    assert np.abs(w.indpast - w.window_idx).max() <= 960
    run = _sweep(linear, stride=2000, neighbor_mode='level')
    lev = run.window_level()
    w = run.windows[len(run.windows) // 2]
    spread = np.abs(lev[w.indpast] - lev[w.window_idx]).max()
    assert spread < 0.2 * lev.std()
    with pytest.raises(ValueError):
        _sweep(linear, neighbor_mode='nearest')


def test_switching_joint_is_two_states_by_time():
    """A joint that changes between linear and 10% softer every two to five
    seconds: models fitted from neighbours in time separate the states."""
    rng = np.random.default_rng(5)
    edges = np.cumsum(rng.uniform(2.0, 5.0, 40))
    rec = simulate_chain(
        duration=30.0, seed=2,
        softness=lambda t: 0.1 if np.searchsorted(edges, t) % 2 else 0.0)
    run = _sweep(rec, stride=150, neighbor_mode='time')
    cl = _cluster(run)
    assert cl.n_clusters == 2
    truth = np.array([np.searchsorted(edges, run.windows[w].time) % 2
                      for w in cl.use_idx])
    agree = (cl.cluster_ids == truth).mean()
    assert max(agree, 1 - agree) > 0.85


def test_forced_cluster_count_and_round_trip(linear_run, tmp_path):
    cl = CVAPhiCluster(linear_run)
    with contextlib.redirect_stdout(io.StringIO()):
        cl.fit(n_clusters=3)
        assert cl.n_clusters == 3 and len(cl.cluster_mean_ev) == 3
        cl = _cluster(linear_run)
        linear_run.save(tmp_path / 'run.h5')
        cl.save(tmp_path / 'clust.h5')
        run2 = CVARun.load(tmp_path / 'run.h5')
        cl2 = CVAPhiCluster.load(tmp_path / 'clust.h5', run2)
    assert run2.cfg.use_input and run2.cfg.neighbor_mode == 'shape'
    assert cl2.n_clusters == 1 and cl2.verdict == cl.verdict
    assert np.allclose(cl2.track_freq, cl.track_freq, equal_nan=True)
    for fig in (cl2.plot_elbow(), cl2.plot_regime_timeseries(),
                cl2.plot_eigenvalues_zplane(), cl2.plot_pca_scatter(),
                cl2.plot_pca_variance(), cl2.plot_cluster_amplitude()):
        assert fig is not None


def test_mode_shapes_with_one_state(linear, linear_run):
    cl = _cluster(linear_run)
    ms = CVAModeShapes(linear_run, cl)
    with contextlib.redirect_stdout(io.StringIO()):
        ms.extract()
        assert ms.plot_regime_comparison() is None
    cm = ms.cluster_modes[0]
    assert cm is not None and len(cm.freq) >= 5
