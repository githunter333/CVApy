"""The regime test must read one state as one and two as two."""
import numpy as np
import pytest

import cva_plotting
from cva_level import track_in_band
from cva_regimes import analyze_regimes, plot_regimes
from cva_residual import noncoherent_history
from cva_sim import simulate_chain

cva_plotting.set_show(False)
DURATION = 40.0


def _regimes(softness, with_residual=False):
    rec = simulate_chain(duration=DURATION, softness=softness, seed=2)
    trk = track_in_band(rec.time, rec.inputd, rec.datout, rec.fs,
                        band_top=1000.0)
    rr = (noncoherent_history(rec.inputd, rec.datout, rec.fs, skip_sec=1.0)
          if with_residual else None)
    return analyze_regimes(trk, rr)


@pytest.fixture(scope='module')
def switching():
    """A joint that changes between linear and 10% softer in tension every
    two to five seconds, regardless of response level."""
    rng = np.random.default_rng(5)
    edges = np.cumsum(rng.uniform(2.0, 5.0, 40))
    return _regimes(lambda t: 0.1 if np.searchsorted(edges, t) % 2 else 0.0,
                    with_residual=True)


def test_linear_record_is_one_state():
    assert not _regimes(0.0).two_states


def test_steady_nonlinear_record_is_one_state():
    assert not _regimes(0.5).two_states


def test_switching_joint_is_two_states(switching):
    assert switching.two_states
    assert 1.5 < switching.features[switching.by].dwell_sec < 6.0


def test_states_have_different_frequencies(switching):
    a, b = switching.state_table()
    rel = np.abs(a['freq'] - b['freq']) / b['freq']
    assert rel.max() > 2e-3
    assert plot_regimes(switching) is not None
