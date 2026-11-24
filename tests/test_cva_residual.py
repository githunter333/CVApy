"""Score the non-coherent time history against simulated records."""
import numpy as np
import pytest

import cva_plotting
from cva_residual import noncoherent_history, plot_residual, track_residual
from cva_sim import simulate_chain

cva_plotting.set_show(False)


@pytest.fixture(scope='module')
def linear():
    rec = simulate_chain(duration=20.0, softness=0.0, seed=3)
    return noncoherent_history(rec.inputd, rec.datout, rec.fs, skip_sec=1.0)


@pytest.fixture(scope='module')
def bilinear():
    rec = simulate_chain(duration=20.0, softness=0.5, seed=3)
    return noncoherent_history(rec.inputd, rec.datout, rec.fs, skip_sec=1.0)


def test_parts_add_up(linear):
    assert np.allclose(linear.linear + linear.resid, linear.total)


def test_linear_record_leaves_little_residual(linear):
    assert linear.share.max() < 0.05


def test_bilinear_joint_leaves_a_residual(linear, bilinear):
    assert (bilinear.share > 3 * linear.share).all()


def test_single_response_channel():
    rec = simulate_chain(duration=8.0, softness=0.0, seed=1)
    rr = noncoherent_history(rec.inputd[:, 0], rec.datout[:, 2], rec.fs)
    assert rr.resid.shape == (len(rec.time), 1)
    assert len(rr.table(verbose=False)) == 1


def test_plot_and_tracks_build(bilinear):
    assert plot_residual(bilinear, channel=3) is not None
    trk = track_residual(bilinear)
    assert len(trk.ref_freq) >= 3
