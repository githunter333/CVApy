"""
Scores the time-block tracker against simulated records with known answers.

Run from the project folder:   python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cva_block import BlockConfig, BlockTracker      # noqa: E402
from cva_sim import simulate_chain                    # noqa: E402

NOISE = 0.01          # 1 % of each channel's rms
SEED = 3


def track(softness, **cfg):
    rec = simulate_chain(10.0, softness=softness, noise=NOISE, seed=SEED)
    trk = BlockTracker(BlockConfig(**cfg)).fit(
        rec.time, rec.inputd, rec.datout, rec.fs)
    return rec, trk, trk.summary(verbose=False)


@pytest.fixture(scope='module')
def linear():
    return track(0.0)


@pytest.fixture(scope='module')
def bilinear():
    return track(0.10)


@pytest.fixture(scope='module')
def softening():
    return track(lambda t: 0.25 * t / 10.0)


def test_linear_finds_the_six_modes_and_nothing_else(linear):
    rec, trk, s = linear
    assert trk.n_modes == 6
    assert np.all(s['persistence'] == 1.0)
    assert np.allclose(s['freq_mean'], rec.f_comp, rtol=2e-3)


def test_linear_is_steady(linear):
    rec, trk, s = linear
    # Nothing changes in a linear record, so neither may the tracks.
    assert np.all(s['freq_std'] < 0.002 * rec.f_comp)
    assert np.all(np.abs(s['trend_hz']) < 0.002 * rec.f_comp)
    assert np.all(s['mac_mean'] > 0.999)


def test_linear_damping(linear):
    rec, trk, s = linear
    # The 100 Hz mode is excluded: four lags at 2500 Hz span too little of
    # its cycle to fix its damping (it reads about 0.9 % against 0.33 %).
    assert np.allclose(s['zeta_mean'][1:], rec.zeta[1:], rtol=0.03)


def test_bilinear_reports_the_blend_of_the_two_states(bilinear):
    rec, trk, s = bilinear
    assert trk.n_modes == 6
    lo = np.minimum(rec.f_comp, rec.f_tens)
    hi = np.maximum(rec.f_comp, rec.f_tens)
    margin = 0.002 * rec.f_comp
    assert np.all(s['freq_mean'] > lo - margin)
    assert np.all(s['freq_mean'] < hi + margin)
    assert np.allclose(s['freq_mean'], rec.effective(0.0)[0], rtol=2e-3)
    # A fixed joint gives a fixed blend.
    assert np.all(np.abs(s['trend_hz']) < 0.002 * rec.f_comp)


def test_softening_joint_is_followed_in_time(softening):
    rec, trk, s = softening
    assert trk.n_modes == 6
    true_drift = (rec.effective(trk.times[-1])[0]
                  - rec.effective(trk.times[0])[0])
    # Modes 2, 3, 5 and 6 carry strain in the joint and must be seen to drop.
    for j in (1, 2, 4, 5):
        assert s['trend_hz'][j] < 0
        assert abs(s['trend_hz'][j] - true_drift[j]) < 0.45 * abs(true_drift[j])
    # Modes 1 and 4 barely involve the joint and must stay nearly put.
    for j in (0, 3):
        assert abs(s['trend_hz'][j]) < 0.004 * rec.f_comp[j]


def test_block_length_does_not_change_the_story(softening):
    rec, trk, s = softening
    _, _, s2 = track(lambda t: 0.25 * t / 10.0, block_sec=1.0)
    assert np.allclose(s['trend_hz'], s2['trend_hz'], atol=1.5)
    assert np.allclose(s['freq_mean'], s2['freq_mean'], rtol=2e-3)


def test_single_response_channel():
    rec = simulate_chain(6.0, softness=0.0, noise=NOISE, seed=SEED)
    one = BlockTracker(BlockConfig(lags=16, freq_max=900.0)).fit(
        rec.time, rec.inputd, rec.datout[:, 5], rec.fs)
    # One channel has no mode shape to tell a real mode from a numerical
    # pole, so the band limit does that job here. Every true mode must be
    # found, and found accurately.
    assert one.n_modes == 6
    assert np.allclose(one.ref_freq, rec.f_comp, rtol=6e-3)


def test_output_only():
    rec = simulate_chain(6.0, softness=0.0, noise=NOISE, seed=SEED)
    none = BlockTracker(BlockConfig(lags=8)).fit(
        rec.time, None, rec.datout, rec.fs)
    # Without the input the random drive is unexplained, so fewer modes
    # survive and they are less exact; the ones reported must still be real.
    assert none.n_modes >= 4
    for f in none.ref_freq:
        assert np.min(np.abs(rec.f_comp - f) / rec.f_comp) < 0.01


def test_short_record_is_refused_plainly():
    rec = simulate_chain(0.3, softness=0.0, seed=SEED)
    with pytest.raises(ValueError, match='shorter than one block'):
        BlockTracker().fit(rec.time, rec.inputd, rec.datout, rec.fs)
