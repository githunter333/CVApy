"""Out-of-band peaks and level dependence, scored against known answers."""
import numpy as np
import pytest
from scipy.signal import lfilter

import cva_plotting
from cva_bands import analyze_bands, plot_band
from cva_level import level_dependence, plot_level, track_in_band
from cva_sim import simulate_chain

cva_plotting.set_show(False)
FS = 2500.0


def _mode(f, zeta, x):
    w = 2 * np.pi * f
    r = np.exp(-zeta * w / FS)
    th = w * np.sqrt(1 - zeta ** 2) / FS
    return lfilter([1.0], [1, -2 * r * np.cos(th), r * r], x)


@pytest.fixture(scope='module')
def rung():
    """A strong 58 Hz mode in band, and two weak modes above the band rung
    by sparse random kicks: 521.5 Hz at 1.2% and 685.7 Hz at 1.75%."""
    rng = np.random.default_rng(0)
    N = 150000
    kicks = np.zeros(N)
    kicks[rng.integers(0, N, 3000)] = rng.standard_normal(3000)
    low = _mode(58.0, 0.002, rng.standard_normal(N))
    m5 = _mode(521.5, 0.012, kicks)
    m6 = _mode(685.7, 0.0175, np.roll(kicks, 7))
    y = (low / low.std() + 0.05 * m5 / m5.std() + 0.05 * m6 / m6.std()
         + 0.002 * rng.standard_normal(N))
    return analyze_bands(y, FS, band_top=416.0)


def test_finds_both_rung_modes(rung):
    f = sorted(b.freq_all for b in rung.bands)
    assert len(f) == 2
    assert abs(f[0] - 521.5) < 3.0 and abs(f[1] - 685.7) < 3.0


def test_damping_of_rung_modes(rung):
    b5, b6 = sorted(rung.bands, key=lambda b: b.f0)
    assert abs(b5.zeta_all - 0.012) < 0.15 * 0.012
    assert abs(b6.zeta_all - 0.0175) < 0.15 * 0.0175


def test_band_history_has_the_band_energy(rung):
    b = rung.bands[0]
    assert b.signal.shape == (150000, 1)
    assert (b.envelope[1000:-1000, 0] >= np.abs(b.signal[1000:-1000, 0]) - 1e-6).all()
    assert plot_band(rung, 0) is not None


def test_nothing_found_when_nothing_is_there():
    rng = np.random.default_rng(1)
    y = _mode(58.0, 0.002, rng.standard_normal(60000))
    assert analyze_bands(y, FS, band_top=416.0).bands == []


def test_linear_chain_shows_no_level_dependence():
    rec = simulate_chain(duration=20.0, softness=0.0, seed=3)
    trk = track_in_band(rec.time, rec.inputd, rec.datout, rec.fs,
                        band_top=1000.0)
    lv = level_dependence(trk, label='linear')
    assert np.nanmax(np.abs(lv.f_slope)) < 0.05      # % per doubling
    assert plot_level([lv]) is not None
