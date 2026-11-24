"""Quadratic and cubic coupling on signals whose coupling is known."""
import numpy as np
import pytest
from scipy.signal import lfilter

import cva_plotting
from cva_hos import analyze_hos, plot_hos, plot_hos_comparison

cva_plotting.set_show(False)
FS = 2500.0
N = 100000


def _narrow(f, zeta, rng):
    w = 2 * np.pi * f
    r = np.exp(-zeta * w / FS)
    th = w * np.sqrt(1 - zeta ** 2) / FS
    x = lfilter([1.0], [1, -2 * r * np.cos(th), r * r], rng.standard_normal(N))
    return x / x.std()


@pytest.fixture(scope='module')
def cases():
    rng = np.random.default_rng(0)
    x = _narrow(60.0, 0.02, rng) + 0.7 * _narrow(250.0, 0.02, rng)
    noise = 0.05 * rng.standard_normal((3, N))
    kw = dict(band_top=400.0, modes=[60.0, 250.0], nperseg=256)
    return dict(
        linear=analyze_hos(x + noise[0], FS, label='linear', **kw),
        quad=analyze_hos(x + 0.3 * x ** 2 + noise[1], FS, label='x^2', **kw),
        cubic=analyze_hos(x + 0.1 * x ** 3 + noise[2], FS, label='x^3', **kw))


def test_linear_signal_reads_as_none(cases):
    assert cases['linear'].source_type() == ['none']


def test_squared_term_reads_as_asymmetric(cases):
    hh = cases['quad']
    assert hh.source_type() == ['asymmetric']
    rows = hh.pair_table(0, verbose=False)
    top = max((r for r in rows if r['kind'] == 'quadratic'),
              key=lambda r: r['excess'])
    assert top['excess'] > 30


def test_cubed_term_reads_as_symmetric(cases):
    hh = cases['cubic']
    assert hh.source_type() == ['symmetric']
    assert hh.quad_excess[0] < 30 < hh.cubic_excess[0]


def test_plots_build(cases):
    assert plot_hos(cases['quad']) is not None
    assert plot_hos_comparison(list(cases.values())) is not None
