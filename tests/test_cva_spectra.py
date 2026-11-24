"""Score the spectral nonlinearity indicators against simulated records."""
import numpy as np
import pytest

import cva_plotting
from cva_sim import simulate_chain
from cva_spectra import analyze_record, plot_comparison, plot_record

cva_plotting.set_show(False)


@pytest.fixture(scope='module')
def linear():
    rec = simulate_chain(duration=20.0, softness=0.0, seed=3)
    return analyze_record(rec.inputd, rec.datout, rec.fs, label='linear',
                          skip_sec=1.0)


@pytest.fixture(scope='module')
def bilinear():
    rec = simulate_chain(duration=20.0, softness=0.5, seed=3)
    return analyze_record(rec.inputd, rec.datout, rec.fs, label='bilinear',
                          skip_sec=1.0)


def test_linear_record_is_coherent(linear):
    assert linear.frac_noncoherent.max() < 0.05
    assert np.abs(linear.skew).max() < 0.1


def test_bilinear_joint_raises_noncoherent_energy(linear, bilinear):
    assert (bilinear.frac_noncoherent > 3 * linear.frac_noncoherent).all()
    assert bilinear.frac_noncoherent.max() > 0.10


def test_two_identical_inputs_do_not_break_multiple_coherence():
    rec = simulate_chain(duration=6.0, softness=0.0, seed=1)
    rng = np.random.default_rng(0)
    extra = rng.standard_normal(len(rec.time))      # unrelated second input
    U = np.column_stack([rec.inputd[:, 0], extra])
    one = analyze_record(rec.inputd, rec.datout, rec.fs)
    two = analyze_record(U, rec.datout, rec.fs)
    assert two.coh.shape == one.coh.shape
    # an unrelated extra input can only add a little spurious coherence
    assert (two.frac_noncoherent <= one.frac_noncoherent + 1e-9).all()


def test_plots_build(linear, bilinear):
    assert plot_record(bilinear) is not None
    assert plot_comparison([linear, bilinear]) is not None
