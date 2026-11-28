"""The coherent / non-coherent split from stored spectra must agree with the
same split from the time histories the spectra were made from."""
import numpy as np
import pytest
from scipy.signal import csd, lfilter

import cva_plotting
from cva_spectra import analyze_record, analyze_spectra, plot_record

cva_plotting.set_show(False)
FS, NSEG = 2000.0, 1024


@pytest.fixture(scope='module')
def record():
    """Two inputs band-limited to 400 Hz, three responses; the third has a
    squared term, so it is partly non-coherent and reaches above the band."""
    rng = np.random.default_rng(3)
    n = 80000
    b = np.sinc(0.4 * (np.arange(-40, 41))) * np.hamming(81) * 0.4
    u = lfilter(b, 1, rng.standard_normal((n, 2)), axis=0)
    y = np.empty((n, 3))
    y[:, 0] = lfilter([1.0], [1, -1.5, 0.9], u[:, 0]) + 0.5 * u[:, 1]
    y[:, 1] = lfilter([1.0], [1, -0.6, 0.8], u[:, 1])
    y[:, 2] = u[:, 0] - u[:, 1] + 3.0 * u[:, 0] ** 2
    y += 0.02 * rng.standard_normal(y.shape)
    return u, y


def _matrices(u, y):
    kw = dict(fs=FS, nperseg=NSEG, noverlap=NSEG // 2, window='hann')
    f, _ = csd(u[:, 0], u[:, 0], **kw)
    X = np.hstack([u - u.mean(0), y - y.mean(0)])
    n = X.shape[1]
    C = np.zeros((n, n, len(f)), complex)       # C[i, j] = E[x_i conj(x_j)]
    for i in range(n):
        for j in range(n):
            C[i, j] = csd(X[:, j], X[:, i], **kw)[1]
    return f, C


def test_spectra_route_matches_time_histories(record):
    u, y = record
    ref = analyze_record(u, y, FS, nperseg=NSEG)
    f, C = _matrices(u, y)
    Cu, Cy, Cyu = C[:2, :2], C[2:, 2:], C[2:, :2]
    H = np.stack([Cyu[:, :, k] @ np.linalg.inv(Cu[:, :, k])
                  for k in range(len(f))], axis=2)

    # frequency responses with the input matrix
    a = analyze_spectra(f, c_resp=Cy, H=H, c_in=Cu, n_avg=155)
    # the same arrays with the frequency axis first
    b = analyze_spectra(f, c_resp=np.moveaxis(Cy, 2, 0),
                        H=np.moveaxis(H, 2, 0), c_in=np.moveaxis(Cu, 2, 0))
    # coherence handed over directly
    c = analyze_spectra(f, s_resp=ref.s_tot, coherence=ref.coh,
                        s_in=ref.s_in)
    # input-to-response cross-spectra, E[conj(u) y]
    Suy = np.transpose(Cyu, (1, 0, 2))     # E[y conj(u)] is the same number
    d = analyze_spectra(f, c_resp=Cy, c_in_resp=Suy, c_in=Cu)

    for r in (a, b, c, d):
        assert r.from_spectra and r.oob_known
        np.testing.assert_allclose(r.frac_noncoherent, ref.frac_noncoherent,
                                   atol=0.003)
        np.testing.assert_allclose(r.frac_out_of_band, ref.frac_out_of_band,
                                   atol=0.003)
        np.testing.assert_allclose(r.rms, ref.rms, rtol=0.01)
        assert abs(r.band[1] - ref.band[1]) < 10
    # the squared term shows: non-coherent and above the band
    assert a.frac_noncoherent[2] > 0.2 > a.frac_noncoherent[1]
    assert a.frac_out_of_band[2] > 0.02 > a.frac_out_of_band[1]
    assert np.isnan(a.skew).all()
    a.table(verbose=False)
    assert plot_record(a) is not None


def test_spectra_that_stop_at_the_band_edge(record):
    u, y = record
    ref = analyze_record(u, y, FS, nperseg=NSEG)
    keep = ref.freq <= ref.band[1]
    r = analyze_spectra(ref.freq[keep], s_resp=ref.s_tot[keep],
                        coherence=ref.coh[keep], s_in=ref.s_in[keep])
    assert not r.oob_known
    assert np.isnan(r.frac_out_of_band).all()
    assert any('cannot be seen' in n for n in r.notes())
    assert plot_record(r) is not None


def test_spectra_route_refuses_what_is_not_enough(record):
    u, y = record
    ref = analyze_record(u, y, FS, nperseg=NSEG)
    f, C = _matrices(u, y)
    with pytest.raises(ValueError, match='only the coherent part'):
        analyze_spectra(f, s_resp=ref.s_tot, s_in=ref.s_in)
    with pytest.raises(ValueError, match='response spectral density'):
        analyze_spectra(f, coherence=ref.coh, s_in=ref.s_in)
    with pytest.raises(ValueError, match='exceeds the measured response'):
        analyze_spectra(f, s_resp=0.2 * ref.s_tot, s_in=ref.s_in,
                        H=np.ones((3, 2, len(f))) * 5, c_in=C[:2, :2])
