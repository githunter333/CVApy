"""The one-call report runs every step and writes its pages."""
import cva_plotting
from cva_report import STEPS, analyze_all, full_report, plot_all
from cva_sim import simulate_chain

cva_plotting.set_show(False)


def test_full_report_writes_every_page(tmp_path):
    rec = simulate_chain(duration=12.0, softness=0.3, seed=1)
    res = full_report(rec.inputd, rec.datout, rec.fs, tmp_path, channel=3,
                      verbose=False)
    for key in ('spectra', 'residual', 'track', 'level', 'hos', 'regimes'):
        assert key in res
    made = {p.name for p in tmp_path.iterdir()}
    assert {'1_spectra.png', '3_level.png', '5_regimes.png',
            'tables.txt'} <= made
    assert '***' not in (tmp_path / 'tables.txt').read_text()


def test_steps_can_be_chosen_and_stopped():
    rec = simulate_chain(duration=8.0, softness=0.0, seed=1)
    seen = []
    res, _ = analyze_all(rec.inputd, rec.datout, rec.fs, steps=['spectra'],
                         progress=lambda pct, name: seen.append(name))
    assert 'spectra' in res and 'hos' not in res and 'residual' not in res
    assert seen[0] == 'spectra' and seen[-1] == 'done'
    assert [t for t, _, _ in plot_all(res)] == ['Nonlinearity']
    res, text = analyze_all(rec.inputd, rec.datout, rec.fs, steps=STEPS,
                            stop=lambda: True)
    assert 'spectra' not in res and 'stopped' in text
