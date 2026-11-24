"""
cva_report.py
=============
Run every nonlinearity analysis on one record.

    from cva_report import full_report
    out = full_report(inputd, datout, fs, 'results/mytest',
                      label='0 dB', names=channel_names, channel=3)

Pages written (PNG) and what each answers:

    1_spectra.png        how much of each response is non-coherent or
                         above the excitation band; skewness, kurtosis
    2_residual_<ch>.png  when the non-coherent response occurs
    3_level.png          do in-band modes change with response level
    3_band_<Hz>.png      each response peak above the excitation band:
                         time history, frequency, damping
    4_coupling_<ch>.png  asymmetric or symmetric nonlinearity
    5_regimes.png        one state or two at the same level
    tables.txt           every table printed by the steps above

Works with one input or several (multiple coherence throughout). A step
that fails is reported in the tables and the rest carry on.

The work is in two halves so a GUI can do the arithmetic on a worker thread
and the drawing on its own:

    res, text = analyze_all(inputd, datout, fs, ...)    # no matplotlib
    pages = plot_all(res, channel=3)                    # [(title, Figure)]
"""
from __future__ import annotations

import contextlib
import io
import traceback
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

STEPS = ('spectra', 'residual', 'level', 'bands', 'coupling', 'regimes')
STEP_TITLES = {'spectra': 'Nonlinearity', 'residual': 'Residual',
               'level': 'Level', 'bands': 'Out-of-band',
               'coupling': 'Coupling', 'regimes': 'Regimes'}


def analyze_all(inputd, datout, fs: float, label: str = '',
                names: Optional[Sequence[str]] = None, channel: int = 0,
                nperseg: int = 0, hos_nperseg: int = 1024,
                track_lags: int = 2, block_sec: float = 0.5,
                skip_sec: float = 0.0, time=None,
                steps: Sequence[str] = STEPS,
                progress: Optional[Callable[[int, str], None]] = None,
                stop: Optional[Callable[[], bool]] = None):
    """
    Compute the requested steps. No figures are made here.

    inputd, datout : (N, n_in) and (N, n_out) arrays.
    channel   : response channel for the single-channel tables.
    nperseg   : Welch segment for the coherence and the residual filter
                (0 = automatic, about 0.3 Hz). With several inputs keep at
                least ten segments per input or the coherence reads high.
    hos_nperseg: segment for the coupling maps; see cva_hos.analyze_hos.
    track_lags: model lags for the in-band mode tracks; the model holds
                lags x n_out states, so half that many modes.
    steps     : any of STEPS. 'level', 'bands' and 'coupling' need the
                excitation band and 'regimes' needs the mode tracks; what
                they need is computed even if not asked for.
    progress  : called as progress(percent, step name) before each step.
    stop      : polled between steps; returning True abandons the rest.

    Returns (res, text): res is a dict keyed 'spectra', 'residual',
    'track', 'level', 'bands', 'hos', 'regimes' (a key is missing if its
    step was skipped or failed); text is everything the steps printed.
    """
    from cva_bands import analyze_bands
    from cva_hos import analyze_hos
    from cva_level import level_dependence, track_in_band
    from cva_regimes import analyze_regimes
    from cva_residual import noncoherent_history
    from cva_spectra import analyze_record

    U = np.atleast_2d(np.asarray(inputd, float).T).T
    Y = np.atleast_2d(np.asarray(datout, float).T).T
    k0 = int(round(skip_sec * fs))
    U, Y = U[k0:], Y[k0:]
    t = ((k0 + np.arange(len(U))) / fs if time is None
         else np.asarray(time, float)[k0:])
    if names is None:
        names = [f"ch {c + 1}" for c in range(Y.shape[1])]
    names = [str(n) for n in names]
    channel = int(np.clip(channel, 0, Y.shape[1] - 1))
    res = dict(_names=names, _channel=channel, _label=label)
    log = io.StringIO()
    want = set(steps)

    def spectra():
        res['spectra'] = analyze_record(U, Y, fs, label=label, names=names,
                                        nperseg=nperseg)
        if 'spectra' in want:
            res['spectra'].table()

    def residual():
        res['residual'] = noncoherent_history(U, Y, fs, label=label,
                                              names=names, nperseg=nperseg)
        res['residual'].table()

    def track():
        top = min(res['spectra'].band[1], 0.45 * fs)
        res['track'] = track_in_band(t, U, Y, fs, top, lags=track_lags,
                                     block_sec=block_sec)
        res['track'].summary()

    def level():
        res['level'] = level_dependence(res['track'], label=label)
        res['level'].table()

    def bands():
        top = res['spectra'].band[1]
        if top >= 0.45 * fs:
            print('\nOut-of-band: the excitation band reaches Nyquist; '
                  'nothing above it to analyse.')
            return
        res['bands'] = analyze_bands(Y, fs, top, label=label, names=names,
                                     block_sec=block_sec)
        res['bands'].table()

    def coupling():
        res['hos'] = analyze_hos(Y, fs, label=label, names=names,
                                 band_top=res['spectra'].band[1],
                                 nperseg=hos_nperseg)
        res['hos'].table()
        res['hos'].pair_table(channel)

    def regimes():
        res['regimes'] = analyze_regimes(res['track'], res.get('residual'),
                                         label=label)
        res['regimes'].summary()

    plan = []
    if want:
        plan.append(('spectra', spectra))
    if 'residual' in want:
        plan.append(('residual', residual))
    if want & {'level', 'regimes'}:
        plan.append(('mode tracks', track))
    for name, fn in (('level', level), ('bands', bands),
                     ('coupling', coupling), ('regimes', regimes)):
        if name in want:
            plan.append((name, fn))

    for i, (name, fn) in enumerate(plan):
        if stop is not None and stop():
            log.write('\n*** stopped ***\n')
            break
        if progress is not None:
            progress(int(100 * i / len(plan)), name)
        try:
            with contextlib.redirect_stdout(log):
                fn()
        except Exception:
            log.write(f"\n*** {name} failed ***\n{traceback.format_exc()}\n")
    if progress is not None:
        progress(100, 'done')
    return res, log.getvalue()


def plot_all(res: dict, channel: Optional[int] = None, f_max: float = 0.0,
             dpi: int = 130, steps: Sequence[str] = STEPS):
    """Draw the pages for whatever analyze_all produced.
    Returns a list of (title, Figure, file stem)."""
    from cva_bands import plot_band
    from cva_hos import plot_hos
    from cva_level import plot_level
    from cva_regimes import plot_regimes
    from cva_residual import plot_residual
    from cva_spectra import plot_record

    c = res['_channel'] if channel is None else channel
    tag = res['_names'][c].replace(' ', '_').replace('/', '-')
    want = set(steps)
    pages = []
    if 'spectra' in want and 'spectra' in res:
        pages.append(('Nonlinearity',
                      plot_record(res['spectra'], f_max=f_max, dpi=dpi),
                      '1_spectra'))
    if 'residual' in want and 'residual' in res:
        pages.append(('Residual',
                      plot_residual(res['residual'], channel=c, f_max=f_max,
                                    dpi=dpi), f"2_residual_{tag}"))
    if 'level' in want and 'level' in res:
        pages.append(('Level', plot_level([res['level']], dpi=dpi),
                      '3_level'))
    if 'bands' in want and 'bands' in res:
        for k, b in enumerate(res['bands'].bands):
            pages.append((f"Band {b.f0:.0f} Hz",
                          plot_band(res['bands'], k, dpi=dpi),
                          f"3_band_{b.f0:.0f}Hz"))
    if 'coupling' in want and 'hos' in res:
        pages.append(('Coupling', plot_hos(res['hos'], channel=c, dpi=dpi),
                      f"4_coupling_{tag}"))
    if 'regimes' in want and 'regimes' in res:
        pages.append(('Regimes', plot_regimes(res['regimes'], dpi=dpi),
                      '5_regimes'))
    return pages


def full_report(inputd, datout, fs: float, out_dir, label: str = '',
                names: Optional[Sequence[str]] = None, channel: int = 0,
                verbose: bool = True, **kw) -> dict:
    """analyze_all, then plot_all saved as PNG into out_dir with the tables
    in tables.txt. Keyword arguments are those of analyze_all."""
    import matplotlib.pyplot as plt

    import cva_plotting
    cva_plotting.set_show(False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res, text = analyze_all(inputd, datout, fs, label=label, names=names,
                            channel=channel, **kw)
    try:
        for _, fig, stem in plot_all(res, steps=kw.get('steps', STEPS)):
            fig.savefig(out_dir / f"{stem}.png", dpi=130)
    except Exception:
        text += f"\n*** plotting failed ***\n{traceback.format_exc()}\n"
    plt.close('all')
    (out_dir / 'tables.txt').write_text(text)
    if verbose:
        print(text)
    return res
