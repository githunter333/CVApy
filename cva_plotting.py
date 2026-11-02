"""
cva_plotting.py
===============
Display policy for CVApy figures.

Every plot method in cva_run / cva_phi_cluster / cva_mode_shapes /
cva_diagnostics builds a Matplotlib Figure, optionally saves it, then calls
``show_if_enabled()`` and *returns the figure*.

Script use (the default) keeps the old behaviour: ``plt.show()`` pops the
window. A GUI embeds the returned figure in its own canvas instead and must
first call ``set_show(False)``, otherwise Matplotlib would also try to open a
second, detached window for every plot.

    import cva_plotting
    cva_plotting.set_show(False)        # GUI owns the figures now
"""
from __future__ import annotations

import matplotlib.pyplot as plt

_SHOW = True


def set_show(flag: bool) -> None:
    """Enable (script) or disable (GUI) the blocking plt.show() calls."""
    global _SHOW
    _SHOW = bool(flag)


def show_enabled() -> bool:
    return _SHOW


def show_if_enabled() -> None:
    if _SHOW:
        plt.show()
