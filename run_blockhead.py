"""
run_blockhead.py
================
Every nonlinearity analysis on the Area III 6-DOF blockhead test (J. S. Cap,
2017): 23 s of random vibration at 12,800 Hz, six table motions as inputs
(X, Y, Z and three rotations).

    python run_blockhead.py            # the 12 "target" channels on the block
    python run_blockhead.py --full     # all 24 block channels
    python run_blockhead.py --nut      # lower nut accelerometer, A10
    python run_blockhead.py --base     # base plate corners, A1-A4

Pages and tables go to results/blockhead/, blockhead24/, blockhead_nut/ and
blockhead_base/.

The six inputs are computed from the base plate channels, so with --base the
non-coherent figures are low by construction; skewness, kurtosis,
out-of-band energy and coupling do not depend on the inputs.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio

from cva_report import full_report

DATA = Path('/Users/nhunterjr/Code/shared/MIMO Work/mimo_blockhead_study/'
            'Case01_XYZRXRYRZ_Group_Accel.mat')
OUT = Path(__file__).parent / 'results'


# set: (response array, channel names, output folder, channel for the
#       single-channel pages)
SETS = {'target': ('arsp_target', 'crsp_target', 'blockhead', 'A17-X'),
        'full': ('arsp_full', 'crsp_full', 'blockhead24', 'A17-X'),
        'nut': ('ainp_center', 'cinp_center', 'blockhead_nut', 'A10-X'),
        'base': ('ainp_corners', 'cinp_corners', 'blockhead_base', 'A1-X')}


def load(which='target'):
    """Inputs, responses, sample rate and channel names for one of SETS.
    True and False are accepted for 'full' and 'target'."""
    which = {True: 'full', False: 'target'}.get(which, which)
    data, chan = SETS[which][:2]
    m = sio.loadmat(DATA)
    fs = float(m['SampleRate'].squeeze())
    names = [str(r[1][0]) for r in m[chan]]
    return m['ainp_6dof'], m[data], fs, names


def main(which='target'):
    which = {True: 'full', False: 'target'}.get(which, which)
    u, y, fs, names = load(which)
    folder, channel = SETS[which][2:]
    # six inputs: 8192-sample segments give about 70 averages (1.6 Hz)
    return full_report(u, y, fs, OUT / folder,
                       label='blockhead 6-DOF, ' + which, names=names,
                       channel=names.index(channel),
                       nperseg=8192, track_lags=8)


if __name__ == '__main__':
    import sys
    picked = [k for k in SETS if '--' + k in sys.argv] or ['target']
    for k in picked:
        main(k)
