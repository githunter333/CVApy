"""
run_blockhead.py
================
Every nonlinearity analysis on the Area III 6-DOF blockhead test (J. S. Cap,
2017): 23 s of random vibration at 12,800 Hz, six table motions as inputs
(X, Y, Z and three rotations), twelve "target" accelerometers on the block
as responses.

    python run_blockhead.py            # the 12 target channels
    python run_blockhead.py --full     # all 24 response channels

Pages and tables go to results/blockhead/ and results/blockhead24/.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio

from cva_report import full_report

DATA = Path('/Users/nhunterjr/Code/shared/MIMO Work/mimo_blockhead_study/'
            'Case01_XYZRXRYRZ_Group_Accel.mat')
OUT = Path(__file__).parent / 'results'


def load(full: bool = False):
    """Inputs, responses, sample rate and channel names. full=False gives
    the 12 "target" accelerometers of the original study; full=True all 24
    (four corners on top of the block, four underneath, X Y Z each)."""
    m = sio.loadmat(DATA)
    fs = float(m['SampleRate'].squeeze())
    key = 'full' if full else 'target'
    names = [str(r[1][0]) for r in m['crsp_' + key]]
    return m['ainp_6dof'], m['arsp_' + key], fs, names


def main(full: bool = False):
    u, y, fs, names = load(full)
    # six inputs: 8192-sample segments give about 70 averages (1.6 Hz)
    return full_report(u, y, fs, OUT / ('blockhead24' if full else 'blockhead'),
                       label='blockhead 6-DOF', names=names,
                       channel=names.index('A17-X'),   # the one skewed channel
                       nperseg=8192, track_lags=8)


if __name__ == '__main__':
    import sys
    main(full='--full' in sys.argv)
