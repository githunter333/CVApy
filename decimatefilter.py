#!/usr/bin/env python3
"""
decimatefilter.py
=================
Decimate a raw acquisition record by an integer factor and write a CVApy-ready
text file: [time, force, accel...].

Usage
-----
    python decimatefilter.py INPUT.txt OUTPUT.txt [-q 2]
                             [--force-col 2] [--accel-col 3 [4 ...]]

Never writes to INPUT. Refuses to overwrite OUTPUT unless --overwrite is given.
(The previous version read and wrote the same hardcoded path, destroying the
source record on every run.)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import decimate


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('-q', '--factor', type=int, default=2,
                    help='decimation factor (default 2)')
    ap.add_argument('--time-col', type=int, default=0)
    ap.add_argument('--force-col', type=int, default=2)
    ap.add_argument('--accel-col', type=int, nargs='+', default=[3],
                    help='one or more acceleration column indices')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)

    if not args.input.exists():
        sys.exit(f'Input not found: {args.input}')
    if args.output.exists() and args.input.resolve() == args.output.resolve():
        sys.exit('Refusing to write output over the input file.')
    if args.output.exists() and not args.overwrite:
        sys.exit(f'Output exists: {args.output}  (pass --overwrite to replace)')

    print(f'Loading {args.input} ...')
    data = np.loadtxt(args.input)
    q = args.factor

    # Time is subsampled, not filtered — it is already monotonic.
    time = data[::q, args.time_col]
    cols = [decimate(data[:, c], q=q, ftype='fir', zero_phase=True)
            for c in [args.force_col] + list(args.accel_col)]

    # decimate() can return one more/fewer sample than time[::q]; trim to match.
    n = min(len(time), *(len(c) for c in cols))
    out = np.column_stack([time[:n]] + [c[:n] for c in cols])

    np.savetxt(args.output, out, fmt='%.8f')
    print(f'  in  : {data.shape[0]} samples, {data.shape[1]} cols')
    print(f'  out : {out.shape[0]} samples, {out.shape[1]} cols  (q={q})')
    print(f'  wrote {args.output}')


if __name__ == '__main__':
    main()
