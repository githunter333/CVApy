"""
format_r6dof.py
===============
Formats r6dof .dat input/output files into the standard CVA pipeline format:
    col 1       : time (s)  — generated from sample rate
    col 2       : force (input)
    cols 3..N+2 : accelerations (outputs)

Naming convention
-----------------
Input file  : <base>in.dat   e.g. r6dofcubic2in.dat
Output file : <base>out.dat  e.g. r6dofcubic2out.dat
Formatted   : <OUTPUT_DIR>/<base>_cva.txt

================================================================
USAGE
================================================================
1. Set DATA_DIR to the folder containing the .dat files
2. Set BASE_NAMES to the list of run base names to format
   e.g. ['r6dofcubic2', 'r6dofcubic3', 'r6doflinear1']
3. Run the script — formatted files appear in DATA_DIR/cva_formatted/

================================================================
PARAMETERS
================================================================
FS          : sample rate (Hz) — used to generate time column
DELIMITER   : column separator in .dat files ('auto' detects whitespace/comma)
SKIP_HEADER : number of header rows to skip (0 = none)

================================================================
Author: Norman / Claude
Date:   2026-04-13
"""

from pathlib import Path
import numpy as np

# =============================================================================
# ===== USER SETTINGS =====
# =============================================================================

DATA_DIR   = Path('/Users/nhunterjr/Documents/Code/shared/Nonlinear Data/Experimental_Data/r6dof')

# List of base names to format — script appends 'in.dat' and 'out.dat'
BASE_NAMES = [
     'r6dofblinin80pct2'
     'r6dofcubic2s',
     'r6dofdeadzone2'

]

FS           = 2048.0   # sample rate Hz
DELIMITER    = 'auto'   # 'auto', ' ', ',', '\t'
SKIP_HEADER  = 0        # header rows to skip
MAX_SAMPLES  = 16384    # max samples to keep (None = keep all)
                        # 16384 = 8 s at 2048 Hz, good balance of speed vs coverage
                        # Set to None to use full file length
N_OUTPUT_COLS = 6       # number of acceleration columns to use (None = use all)
                        # e.g. 6 = use first 6 columns even if file has 9

# =============================================================================
# ===== HELPERS =====
# =============================================================================

def load_dat(filepath: Path, delimiter: str, skip_header: int) -> np.ndarray:
    """Load a .dat text file, auto-detecting delimiter if needed."""
    if delimiter == 'auto':
        data = np.loadtxt(str(filepath), skiprows=skip_header)
    else:
        data = np.loadtxt(str(filepath), delimiter=delimiter,
                          skiprows=skip_header)
    return np.atleast_2d(data)


def format_run(base: str, data_dir: Path, fs: float,
               delimiter: str, skip_header: int,
               output_dir: Path,
               max_samples:   int = None,
               n_output_cols: int = None) -> Path:
    """
    Format one run's input/output .dat files into a single CVA text file.

    Parameters
    ----------
    base       : base name e.g. 'r6dofcubic2'
    data_dir   : directory containing the .dat files
    fs         : sample rate Hz
    delimiter  : column delimiter or 'auto'
    skip_header: header rows to skip
    output_dir : where to write the formatted file

    Returns
    -------
    Path of the formatted output file
    """
    in_file  = data_dir / f'{base}in.dat'
    out_file = data_dir / f'{base}out.dat'

    # Validate files exist
    if not in_file.exists():
        raise FileNotFoundError(f'Input file not found: {in_file}')
    if not out_file.exists():
        raise FileNotFoundError(f'Output file not found: {out_file}')

    print(f'\nFormatting: {base}')
    print(f'  Input  : {in_file.name}')
    print(f'  Output : {out_file.name}')

    # Load
    force_data = load_dat(in_file,  delimiter, skip_header)
    accel_data = load_dat(out_file, delimiter, skip_header)

    # Ensure 2D and correct orientation (rows = samples)
    if force_data.ndim == 1:
        force_data = force_data.reshape(-1, 1)
    if force_data.shape[0] == 1:
        force_data = force_data.T
    if accel_data.shape[0] == 1:
        accel_data = accel_data.T

    # Extract force column (first column)
    force = force_data[:, 0:1]    # (N, 1)

    # All columns of output file are accelerations
    accel = accel_data            # (N, n_out)

    # Trim to requested number of output columns
    if n_output_cols is not None and accel.shape[1] > n_output_cols:
        print(f'  Trimming output cols: {accel.shape[1]} -> {n_output_cols}')
        accel = accel[:, :n_output_cols]

    # Check lengths match
    n_force = force.shape[0]
    n_accel = accel.shape[0]
    if n_force != n_accel:
        print(f'  WARNING: length mismatch — force={n_force}, accel={n_accel}')
        n_samples = min(n_force, n_accel)
        force = force[:n_samples, :]
        accel = accel[:n_samples, :]
        print(f'  Truncating to {n_samples} samples')
    else:
        n_samples = n_force

    # Generate time column
    time = np.arange(n_samples).reshape(-1, 1) / fs   # (N, 1)

    # Truncate if requested
    if max_samples is not None and n_samples > max_samples:
        time  = time[:max_samples, :]
        force = force[:max_samples, :]
        accel = accel[:max_samples, :]
        print(f'  Truncated : {n_samples} -> {max_samples} samples'
              f'  ({max_samples/fs:.2f} s)')
        n_samples = max_samples

    # Assemble: [time, force, accel1, accel2, ...]
    combined = np.hstack([time, force, accel])

    # Save
    out_path = output_dir / f'{base}_cva.txt'
    np.savetxt(str(out_path), combined, fmt='%.8f')

    print(f'  Samples   : {n_samples}  ({n_samples/fs:.2f} s)')
    print(f'  Channels  : {accel.shape[1]} accelerations')
    print(f'  Saved     : {out_path.name}')

    return out_path


# =============================================================================
# ===== MAIN =====
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('  r6dof File Formatter — CVA Pipeline')
    print('=' * 60)
    print(f'  Source dir : {DATA_DIR}')
    print(f'  Sample rate: {FS} Hz')
    print(f'  Runs       : {len(BASE_NAMES)}')

    # Create output directory
    output_dir = DATA_DIR / 'cva_formatted'
    output_dir.mkdir(exist_ok=True)
    print(f'  Output dir : {output_dir}')

    formatted_files = []
    errors          = []

    for base in BASE_NAMES:
        try:
            out_path = format_run(
                base          = base,
                data_dir      = DATA_DIR,
                fs            = FS,
                delimiter     = DELIMITER,
                skip_header   = SKIP_HEADER,
                output_dir    = output_dir,
                max_samples   = MAX_SAMPLES,
                n_output_cols = N_OUTPUT_COLS,
            )
            formatted_files.append(out_path)
        except Exception as e:
            print(f'  ERROR on {base}: {e}')
            errors.append((base, str(e)))

    # Summary
    print('\n' + '=' * 60)
    print(f'  Done. {len(formatted_files)} files formatted.')
    if errors:
        print(f'  {len(errors)} errors:')
        for base, msg in errors:
            print(f'    {base}: {msg}')

    print('\nFormatted files:')
    for p in formatted_files:
        print(f'  {p}')

    print('\nTo run CVA pipeline on a formatted file, set in run_cva_pipeline.py:')
    if formatted_files:
        print(f"  DATA_FILE = '{formatted_files[0]}'")
    print('=' * 60)