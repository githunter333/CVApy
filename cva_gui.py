#!/usr/bin/env python3
"""
cva_gui.py
==========
PySide6 desktop front end for the CVApy three-stage pipeline.

    conda activate cvapy
    python cva_gui.py

Layout
------
Left   : parameter panel (record, analysis window, CVA, clustering, modes)
Right  : one tab per figure, each with the standard Matplotlib toolbar
Bottom : log pane fed by the pipeline's own stdout, plus a progress bar

Workflow
--------
Choosing a record inspects it at once — header rows, delimiter, channel names
and units, whether there is a time column, whether an in/out sibling exists —
and fills the Record panel from what it finds. The record then loads in full
and appears as a time history and a spectral density.

The analysis window is chosen from the time history: drag across any panel,
type the times, or press Suggest. Run Pipeline applies the window and reuses
the already-parsed arrays rather than reading the files again.

See cva_record.py for the three record shapes this handles.

The pipeline runs on a QThread, so the window stays responsive during the
multi-minute CVA sweep. Figures are built on the GUI thread afterwards and
embedded with FigureCanvasQTAgg; the Matplotlib backend is Agg and
cva_plotting.set_show(False) is set, so no detached plot windows ever open.
"""
from __future__ import annotations

import signal
import sys
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path

import matplotlib

# Figures are embedded in our own canvases — never shown in their own window.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cva_plotting

cva_plotting.set_show(False)          # the GUI owns every figure

import cva_record
from cva_record import RecordSpec, default_spec, sniff_record
from cva_mode_shapes import CVAModeShapes
from cva_phi_cluster import CVAPhiCluster
from cva_run import CVAConfig, CVARun

ROLES = ('Time', 'Excitation', 'Response', 'Ignore')

# What 'Suggest' is willing to spend on a sweep, in windows. Only a ceiling:
# a transient whose ringdown is shorter gets the shorter window.
SUGGEST_MAX_WINDOWS = 5000

plt.rcParams['figure.max_open_warning'] = 0

DEFAULT_RESULTS_ROOT = Path('/Users/nhunterjr/Documents/DEVONthink/CVA_Results')


# =============================================================================
# stdout -> line buffer
# =============================================================================

class _LineBuffer:
    """File-like object collecting completed lines for the GUI to drain.

    Deliberately holds no Qt objects. The CVA sweep prints from joblib
    worker threads, and emitting a Qt signal from a thread Qt did not create
    hangs the event loop — the GUI polls this buffer on a timer instead.
    deque.append / popleft are atomic, so no lock is needed.
    """

    def __init__(self, sink: deque):
        self._sink = sink
        self._buf = ''

    def write(self, text):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._sink.append(line)

    def flush(self):
        if self._buf:
            self._sink.append(self._buf)
            self._buf = ''


# =============================================================================
# Worker — runs the pipeline off the GUI thread
# =============================================================================

class PipelineWorker(QObject):
    done     = Signal(object, object, object)   # run, clust, ms
    failed   = Signal(str)

    def __init__(self, params, existing=None, cache=None):
        super().__init__()
        self.p = params
        self.run_obj, self.clust_obj, self.ms_obj = existing or (None, None, None)
        self.cache = dict(cache) if cache else {}   # arrays read by the preview
        self.lines = deque()           # log lines, drained by the GUI timer
        self.status = (0, 'Starting …')      # (percent, stage text)
        self._stop = False

    def request_stop(self):
        """Cooperative stop — polled once per window during the sweep and between
        stages, so a long run gives up within a window rather than at the
        end of the stage."""
        self._stop = True

    def stop_requested(self):
        return self._stop

    def _set(self, pct, label):
        self.status = (pct, label)

    def _cva_progress(self, done, total, elapsed, eta):
        """Called from CVARun.run(), on joblib worker threads as well as
        this one — so it only writes an attribute. Maps the sweep onto
        5-55% of the bar."""
        self._set(5 + int(50.0 * done / max(1, total)),
                  f'Stage 1 — CVA sweep   {done}/{total}   '
                  f'({done/max(1, total):.0%})   elapsed {elapsed:.0f}s'
                  f'   eta {eta:.0f}s')

    @Slot()
    def execute(self):
        p = self.p
        saved_stdout = sys.stdout
        sys.stdout = _LineBuffer(self.lines)
        try:
            run, clust, ms = self.run_obj, self.clust_obj, self.ms_obj

            # ---------------- Stage 1: CVA ----------------
            if p['run_cva']:
                self._set(1, 'Stage 1 — reading record')
                print('--- Step 1: CVA Run ---')
                cfg = CVAConfig(
                    lags=p['lags'], tol=p['tol'], neighbors=p['neighbors'],
                    ranklock=p['ranklock'], eval_stride=p['eval_stride'],
                    fs=p['fs'], freq_min=p['freq_min'], freq_max=p['freq_max'],
                    sig_threshold_fraction=p['sig_threshold_fraction'],
                    sig_peak_mindist=p['sig_peak_mindist'])
                run = CVARun(cfg)
                if self.cache:
                    print('Reusing the record read for the preview')
                print(f'Record: {p["spec"].summary()}')
                run.load_record(p['spec'], cache=self.cache)
                self._set(3, 'Stage 1 — building database')
                run.build_database()
                run.run(n_jobs=p['n_jobs'], progress_cb=self._cva_progress,
                        stop_cb=self.stop_requested)
                self._set(55, 'Stage 1 — eigenvalue significance')
                run.eigenvalue_significance()
                if p['save_hdf5']:
                    run.save(str(p['h5_run']))
            elif run is None:
                self._set(0, 'Stage 1 — loading')
                print(f'--- Step 1: Loading {Path(p["h5_run"]).name} ---')
                run = CVARun.load(str(p['h5_run']))

            if self._stop:
                raise InterruptedError('Stopped after stage 1')

            # ---------------- Stage 2: clustering ----------------
            if p['run_cluster']:
                self._set(60, 'Stage 2 — clustering Phi')
                print('\n--- Step 2: Phi Clustering ---')
                clust = CVAPhiCluster(run, k_max=p['k_max'],
                                      n_pca_var=p['n_pca_var'],
                                      kmeans_rep=p['kmeans_rep'],
                                      f_min=p['f_min'], f_max=p['f_max'])
                clust.fit(n_clusters=p['n_clusters'])
                if p['save_hdf5']:
                    clust.save(str(p['h5_clust']))
            elif clust is None and Path(p['h5_clust']).exists():
                print(f'--- Step 2: Loading {Path(p["h5_clust"]).name} ---')
                clust = CVAPhiCluster.load(str(p['h5_clust']), run)

            if self._stop:
                raise InterruptedError('Stopped after stage 2')

            # ---------------- Stage 3: mode shapes ----------------
            if p['run_modes'] and clust is not None:
                self._set(85, 'Stage 3 — extracting mode shapes')
                print('\n--- Step 3: Mode Shape Extraction ---')
                ms = CVAModeShapes(run, clust, min_im_frac=p['min_im_frac'],
                                   f_min=p['f_min'], zeta_max=p['zeta_max'],
                                   n_modes_max=p['n_modes_max'])
                ms.extract()
                if p['save_hdf5']:
                    ms.save(str(p['h5_modes']))
            elif ms is None and clust is not None and Path(p['h5_modes']).exists():
                print(f'--- Step 3: Loading {Path(p["h5_modes"]).name} ---')
                ms = CVAModeShapes.load(str(p['h5_modes']), run, clust)

            self._set(100, 'Complete')
            sys.stdout.flush()
            self.done.emit(run, clust, ms)

        except InterruptedError as e:
            sys.stdout.flush()
            self.lines.append(f'\n*** {e} ***')
            self._set(0, 'Stopped')
            self.done.emit(self.run_obj, self.clust_obj, self.ms_obj)
        except Exception:
            sys.stdout.flush()
            self.failed.emit(traceback.format_exc())
        finally:
            sys.stdout = saved_stdout


# =============================================================================
# Preview worker — parses the record so its spectra can be plotted
# =============================================================================

class PreviewWorker(QObject):
    """Reads a record off the GUI thread and hands back a loaded CVARun.

    The run holds time / input / output arrays but no CVA results, which is
    all plot_spectral_density() needs. The parsed array comes back too, so
    the pipeline can skip a second np.loadtxt of the same file.
    """

    done   = Signal(object, object)      # run, {path: array}
    failed = Signal(str)

    def __init__(self, params, cache=None):
        super().__init__()
        self.p = params
        self.cache = dict(cache) if cache else {}
        self.lines = deque()
        self.status = (0, 'Reading record …')

    def request_stop(self):
        """np.loadtxt cannot be interrupted — stopping just means waiting."""

    @Slot()
    def execute(self):
        p = self.p
        saved_stdout = sys.stdout
        sys.stdout = _LineBuffer(self.lines)
        try:
            cfg = CVAConfig(
                lags=p['lags'], tol=p['tol'], neighbors=p['neighbors'],
                ranklock=p['ranklock'], eval_stride=p['eval_stride'],
                fs=p['fs'], freq_min=p['freq_min'], freq_max=p['freq_max'],
                sig_threshold_fraction=p['sig_threshold_fraction'],
                sig_peak_mindist=p['sig_peak_mindist'])
            run = CVARun(cfg)
            # The preview always reads the whole record: the time window is
            # chosen by looking at it, so it cannot be applied beforehand.
            spec = replace(p['spec'], t_start=None, t_end=None)
            print(f'Record: {spec.summary()}')
            run.load_record(spec, cache=self.cache)
            sys.stdout.flush()
            self.done.emit(run, self.cache)
        except Exception:
            sys.stdout.flush()
            self.failed.emit(traceback.format_exc())
        finally:
            sys.stdout = saved_stdout


# =============================================================================
# Figure tab
# =============================================================================

class FigureTab(QWidget):
    def __init__(self, fig: Figure, parent=None):
        super().__init__(parent)
        self.fig = fig
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar2QT(canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(toolbar)
        lay.addWidget(canvas)
        canvas.draw_idle()


# =============================================================================
# Parameter panel
# =============================================================================

class ParamPanel(QScrollArea):

    data_chosen       = Signal(str)   # a record was picked from a dialog
    window_changed    = Signal()      # t_start / t_end edited by hand
    suggest_requested = Signal()      # 'Suggest' pressed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        inner = QWidget()
        self.setWidget(inner)
        root = QVBoxLayout(inner)
        self.w = {}

        # ---------- record ----------
        self._info = None          # sniff of the selected file
        self._pair_info = None     # sniff of its in/out sibling, if any

        g = QGroupBox('Record')
        f = QFormLayout(g)
        self.w['data_file'] = QLineEdit()
        self.w['data_file'].setPlaceholderText('select a record …')
        browse = QPushButton('Browse …')
        browse.clicked.connect(self._pick_data)
        row = QHBoxLayout()
        row.addWidget(self.w['data_file'])
        row.addWidget(browse)
        rw = QWidget()
        rw.setLayout(row)
        f.addRow('File', rw)

        # Only used for the '<base>in.dat' / '<base>out.dat' records, where
        # the excitation and the responses are in separate files. Filled in
        # automatically when the sibling is found on disk.
        self.w['pair_file'] = QLineEdit()
        self.w['pair_file'].setPlaceholderText('(all channels in one file)')
        pair_btn = QPushButton('…')
        pair_btn.setFixedWidth(30)
        pair_btn.clicked.connect(self._pick_pair)
        rowp = QHBoxLayout()
        rowp.addWidget(self.w['pair_file'])
        rowp.addWidget(pair_btn)
        rwp = QWidget()
        rwp.setLayout(rowp)
        f.addRow('Paired with', rwp)

        self.lbl_record = QLabel('— no record selected —')
        self.lbl_record.setWordWrap(True)
        self.lbl_record.setStyleSheet('color: #666;')
        f.addRow('Layout', self.lbl_record)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(['From', 'Channel', 'Units', 'Role'])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setMinimumHeight(150)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        f.addRow(self.tbl)

        f.addRow('fs (0 = from time col)', self._text('fs', '0'))
        f.addRow('decimate (1 = none)', self._spin('decimate', 1, 1, 200))
        root.addWidget(g)

        # ---------- analysis window ----------
        g = QGroupBox('Analysis window')
        f = QFormLayout(g)
        self.w['signal_type'] = QComboBox()
        self.w['signal_type'].addItems(['Auto', 'Transient', 'Random'])
        f.addRow('signal type', self.w['signal_type'])
        f.addRow('t start (s)', self._text('t_start', ''))
        f.addRow('t end (s)', self._text('t_end', ''))

        btn_full = QPushButton('Full record')
        btn_full.clicked.connect(self.window_full)
        btn_sugg = QPushButton('Suggest')
        btn_sugg.setToolTip('A window to suit the detected signal type — for '
                            'a transient, the impact and its ringdown')
        btn_sugg.clicked.connect(self.suggest_requested.emit)
        roww = QHBoxLayout()
        roww.addWidget(btn_full)
        roww.addWidget(btn_sugg)
        rww = QWidget()
        rww.setLayout(roww)
        f.addRow('', rww)

        self.lbl_cost = QLabel('—')
        self.lbl_cost.setWordWrap(True)
        f.addRow('Sweep size', self.lbl_cost)
        root.addWidget(g)

        for key in ('t_start', 't_end'):
            self.w[key].editingFinished.connect(self.window_changed.emit)
        self.w['fs'].editingFinished.connect(self._refresh_cost)
        self.w['decimate'].valueChanged.connect(self._refresh_cost)
        root.addWidget(self._output_group())

        # ---------- CVA ----------
        g = QGroupBox('CVA')
        f = QFormLayout(g)
        f.addRow('lags', self._spin('lags', 50, 2, 512))
        f.addRow('tol', self._text('tol', '1e-2'))
        f.addRow('neighbors', self._spin('neighbors', 1000, 10, 100000))
        f.addRow('ranklock (0=auto)', self._spin('ranklock', 0, 0, 512))
        f.addRow('eval stride', self._spin('eval_stride', 1, 1, 10000))
        f.addRow('n_jobs (-1=all)', self._spin('n_jobs', 1, -1, 64))
        f.addRow('freq min (Hz)', self._text('freq_min', '0'))
        f.addRow('freq max (0=Nyq)', self._text('freq_max', '0'))
        f.addRow('sig threshold frac', self._text('sig_threshold_fraction', '0.25'))
        f.addRow('sig peak min dist', self._spin('sig_peak_mindist', 3, 1, 50))
        root.addWidget(g)

        # ---------- clustering ----------
        g = QGroupBox('Clustering')
        f = QFormLayout(g)
        f.addRow('k max', self._spin('k_max', 8, 2, 32))
        f.addRow('PCA variance %', self._text('n_pca_var', '95.0'))
        f.addRow('k-means repeats', self._spin('kmeans_rep', 20, 1, 500))
        f.addRow('n clusters (0=auto)', self._spin('n_clusters', 0, 0, 32))
        f.addRow('band f_min (Hz)', self._text('f_min', '20'))
        f.addRow('band f_max (Hz)', self._text('f_max', '400'))
        root.addWidget(g)

        # ---------- mode shapes ----------
        g = QGroupBox('Mode shapes')
        f = QFormLayout(g)
        f.addRow('min imag fraction', self._text('min_im_frac', '0.15'))
        f.addRow('zeta max', self._text('zeta_max', '0.5'))
        f.addRow('max modes plotted', self._spin('n_modes_max', 12, 1, 64))
        root.addWidget(g)

        # ---------- stages ----------
        g = QGroupBox('Stages')
        f = QVBoxLayout(g)
        for key, label, default in (
                ('run_cva',     'Stage 1 — run CVA (else load HDF5)', True),
                ('run_cluster', 'Stage 2 — cluster Phi',              True),
                ('run_modes',   'Stage 3 — extract mode shapes',      True),
                ('save_hdf5',   'Save results to HDF5',               True)):
            cb = QCheckBox(label)
            cb.setChecked(default)
            self.w[key] = cb
            f.addWidget(cb)
        root.addWidget(g)

        # ---------- plots ----------
        g = QGroupBox('Plots')
        f = QVBoxLayout(g)
        for key, label in (
                ('plot_time_history',   'Time history (drag to set the window)'),
                ('plot_psd',            'Spectral density (excitation + response)'),
                ('plot_significance',   'Eigenvalue significance'),
                ('plot_freq_deviation', 'Frequency deviation'),
                ('plot_reconstruction', 'Modal reconstruction'),
                ('plot_iterated',       'Iterated prediction'),
                ('plot_clusters',       'Cluster diagnostics (6)'),
                ('plot_modes',          'Mode shapes (3+)')):
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.w[key] = cb
            f.addWidget(cb)

        pf = QFormLayout()
        pf.addRow('history input ch', self._spin('th_in_ch', 0, 0, 256))
        pf.addRow('history responses', self._spin('th_max_out', 3, 1, 32))
        pf.addRow('PSD nperseg (0=auto)', self._spin('psd_nperseg', 0, 0, 65536))
        pf.addRow('PSD f max (0=Nyq)', self._text('psd_fmax', '0'))
        pf.addRow('z threshold', self._text('z_threshold', '0.98'))
        pf.addRow('band half (Hz)', self._text('band_half', '20.0'))
        pf.addRow('MA window', self._spin('ma_window', 20, 1, 1000))
        pf.addRow('channel', self._spin('channel', 0, 0, 256))
        pf.addRow('iter horizon K', self._spin('iter_K', 20, 1, 2000))
        pf.addRow('iter stride', self._spin('iter_stride', 100, 1, 100000))
        pf.addRow('figure DPI', self._spin('dpi', 150, 50, 600))
        holder = QWidget()
        holder.setLayout(pf)
        f.addWidget(holder)
        root.addWidget(g)

        # Wired here rather than at creation: the CVA group, which owns
        # lags and the stride, is built after the record group.
        for key in ('lags', 'eval_stride', 'neighbors'):
            self.w[key].valueChanged.connect(self._refresh_cost)

        root.addStretch(1)

    # ---------- widget helpers ----------
    def _spin(self, key, default, lo, hi):
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(default)
        self.w[key] = sb
        return sb

    def _text(self, key, default):
        le = QLineEdit(default)
        le.setMaximumWidth(120)
        self.w[key] = le
        return le

    def _pick_data(self):
        start = self.w['data_file'].text() or str(Path.home())
        fn, _ = QFileDialog.getOpenFileName(
            self, 'Select data record', start,
            'Text data (*.txt *.dat *.csv);;All files (*)')
        if fn:
            self.w['data_file'].setText(fn)
            self.w['pair_file'].setText('')
            self._sniff(fn)
            self.data_chosen.emit(fn)

    def _pick_results(self):
        d = QFileDialog.getExistingDirectory(
            self, 'Results root', self.w['results_root'].text())
        if d:
            self.w['results_root'].setText(d)

    def _output_group(self):
        g = QGroupBox('Output')
        f = QFormLayout(g)
        self.w['results_root'] = QLineEdit(str(DEFAULT_RESULTS_ROOT))
        pick_out = QPushButton('…')
        pick_out.setFixedWidth(30)
        pick_out.clicked.connect(self._pick_results)
        row = QHBoxLayout()
        row.addWidget(self.w['results_root'])
        row.addWidget(pick_out)
        rw = QWidget()
        rw.setLayout(row)
        f.addRow('Results root', rw)
        return g

    def _pick_pair(self):
        start = self.w['data_file'].text() or str(Path.home())
        fn, _ = QFileDialog.getOpenFileName(
            self, 'Select the matching response file', start,
            'Data (*.txt *.dat *.csv);;All files (*)')
        if fn:
            self.w['pair_file'].setText(fn)
            self._sniff(self.w['data_file'].text(), keep_pair=True)

    # ---------- record inspection ----------
    def _role_infos(self):
        """(excitation file, response file) — the second is None when one
        file holds every channel."""
        if self._info is None:
            return None, None
        if self._pair_info is None:
            return self._info, None
        if self._info.kind == 'response':
            return self._pair_info, self._info
        return self._info, self._pair_info

    def _sniff(self, path, keep_pair=False):
        """Describe the chosen record and fill the channel table from it."""
        try:
            self._info = sniff_record(path)
        except Exception as exc:
            self._info = self._pair_info = None
            self.tbl.setRowCount(0)
            self.lbl_record.setText(f'Could not read this file — {exc}')
            self.lbl_cost.setText('—')
            return

        pair = None
        if keep_pair and self.w['pair_file'].text().strip():
            pair = Path(self.w['pair_file'].text().strip())
        elif self._info.pair is not None:
            pair = self._info.pair
        self._pair_info = None
        if pair is not None and Path(pair) != Path(self._info.path):
            try:
                self._pair_info = sniff_record(pair)
            except Exception:
                self._pair_info = None
        self.w['pair_file'].setText(
            str(self._pair_info.path) if self._pair_info else '')

        spec = default_spec(self._info, pair_info=self._pair_info)
        self._fill_table(spec)

        # A record that carries its own time column sets the rate; the fs
        # field stays at 0 so the time column is believed rather than
        # overridden by whatever was left in the box.
        if self._info.time_col is not None:
            self.w['fs'].setText('0')
        elif not self._num_or_none('fs'):
            self.w['fs'].setText('')

        in_info, out_info = self._role_infos()
        txt = in_info.describe()
        if out_info is not None:
            txt += f'\nresponses: {out_info.path.name} — {out_info.describe()}'
        if in_info.time_col is None:
            txt += '\nNo time column — enter a sample rate below.'
        self.lbl_record.setText(txt)
        self.window_full()

    def _fill_table(self, spec):
        in_info, out_info = self._role_infos()
        rows = []
        if out_info is None:
            for c in range(in_info.n_col):
                role = ('Time' if c == spec.time_col else
                        'Excitation' if c in spec.in_cols else
                        'Response' if c in spec.out_cols else 'Ignore')
                rows.append(('', c, in_info, role))
        else:
            for c in range(in_info.n_col):
                rows.append(('in', c, in_info,
                             'Excitation' if c in spec.in_cols else 'Ignore'))
            for c in range(out_info.n_col):
                rows.append(('out', c, out_info,
                             'Response' if c in spec.out_cols else 'Ignore'))

        self.tbl.setRowCount(len(rows))
        for r, (src, c, info, role) in enumerate(rows):
            for col, text in ((0, src or '—'),
                              (1, info.channel_names()[c]),
                              (2, info.channel_units()[c])):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl.setItem(r, col, item)
            combo = QComboBox()
            combo.addItems(ROLES)
            combo.setCurrentText(role)
            combo.setProperty('src', src)
            combo.setProperty('col', c)
            combo.currentTextChanged.connect(self._refresh_cost)
            self.tbl.setCellWidget(r, 3, combo)
        self._refresh_cost()

    # ---------- window ----------
    def _num_or_none(self, key):
        raw = self.w[key].text().strip()
        if raw == '':
            return None
        try:
            v = float(raw)
        except ValueError:
            return None
        return v or None

    def needs_fs(self):
        """True when the record cannot be read until a rate is entered."""
        if self._info is None:
            return False
        in_info, _ = self._role_infos()
        if in_info is None or in_info.time_col is not None:
            return False
        return not self._num_or_none('fs')

    def window_full(self):
        self.w['t_start'].setText('')
        self.w['t_end'].setText('')
        self._refresh_cost()
        self.window_changed.emit()

    def set_window(self, t0, t1):
        self.w['t_start'].setText(f'{t0:.5f}')
        self.w['t_end'].setText(f'{t1:.5f}')
        self._refresh_cost()
        self.window_changed.emit()

    def _effective_fs(self):
        fs = self._num_or_none('fs')
        if fs:
            return fs
        in_info, _ = self._role_infos()
        return in_info.fs if in_info else None

    def _refresh_cost(self, *_):
        """How big a sweep the current record, decimation and window imply."""
        if self._info is None or 'lags' not in self.w:
            return
        in_info, out_info = self._role_infos()
        base = out_info or in_info
        dec = max(1, self.w['decimate'].value())
        fs = self._effective_fs()
        fs_eff = fs / dec if fs else None
        n = max(0, base.n_samp // dec)

        n_sel = n
        if fs_eff:
            t0 = in_info.t0 if in_info.time_col is not None else 0.0
            lo = self._num_or_none('t_start')
            hi = self._num_or_none('t_end')
            lo = t0 if lo is None else lo
            hi = (t0 + n / fs_eff) if hi is None else hi
            n_sel = min(n, max(0, int(round((hi - lo) * fs_eff))))

        lags = self.w['lags'].value()
        stride = max(1, self.w['eval_stride'].value())
        n_win = max(0, -(-(n_sel - lags) // stride)) if n_sel > lags else 0

        txt = f'{n_sel} samples'
        if fs_eff:
            txt += f'  ({n_sel / fs_eff:.3f} s at {fs_eff:.1f} Hz)'
        txt += f'  ->  {n_win} CVA windows'
        if n_sel <= self.w['neighbors'].value() + lags:
            txt += (f'\n⚠ too short — the sweep needs more than '
                    f'neighbors + lags = {self.w["neighbors"].value() + lags} '
                    f'samples')
        elif n_win > 20000:
            txt += '\n⚠ a sweep this long runs for hours — decimate, ' \
                   'shorten the window, or raise the eval stride'
        self.lbl_cost.setText(txt)

    # ---------- spec ----------
    def record_spec(self) -> RecordSpec:
        """Turn the panel's state into a RecordSpec for cva_record."""
        path = Path(self.w['data_file'].text().strip())
        if self._info is None or Path(self._info.path) != path:
            self._sniff(str(path), keep_pair=True)
        if self._info is None:
            raise ValueError(f'Could not read {path.name}.')

        in_info, out_info = self._role_infos()
        time_col, in_cols, out_cols = None, [], []
        for r in range(self.tbl.rowCount()):
            combo = self.tbl.cellWidget(r, 3)
            src, col = combo.property('src'), int(combo.property('col'))
            role = combo.currentText()
            if role == 'Time':
                if src == 'out':
                    raise ValueError(
                        'The time column must come from the excitation file.')
                time_col = col
            elif role == 'Excitation':
                if src == 'out':
                    raise ValueError(
                        f'{out_info.path.name} holds the responses, so its '
                        f'columns cannot be the excitation. Swap the two '
                        f'files if they are the wrong way round.')
                in_cols.append(col)
            elif role == 'Response':
                if out_info is not None and src != 'out':
                    raise ValueError(
                        f'{in_info.path.name} holds the excitation, so its '
                        f'columns cannot be responses.')
                out_cols.append(col)

        if not in_cols:
            raise ValueError('No channel is marked as the excitation.')
        if not out_cols:
            raise ValueError('No channel is marked as a response.')
        if time_col is None and not self._num_or_none('fs'):
            raise ValueError(
                f'{Path(in_info.path).name} has no time column, so it needs a '
                f'sample rate — type one into the fs box and press Enter.')

        return RecordSpec(
            input_path=Path(in_info.path),
            response_path=Path(out_info.path) if out_info else None,
            time_col=time_col,
            in_cols=in_cols, out_cols=out_cols,
            fs=self._num_or_none('fs'),
            decimate=max(1, self.w['decimate'].value()),
            t_start=self._num_or_none('t_start'),
            t_end=self._num_or_none('t_end'))

    # ---------- collect ----------
    def values(self):
        def num(key, cast=float):
            raw = self.w[key].text().strip()
            if raw == '':
                return cast(0)
            return cast(raw)

        p = {}
        for key, widget in self.w.items():
            if isinstance(widget, QSpinBox):
                p[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                p[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                p[key] = widget.currentText()
            else:
                p[key] = widget.text().strip()

        for key in ('tol', 'fs', 'freq_min', 'freq_max',
                    'sig_threshold_fraction', 'n_pca_var', 'f_min', 'f_max',
                    'min_im_frac', 'zeta_max', 'z_threshold', 'band_half',
                    'psd_fmax'):
            p[key] = num(key)

        p['fs'] = p['fs'] or None
        if not p['data_file']:
            raise ValueError('No data file selected.')
        if not Path(p['data_file']).exists():
            raise ValueError(f'Data file not found:\n{p["data_file"]}')

        p['t_start'] = self._num_or_none('t_start')
        p['t_end'] = self._num_or_none('t_end')
        p['spec'] = self.record_spec()

        stem = Path(p['data_file']).stem
        run_dir = Path(p['results_root']) / stem
        p['stem'] = stem
        p['run_dir'] = run_dir
        p['fig_dir'] = run_dir / 'figures'
        p['h5_run'] = run_dir / f'{stem}_cva.h5'
        p['h5_clust'] = run_dir / f'{stem}_clusters.h5'
        p['h5_modes'] = run_dir / f'{stem}_modes.h5'
        return p


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CVApy — Local Canonical Variate Analysis')
        self.resize(1500, 950)

        self.run_obj = None
        self.clust_obj = None
        self.ms_obj = None
        self.thread = None
        self.worker = None
        self._figs = []
        self._fail_title = 'Pipeline failed'   # retitled per operation
        self._cache = {}          # {path string: parsed array}
        self._stamps = {}         # {path string: (mtime, size)} for the above
        self._spans = []          # drag-to-select handles on the time history

        self.params = ParamPanel()
        self.params.setMinimumWidth(330)
        self.params.setMaximumWidth(430)
        self.params.data_chosen.connect(self._on_data_chosen)
        self.params.suggest_requested.connect(self.on_suggest)
        # Typing a rate is the one thing standing between a no-time-column
        # record and being read, so Enter there just loads it.
        self.params.w['fs'].returnPressed.connect(self.load_preview)
        self.params.window_changed.connect(self.refresh_window_view)

        self.btn_preview = QPushButton('Load record')
        self.btn_preview.setToolTip(
            'Read the whole record and plot its time history and spectral '
            'density, so the analysis window can be chosen from them')
        self.btn_preview.clicked.connect(self.load_preview)
        self.btn_run = QPushButton('Run Pipeline')
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.start_pipeline)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)
        self.btn_plots = QPushButton('Regenerate Plots')
        self.btn_plots.setEnabled(False)
        self.btn_plots.clicked.connect(self.build_figures)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_plots)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.addWidget(self.params)
        ll.addLayout(btns)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)
        self.tabs.setDocumentMode(True)
        placeholder = QLabel(
            'Select a record — its time history and spectral density appear '
            'here.\n\nDrag across the time history to set the analysis '
            'window, or press Suggest.\nThen press Run Pipeline.')
        placeholder.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(placeholder, 'Figures')

        self.logpane = QPlainTextEdit()
        self.logpane.setReadOnly(True)
        self.logpane.setMaximumBlockCount(5000)
        self.logpane.setFont(QFont('Menlo', 11))

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(self.logpane)
        right.setSizes([680, 240])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([370, 1130])
        self.setCentralWidget(split)

        # Workers hand their output over through plain buffers; this timer
        # is what moves it onto the screen (see _LineBuffer).
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._drain_worker)
        self._last_status = None

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setMinimumWidth(240)
        self.progress.setTextVisible(True)
        self.progress.setFormat('%p%')
        self.progress.setValue(0)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage('Ready')

        self._build_menus()

    # ------------------------------------------------------------------
    def _build_menus(self):
        m = self.menuBar().addMenu('&File')
        a = QAction('Open Data Record …', self)
        a.triggered.connect(self.params._pick_data)
        m.addAction(a)
        a = QAction('Save All Figures', self)
        a.triggered.connect(self.save_figures)
        m.addAction(a)
        m.addSeparator()
        a = QAction('Quit', self)
        a.triggered.connect(self.close)
        m.addAction(a)

        m = self.menuBar().addMenu('&Run')
        a = QAction('Load Record + Plots', self)
        a.triggered.connect(self.load_preview)
        m.addAction(a)
        a = QAction('Run Pipeline', self)
        a.triggered.connect(self.start_pipeline)
        m.addAction(a)
        a = QAction('Regenerate Plots', self)
        a.triggered.connect(self.build_figures)
        m.addAction(a)

    # ------------------------------------------------------------------
    @Slot(str)
    def append_log(self, line):
        self.logpane.appendPlainText(line)

    @Slot(int, str)
    def set_progress(self, pct, label):
        self.progress.setValue(max(0, min(100, pct)))
        self.statusBar().showMessage(label)

    # ------------------------------------------------------------------
    def _drain_worker(self):
        """Move whatever the worker has produced into the log and the bar."""
        w = self.worker
        if w is None:
            return
        lines = getattr(w, 'lines', None)
        if lines:
            while True:
                try:
                    self.append_log(lines.popleft())
                except IndexError:
                    break
        status = getattr(w, 'status', None)
        if status is not None and status != self._last_status:
            self._last_status = status
            self.set_progress(*status)

    # ------------------------------------------------------------------
    def _busy(self, busy, label=''):
        """Indeterminate bar for work with no measurable progress."""
        if busy:
            self.progress.setRange(0, 0)
            self.statusBar().showMessage(label)
        else:
            self.progress.setRange(0, 100)

    # ------------------------------------------------------------------
    @Slot(str)
    def _on_data_chosen(self, path):
        """
        Load the record the moment it is picked — unless it cannot be read
        yet.

        A record with no time column needs a sample rate, and firing the
        load anyway threw a modal failure dialog over the very panel the
        rate has to be typed into.
        """
        if self.params.needs_fs():
            msg = (f'{Path(path).name} has no time column — type a sample '
                   f'rate into fs and press Enter.')
            self.statusBar().showMessage(msg)
            self.append_log(f'--- {msg} ---')
            self.params.w['fs'].setFocus()
            self.params.w['fs'].selectAll()
            return
        self.load_preview()

    def load_preview(self, *_):
        """Parse the selected record and plot its spectral densities."""
        if self.thread is not None:
            return
        try:
            p = self.params.values()
        except ValueError as e:
            QMessageBox.warning(self, 'Check parameters', str(e))
            return

        self._fail_title = 'Could not read the record'
        self.logpane.clear()
        self.append_log(f'--- Loading {Path(p["data_file"]).name} ---')
        self.btn_run.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_plots.setEnabled(False)
        self._busy(True, 'Reading record …')

        self.thread = QThread(self)
        self.worker = PreviewWorker(p, cache=self._cached_arrays())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.execute)
        self.worker.done.connect(self.on_preview_done)
        self.worker.failed.connect(self.on_failed)
        self._last_status = None
        self._poll.start()
        self.thread.start()

    @Slot(object, object)
    def on_preview_done(self, run, cache):
        self._busy(False)
        self._teardown_thread()
        self._remember(cache)
        # A new record invalidates anything held from the previous one.
        self.run_obj, self.clust_obj, self.ms_obj = run, None, None
        self.btn_plots.setEnabled(True)
        self.progress.setValue(0)
        self.build_figures()
        # after build_figures, which writes its own status line
        self.statusBar().showMessage(
            f'{run.data_file.name} — {run.n_in} input'
            f'{"s" if run.n_in > 1 else ""}, {run.n_out} responses, '
            f'fs={run.fs:.1f} Hz, {len(run.time)} samples')

    @staticmethod
    def _stamp(path):
        try:
            st = Path(path).stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return (0, 0)

    def _remember(self, cache):
        """Keep the arrays the preview read, with a stamp of each file."""
        self._cache = dict(cache or {})
        self._stamps = {k: self._stamp(k) for k in self._cache}

    def _cached_arrays(self):
        """The remembered arrays, minus any whose file changed on disk."""
        return {k: v for k, v in self._cache.items()
                if self._stamps.get(k) == self._stamp(k)}

    # ------------------------------------------------------------------
    def start_pipeline(self):
        if self.thread is not None:
            return
        try:
            p = self.params.values()
        except ValueError as e:
            QMessageBox.warning(self, 'Check parameters', str(e))
            return

        self._fail_title = 'Pipeline failed'
        p['run_dir'].mkdir(parents=True, exist_ok=True)
        p['fig_dir'].mkdir(parents=True, exist_ok=True)

        self.logpane.clear()
        self.append_log('=' * 62)
        self.append_log(f'  CVA Pipeline — {p["stem"]}')
        self.append_log(f'  data    : {p["data_file"]}')
        self.append_log(f'  results : {p["run_dir"]}')
        self.append_log(f'  lags={p["lags"]}  tol={p["tol"]:.0e}  '
                        f'neighbors={p["neighbors"]}  n_jobs={p["n_jobs"]}')
        self.append_log('=' * 62)

        self.btn_run.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_plots.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        # A preview-only run holds data but no windows — nothing to reuse.
        existing = (self.run_obj, self.clust_obj, self.ms_obj)
        if self.run_obj is not None and not self.run_obj.windows:
            existing = (None, None, None)

        self.thread = QThread(self)
        self.worker = PipelineWorker(
            p, existing=existing, cache=self._cached_arrays())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.execute)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self._last_status = None
        self._poll.start()
        self.thread.start()

    def stop_pipeline(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage(
                'Stopping — the sweep gives up at the next window …')

    def _teardown_thread(self):
        self._drain_worker()          # whatever arrived since the last tick
        self._poll.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_stop.setEnabled(False)

    @Slot(object, object, object)
    def on_done(self, run, clust, ms):
        # A stopped run hands back None — keep the record already loaded so
        # its spectra stay on screen instead of emptying the window.
        if run is not None:
            self.run_obj, self.clust_obj, self.ms_obj = run, clust, ms
        self._teardown_thread()
        self.btn_plots.setEnabled(self.run_obj is not None)
        if run is not None:
            self.append_log('\n' + repr(run))
        if clust is not None:
            self.append_log(repr(clust))
        if ms is not None:
            self.append_log(repr(ms))
        self.build_figures()

    @Slot(str)
    def on_failed(self, tb):
        self._busy(False)
        self._teardown_thread()
        self.progress.setValue(0)
        self.statusBar().showMessage('Failed')
        title = getattr(self, '_fail_title', 'Pipeline failed')
        self.append_log(f'\n*** {title.upper()} ***\n' + tb)
        QMessageBox.critical(self, title, tb.strip().splitlines()[-1])

    # ------------------------------------------------------------------
    def build_figures(self):
        if self.run_obj is None:
            return
        try:
            p = self.params.values()
        except ValueError as e:
            QMessageBox.warning(self, 'Check parameters', str(e))
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('Building figures …')
        try:
            for f in self._figs:
                plt.close(f)
            self._figs = []
            self.tabs.clear()

            run, clust, ms = self.run_obj, self.clust_obj, self.ms_obj
            has_cva = bool(run.windows)
            jobs = []
            th_fig = None

            window = self._window_from(p, run)
            if p['plot_time_history'] and run.datout is not None:
                jobs.append(('Time history', lambda: run.plot_time_history(
                    in_ch=p['th_in_ch'], max_out=p['th_max_out'],
                    window=window, dpi=p['dpi'])))
            if p['plot_psd'] and run.datout is not None:
                jobs.append(('Spectral density',
                             lambda: run.plot_spectral_density(
                                 nperseg=p['psd_nperseg'],
                                 f_max=p['psd_fmax'], dpi=p['dpi'])))
            if p['plot_significance'] and has_cva:
                jobs.append(('Significance', lambda: run.plot_significance(
                    z_threshold=p['z_threshold'] or None, dpi=p['dpi'])))
            if p['plot_freq_deviation'] and has_cva:
                jobs.append(('Freq deviation', lambda: run.plot_frequency_deviation(
                    band_half=p['band_half'], ma_window=p['ma_window'],
                    dpi=p['dpi'])))
            if p['plot_reconstruction'] and has_cva:
                jobs.append(('Reconstruction', lambda: run.plot_reconstruction(
                    channel=p['channel'], dpi=p['dpi'])))
            if p['plot_iterated'] and has_cva:
                jobs.append(('Iterated pred.', lambda: run.plot_iterated_prediction(
                    channel=p['channel'], iter_K=p['iter_K'],
                    iter_stride=p['iter_stride'], dpi=p['dpi'])))
            if p['plot_clusters'] and clust is not None:
                jobs += [
                    ('PCA variance', lambda: clust.plot_pca_variance(dpi=p['dpi'])),
                    ('Elbow',        lambda: clust.plot_elbow(dpi=p['dpi'])),
                    ('Regimes',      lambda: clust.plot_regime_timeseries(dpi=p['dpi'])),
                    ('z-plane',      lambda: clust.plot_eigenvalues_zplane(dpi=p['dpi'])),
                    ('Cluster ampl.', lambda: clust.plot_cluster_amplitude(dpi=p['dpi'])),
                    ('PCA scatter',  lambda: clust.plot_pca_scatter(dpi=p['dpi'])),
                ]
            if p['plot_modes'] and ms is not None:
                jobs += [
                    ('Regime modes', lambda: ms.plot_regime_comparison(dpi=p['dpi'])),
                    ('Modes/cluster', lambda: ms.plot_per_cluster(dpi=p['dpi'])),
                    ('Mode magnitude', lambda: ms.plot_magnitude_overlay(dpi=p['dpi'])),
                ]

            for title, fn in jobs:
                try:
                    result = fn()
                except Exception:
                    self.append_log(f'[plot] {title} failed:\n'
                                    + traceback.format_exc())
                    continue
                figs = result if isinstance(result, list) else [result]
                for i, fig in enumerate(figs):
                    if not isinstance(fig, Figure):
                        continue
                    name = title if len(figs) == 1 else f'{title} {i}'
                    self._figs.append(fig)
                    self.tabs.addTab(FigureTab(fig), name)
                    if title == 'Time history':
                        th_fig = fig
                QApplication.processEvents()

            self._attach_spans(th_fig)

            if self.tabs.count() == 0:
                lbl = QLabel('No figures — enable some in the Plots group.')
                lbl.setAlignment(Qt.AlignCenter)
                self.tabs.addTab(lbl, 'Figures')
            self.statusBar().showMessage(
                f'{len(self._figs)} figures')
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    # analysis window
    # ------------------------------------------------------------------
    @staticmethod
    def _window_from(p, run):
        """(t_start, t_end) to shade, filling blanks from the record."""
        if p['t_start'] is None and p['t_end'] is None:
            return None
        lo = float(run.time[0]) if p['t_start'] is None else p['t_start']
        hi = float(run.time[-1]) if p['t_end'] is None else p['t_end']
        return (lo, hi)

    def _attach_spans(self, fig):
        """Let the window be dragged out on any panel of the time history."""
        self._spans = []
        if fig is None:
            return

        def on_select(lo, hi):
            if hi > lo:
                self.params.set_window(lo, hi)

        for ax in fig.axes:
            self._spans.append(SpanSelector(
                ax, on_select, 'horizontal', useblit=False,
                props=dict(alpha=0.25, facecolor='seagreen'),
                drag_from_anywhere=True))

    @Slot()
    def on_suggest(self):
        if self.run_obj is None or self.run_obj.datout is None:
            QMessageBox.information(
                self, 'Load a record first',
                'Press Load record first — Suggest needs to see the '
                'signals before it can pick a window.')
            return
        kind = self.params.w['signal_type'].currentText().lower()
        # Keep the suggestion to a sweep worth starting. A stationary record
        # would otherwise be offered whole, which on a 20 s 10 kHz record is
        # 200 000 windows.
        lags = self.params.w['lags'].value()
        stride = max(1, self.params.w['eval_stride'].value())
        budget = SUGGEST_MAX_WINDOWS * stride + lags
        t0, t1 = cva_record.suggest_window(
            self.run_obj.time, self.run_obj.inputd, self.run_obj.datout,
            kind=None if kind == 'auto' else kind, max_samples=budget)
        st = cva_record.classify(self.run_obj.inputd, self.run_obj.datout)
        n_sel = int(((self.run_obj.time >= t0)
                     & (self.run_obj.time <= t1)).sum())
        note = (f'  (capped at {SUGGEST_MAX_WINDOWS} windows)'
                if n_sel >= budget else '')
        self.append_log(f'Suggested window {t0:.5f} .. {t1:.5f} s   '
                        f'({st}){note}')
        self.params.set_window(t0, t1)

    @Slot()
    def refresh_window_view(self):
        """Redraw just the time history after the window changed."""
        if self.run_obj is None or self.run_obj.datout is None:
            return
        idx = next((i for i in range(self.tabs.count())
                    if self.tabs.tabText(i) == 'Time history'), None)
        if idx is None:
            return
        try:
            p = self.params.values()
        except ValueError:
            return
        old_tab = self.tabs.widget(idx)
        try:
            fig = self.run_obj.plot_time_history(
                in_ch=p['th_in_ch'], max_out=p['th_max_out'],
                window=self._window_from(p, self.run_obj), dpi=p['dpi'])
        except Exception:
            self.append_log('[plot] Time history failed:\n'
                            + traceback.format_exc())
            return
        if getattr(old_tab, 'fig', None) is not None:
            if old_tab.fig in self._figs:
                self._figs.remove(old_tab.fig)
            plt.close(old_tab.fig)
        self._figs.append(fig)
        self.tabs.removeTab(idx)
        self.tabs.insertTab(idx, FigureTab(fig), 'Time history')
        self.tabs.setCurrentIndex(idx)
        self._attach_spans(fig)

    # ------------------------------------------------------------------
    def save_figures(self):
        if not self._figs:
            QMessageBox.information(self, 'Nothing to save', 'No figures yet.')
            return
        d = QFileDialog.getExistingDirectory(self, 'Save figures to')
        if not d:
            return
        dpi = self.params.w['dpi'].value()
        out = Path(d)
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if not isinstance(tab, FigureTab):
                continue
            safe = self.tabs.tabText(i).replace(' ', '_').replace('.', '')
            path = out / f'{i:02d}_{safe}.png'
            tab.fig.savefig(path, dpi=dpi, bbox_inches='tight')
            self.append_log(f'  Saved: {path.name}')
        self.statusBar().showMessage(f'Saved {len(self._figs)} figures to {out}')

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.thread is not None:
            r = QMessageBox.question(
                self, 'Pipeline running',
                'The pipeline is still running. Quit anyway?')
            if r != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.thread.quit()
            self.thread.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('CVApy')

    # Qt's event loop would otherwise swallow Ctrl+C in the launching
    # terminal; the idle timer gives Python a chance to run the handler.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QTimer()
    sigint_timer.start(200)
    sigint_timer.timeout.connect(lambda: None)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
