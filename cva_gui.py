#!/usr/bin/env python3
"""
cva_gui.py
==========
PySide6 desktop front end for the CVApy three-stage pipeline.

    conda activate cvapy
    python cva_gui.py

Layout
------
Left   : parameter panel (data layout, CVA, clustering, mode shapes, plots)
Right  : one tab per figure, each with the standard Matplotlib toolbar
Bottom : log pane fed by the pipeline's own stdout, plus a progress bar

The pipeline runs on a QThread, so the window stays responsive during the
multi-minute CVA sweep. Figures are built on the GUI thread afterwards and
embedded with FigureCanvasQTAgg; the Matplotlib backend is Agg and
cva_plotting.set_show(False) is set, so no detached plot windows ever open.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib

# Figures are embedded in our own canvases — never shown in their own window.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cva_plotting

cva_plotting.set_show(False)          # the GUI owns every figure

from cva_mode_shapes import CVAModeShapes
from cva_phi_cluster import CVAPhiCluster
from cva_run import CVAConfig, CVARun

plt.rcParams['figure.max_open_warning'] = 0

DEFAULT_RESULTS_ROOT = Path('/Users/nhunterjr/Documents/DEVONthink/CVA_Results')


# =============================================================================
# stdout -> Qt signal
# =============================================================================

class _SignalStream:
    """File-like object that emits each completed line on a Qt signal."""

    def __init__(self, emit_line):
        self._emit = emit_line
        self._buf = ''

    def write(self, text):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._emit(line)

    def flush(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ''


# =============================================================================
# Worker — runs the pipeline off the GUI thread
# =============================================================================

class PipelineWorker(QObject):
    log      = Signal(str)
    progress = Signal(int, str)
    done     = Signal(object, object, object)   # run, clust, ms
    failed   = Signal(str)

    def __init__(self, params, existing=None):
        super().__init__()
        self.p = params
        self.run_obj, self.clust_obj, self.ms_obj = existing or (None, None, None)
        self._stop = False

    def request_stop(self):
        """Cooperative stop — honoured between stages, not mid-sweep."""
        self._stop = True

    def _emit_log(self, line):
        self.log.emit(line)
        # The CVA sweep prints lines like '   30.0%  (45/150)  elapsed=...'
        s = line.strip()
        if s and s[0].isdigit() and '%' in s.split()[0]:
            try:
                self.progress.emit(int(float(s.split('%')[0])), 'CVA sweep')
            except ValueError:
                pass

    @Slot()
    def execute(self):
        p = self.p
        saved_stdout = sys.stdout
        sys.stdout = _SignalStream(self._emit_log)
        try:
            run, clust, ms = self.run_obj, self.clust_obj, self.ms_obj

            # ---------------- Stage 1: CVA ----------------
            if p['run_cva']:
                self.progress.emit(0, 'Stage 1 — CVA')
                print('--- Step 1: CVA Run ---')
                cfg = CVAConfig(
                    lags=p['lags'], tol=p['tol'], neighbors=p['neighbors'],
                    ranklock=p['ranklock'], eval_stride=p['eval_stride'],
                    fs=p['fs'], freq_min=p['freq_min'], freq_max=p['freq_max'],
                    sig_threshold_fraction=p['sig_threshold_fraction'],
                    sig_peak_mindist=p['sig_peak_mindist'])
                run = CVARun(cfg)
                run.load_data(p['data_file'], col_time=p['col_time'],
                              col_force=p['col_force'],
                              col_accel_start=p['col_accel_start'],
                              n_in=p['n_in'])
                run.build_database()
                run.run(n_jobs=p['n_jobs'])
                run.eigenvalue_significance()
                if p['save_hdf5']:
                    run.save(str(p['h5_run']))
            elif run is None:
                self.progress.emit(0, 'Stage 1 — loading')
                print(f'--- Step 1: Loading {Path(p["h5_run"]).name} ---')
                run = CVARun.load(str(p['h5_run']))

            if self._stop:
                raise InterruptedError('Stopped after stage 1')

            # ---------------- Stage 2: clustering ----------------
            if p['run_cluster']:
                self.progress.emit(55, 'Stage 2 — clustering')
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
                self.progress.emit(80, 'Stage 3 — mode shapes')
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

            self.progress.emit(100, 'Complete')
            sys.stdout.flush()
            self.done.emit(run, clust, ms)

        except InterruptedError as e:
            sys.stdout.flush()
            self.log.emit(f'\n*** {e} ***')
            self.done.emit(self.run_obj, self.clust_obj, self.ms_obj)
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        inner = QWidget()
        self.setWidget(inner)
        root = QVBoxLayout(inner)
        self.w = {}

        # ---------- data ----------
        g = QGroupBox('Data')
        f = QFormLayout(g)
        self.w['data_file'] = QLineEdit()
        self.w['data_file'].setPlaceholderText('select a .txt record …')
        browse = QPushButton('Browse …')
        browse.clicked.connect(self._pick_data)
        row = QHBoxLayout()
        row.addWidget(self.w['data_file'])
        row.addWidget(browse)
        rw = QWidget()
        rw.setLayout(row)
        f.addRow('File', rw)

        self.w['results_root'] = QLineEdit(str(DEFAULT_RESULTS_ROOT))
        pick_out = QPushButton('…')
        pick_out.setFixedWidth(30)
        pick_out.clicked.connect(self._pick_results)
        row2 = QHBoxLayout()
        row2.addWidget(self.w['results_root'])
        row2.addWidget(pick_out)
        rw2 = QWidget()
        rw2.setLayout(row2)
        f.addRow('Results root', rw2)

        f.addRow('time col', self._spin('col_time', 0, 0, 64))
        f.addRow('force col', self._spin('col_force', 1, 0, 64))
        f.addRow('accel start col', self._spin('col_accel_start', 2, 0, 64))
        f.addRow('n inputs', self._spin('n_in', 1, 1, 64))
        root.addWidget(g)

        # ---------- CVA ----------
        g = QGroupBox('CVA')
        f = QFormLayout(g)
        f.addRow('lags', self._spin('lags', 50, 2, 512))
        f.addRow('tol', self._text('tol', '1e-2'))
        f.addRow('neighbors', self._spin('neighbors', 1000, 10, 100000))
        f.addRow('ranklock (0=auto)', self._spin('ranklock', 0, 0, 512))
        f.addRow('eval stride', self._spin('eval_stride', 1, 1, 10000))
        f.addRow('fs (Hz)', self._text('fs', '5000'))
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

    def _pick_results(self):
        d = QFileDialog.getExistingDirectory(
            self, 'Results root', self.w['results_root'].text())
        if d:
            self.w['results_root'].setText(d)

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
            else:
                p[key] = widget.text().strip()

        for key in ('tol', 'fs', 'freq_min', 'freq_max',
                    'sig_threshold_fraction', 'n_pca_var', 'f_min', 'f_max',
                    'min_im_frac', 'zeta_max', 'z_threshold', 'band_half'):
            p[key] = num(key)

        p['fs'] = p['fs'] or None
        if not p['data_file']:
            raise ValueError('No data file selected.')
        if not Path(p['data_file']).exists():
            raise ValueError(f'Data file not found:\n{p["data_file"]}')

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

        self.params = ParamPanel()
        self.params.setMinimumWidth(330)
        self.params.setMaximumWidth(430)

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
        placeholder = QLabel('Select a data file and press Run Pipeline.\n\n'
                             'Figures appear here, one tab each.')
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

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
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
    def start_pipeline(self):
        if self.thread is not None:
            return
        try:
            p = self.params.values()
        except ValueError as e:
            QMessageBox.warning(self, 'Check parameters', str(e))
            return

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
        self.btn_plots.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.thread = QThread(self)
        self.worker = PipelineWorker(
            p, existing=(self.run_obj, self.clust_obj, self.ms_obj))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.execute)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.set_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

    def stop_pipeline(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.statusBar().showMessage('Stopping after current stage …')

    def _teardown_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    @Slot(object, object, object)
    def on_done(self, run, clust, ms):
        self.run_obj, self.clust_obj, self.ms_obj = run, clust, ms
        self._teardown_thread()
        self.btn_plots.setEnabled(run is not None)
        if run is not None:
            self.append_log('\n' + repr(run))
        if clust is not None:
            self.append_log(repr(clust))
        if ms is not None:
            self.append_log(repr(ms))
        self.build_figures()

    @Slot(str)
    def on_failed(self, tb):
        self._teardown_thread()
        self.progress.setValue(0)
        self.statusBar().showMessage('Failed')
        self.append_log('\n*** PIPELINE FAILED ***\n' + tb)
        QMessageBox.critical(self, 'Pipeline failed', tb.strip().splitlines()[-1])

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
            jobs = []

            if p['plot_significance']:
                jobs.append(('Significance', lambda: run.plot_significance(
                    z_threshold=p['z_threshold'] or None, dpi=p['dpi'])))
            if p['plot_freq_deviation']:
                jobs.append(('Freq deviation', lambda: run.plot_frequency_deviation(
                    band_half=p['band_half'], ma_window=p['ma_window'],
                    dpi=p['dpi'])))
            if p['plot_reconstruction']:
                jobs.append(('Reconstruction', lambda: run.plot_reconstruction(
                    channel=p['channel'], dpi=p['dpi'])))
            if p['plot_iterated']:
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
                QApplication.processEvents()

            if self.tabs.count() == 0:
                lbl = QLabel('No figures — enable some in the Plots group.')
                lbl.setAlignment(Qt.AlignCenter)
                self.tabs.addTab(lbl, 'Figures')
            self.statusBar().showMessage(
                f'{len(self._figs)} figures')
        finally:
            QApplication.restoreOverrideCursor()

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
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
