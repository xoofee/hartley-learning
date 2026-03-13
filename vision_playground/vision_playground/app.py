"""
Main window: dockable layout, camera preview, gallery, log, calibration, 3D plot.

Composes all widgets; single place for dock setup and wiring.
Saves/restores dock layout, geometry, camera selection, and chessboard config.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSettings, QEvent, QObject, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QDockWidget,
    QPushButton,
    QScrollArea,
    QFrame,
    QStackedWidget,
    QTabWidget,
    QSizePolicy,
    QMenuBar,
    QMenu,
    QAction,
    QActionGroup,
)

from .state import AppState
from .logging_ui import (
    set_log_sink,
    set_minimum_level,
    get_minimum_level,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
)
from .widgets import LogOutputWidget, ImageViewWidget
from .widgets.log_output import (
    DISPLAY_FILTER_DEBUG,
    DISPLAY_FILTER_INFO,
    DISPLAY_FILTER_WARNING,
    DISPLAY_FILTER_ERROR,
)
from .camera_preview import CameraPreviewWidget
from .gallery import GalleryWidget
from .video_gallery import VideoGalleryWidget
from .video_document import VideoDocumentWidget
from .calibration import CalibrationWidget
from .plot3d import Calib3DPlot
from .calibration_result_view import CalibrationResultImagesWidget
from . import logging_ui
from .plugins.registry import get_demo_by_id, get_demos
from .plugins.demos import register_builtin_demos, build_demos_button_group
from .view_angles import image_coords_to_yaw_pitch_deg

SETTINGS_ORG = "hartley-learning"
SETTINGS_APP = "chessboard_calib"

register_builtin_demos()


class _InteractiveDemoEventFilter(QObject):
    """Event filter: forward mouse events to the active demo when it has handle_mouse_event.
    Installed on QApplication. For move/release we also accept when cursor is over doc (obj may be window after consumed press)."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main = main_window

    def eventFilter(self, obj, event) -> bool:
        demo_id = self._main._state.current_demo_id
        demo = get_demo_by_id(demo_id)
        if demo is None or not hasattr(demo, "handle_mouse_event"):
            return False
        t = event.type()
        if t not in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
            return False
        tabbed = self._main._tabbed_view
        doc = tabbed.current_document()
        if doc is None:
            return False

        # Use global pos so coordinates are consistent for press/move/release (e.g. during drag)
        pos_in_doc = doc.mapFromGlobal(event.globalPos())
        in_doc_rect = doc.rect().contains(pos_in_doc)

        def _target_in_doc() -> bool:
            if not isinstance(obj, QWidget):
                return False
            def _is_under(w, ancestor):
                while w:
                    if w is ancestor:
                        return True
                    w = w.parentWidget()
                return False
            return obj is tabbed or obj is doc or _is_under(obj, doc)

        # Accept when: event target is doc (or child), OR for move/release when cursor is over doc (obj may be window)
        if not _target_in_doc():
            if t == QEvent.MouseButtonPress or not in_doc_rect:
                return False
        event_type = "press" if t == QEvent.MouseButtonPress else ("release" if t == QEvent.MouseButtonRelease else "move")
        if demo.handle_mouse_event(
            self._main._get_demo_context(),
            doc,
            pos_in_doc.x(),
            pos_in_doc.y(),
            event_type,
        ):
            event.accept()
            return True
        return False


def _settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    """Read a boolean from QSettings; handles string values from some backends."""
    v = settings.value(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).lower() not in ("0", "false", "no", "")


def _next_capture_path(folder: Path) -> Path:
    """Return next available path {folder.name}_0001.jpg, ..."""
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder.name or "img"
    for i in range(1, 10000):
        p = folder / f"{prefix}_{i:04d}.jpg"
        if not p.exists():
            return p
    return folder / f"{prefix}_9999.jpg"


class ImageDocumentWidget(QWidget):
    """Single-document image view; path and optional homography for rotate demo. Uses ImageViewWidget."""

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = path
        self._img_bgr = None
        self._load_image()
        layout = QVBoxLayout(self)
        self._view = ImageViewWidget(self)
        self._view.set_image(self._img_bgr, placeholder="Failed to load image")
        layout.addWidget(self._view)

    def _load_image(self) -> None:
        img = cv2.imread(str(self._path))
        self._img_bgr = img if img is not None else None

    def path(self) -> Path:
        return self._path

    def image_bgr(self) -> np.ndarray | None:
        return self._img_bgr

    def set_homography(self, H: np.ndarray | None) -> None:
        """Set 3x3 homography for display (e.g. from rotate demo). None = show original."""
        self._view.set_homography(H)

    def map_to_image_coords(self, wx: float, wy: float) -> tuple[float, float] | None:
        """Map widget coords (document space) to image pixel (x, y); None if outside image."""
        view_pt = self._view.mapFrom(self, QPoint(int(wx), int(wy)))
        return self._view.map_to_image_coords(float(view_pt.x()), float(view_pt.y()))

    def image_view(self) -> ImageViewWidget:
        """Reusable image view (for connecting hover_image_coords, etc.)."""
        return self._view


class CentralTabbedView(QWidget):
    """Tabbed image documents; closable tabs; open by path or switch to existing tab."""

    image_coords_changed = pyqtSignal(object)  # tuple[float, float] | None

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self._close_tab)
        self._tab_widget.currentChanged.connect(self._on_current_tab_changed)
        self._tab_widget.tabBar().installEventFilter(self)
        self._path_to_index: dict = {}
        self._current_coords_signal = None
        layout.addWidget(self._tab_widget)
        self._placeholder = QLabel("Open an image from a gallery (click a thumbnail)")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._placeholder.setMinimumSize(400, 300)
        self._show_placeholder()

    def eventFilter(self, obj, event) -> bool:
        if obj == self._tab_widget.tabBar() and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.MiddleButton:
                idx = self._tab_widget.tabBar().tabAt(event.pos())
                if 0 <= idx < self._tab_widget.count():
                    w = self._tab_widget.widget(idx)
                    if w is not self._placeholder:
                        self._close_tab(idx)
                        return True
        return super().eventFilter(obj, event)

    def _on_current_tab_changed(self, index: int) -> None:
        if self._current_coords_signal is not None:
            try:
                self._current_coords_signal.disconnect(self._forward_image_coords)
            except TypeError:
                pass
            self._current_coords_signal = None
        doc = self.current_document()
        if doc is not None and isinstance(doc, ImageDocumentWidget):
            self._current_coords_signal = doc.image_view().hover_image_coords
            self._current_coords_signal.connect(self._forward_image_coords)
        else:
            self.image_coords_changed.emit(None)

    def _forward_image_coords(self, pt: tuple[float, float] | None) -> None:
        self.image_coords_changed.emit(pt)

    def _show_placeholder(self) -> None:
        if self._tab_widget.count() == 0:
            self._tab_widget.addTab(self._placeholder, "")
            self._tab_widget.tabBar().setTabButton(0, self._tab_widget.tabBar().RightSide, None)

    def _hide_placeholder(self) -> None:
        if self._tab_widget.count() == 1 and self._tab_widget.indexOf(self._placeholder) >= 0:
            self._tab_widget.removeTab(0)
        self._path_to_index.pop(None, None)

    def open_path(self, path: Path) -> None:
        path = path.resolve()
        if path in self._path_to_index:
            self._tab_widget.setCurrentIndex(self._path_to_index[path])
            return
        self._hide_placeholder()
        doc = ImageDocumentWidget(path)
        idx = self._tab_widget.addTab(doc, path.name)
        self._tab_widget.setCurrentIndex(idx)
        self._path_to_index[path] = idx
        self._rebuild_path_index()

    def _rebuild_path_index(self) -> None:
        self._path_to_index.clear()
        for i in range(self._tab_widget.count()):
            w = self._tab_widget.widget(i)
            if w is not self._placeholder and hasattr(w, "path"):
                self._path_to_index[w.path()] = i

    def _close_tab(self, index: int) -> None:
        w = self._tab_widget.widget(index)
        if isinstance(w, VideoDocumentWidget):
            w.release()
        self._tab_widget.removeTab(index)
        self._rebuild_path_index()
        if self._tab_widget.count() == 0:
            self._show_placeholder()

    def current_document(self) -> ImageDocumentWidget | VideoDocumentWidget | None:
        w = self._tab_widget.currentWidget()
        if w is not None and (isinstance(w, ImageDocumentWidget) or isinstance(w, VideoDocumentWidget)):
            return w
        return None

    def open_video_path(self, path: Path) -> None:
        path = path.resolve()
        if path in self._path_to_index:
            self._tab_widget.setCurrentIndex(self._path_to_index[path])
            return
        self._hide_placeholder()
        doc = VideoDocumentWidget(path)
        idx = self._tab_widget.addTab(doc, path.name)
        self._tab_widget.setCurrentIndex(idx)
        self._path_to_index[path] = idx
        self._rebuild_path_index()

    def get_current_video_document(self) -> VideoDocumentWidget | None:
        w = self._tab_widget.currentWidget()
        return w if isinstance(w, VideoDocumentWidget) else None

    def current_path(self) -> Path | None:
        doc = self.current_document()
        return doc.path() if doc is not None else None

    def get_open_documents_sorted(self) -> list[tuple[Path, np.ndarray]]:
        """Return (path, image_bgr) for each open tab, sorted by path name. Skips placeholder and failed loads."""
        out: list[tuple[Path, np.ndarray]] = []
        for i in range(self._tab_widget.count()):
            w = self._tab_widget.widget(i)
            if w is self._placeholder or not isinstance(w, ImageDocumentWidget):
                continue
            path = w.path()
            img = w.image_bgr()
            if img is None or img.size == 0:
                continue
            out.append((path, img))
        out.sort(key=lambda x: x[0].name)
        return out

    def add_custom_tab(self, widget: QWidget, label: str) -> int:
        """Add a closable tab with an arbitrary widget, or switch to it if a tab with this label exists. Returns tab index."""
        for i in range(self._tab_widget.count()):
            if self._tab_widget.widget(i) is widget:
                self._tab_widget.setCurrentIndex(i)
                return i
            if self._tab_widget.tabText(i) == label and self._tab_widget.widget(i) is not self._placeholder:
                self._tab_widget.setCurrentIndex(i)
                return i
        self._hide_placeholder()
        idx = self._tab_widget.addTab(widget, label)
        self._tab_widget.setCurrentIndex(idx)
        return idx

    def get_open_paths_in_tab_order(self) -> list[Path]:
        """Return paths of open documents in tab order (for persistence)."""
        paths: list[Path] = []
        for i in range(self._tab_widget.count()):
            w = self._tab_widget.widget(i)
            if w is not self._placeholder and isinstance(w, ImageDocumentWidget):
                paths.append(w.path())
        return paths

    def get_current_tab_index(self) -> int:
        """Return current tab index (0-based; 0 when only placeholder)."""
        return self._tab_widget.currentIndex()

    def restore_open_paths(self, paths: list[str], current_index: int = 0) -> None:
        """Open saved paths in order and set current tab. Skips missing files."""
        for s in paths:
            p = Path(str(s)).resolve()
            if p.is_file():
                self.open_path(p)
        if self._tab_widget.indexOf(self._placeholder) < 0:
            n = self._tab_widget.count()
            if n > 0:
                idx = min(max(0, current_index), n - 1)
                self._tab_widget.setCurrentIndex(idx)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._state = AppState()
        self._state.refresh_gallery_paths()
        self._init_ui()
        self._connect()
        self._restore_settings()

    def _init_ui(self) -> None:
        self.setWindowTitle("Vision Playground")
        self.setGeometry(100, 100, 1280, 800)

        # Central: tabbed view (images, videos, calibration result, custom tabs — all closable)
        self._images_base = Path(__file__).resolve().parent.parent / "images"
        self._videos_base = Path(__file__).resolve().parent.parent / "videos"
        self._tabbed_view = CentralTabbedView()
        self._calib_result_view = CalibrationResultImagesWidget(self._state)
        self.setCentralWidget(self._tabbed_view)

        # Dock: Camera preview (no Take photo; galleries have Capture)
        self._camera_preview = CameraPreviewWidget()
        dock_cam = QDockWidget("Camera preview", self)
        dock_cam.setObjectName("CameraPreview")
        dock_cam.setWidget(self._camera_preview)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_cam)

        # Dock: Calibration gallery (images/calib)
        self._gallery_calib = GalleryWidget(
            self._images_base / "calib",
            title="Calibration",
            on_capture_requested=self._on_capture_requested,
        )
        dock_calib_gal = QDockWidget("Calibration gallery", self)
        dock_calib_gal.setObjectName("CalibrationGallery")
        dock_calib_gal.setWidget(self._gallery_calib)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_calib_gal)

        # Dock: Work gallery (images/work)
        self._gallery_work = GalleryWidget(
            self._images_base / "work",
            title="Work",
            on_capture_requested=self._on_capture_requested,
        )
        dock_work_gal = QDockWidget("Work gallery", self)
        dock_work_gal.setObjectName("WorkGallery")
        dock_work_gal.setWidget(self._gallery_work)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_work_gal)

        # Dock: Video gallery (videos/*.mp4)
        self._video_gallery = VideoGalleryWidget(
            self._videos_base,
            title="Videos",
        )
        dock_video_gal = QDockWidget("Video gallery", self)
        dock_video_gal.setObjectName("VideoGallery")
        dock_video_gal.setWidget(self._video_gallery)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_video_gal)

        # Dock: Log
        self._log_widget = LogOutputWidget(title="Log", show_clear_button=True)
        self._log_widget.set_display_filter_changed_callback(self._on_log_display_filter_changed)
        set_log_sink(self._log_widget)
        dock_log = QDockWidget("Log", self)
        dock_log.setObjectName("Log")
        dock_log.setWidget(self._log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_log)

        # Dock: Calibration (params + button; uses calibration gallery only)
        self._calib_widget = CalibrationWidget(
            self._state,
            get_gallery_paths=lambda: self._gallery_calib.get_paths(),
            get_current_image_size=self._get_current_image_size,
        )
        dock_calib = QDockWidget("Calibration", self)
        dock_calib.setObjectName("Calibration")
        dock_calib.setWidget(self._calib_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_calib)

        # Dock: 3D plot
        self._plot3d = Calib3DPlot(self._state)
        dock_3d = QDockWidget("3D plot", self)
        dock_3d.setObjectName("Plot3D")
        dock_3d.setWidget(self._plot3d)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_3d)

        # Dock: Demos (buttons + demo-specific pane)
        demos_group, self._demos_button_group, self._demos_buttons = build_demos_button_group(self)
        self._demo_ids_order = [d.id() for d in get_demos()]
        self._demo_pane_stack = QStackedWidget()
        for demo_id in self._demo_ids_order:
            demo = get_demo_by_id(demo_id)
            pane = demo.get_pane_widget(self._get_demo_context()) if demo else None
            self._demo_pane_stack.addWidget(pane if pane is not None else QWidget())
        demos_container = QWidget()
        demos_layout = QVBoxLayout(demos_container)
        demos_layout.addWidget(demos_group)
        demos_layout.addWidget(QLabel("Options:"))
        demos_layout.addWidget(self._demo_pane_stack)
        dock_demos = QDockWidget("Demos", self)
        dock_demos.setObjectName("Demos")
        dock_demos.setWidget(demos_container)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_demos)
        for demo_id, btn in self._demos_buttons.items():
            btn.clicked.connect(lambda checked, did=demo_id: self._on_demo_clicked(did))

        # Tabify right docks
        self.tabifyDockWidget(dock_cam, dock_calib_gal)
        self.tabifyDockWidget(dock_calib_gal, dock_work_gal)
        self.tabifyDockWidget(dock_work_gal, dock_video_gal)
        self.tabifyDockWidget(dock_video_gal, dock_calib)
        self.tabifyDockWidget(dock_calib, dock_demos)

        # View menu: one checkable action per dock (persistence via windowState)
        self._docks_by_name: dict[str, QDockWidget] = {}
        self._dock_visibility_actions: dict[str, QAction] = {}
        view_menu = QMenu("&View", self)
        dock_list = [
            (dock_cam, "Camera preview"),
            (dock_calib_gal, "Calibration gallery"),
            (dock_work_gal, "Work gallery"),
            (dock_video_gal, "Video gallery"),
            (dock_log, "Log"),
            (dock_calib, "Calibration"),
            (dock_3d, "3D plot"),
            (dock_demos, "Demos"),
        ]
        for dock, label in dock_list:
            name = dock.objectName()
            self._docks_by_name[name] = dock
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, d=dock: d.setVisible(checked))
            dock.visibilityChanged.connect(lambda visible, a=action: a.setChecked(visible))
            view_menu.addAction(action)
            self._dock_visibility_actions[name] = action
        # Options menu: status bar toggles
        self._show_image_coords = True
        self._show_yaw_pitch = False
        self._last_image_coords: tuple[float, float] | None = None
        options_menu = QMenu("&Options", self)
        self._action_show_image_coords = QAction("Show image coords (x, y)", self)
        self._action_show_image_coords.setCheckable(True)
        self._action_show_image_coords.setChecked(True)
        self._action_show_image_coords.triggered.connect(self._on_toggle_show_image_coords)
        options_menu.addAction(self._action_show_image_coords)
        self._action_show_yaw_pitch = QAction("Show yaw / pitch (degrees)", self)
        self._action_show_yaw_pitch.setCheckable(True)
        self._action_show_yaw_pitch.setChecked(False)
        self._action_show_yaw_pitch.triggered.connect(self._on_toggle_show_yaw_pitch)
        options_menu.addAction(self._action_show_yaw_pitch)

        # Log menu: minimum log level (exclusive, like radio)
        log_menu = QMenu("&Log", self)
        self._log_level_group = QActionGroup(self)
        self._log_level_group.setExclusive(True)
        self._log_level_actions: dict[int, QAction] = {}
        for level, label in [(DEBUG, "Debug"), (INFO, "Info"), (WARNING, "Warning"), (ERROR, "Error")]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(level == INFO)
            action.triggered.connect(lambda checked, lvl=level: self._on_log_level_changed(lvl))
            self._log_level_group.addAction(action)
            log_menu.addAction(action)
            self._log_level_actions[level] = action
        log_menu.setToolTip("Minimum log level: only messages at or above this level are recorded.")

        menu_bar = QMenuBar(self)
        menu_bar.addMenu(view_menu)
        menu_bar.addMenu(options_menu)
        menu_bar.addMenu(log_menu)
        self.setMenuBar(menu_bar)

        self._status_bar = self.statusBar()
        self._status_bar.showMessage("")

        self._interactive_demo_event_filter = _InteractiveDemoEventFilter(self)
        self._plot3d.redraw()

    def _restore_settings(self) -> None:
        """Restore geometry, dock state, chessboard config, and camera selection."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        # Geometry and dock layout
        geom = settings.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        state = settings.value("windowState")
        if state is not None:
            self.restoreState(state)
            for name, action in self._dock_visibility_actions.items():
                dock = self._docks_by_name.get(name)
                if dock is not None:
                    action.setChecked(dock.isVisible())
        # Chessboard: restore to state then sync UI
        try:
            cols = int(settings.value("chessboard_cols", self._state.chessboard.cols))
            rows = int(settings.value("chessboard_rows", self._state.chessboard.rows))
            square_size = float(settings.value("chessboard_square_size", self._state.chessboard.square_size))
        except (TypeError, ValueError):
            pass
        else:
            self._state.chessboard.cols = cols
            self._state.chessboard.rows = rows
            self._state.chessboard.square_size = square_size
            self._calib_widget._params_widget.sync_from_params()
        # Camera: restore by device_id or fallback index (cameras may have changed)
        saved_device_id = settings.value("camera_device_id")
        try:
            saved_index = int(settings.value("camera_index", -1))
        except (TypeError, ValueError):
            saved_index = None
        else:
            if saved_index < 0:
                saved_index = None
        self._camera_preview.set_selected_camera_from_save(saved_device_id, saved_index)
        # Resolution
        try:
            rw = int(settings.value("camera_resolution_width", -1))
            rh = int(settings.value("camera_resolution_height", -1))
        except (TypeError, ValueError):
            rw = rh = -1
        if rw > 0 and rh > 0:
            self._camera_preview.set_selected_resolution_from_save(rw, rh)
        # Load persisted K and dist (minimal calibration for realtime pose / 3D)
        self._load_persisted_calibration(settings)
        self._calib_widget.refresh_calibration_display()
        # Options: status bar toggles (persistent)
        self._show_image_coords = _settings_bool(settings, "tools_show_image_coords", True)
        self._show_yaw_pitch = _settings_bool(settings, "tools_show_yaw_pitch", False)
        self._action_show_image_coords.setChecked(self._show_image_coords)
        self._action_show_yaw_pitch.setChecked(self._show_yaw_pitch)
        self._refresh_status_bar()
        # Log: minimum level and display filter (persistent)
        log_level = settings.value("log_level", INFO)
        try:
            log_level = int(log_level)
        except (TypeError, ValueError):
            log_level = INFO
        log_level = max(DEBUG, min(ERROR, log_level))
        set_minimum_level(log_level)
        act = self._log_level_actions.get(log_level)
        if act is not None:
            act.setChecked(True)
        display_filter = settings.value("log_display_filter", DISPLAY_FILTER_DEBUG)
        try:
            display_filter = int(display_filter)
        except (TypeError, ValueError):
            display_filter = DISPLAY_FILTER_DEBUG
        display_filter = max(DISPLAY_FILTER_DEBUG, min(DISPLAY_FILTER_ERROR, display_filter))
        self._log_widget.set_display_filter_level(display_filter)
        # Center widget: restore open images and current tab
        open_paths = settings.value("center_open_paths")
        current_tab = settings.value("center_current_tab_index", 0)
        if open_paths and isinstance(open_paths, list) and len(open_paths) > 0:
            try:
                current_tab = int(current_tab) if current_tab is not None else 0
            except (TypeError, ValueError):
                current_tab = 0
            self._tabbed_view.restore_open_paths(open_paths, current_tab)

    def _load_persisted_calibration(self, settings: QSettings) -> None:
        """Restore K, dist, and image_size from settings into state.calibration if present."""
        from .state import CalibrationResult
        K_list = settings.value("calib_K")
        dist_list = settings.value("calib_dist")
        if not K_list or not dist_list:
            return
        try:
            K = np.array(K_list, dtype=np.float64).reshape(3, 3)
            dist = np.array(dist_list, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return
        if K.shape != (3, 3) or len(dist) < 5:
            return
        image_size = None
        size_list = settings.value("calib_image_size")
        if size_list and len(size_list) >= 2:
            try:
                image_size = (int(size_list[0]), int(size_list[1]))
            except (TypeError, ValueError):
                pass
        self._state.calibration_is_fake = False
        self._state.calibration = CalibrationResult(
            K=K,
            dist=dist,
            rvecs=[],
            tvecs=[],
            reproj_err=0.0,
            image_paths=[],
            image_size=image_size,
        )

    def _save_settings(self) -> None:
        """Save geometry, dock state, chessboard config, and camera selection."""
        self._calib_widget._params_widget.apply_to_params()
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("chessboard_cols", self._state.chessboard.cols)
        settings.setValue("chessboard_rows", self._state.chessboard.rows)
        settings.setValue("chessboard_square_size", self._state.chessboard.square_size)
        device_id, opencv_index = self._camera_preview.get_selected_camera_for_save()
        if device_id is not None:
            settings.setValue("camera_device_id", device_id)
        if opencv_index is not None:
            settings.setValue("camera_index", opencv_index)
        res = self._camera_preview.get_selected_resolution_for_save()
        if res is not None:
            settings.setValue("camera_resolution_width", res[0])
            settings.setValue("camera_resolution_height", res[1])
        if self._state.calibration is not None and not self._state.calibration_is_fake:
            settings.setValue("calib_K", self._state.calibration.K.ravel().tolist())
            settings.setValue("calib_dist", self._state.calibration.dist.ravel().tolist())
            if self._state.calibration.image_size is not None:
                settings.setValue("calib_image_size", list(self._state.calibration.image_size))
        settings.setValue("tools_show_image_coords", self._show_image_coords)
        settings.setValue("tools_show_yaw_pitch", self._show_yaw_pitch)
        settings.setValue("log_level", get_minimum_level())
        settings.setValue("log_display_filter", self._log_widget.get_display_filter_level())
        # Center widget: open image paths in tab order and current tab
        open_paths = self._tabbed_view.get_open_paths_in_tab_order()
        settings.setValue("center_open_paths", [str(p) for p in open_paths])
        settings.setValue("center_current_tab_index", self._tabbed_view.get_current_tab_index())

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _connect(self) -> None:
        self._gallery_calib.image_selected.connect(self._on_gallery_image_selected)
        self._gallery_work.image_selected.connect(self._on_gallery_image_selected)
        self._video_gallery.video_selected.connect(self._on_video_gallery_selected)
        self._calib_widget.calibration_done.connect(self._on_calibration_done)
        self._camera_preview.frame_available.connect(self._on_preview_frame)
        self._camera_preview.preview_running_changed.connect(self._on_preview_running_changed)
        self._tabbed_view.image_coords_changed.connect(self._on_image_coords_changed)

    def _on_image_coords_changed(self, pt: tuple[float, float] | None) -> None:
        self._last_image_coords = pt
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        parts = []
        if self._show_image_coords:
            if self._last_image_coords is not None:
                x, y = self._last_image_coords
                parts.append(f"Image: ({int(x)}, {int(y)})")
            else:
                parts.append("Image: —")
        if self._show_yaw_pitch:
            K = self._state.calibration.K if self._state.calibration is not None else None
            if K is not None and self._last_image_coords is not None:
                try:
                    yaw_deg, pitch_deg = image_coords_to_yaw_pitch_deg(
                        self._last_image_coords[0], self._last_image_coords[1], K
                    )
                    parts.append(f"Yaw: {yaw_deg:.1f}° Pitch: {pitch_deg:.1f}°")
                except Exception:
                    parts.append("Yaw: — Pitch: —")
            else:
                parts.append("Yaw: — Pitch: —")
        self._status_bar.showMessage("  |  ".join(parts) if parts else "")

    def _on_toggle_show_image_coords(self, checked: bool) -> None:
        self._show_image_coords = checked
        self._refresh_status_bar()

    def _on_toggle_show_yaw_pitch(self, checked: bool) -> None:
        self._show_yaw_pitch = checked
        self._refresh_status_bar()

    def _on_log_level_changed(self, level: int) -> None:
        set_minimum_level(level)
        self._save_log_settings()

    def _on_log_display_filter_changed(self) -> None:
        self._save_log_settings()

    def _save_log_settings(self) -> None:
        """Persist only log level and log display filter (called on Log menu or filter combo change)."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue("log_level", get_minimum_level())
        settings.setValue("log_display_filter", self._log_widget.get_display_filter_level())

    def _request_rotate_reset(self) -> None:
        """Reset rotation for the rotate-image demo (from pane button)."""
        demo = get_demo_by_id("rotate_image")
        doc = self._tabbed_view.current_document()
        if demo is not None and hasattr(demo, "reset_rotation") and doc is not None:
            demo.reset_rotation(doc)

    def _get_current_image_size(self) -> tuple[int, int] | None:
        """Return (width, height) of the current center image or video frame, or None."""
        doc = self._tabbed_view.current_document()
        if doc is None:
            return None
        if hasattr(doc, "image_bgr"):
            img = doc.image_bgr()
        elif hasattr(doc, "frame_bgr"):
            img = doc.frame_bgr()
        else:
            return None
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        return (w, h)

    def _get_demo_context(self) -> dict:
        return {
            "state": self._state,
            "get_current_document": lambda: self._tabbed_view.current_document(),
            "get_current_path": lambda: self._tabbed_view.current_path(),
            "get_current_video_document": lambda: self._tabbed_view.get_current_video_document(),
            "get_work_folder": lambda: self._images_base / "work",
            "get_open_documents_sorted": lambda: self._tabbed_view.get_open_documents_sorted(),
            "open_path": self._tabbed_view.open_path,
            "add_center_tab": self._tabbed_view.add_custom_tab,
            "get_K": lambda: self._state.calibration.K if self._state.calibration is not None else None,
            "get_calibration_image_size": lambda: self._state.calibration.image_size if self._state.calibration is not None else None,
            "switch_demo": self._switch_demo,
            "request_rotate_reset": self._request_rotate_reset,
        }

    def _switch_demo(self, demo_id: str) -> None:
        """Switch to another demo (e.g. from rotate when K is missing)."""
        btn = self._demos_buttons.get(demo_id)
        if btn is not None:
            btn.setChecked(True)
            self._on_demo_clicked(demo_id)

    def _on_demo_clicked(self, demo_id: str) -> None:
        prev_id = self._state.current_demo_id
        if prev_id == demo_id:
            return
        prev = get_demo_by_id(prev_id)
        if prev is not None:
            prev.on_deactivated()
        if prev_id == "realtime_pose":
            self._state.realtime_display_frame = None
        if prev is not None and hasattr(prev, "handle_mouse_event"):
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self._interactive_demo_event_filter)
        self._state.current_demo_id = demo_id
        self._state.realtime_pose = None
        try:
            self._demo_pane_stack.setCurrentIndex(self._demo_ids_order.index(demo_id))
        except ValueError:
            pass
        current = get_demo_by_id(demo_id)
        if current is not None:
            current.on_activated(self._get_demo_context())
        if current is not None and hasattr(current, "handle_mouse_event"):
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self._interactive_demo_event_filter)
        self._plot3d.redraw()

    def _on_preview_frame(self, frame) -> None:
        self._state.latest_camera_frame = frame
        current = get_demo_by_id(self._state.current_demo_id)
        if current is not None and hasattr(current, "on_frame"):
            current.on_frame(frame, self._get_demo_context())
        if self._state.current_demo_id == "realtime_pose" and self._state.realtime_display_frame is not None:
            self._camera_preview.set_frame_to_show(self._state.realtime_display_frame)
        else:
            self._camera_preview.set_frame_to_show(frame)
        if self._state.current_demo_id == "realtime_pose":
            self._plot3d.redraw()

    def _on_preview_running_changed(self, running: bool) -> None:
        self._gallery_calib.set_capture_enabled(running)
        self._gallery_work.set_capture_enabled(running)

    def _on_capture_requested(self, folder: Path) -> None:
        """Save current camera frame to the given gallery folder."""
        frame = self._state.latest_camera_frame
        if frame is None or frame.size == 0:
            logging_ui.log("Start camera preview first, then Capture.")
            return
        path = _next_capture_path(folder)
        if cv2.imwrite(str(path), frame):
            logging_ui.log(f"Saved: {path.name}")
            if folder == self._images_base / "calib":
                self._gallery_calib.reload()
            else:
                self._gallery_work.reload()
        else:
            logging_ui.log("Failed to save image.")

    def _on_gallery_image_selected(self, path: Path) -> None:
        self._state.selected_image_path = path
        self._tabbed_view.open_path(path)

    def _on_video_gallery_selected(self, path: Path) -> None:
        self._tabbed_view.open_video_path(path)

    def _on_calibration_done(self) -> None:
        self._calib_result_view.set_calibration_result(
            self._state.calibration,
            self._state.chessboard,
        )
        # For real calibration, add or switch to calibration result tab (closable like other docs)
        if not self._state.calibration_is_fake:
            self._tabbed_view.add_custom_tab(self._calib_result_view, "Calibration result")
        self._plot3d.redraw()


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
