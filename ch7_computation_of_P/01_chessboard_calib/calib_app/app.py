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
from PyQt5.QtCore import Qt, QSettings, QEvent, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QDockWidget,
    QScrollArea,
    QFrame,
    QStackedWidget,
    QTabWidget,
    QSizePolicy,
)

from .state import AppState
from .logging_ui import set_log_sink
from .widgets import LogOutputWidget
from .camera_preview import CameraPreviewWidget
from .gallery import GalleryWidget
from .calibration import CalibrationWidget
from .plot3d import Calib3DPlot
from .calibration_result_view import CalibrationResultImagesWidget
from . import logging_ui
from .plugins.registry import get_demo_by_id
from .plugins.demos import register_builtin_demos, build_demos_button_group

SETTINGS_ORG = "hartley-learning"
SETTINGS_APP = "chessboard_calib"

register_builtin_demos()


class _RotateDemoEventFilter(QObject):
    """Event filter for tabbed view: forward mouse events to rotate demo when active."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main = main_window

    def eventFilter(self, obj, event) -> bool:
        if self._main._state.current_demo_id != "rotate_image":
            return False
        t = event.type()
        if t not in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
            return False
        tabbed = self._main._tabbed_view
        if obj != tabbed:
            return False
        doc = tabbed.current_document()
        if doc is None:
            return False
        pos_in_doc = doc.mapFromGlobal(obj.mapToGlobal(event.pos()))
        event_type = "press" if t == QEvent.MouseButtonPress else ("release" if t == QEvent.MouseButtonRelease else "move")
        demo = get_demo_by_id("rotate_image")
        if demo is None or not hasattr(demo, "handle_mouse_event"):
            return False
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
    """Single-document image view; path and optional homography for rotate demo."""

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = path
        self._img_bgr = None
        self._H = None
        self._load_image()
        layout = QVBoxLayout(self)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(400, 300)
        self._label.setStyleSheet("background-color: #f0f0f0; color: #444;")
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._label)
        self._update_display()

    def _load_image(self) -> None:
        img = cv2.imread(str(self._path))
        self._img_bgr = img if img is not None else None

    def path(self) -> Path:
        return self._path

    def image_bgr(self) -> np.ndarray | None:
        return self._img_bgr

    def set_homography(self, H: np.ndarray | None) -> None:
        """Set 3x3 homography for display (e.g. from rotate demo). None = show original."""
        self._H = H
        self._update_display()

    def _update_display(self) -> None:
        if self._img_bgr is None:
            self._label.setText("Failed to load image")
            self._label.setPixmap(QPixmap())
            return
        img = self._img_bgr
        if self._H is not None:
            h, w = img.shape[:2]
            img = cv2.warpPerspective(img, self._H, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        self._img_wh = (w, h)
        lw, lh = self._label.width(), self._label.height()
        if lw > 0 and lh > 0:
            scale = min(lw / w, lh / h, 1.0)
            if scale < 1.0:
                nw, nh = int(w * scale), int(h * scale)
                if nw < 1: nw = 1
                if nh < 1: nh = 1
                img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                h, w = nh, nw
        self._pixmap_wh = (w, h)
        bytes_per_line = 3 * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._label.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._label.setText("")

    def map_to_image_coords(self, wx: float, wy: float) -> tuple[float, float] | None:
        """Map widget coords to image pixel (x, y); None if outside image."""
        if not hasattr(self, "_img_wh") or not hasattr(self, "_pixmap_wh"):
            return None
        iw, ih = self._img_wh
        nw, nh = self._pixmap_wh
        ww, wh = self.width(), self.height()
        left = (ww - nw) / 2
        top = (wh - nh) / 2
        px, py = wx - left, wy - top
        if 0 <= px < nw and 0 <= py < nh:
            return (px * iw / nw, py * ih / nh)
        return None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()


class CentralTabbedView(QWidget):
    """Tabbed image documents; closable tabs; open by path or switch to existing tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self._close_tab)
        self._path_to_index: dict = {}
        layout.addWidget(self._tab_widget)
        self._placeholder = QLabel("Open an image from a gallery (click a thumbnail)")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._placeholder.setMinimumSize(400, 300)
        self._show_placeholder()

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
        self._tab_widget.removeTab(index)
        self._rebuild_path_index()
        if self._tab_widget.count() == 0:
            self._show_placeholder()

    def current_document(self) -> ImageDocumentWidget | None:
        w = self._tab_widget.currentWidget()
        if w is not None and isinstance(w, ImageDocumentWidget):
            return w
        return None

    def current_path(self) -> Path | None:
        doc = self.current_document()
        return doc.path() if doc is not None else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._state = AppState()
        self._state.refresh_gallery_paths()
        self._init_ui()
        self._connect()
        self._restore_settings()

    def _init_ui(self) -> None:
        self.setWindowTitle("Chessboard calibration")
        self.setGeometry(100, 100, 1280, 800)

        # Central: stacked — tabbed image docs | calibration result images
        self._images_base = Path(__file__).resolve().parent.parent / "images"
        self._tabbed_view = CentralTabbedView()
        self._calib_result_view = CalibrationResultImagesWidget(self._state)
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self._tabbed_view)  # index 0
        self._central_stack.addWidget(self._calib_result_view)  # index 1
        self.setCentralWidget(self._central_stack)

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

        # Dock: Log
        self._log_widget = LogOutputWidget(title="Log", show_clear_button=True)
        set_log_sink(self._log_widget)
        dock_log = QDockWidget("Log", self)
        dock_log.setObjectName("Log")
        dock_log.setWidget(self._log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_log)

        # Dock: Calibration (params + button; uses calibration gallery only)
        self._calib_widget = CalibrationWidget(
            self._state,
            get_gallery_paths=lambda: self._gallery_calib.get_paths(),
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

        # Dock: Demos
        demos_group, self._demos_button_group, self._demos_buttons = build_demos_button_group(self)
        dock_demos = QDockWidget("Demos", self)
        dock_demos.setObjectName("Demos")
        dock_demos.setWidget(demos_group)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_demos)
        for demo_id, btn in self._demos_buttons.items():
            btn.clicked.connect(lambda checked, did=demo_id: self._on_demo_clicked(did))

        # Tabify right docks
        self.tabifyDockWidget(dock_cam, dock_calib_gal)
        self.tabifyDockWidget(dock_calib_gal, dock_work_gal)
        self.tabifyDockWidget(dock_work_gal, dock_calib)
        self.tabifyDockWidget(dock_calib, dock_demos)

        self._rotate_event_filter = _RotateDemoEventFilter(self)
        logging_ui.log("Calibration app started. Add images and run Calibrate from gallery.")
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

    def _load_persisted_calibration(self, settings: QSettings) -> None:
        """Restore K and dist from settings into state.calibration if present."""
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
        self._state.calibration = CalibrationResult(
            K=K,
            dist=dist,
            rvecs=[],
            tvecs=[],
            reproj_err=0.0,
            image_paths=[],
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
        if self._state.calibration is not None:
            settings.setValue("calib_K", self._state.calibration.K.ravel().tolist())
            settings.setValue("calib_dist", self._state.calibration.dist.ravel().tolist())

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _connect(self) -> None:
        self._gallery_calib.image_selected.connect(self._on_gallery_image_selected)
        self._gallery_work.image_selected.connect(self._on_gallery_image_selected)
        self._calib_widget.calibration_done.connect(self._on_calibration_done)
        self._camera_preview.frame_available.connect(self._on_preview_frame)

    def _get_demo_context(self) -> dict:
        return {
            "state": self._state,
            "get_current_document": lambda: self._tabbed_view.current_document(),
            "get_current_path": lambda: self._tabbed_view.current_path(),
            "get_work_folder": lambda: self._images_base / "work",
            "get_K": lambda: self._state.calibration.K if self._state.calibration is not None else None,
            "switch_demo": self._switch_demo,
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
        if prev_id == "rotate_image":
            self._tabbed_view.removeEventFilter(self._rotate_event_filter)
        self._state.current_demo_id = demo_id
        self._state.realtime_pose = None
        current = get_demo_by_id(demo_id)
        if current is not None:
            current.on_activated(self._get_demo_context())
        if demo_id == "rotate_image":
            self._tabbed_view.installEventFilter(self._rotate_event_filter)
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

    def _on_calibration_done(self) -> None:
        self._calib_result_view.set_calibration_result(
            self._state.calibration,
            self._state.chessboard,
        )
        self._central_stack.setCurrentIndex(1)
        self._plot3d.redraw()


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
