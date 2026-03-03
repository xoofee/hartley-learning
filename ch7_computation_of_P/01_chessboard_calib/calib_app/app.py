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
from PyQt5.QtCore import Qt, QSettings
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


def _next_capture_path(folder: Path) -> Path:
    """Return next available path calib_001.jpg, calib_002.jpg, ..."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        p = folder / f"calib_{i:04d}.jpg"
        if not p.exists():
            return p
    return folder / "calib_9999.jpg"


class ImageViewWidget(QWidget):
    """Central area: show selected gallery image or placeholder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(400, 300)
        self._label.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._label.setText("Select an image from the gallery or take a photo")
        self._label.setFrameStyle(QFrame.StyledPanel)
        layout.addWidget(self._label)

    def set_image_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._label.setText("Select an image from the gallery or take a photo")
            self._label.setPixmap(QPixmap())
            return
        img = cv2.imread(str(path))
        if img is None:
            self._label.setText("Failed to load image")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._label.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._label.setText("")

    def clear(self) -> None:
        self.set_image_path(None)


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

        # Central: stacked — gallery image view | calibration result images
        self._image_view = ImageViewWidget()
        self._calib_result_view = CalibrationResultImagesWidget(self._state)
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self._image_view)  # index 0
        self._central_stack.addWidget(self._calib_result_view)  # index 1
        self.setCentralWidget(self._central_stack)

        # Dock: Camera preview (shows raw when demo "none", processed when "Realtime pose")
        self._camera_preview = CameraPreviewWidget(on_capture=self._on_capture)
        dock_cam = QDockWidget("Camera preview", self)
        dock_cam.setObjectName("CameraPreview")
        dock_cam.setWidget(self._camera_preview)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_cam)

        # Dock: Gallery
        self._gallery = GalleryWidget(self._state.gallery_folder)
        dock_gal = QDockWidget("Gallery", self)
        dock_gal.setObjectName("Gallery")
        dock_gal.setWidget(self._gallery)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_gal)

        # Dock: Log
        self._log_widget = LogOutputWidget(title="Log", show_clear_button=True)
        set_log_sink(self._log_widget)
        dock_log = QDockWidget("Log", self)
        dock_log.setObjectName("Log")
        dock_log.setWidget(self._log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_log)

        # Dock: Calibration (params + button)
        self._calib_widget = CalibrationWidget(
            self._state,
            get_gallery_paths=lambda: self._gallery.get_paths(),
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

        # Tabify or stack right docks for cleaner default layout
        self.tabifyDockWidget(dock_cam, dock_gal)
        self.tabifyDockWidget(dock_gal, dock_calib)
        self.tabifyDockWidget(dock_calib, dock_demos)

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
        self._gallery.image_selected.connect(self._on_gallery_image_selected)
        self._calib_widget.calibration_done.connect(self._on_calibration_done)
        self._camera_preview.frame_available.connect(self._on_preview_frame)

    def _get_demo_context(self) -> dict:
        return {"state": self._state}

    def _on_demo_clicked(self, demo_id: str) -> None:
        prev_id = self._state.current_demo_id
        if prev_id == demo_id:
            return
        prev = get_demo_by_id(prev_id)
        if prev is not None:
            prev.on_deactivated()
        if prev_id == "realtime_pose":
            self._state.realtime_display_frame = None
        self._state.current_demo_id = demo_id
        self._state.realtime_pose = None
        current = get_demo_by_id(demo_id)
        if current is not None:
            current.on_activated(self._get_demo_context())
        self._plot3d.redraw()

    def _on_preview_frame(self, frame) -> None:
        current = get_demo_by_id(self._state.current_demo_id)
        if current is not None and hasattr(current, "on_frame"):
            current.on_frame(frame, self._get_demo_context())
        # Show processed frame in preview when Realtime pose, else raw
        if self._state.current_demo_id == "realtime_pose" and self._state.realtime_display_frame is not None:
            self._camera_preview.set_frame_to_show(self._state.realtime_display_frame)
        else:
            self._camera_preview.set_frame_to_show(frame)
        if self._state.current_demo_id == "realtime_pose":
            self._plot3d.redraw()

    def _on_capture(self, frame) -> None:
        path = _next_capture_path(self._state.gallery_folder)
        if cv2.imwrite(str(path), frame):
            logging_ui.log(f"Saved: {path.name}")
            self._gallery.reload()
        else:
            logging_ui.log("Failed to save image.")

    def _on_gallery_image_selected(self, path: Path) -> None:
        self._state.selected_image_path = path
        self._image_view.set_image_path(path)

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
