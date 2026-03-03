"""
Main window: dockable layout, camera preview, gallery, log, calibration, 3D plot.

Composes all widgets; single place for dock setup and wiring.
Saves/restores dock layout, geometry, camera selection, and chessboard config.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
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
)

from .state import AppState
from .logging_ui import set_log_sink
from .widgets import LogOutputWidget
from .camera_preview import CameraPreviewWidget
from .gallery import GalleryWidget
from .calibration import CalibrationWidget
from .plot3d import Calib3DPlot
from . import logging_ui

SETTINGS_ORG = "hartley-learning"
SETTINGS_APP = "chessboard_calib"


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

        # Central: image view
        self._image_view = ImageViewWidget()
        self.setCentralWidget(self._image_view)

        # Dock: Camera preview
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

        # Tabify or stack right docks for cleaner default layout
        self.tabifyDockWidget(dock_cam, dock_gal)
        self.tabifyDockWidget(dock_gal, dock_calib)

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

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _connect(self) -> None:
        self._gallery.image_selected.connect(self._on_gallery_image_selected)
        self._calib_widget.calibration_done.connect(self._on_calibration_done)

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
        self._plot3d.redraw()


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
