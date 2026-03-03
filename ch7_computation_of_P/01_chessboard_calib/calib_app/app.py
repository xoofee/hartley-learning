"""
Main window: dockable layout, camera preview, gallery, log, calibration, 3D plot.

Composes all widgets; single place for dock setup and wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt
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
        self._label.setStyleSheet("background-color: #1a1a1a; color: #666;")
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

    def _init_ui(self) -> None:
        self.setWindowTitle("Chessboard calibration")
        self.setGeometry(100, 100, 1280, 800)

        # Central: image view
        self._image_view = ImageViewWidget()
        self.setCentralWidget(self._image_view)

        # Dock: Camera preview
        self._camera_preview = CameraPreviewWidget(on_capture=self._on_capture)
        dock_cam = QDockWidget("Camera preview", self)
        dock_cam.setWidget(self._camera_preview)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_cam)

        # Dock: Gallery
        self._gallery = GalleryWidget(self._state.gallery_folder)
        dock_gal = QDockWidget("Gallery", self)
        dock_gal.setWidget(self._gallery)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_gal)

        # Dock: Log
        self._log_widget = LogOutputWidget(title="Log", show_clear_button=True)
        set_log_sink(self._log_widget)
        dock_log = QDockWidget("Log", self)
        dock_log.setWidget(self._log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_log)

        # Dock: Calibration (params + button)
        self._calib_widget = CalibrationWidget(
            self._state,
            get_gallery_paths=lambda: self._gallery.get_paths(),
        )
        dock_calib = QDockWidget("Calibration", self)
        dock_calib.setWidget(self._calib_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_calib)

        # Dock: 3D plot
        self._plot3d = Calib3DPlot(self._state)
        dock_3d = QDockWidget("3D plot", self)
        dock_3d.setWidget(self._plot3d)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_3d)

        # Tabify or stack right docks for cleaner default layout
        self.tabifyDockWidget(dock_cam, dock_gal)
        self.tabifyDockWidget(dock_gal, dock_calib)

        logging_ui.log("Calibration app started. Add images and run Calibrate from gallery.")
        self._plot3d.redraw()

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
