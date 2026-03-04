"""
Camera preview: enumerate cameras via Qt (no OpenCV probe), preview/capture via OpenCV.

Single responsibility: camera enumeration (QCameraInfo), capture, and preview widget.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFrame,
)

try:
    from PyQt5.QtMultimedia import QCameraInfo
except ImportError:
    QCameraInfo = None  # type: ignore[misc, assignment]

# Resolutions offered in the resolution combo (width, height)
RESOLUTION_CHOICES = [
    (3840, 2160),
    (1920, 1080),
    (1280, 800),
    (1280, 720),
    (1024, 768),
    (640, 480),
]


def get_cameras_qt() -> List[Tuple[int, str, str]]:
    """
    Return list of (opencv_index, display_name, device_id) using Qt's QCameraInfo.
    device_id is info.deviceName() for save/restore when camera list changes.
    """
    if QCameraInfo is None:
        return []
    infos = QCameraInfo.availableCameras()
    result: List[Tuple[int, str, str]] = []
    for i, info in enumerate(infos):
        name = info.description().strip() or info.deviceName() or f"Camera {i}"
        device_id = info.deviceName() or ""
        result.append((i, name, device_id))
    return result


class CameraPreviewWidget(QWidget):
    """Camera selection, start/stop preview, and capture button."""

    frame_available = pyqtSignal(object)  # BGR frame (numpy array) when preview is running

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._camera_list: List[Tuple[int, str, str]] = []  # (opencv_index, display_name, device_id)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("Camera:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(120)
        self._refresh_cameras()
        row.addWidget(self._combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_cameras)
        row.addWidget(refresh_btn)
        row.addStretch()
        layout.addLayout(row)

        self._preview_label = QLabel()
        self._preview_label.setMinimumSize(320, 240)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._preview_label.setText("Preview off")
        self._preview_label.setFrameStyle(QFrame.StyledPanel)
        layout.addWidget(self._preview_label)

        btn_row = QHBoxLayout()
        self._res_combo = QComboBox()
        self._res_combo.setMinimumWidth(100)
        for w, h in RESOLUTION_CHOICES:
            self._res_combo.addItem(f"{w}×{h}", (w, h))
        btn_row.addWidget(self._res_combo)
        self._start_btn = QPushButton("Start preview")
        self._start_btn.clicked.connect(self._toggle_preview)
        btn_row.addWidget(self._start_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _refresh_cameras(self) -> None:
        was_running = self._timer.isActive()
        if was_running:
            self._stop_preview()
        self._camera_list = get_cameras_qt()
        self._combo.clear()
        for opencv_index, name, _ in self._camera_list:
            self._combo.addItem(f"{name} ({opencv_index})", opencv_index)
        if not self._camera_list:
            self._combo.addItem("No cameras found", -1)
        if self._camera_list and was_running:
            self._start_preview()

    def get_selected_camera_for_save(self) -> Tuple[Optional[str], Optional[int]]:
        """Return (device_id, opencv_index) for the current selection, for persisting."""
        idx = self._current_index()
        if idx is None:
            return None, None
        for opencv_index, _name, device_id in self._camera_list:
            if opencv_index == idx:
                return (device_id or None), opencv_index
        return None, idx

    def set_selected_camera_from_save(
        self,
        device_id: Optional[str],
        fallback_index: Optional[int],
    ) -> None:
        """
        Restore selection: prefer camera with device_id; else use fallback_index if in range.
        Call after _refresh_cameras() so _camera_list is populated.
        """
        if not self._camera_list:
            return
        combo_index = 0
        if device_id:
            for i, (_idx, _name, did) in enumerate(self._camera_list):
                if did == device_id:
                    combo_index = i
                    break
        elif fallback_index is not None:
            for i, (idx, _name, _did) in enumerate(self._camera_list):
                if idx == fallback_index:
                    combo_index = i
                    break
        if 0 <= combo_index < self._combo.count():
            self._combo.setCurrentIndex(combo_index)

    def get_selected_resolution_for_save(self) -> Optional[Tuple[int, int]]:
        """Return (width, height) of current resolution combo for persisting."""
        data = self._res_combo.currentData()
        if data is not None:
            return data
        return None

    def set_selected_resolution_from_save(self, width: Optional[int], height: Optional[int]) -> None:
        """Restore resolution combo to (width, height) if it exists in the list."""
        if width is None or height is None:
            return
        for i in range(self._res_combo.count()):
            data = self._res_combo.itemData(i)
            if data is not None and data[0] == width and data[1] == height:
                self._res_combo.setCurrentIndex(i)
                break

    def set_frame_to_show(self, frame_bgr: Optional[np.ndarray]) -> None:
        """Display a BGR frame (scaled to fit, aspect ratio preserved). None = show placeholder."""
        if frame_bgr is None or frame_bgr.size == 0:
            self._preview_label.setText("Preview off")
            self._preview_label.setPixmap(QPixmap())
            return
        if len(frame_bgr.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if w <= 0 or h <= 0:
            return
        label_w = self._preview_label.width()
        label_h = self._preview_label.height()
        scale = min(label_w / w, label_h / h, 1.0)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            if nw < 1:
                nw = 1
            if nh < 1:
                nh = 1
            frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = nh, nw
        bytes_per_line = 3 * w
        qimg = QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self._preview_label.setPixmap(QPixmap.fromImage(qimg))
        self._preview_label.setText("")

    def _current_index(self) -> Optional[int]:
        if self._combo.count() == 0:
            return None
        idx = self._combo.currentData()
        if idx is None or idx < 0:
            return None
        return int(idx)

    def _toggle_preview(self) -> None:
        if self._timer.isActive():
            self._stop_preview()
        else:
            self._start_preview()

    def _selected_resolution(self) -> Tuple[int, int]:
        """Return (width, height) for the current resolution combo selection."""
        if self._res_combo.count() == 0:
            return 1280, 720
        data = self._res_combo.currentData()
        if data is not None:
            return data
        return 1280, 720

    def _start_preview(self) -> None:
        idx = self._current_index()
        if idx is None:
            return
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            return
        w, h = self._selected_resolution()
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._timer.start(30)
        self._start_btn.setText("Stop preview")

    def _stop_preview(self) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._start_btn.setText("Start preview")
        self._preview_label.setText("Preview off")
        self._preview_label.setPixmap(QPixmap())

    def _on_timer(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret:
            return
        self.frame_available.emit(frame)
        # Display is updated by the main window via set_frame_to_show (raw or processed)

    def closeEvent(self, event) -> None:
        self._stop_preview()
        super().closeEvent(event)
