"""
Camera preview: enumerate cameras via Qt (no OpenCV probe), preview/capture via OpenCV.

Single responsibility: camera enumeration (QCameraInfo), capture, and preview widget.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
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

    def __init__(
        self,
        on_capture: Optional[Callable[[np.ndarray], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._on_capture = on_capture
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
        self._preview_label.setStyleSheet("background-color: #222; color: #888;")
        self._preview_label.setText("Preview off")
        self._preview_label.setFrameStyle(QFrame.StyledPanel)
        layout.addWidget(self._preview_label)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start preview")
        self._start_btn.clicked.connect(self._toggle_preview)
        self._capture_btn = QPushButton("Take photo")
        self._capture_btn.clicked.connect(self._capture)
        self._capture_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._capture_btn)
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

    def _start_preview(self) -> None:
        idx = self._current_index()
        if idx is None:
            return
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            return
        self._timer.start(30)
        self._start_btn.setText("Stop preview")
        self._capture_btn.setEnabled(True)

    def _stop_preview(self) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._start_btn.setText("Start preview")
        self._capture_btn.setEnabled(False)
        self._preview_label.setText("Preview off")
        self._preview_label.setPixmap(QPixmap())

    def _on_timer(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        self._preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _capture(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if ret and self._on_capture is not None:
            self._on_capture(frame.copy())

    def closeEvent(self, event) -> None:
        self._stop_preview()
        super().closeEvent(event)
