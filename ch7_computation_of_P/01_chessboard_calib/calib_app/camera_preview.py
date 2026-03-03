"""
Camera preview: enumerate cameras (prefer USB), start/stop preview, capture frame.

Single responsibility: camera enumeration, capture, and preview widget.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

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


def enumerate_cameras(max_index: int = 10) -> List[int]:
    """
    Return list of working camera indices. Prefer non-zero indices (often USB when 0 is integrated).
    """
    working = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                working.append(i)
    # Prefer index > 0 (USB) over 0 (often integrated)
    return sorted(working, key=lambda i: (i == 0, i))


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
        indices = enumerate_cameras()
        self._combo.clear()
        for i in indices:
            self._combo.addItem(f"Camera {i}", i)
        if indices and was_running:
            self._start_preview()

    def _current_index(self) -> Optional[int]:
        if self._combo.count() == 0:
            return None
        return self._combo.currentData()

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
