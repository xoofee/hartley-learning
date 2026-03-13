"""
Video document: tab content with video player (play/pause, progress bar, time).
Paused at first frame when opened.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QFrame,
    QSizePolicy,
)


def _format_time(ms: int) -> str:
    """Format milliseconds as M:SS."""
    if ms < 0:
        ms = 0
    s = ms // 1000
    m = s // 60
    s = s % 60
    return f"{m}:{s:02d}"


class VideoDocumentWidget(QFrame):
    """
    Single video document: player with play/pause, stop, progress bar (seek), time display.
    Exposes path(), frame_bgr() (current frame), get_frame_at_ms(ms) for SFM/demos.
    """

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_frame_bgr: Optional[np.ndarray] = None
        self._duration_ms = 0
        self._position_ms = 0
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._open_capture()
        self._init_ui()
        self._seek_to_ms(0)
        self._update_ui()

    def _open_capture(self) -> None:
        self._cap = cv2.VideoCapture(str(self._path))
        if self._cap.isOpened():
            self._duration_ms = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1e-6, self._cap.get(cv2.CAP_PROP_FPS)) * 1000)
        else:
            self._duration_ms = 0

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._video_label = QLabel()
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self._video_label.setText("No video")
        layout.addWidget(self._video_label)

        controls = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(60)
        self._play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(50)
        self._stop_btn.clicked.connect(self._stop)
        controls.addWidget(self._stop_btn)
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setMinimumWidth(100)
        controls.addWidget(self._time_label)
        controls.addStretch()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(10000)
        self._slider.setValue(0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.setMinimumWidth(200)
        controls.addWidget(self._slider, 1)
        layout.addLayout(controls)
        self._slider_pressed = False

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        self._playing = True
        self._play_btn.setText("Pause")
        self._timer.start(33)

    def _pause(self) -> None:
        self._playing = False
        self._play_btn.setText("Play")
        self._timer.stop()

    def _stop(self) -> None:
        self._pause()
        self._seek_to_ms(0)

    def _on_tick(self) -> None:
        if self._cap is None or not self._cap.isOpened() or self._slider_pressed:
            return
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._pause()
            self._seek_to_ms(self._duration_ms)
            return
        self._current_frame_bgr = frame
        self._position_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        self._update_ui()
        self._update_slider_from_position()

    def _seek_to_ms(self, ms: int) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        ms = max(0, min(ms, self._duration_ms))
        self._cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ret, frame = self._cap.read()
        self._current_frame_bgr = frame if ret and frame is not None else self._current_frame_bgr
        self._position_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        self._update_ui()
        self._update_slider_from_position()

    def _update_slider_from_position(self) -> None:
        if self._slider_pressed or self._duration_ms <= 0:
            return
        v = int(10000 * self._position_ms / self._duration_ms)
        self._slider.blockSignals(True)
        self._slider.setValue(v)
        self._slider.blockSignals(False)

    def _on_slider_moved(self, value: int) -> None:
        if self._duration_ms <= 0:
            return
        ms = int(self._duration_ms * value / 10000)
        self._seek_to_ms(ms)

    def _on_slider_pressed(self) -> None:
        self._slider_pressed = True

    def _on_slider_released(self) -> None:
        self._slider_pressed = False
        v = self._slider.value()
        if self._duration_ms > 0:
            ms = int(self._duration_ms * v / 10000)
            self._seek_to_ms(ms)

    def _update_display(self) -> None:
        if self._current_frame_bgr is not None and self._current_frame_bgr.size > 0:
            h, w = self._current_frame_bgr.shape[:2]
            rgb = cv2.cvtColor(self._current_frame_bgr, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())
            self._video_label.setPixmap(
                pix.scaled(
                    self._video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self._video_label.setText("No frame")

    def _update_ui(self) -> None:
        self._time_label.setText(f"{_format_time(self._position_ms)} / {_format_time(self._duration_ms)}")
        self._update_display()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()

    def path(self) -> Path:
        return self._path

    def frame_bgr(self) -> Optional[np.ndarray]:
        """Current frame (BGR) for demos."""
        return self._current_frame_bgr.copy() if self._current_frame_bgr is not None else None

    def get_frame_at_ms(self, ms: int) -> Optional[np.ndarray]:
        """Get frame at given position (ms); restores playback position and display afterward."""
        if self._cap is None or not self._cap.isOpened():
            return None
        prev_ms = self._position_ms
        self._cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ret, frame = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_MSEC, prev_ms)
        ret_restore, frame_restore = self._cap.read()
        self._position_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        if ret_restore and frame_restore is not None:
            self._current_frame_bgr = frame_restore
            self._update_ui()
            self._update_slider_from_position()
        return frame if ret and frame is not None else None

    def duration_ms(self) -> int:
        return self._duration_ms

    def position_ms(self) -> int:
        return self._position_ms

    def release(self) -> None:
        """Release video capture (call when tab is closed)."""
        self._timer.stop()
        self._playing = False
        self._play_btn.setText("Play")
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def closeEvent(self, event) -> None:
        self.release()
        super().closeEvent(event)
