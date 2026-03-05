"""
Rotate image demo: drag on a work-gallery image to rotate (H = K @ R @ inv(K)).
Uses saved K; point under mouse stays fixed. Requires K and image from work gallery.

BUG:

the accumulated rotation is not correct!

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSpinBox,
    QFormLayout,
    QLabel,
)

from ..registry import Demo
from ...view_angles import direction_to_yaw_pitch, R_from_yaw_pitch
from ...logging_ui import log

def _rotation_from_d1_to_d2(d1, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    v = np.cross(d1, d2)
    c = np.dot(d1, d2)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation
            # find orthogonal axis
            if abs(d1[0]) < abs(d1[2]):
                axis = np.array([0, -d1[2], d1[1]])
            else:
                axis = np.array([-d1[1], d1[0], 0])
            axis /= np.linalg.norm(axis)    # important! Rodrigues requires unit vector
            rvec = axis * np.pi
            R, _ = cv2.Rodrigues(rvec)
            return R

    axis = v / s
    angle = np.arctan2(s, c)
    rvec = axis * angle
    R, _ = cv2.Rodrigues(rvec)
    return R

class RotateImageDemo(Demo):
    def id(self) -> str:
        return "rotate_image"

    def label(self) -> str:
        return "Rotate image"

    def on_activated(self, context: dict) -> None:
        K = context.get("get_K") and context["get_K"]()
        if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
            QMessageBox.warning(
                None,
                "Rotate image",
                "Calibration matrix K is not available. Load or run calibration first.",
            )
            switch = context.get("switch_demo")
            if switch is not None:
                switch("none")
            return
        self._K = np.asarray(K, dtype=np.float64)
        self._K_inv = np.linalg.inv(self._K)
        self._yaw_acc: float = 0.0
        self._pitch_acc: float = 0.0
        self._start_xy = None
        self._start_d0 = None
        self._spin_yaw = getattr(self, "_spin_yaw", None)
        self._spin_pitch = getattr(self, "_spin_pitch", None)
        self._context = context
        self._update_pane_spins()

    def on_deactivated(self) -> None:
        doc = None
        if hasattr(self, "_current_document") and self._current_document is not None:
            doc = self._current_document
        if doc is not None and hasattr(doc, "set_homography"):
            doc.set_homography(None)
        self._yaw_acc = 0.0
        self._pitch_acc = 0.0
        self._start_xy = None
        self._start_d0 = None
        self._current_document = None
        self._context = None

    def reset_rotation(self, document: Any) -> None:
        """Clear rotation and show original image (no homography)."""
        if hasattr(document, "set_homography"):
            document.set_homography(None)
        self._yaw_acc = 0.0
        self._pitch_acc = 0.0
        self._start_xy = None
        self._start_d0 = None
        self._update_pane_spins()

    def _update_pane_spins(self, yaw_rad: Optional[float] = None, pitch_rad: Optional[float] = None) -> None:
        """Sync spin boxes to given angles (rad) or to _yaw_acc, _pitch_acc. Blocks signals to avoid feedback."""
        if self._spin_yaw is None or self._spin_pitch is None:
            return
        y = np.degrees(yaw_rad if yaw_rad is not None else self._yaw_acc)
        p = np.degrees(pitch_rad if pitch_rad is not None else self._pitch_acc)
        self._spin_yaw.blockSignals(True)
        self._spin_pitch.blockSignals(True)
        self._spin_yaw.setValue(int(round(y)))
        self._spin_pitch.setValue(int(round(np.clip(p, -90, 90))))
        self._spin_yaw.blockSignals(False)
        self._spin_pitch.blockSignals(False)

    def _apply_yaw_pitch_from_spins(self) -> None:
        """Apply current spin values (degrees) as rotation to the current document."""
        if self._spin_yaw is None or self._spin_pitch is None or self._context is None:
            return
        get_doc = self._context.get("get_current_document")
        doc = get_doc() if callable(get_doc) else None
        if doc is None or not hasattr(doc, "set_homography"):
            return
        yaw_deg = self._spin_yaw.value()
        pitch_deg = self._spin_pitch.value()
        self._yaw_acc = np.radians(float(yaw_deg))
        self._pitch_acc = np.radians(float(pitch_deg))
        R = R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
        H = self._K @ R.T @ self._K_inv
        doc.set_homography(H)

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        """Pane with Reset button and yaw/pitch spin boxes (1° step)."""
        request_reset = context.get("request_rotate_reset")
        if not callable(request_reset):
            return None
        self._context = context
        widget = QWidget()
        layout = QVBoxLayout(widget)
        btn = QPushButton("Reset rotation")
        btn.setToolTip("Show the original image (no rotation)")
        btn.clicked.connect(request_reset)
        layout.addWidget(btn)
        form = QFormLayout()
        self._spin_yaw = QSpinBox()
        self._spin_yaw.setRange(-180, 180)
        self._spin_yaw.setSingleStep(1)
        self._spin_yaw.setSuffix(" °")
        self._spin_yaw.setValue(0)
        self._spin_yaw.valueChanged.connect(self._apply_yaw_pitch_from_spins)
        form.addRow(QLabel("Yaw:"), self._spin_yaw)
        self._spin_pitch = QSpinBox()
        self._spin_pitch.setRange(-90, 90)
        self._spin_pitch.setSingleStep(1)
        self._spin_pitch.setSuffix(" °")
        self._spin_pitch.setValue(0)
        self._spin_pitch.valueChanged.connect(self._apply_yaw_pitch_from_spins)
        form.addRow(QLabel("Pitch:"), self._spin_pitch)
        layout.addLayout(form)
        layout.addStretch()
        return widget

    def handle_mouse_event(
        self,
        context: dict,
        document: Any,
        x: float,
        y: float,
        event_type: str,
    ) -> bool:
        """Return True if event was handled."""
        get_path = context.get("get_current_path")
        get_work = context.get("get_work_folder")
        if not get_path or not get_work:
            return False
        path = get_path()
        work_folder = Path(get_work()).resolve()
        if path is None or document is None:
            return False
        try:
            path_resolved = Path(path).resolve()
            work_resolved = Path(work_folder).resolve()
            try:
                path_resolved.relative_to(work_resolved)
            except ValueError:
                return False
        except Exception:
            return False
        if not hasattr(document, "map_to_image_coords") or not hasattr(document, "set_homography"):
            return False
        pt = document.map_to_image_coords(x, y)
        if pt is None:
            return False
        ix, iy = pt
        if event_type == "press":
            self._start_xy = (ix, iy)
            d0 = self._K_inv @ np.array([ix, iy, 1.0], dtype=np.float64)
            self._start_d0 = d0 / (np.linalg.norm(d0) + 1e-10)
            self._current_document = document
            return True
        if event_type == "release":
            if self._start_d0 is not None and self._current_document is document:
                d = self._K_inv @ np.array([ix, iy, 1.0], dtype=np.float64)
                d = d / (np.linalg.norm(d) + 1e-10)
                R_acc = R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
                R_delta = _rotation_from_d1_to_d2(self._start_d0, d)  # R @ d0 = d: point follows cursor (camera right => image left)
                R_total = R_delta @ R_acc
                view_dir = R_total.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self._yaw_acc, self._pitch_acc = direction_to_yaw_pitch(view_dir)
                R = R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
                H = self._K @ R.T @ self._K_inv
                document.set_homography(H)
                self._update_pane_spins()
            self._start_xy = None
            self._start_d0 = None
            return True
        if event_type == "move" and self._start_d0 is not None:
            d0 = self._start_d0
            d = self._K_inv @ np.array([ix, iy, 1.0], dtype=np.float64)
            d = d / (np.linalg.norm(d) + 1e-10)
            R_acc = R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
            R_delta = _rotation_from_d1_to_d2(d0, d)  # R @ d0 = d: point follows cursor
            R_total = R_delta @ R_acc
            view_dir = R_total.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            yaw_d, pitch_d = direction_to_yaw_pitch(view_dir)

            log(f'R_ccc: {R_acc}')
            log(f'R_delta: {R_delta}')
            log(f' d0: {d0} , d: {d}')
            log(f' view_dir: {view_dir}')
            log(f"yaw_d: {np.degrees(yaw_d):.1f}, pitch_d: {np.degrees(pitch_d):.1f}")

            R = R_from_yaw_pitch(yaw_d, pitch_d)
            H = self._K @ R.T @ self._K_inv
            document.set_homography(H)
            self._update_pane_spins(yaw_d, pitch_d)
            return True
        return False
