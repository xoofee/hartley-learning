"""
Rotate image demo: drag on a work-gallery image to rotate (H = K @ R @ inv(K)).
Uses saved K; point under mouse stays fixed. Requires K and image from work gallery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QPushButton

from ..registry import Demo


def _rotation_from_d1_to_d2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ d1 = d2 (unit vectors). Camera right => image left, so rotate image: R maps d0->d (point follows cursor)."""
    d1 = np.asarray(d1, dtype=np.float64).ravel()[:3]
    d2 = np.asarray(d2, dtype=np.float64).ravel()[:3]
    d1 = d1 / (np.linalg.norm(d1) + 1e-10)
    d2 = d2 / (np.linalg.norm(d2) + 1e-10)
    dot = float(np.dot(d1, d2))
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        axis = np.array([-d1[1], d1[0], 0.0])
        n = np.linalg.norm(axis)
        if n < 1e-10:
            axis = np.array([1, 0, 0])
        else:
            axis = axis / n
        rvec = axis * np.pi
        R, _ = cv2.Rodrigues(rvec)
        return R
    axis = np.cross(d1, d2)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    rvec = axis * angle
    R, _ = cv2.Rodrigues(rvec)
    return R


def _yaw_pitch_from_direction(d: np.ndarray) -> tuple[float, float]:
    """Extract (yaw, pitch) from unit view direction d. No roll (upright).
    Camera: x right, y down, z forward. Yaw = around Y (azimuth), pitch = around X (elevation).
    d = Ry(yaw) @ Rx(pitch) @ [0,0,1] = [cos(pitch)*sin(yaw), -sin(pitch), cos(pitch)*cos(yaw)]."""
    d = np.asarray(d, dtype=np.float64).ravel()[:3]
    d = d / (np.linalg.norm(d) + 1e-10)
    pitch = np.arcsin(np.clip(-float(d[1]), -1.0, 1.0))
    yaw = np.arctan2(float(d[0]), float(d[2]))
    return float(yaw), float(pitch)


def _R_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """R such that R @ [0,0,1] = view direction. Yaw around Y, pitch around X (camera: x right, y down, z forward)."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # Ry(yaw) around Y; Rx(pitch) around X. R = Ry @ Rx => R @ [0,0,1] = [cp*sy, -sp, cp*cy]
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return Ry @ Rx


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
        self._start_xy: Optional[tuple[float, float]] = None
        self._start_d0: Optional[np.ndarray] = None  # normalized ray at press

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

    def reset_rotation(self, document: Any) -> None:
        """Clear rotation and show original image (no homography)."""
        if hasattr(document, "set_homography"):
            document.set_homography(None)
        self._yaw_acc = 0.0
        self._pitch_acc = 0.0
        self._start_xy = None
        self._start_d0 = None

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        """Pane with Reset rotation button."""
        request_reset = context.get("request_rotate_reset")
        if not callable(request_reset):
            return None
        widget = QWidget()
        layout = QVBoxLayout(widget)
        btn = QPushButton("Reset rotation")
        btn.setToolTip("Show the original image (no rotation)")
        btn.clicked.connect(request_reset)
        layout.addWidget(btn)
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
                R_acc = _R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
                R_delta = _rotation_from_d1_to_d2(self._start_d0, d)  # R @ d0 = d: point follows cursor (camera right => image left)
                R_total = R_delta @ R_acc
                view_dir = R_total.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self._yaw_acc, self._pitch_acc = _yaw_pitch_from_direction(view_dir)
                R = _R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
                H = self._K @ R.T @ self._K_inv
                document.set_homography(H)
            self._start_xy = None
            self._start_d0 = None
            return True
        if event_type == "move" and self._start_d0 is not None:
            d0 = self._start_d0
            d = self._K_inv @ np.array([ix, iy, 1.0], dtype=np.float64)
            d = d / (np.linalg.norm(d) + 1e-10)
            R_acc = _R_from_yaw_pitch(self._yaw_acc, self._pitch_acc)
            R_delta = _rotation_from_d1_to_d2(d0, d)  # R @ d0 = d: point follows cursor
            R_total = R_delta @ R_acc
            view_dir = R_total.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            yaw_d, pitch_d = _yaw_pitch_from_direction(view_dir)
            R = _R_from_yaw_pitch(yaw_d, pitch_d)
            H = self._K @ R.T @ self._K_inv
            document.set_homography(H)
            return True
        return False
