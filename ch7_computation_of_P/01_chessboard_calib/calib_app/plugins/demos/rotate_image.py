"""
Rotate image demo: drag on a work-gallery image to rotate (H = K @ R @ inv(K)).
Uses saved K; point under mouse stays fixed. Requires K and image from work gallery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox

from ..registry import Demo


def _rotation_from_d1_to_d2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ d1 = d2 (unit vectors)."""
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
        self._accumulated_R = np.eye(3)
        self._start_xy: Optional[tuple[float, float]] = None

    def on_deactivated(self) -> None:
        doc = None
        if hasattr(self, "_current_document") and self._current_document is not None:
            doc = self._current_document
        if doc is not None and hasattr(doc, "set_homography"):
            doc.set_homography(None)
        self._accumulated_R = np.eye(3)
        self._start_xy = None
        self._current_document = None

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
        d = self._K_inv @ np.array([ix, iy, 1.0], dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-10)
        if event_type == "press":
            self._start_xy = (ix, iy)
            self._current_document = document
            return True
        if event_type == "release":
            self._start_xy = None
            return True
        if event_type == "move" and self._start_xy is not None:
            sx, sy = self._start_xy
            d0 = self._K_inv @ np.array([sx, sy, 1.0], dtype=np.float64)
            d0 = d0 / (np.linalg.norm(d0) + 1e-10)
            # Drag rotates the view (reversal of camera rotate): point under mouse stays under mouse.
            # So R maps current ray to start ray: R @ d = d0 (content moves with drag).
            R_delta = _rotation_from_d1_to_d2(d, d0)
            self._accumulated_R = R_delta @ self._accumulated_R
            # H maps display coords to source: p_src = H @ p_dst, so H = K @ R.T @ inv(K)
            H = self._K @ self._accumulated_R.T @ self._K_inv
            document.set_homography(H)
            return True
        return False
