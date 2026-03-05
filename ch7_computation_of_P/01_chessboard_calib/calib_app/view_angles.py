"""
Shared view-angle helpers: direction <-> yaw/pitch (camera x right, y down, z forward).
Used by status bar (yaw/pitch from image coords + K) and rotate-image demo.
"""
from __future__ import annotations

import numpy as np


def direction_to_yaw_pitch(d: np.ndarray) -> tuple[float, float]:
    """Extract (yaw, pitch) in radians from unit view direction d. No roll (upright).
    Camera: x right, y down, z forward. Yaw = around Y (azimuth), pitch = around X (elevation).
    d = Ry(yaw) @ Rx(pitch) @ [0,0,1] = [cos(pitch)*sin(yaw), -sin(pitch), cos(pitch)*cos(yaw)]."""
    d = np.asarray(d, dtype=np.float64).ravel()[:3]
    d = d / (np.linalg.norm(d) + 1e-10)
    pitch = np.arcsin(np.clip(-float(d[1]), -1.0, 1.0))
    yaw = np.arctan2(float(d[0]), float(d[2]))
    return float(yaw), float(pitch)


def image_coords_to_yaw_pitch_deg(ix: float, iy: float, K: np.ndarray) -> tuple[float, float]:
    """Convert image pixel (ix, iy) and 3x3 K to (yaw_deg, pitch_deg). One decimal place is applied by caller if needed."""
    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    d = K_inv @ np.array([ix, iy, 1.0], dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-10)
    yaw_rad, pitch_rad = direction_to_yaw_pitch(d)
    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


def R_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """R such that R @ [0,0,1] = view direction. Yaw around Y, pitch around X (camera: x right, y down, z forward)."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return Ry @ Rx
