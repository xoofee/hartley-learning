"""
3D plot: chessboard grid, camera pyramids (from K with fixed height), world and camera axes (RGB).

Single responsibility: 3D visualization of calibration result.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .state import AppState, CalibrationResult

# Pyramid size: fixed height; base aspect from K (fx, fy). Scale so pyramid is visible but not huge.
PYRAMID_HEIGHT = 0.4
PYRAMID_SCALE = 50.0  # larger = wider pyramid for same K


def camera_pyramid_from_K(
    R_cam_to_world: np.ndarray,
    t_cam_to_world: np.ndarray,
    K: np.ndarray,
    height: float = PYRAMID_HEIGHT,
    scale: float = PYRAMID_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build camera pyramid in world frame from K only (no sensor size).
    R_cam_to_world, t_cam_to_world: pose of camera in world (C = t, R = R_cw.T).
    Returns (base_corners (4,3), apex (3,)).
    Base aspect and proportions consistent with K: height/bottom_width ~ fx, height/bottom_length ~ fy.
    """
    fx, fy = K[0, 0], K[1, 1]
    if fx <= 0:
        fx = 1.0
    if fy <= 0:
        fy = 1.0
    half_w = height * scale / fx
    half_h = height * scale / fy
    # In camera frame: apex at origin, base at z = height
    b1 = np.array([-half_w, -half_h, height])
    b2 = np.array([half_w, -half_h, height])
    b3 = np.array([half_w, half_h, height])
    b4 = np.array([-half_w, half_h, height])
    base_cam = np.array([b1, b2, b3, b4])
    apex_cam = np.array([0.0, 0.0, 0.0])
    # To world
    base_world = (R_cam_to_world @ base_cam.T).T + t_cam_to_world.ravel()
    apex_world = (R_cam_to_world @ apex_cam) + t_cam_to_world.ravel()
    return base_world, apex_world


def draw_axes(ax, R_cam_to_world: np.ndarray, t_cam_to_world: np.ndarray, length: float = 0.3) -> None:
    """Draw RGB axes (X=R, Y=G, Z=B) at pose (R, t)."""
    o = t_cam_to_world.ravel()
    for i, color in enumerate(["red", "green", "blue"]):
        d = R_cam_to_world[:, i] * length
        ax.plot(
            [o[0], o[0] + d[0]],
            [o[1], o[1] + d[1]],
            [o[2], o[2] + d[2]],
            color=color,
            linewidth=2,
        )


def chessboard_grid_vertices(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Return (N, 3) world points for chessboard inner corners (z=0)."""
    pts = []
    for j in range(rows):
        for i in range(cols):
            pts.append([i * square_size, j * square_size, 0.0])
    return np.array(pts)


def draw_chessboard_grid(ax, cols: int, rows: int, square_size: float) -> None:
    """Draw chessboard grid in 3D at z=0 (wireframe)."""
    pts = chessboard_grid_vertices(cols, rows, square_size)
    # Draw edges: horizontal and vertical segments
    for j in range(rows):
        for i in range(cols - 1):
            a, b = pts[j * cols + i], pts[j * cols + i + 1]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "k-", linewidth=0.8)
    for i in range(cols):
        for j in range(rows - 1):
            a, b = pts[j * cols + i], pts[(j + 1) * cols + i]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "k-", linewidth=0.8)
    # Outer border
    p0 = pts[0]
    p1 = pts[cols - 1]
    p2 = pts[cols * rows - 1]
    p3 = pts[cols * (rows - 1)]
    for seg in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
        ax.plot(
            [seg[0][0], seg[1][0]],
            [seg[0][1], seg[1][1]],
            [seg[0][2], seg[1][2]],
            "k-",
            linewidth=1.2,
        )


class Calib3DPlot(QWidget):
    """3D plot widget: chessboard, camera pyramids, world and camera axes."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        import matplotlib.pyplot as plt
        self._fig = plt.figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._set_axes_limits()

    def _set_axes_limits(self) -> None:
        """Set 3D limits to fit chessboard and cameras."""
        cb = self._state.chessboard
        w = cb.cols * cb.square_size
        h = cb.rows * cb.square_size
        margin = max(w, h) * 0.5 + 1.0
        self._ax.set_xlim(-margin, w + margin)
        self._ax.set_ylim(-margin, h + margin)
        self._ax.set_zlim(-0.5, margin * 1.2)
        self._ax.set_box_aspect((1, 1, 1))

    def redraw(self) -> None:
        """Redraw chessboard, cameras, and axes from state.calibration."""
        self._ax.cla()
        cb = self._state.chessboard
        # World axes at origin (RGB)
        R_id = np.eye(3)
        t_id = np.zeros(3)
        draw_axes(self._ax, R_id, t_id, length=cb.square_size * 2)
        # Chessboard grid
        draw_chessboard_grid(self._ax, cb.cols, cb.rows, cb.square_size)
        cal = self._state.calibration
        if cal is not None:
            for rvec, tvec in zip(cal.rvecs, cal.tvecs):
                R_cam, _ = cv2.Rodrigues(rvec)
                # OpenCV: R, t map world -> camera. So camera center in world C = -R.T @ t
                C = (-R_cam.T @ tvec).ravel()
                R_cam_to_world = R_cam.T
                t_cam_to_world = C
                base_w, apex_w = camera_pyramid_from_K(
                    R_cam_to_world,
                    t_cam_to_world,
                    cal.K,
                )
                verts = [
                    [apex_w, base_w[0], base_w[1]],
                    [apex_w, base_w[1], base_w[2]],
                    [apex_w, base_w[2], base_w[3]],
                    [apex_w, base_w[3], base_w[0]],
                    [base_w[0], base_w[1], base_w[2], base_w[3]],
                ]
                self._ax.add_collection3d(
                    Poly3DCollection(
                        verts,
                        facecolors="cyan",
                        edgecolors="blue",
                        alpha=0.4,
                    )
                )
                draw_axes(self._ax, R_cam_to_world, t_cam_to_world, length=PYRAMID_HEIGHT * 1.5)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_title("Chessboard and camera poses")
        self._set_axes_limits()
        self._canvas.draw_idle()
