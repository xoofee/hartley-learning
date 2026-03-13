"""
Scene geometry in world coordinates.

Single responsibility: define 3D scene primitives (square, triangle, rectangle).
"""
from __future__ import annotations

import numpy as np


def get_scene_square(side: float = 1.0, z: float = 0.0) -> np.ndarray:
    """Horizontal square in plane z=0, centered at origin. (4, 3)."""
    h = side / 2
    return np.array([
        [-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]
    ])


def get_scene_triangle(width: float = 0.8, height: float = 1.0, x_plane: float = 1.5) -> np.ndarray:
    """Vertical triangle in plane x=x_plane. (3, 3)."""
    w2 = width / 2
    return np.array([
        [x_plane, -w2, 0],
        [x_plane, w2, 0],
        [x_plane, 0, height],
    ])


def get_scene_rectangle(
    width_y: float = 0.4,
    height_z: float = 0.4,
    x_plane: float = 1.5,
    y_center: float = 0.8,
    z_center: float = 0.4,
) -> np.ndarray:
    """Vertical rectangle in plane x=x_plane (same plane as triangle), near the triangle. (4, 3)."""
    hy, hz = width_y / 2, height_z / 2
    return np.array([
        [x_plane, y_center - hy, z_center - hz],
        [x_plane, y_center + hy, z_center - hz],
        [x_plane, y_center + hy, z_center + hz],
        [x_plane, y_center - hy, z_center + hz],
    ])
