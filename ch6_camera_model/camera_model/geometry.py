"""
3D geometry: backproject ray, camera pyramid, plane–box intersection, P row planes.

Single responsibility: 3D visualization and ray/plane geometry.
"""
from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import pinhole


def backproject_image_point_to_ray(
    u: float,
    v: float,
    K: np.ndarray,
    R_cw: np.ndarray,
    t_world_to_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backproject image point (u, v) in pixel coords to a 3D ray in world coordinates.
    P = K [R|t], R = R_world_to_cam, t = -R @ C.
    Returns (C, d) where C is camera center, d is normalized ray direction (both in world coords).
    Ray: X(λ) = C + λ * d for λ > 0.
    """
    x_h = np.array([u, v, 1.0])
    d_cam = np.linalg.solve(K, x_h)
    d_world = R_cw.T @ d_cam
    d_norm = np.linalg.norm(d_world)
    if d_norm < 1e-12:
        d_world = np.array([0.0, 0.0, 1.0])
    else:
        d_world = d_world / d_norm
    C = (-R_cw.T @ t_world_to_cam).ravel()[:3]
    return C, d_world


def get_camera_pyramid(
    C: np.ndarray,
    R_cam: np.ndarray,
    scale: float = 0.5,
    depth: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_corners (4,3), apex C). Base is in front of camera at depth."""
    b1 = np.array([-scale, -scale, depth])
    b2 = np.array([scale, -scale, depth])
    b3 = np.array([scale, scale, depth])
    b4 = np.array([-scale, scale, depth])
    pyramid_points_cam = np.array([b1, b2, b3, b4])
    R_wc = R_cam.T
    pyramid_points_cam_world = (R_wc @ pyramid_points_cam.T).T + C
    return pyramid_points_cam_world, C


def _plane_box_intersection_polygon(
    n: np.ndarray,
    d: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> np.ndarray:
    """
    Intersection of plane n·x + d = 0 with axis-aligned box [xlim]×[ylim]×[zlim].
    Returns (N, 3) array of polygon vertices (ordered) or empty (0, 3) if no intersection.
    """
    x0, x1 = xlim[0], xlim[1]
    y0, y1 = ylim[0], ylim[1]
    z0, z1 = zlim[0], zlim[1]
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    pts = []
    for i, j in edges:
        A, B = corners[i], corners[j]
        denom = n @ (B - A)
        if abs(denom) < 1e-12:
            continue
        t = -(n @ A + d) / denom
        if 0 <= t <= 1:
            p = A + t * (B - A)
            pts.append(p)
    if len(pts) < 3:
        return np.zeros((0, 3))
    pts = np.array(pts)
    pts = np.unique(pts.round(decimals=9), axis=0)
    if len(pts) < 3:
        return np.zeros((0, 3))
    center = pts.mean(axis=0)
    u_vec = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u_vec = u_vec - (u_vec @ n) * n
    u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-12)
    v_vec = np.cross(n, u_vec)
    v_vec = v_vec / (np.linalg.norm(v_vec) + 1e-12)
    c2 = np.array([(pts @ u_vec).mean(), (pts @ v_vec).mean()])
    angles = np.arctan2(pts @ v_vec - c2[1], pts @ u_vec - c2[0])
    order = np.argsort(angles)
    return pts[order]


def draw_P_row_planes(
    ax3d,
    P: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    """Draw the three planes defined by the rows of P (3x4) in 3D, clipped to the given box."""
    colors = ["red", "green", "blue"]
    for i in range(3):
        row = P[i, :]
        n, d = row[:3], row[3]
        verts = _plane_box_intersection_polygon(n, d, xlim, ylim, zlim)
        if len(verts) >= 3:
            ax3d.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=colors[i],
                    edgecolors=colors[i],
                    alpha=0.25,
                    linewidths=0.5,
                )
            )
