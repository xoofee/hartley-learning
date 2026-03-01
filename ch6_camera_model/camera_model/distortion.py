"""
Lens distortion (OpenCV-style: radial k1,k2,k3 + tangential p1,p2).

Single responsibility: apply/undistort pixel coordinates and polygon edges.
"""
from __future__ import annotations

import numpy as np

from .pinhole import project_one


def distortion_params_nonzero(k1: float, k2: float, k3: float, p1: float, p2: float) -> bool:
    """True if any distortion parameter is non-zero."""
    return (
        abs(k1) > 1e-12
        or abs(k2) > 1e-12
        or abs(k3) > 1e-12
        or abs(p1) > 1e-12
        or abs(p2) > 1e-12
    )


def apply_distortion(
    u: float | np.ndarray,
    v: float | np.ndarray,
    K: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply lens distortion to ideal pixel coords (u, v).
    OpenCV model: radial (k1,k2,k3) + tangential (p1,p2).
    (u, v) can be scalars or arrays; returns (u_dist, v_dist) same shape.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (np.asarray(u, dtype=np.float64) - cx) / fx
    y = (np.asarray(v, dtype=np.float64) - cy) / fy
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r4
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x1 = x * radial
    y1 = y * radial
    x2 = x1 + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y2 = y1 + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u_dist = x2 * fx + cx
    v_dist = y2 * fy + cy
    return u_dist, v_dist


def undistort_point(
    u_dist: float,
    v_dist: float,
    K: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    max_iter: int = 20,
) -> tuple[float, float]:
    """
    Undistort distorted pixel (u_dist, v_dist) to ideal (u, v).
    Iterative: start from (u_dist, v_dist), then correct so that apply_distortion(u,v) ≈ (u_dist, v_dist).
    """
    u, v = float(u_dist), float(v_dist)
    for _ in range(max_iter):
        u_d, v_d = apply_distortion(u, v, K, k1, k2, k3, p1, p2)
        du = u_dist - u_d
        dv = v_dist - v_d
        u += du
        v += dv
        if abs(du) < 1e-6 and abs(dv) < 1e-6:
            break
    return u, v


def distort_uv_if_needed(
    uv: np.ndarray,
    K: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> np.ndarray:
    """uv (N, 2) ideal pixel coords. If any of (k1,k2,k3,p1,p2) non-zero, apply distortion; else return uv."""
    if not distortion_params_nonzero(k1, k2, k3, p1, p2):
        return uv
    u, v = uv[:, 0], uv[:, 1]
    u_d, v_d = apply_distortion(u, v, K, k1, k2, k3, p1, p2)
    return np.column_stack([u_d, v_d])


def polygon_3d_edges_to_distorted(
    points_world: np.ndarray,
    P: np.ndarray,
    K: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    n_samples_per_edge: int = 32,
) -> np.ndarray:
    """
    Closed 3D polygon -> distorted polygon with curved edges and correct visibility.
    Samples each edge in 3D, clips to half-space in front of camera (w > 0), then projects and distorts.
    """
    n = len(points_world)
    out = []
    for i in range(n):
        j = (i + 1) % n
        A, B = points_world[i], points_world[j]
        u_a, v_a, w_a = project_one(P, A)
        u_b, v_b, w_b = project_one(P, B)
        if w_a <= 0 and w_b <= 0:
            continue
        if w_a > 0 and w_b > 0:
            t_lo, t_hi = 0.0, 1.0
        else:
            t_cross = w_a / (w_a - w_b) if (w_a - w_b) != 0 else 0.0
            t_cross = max(0.0, min(1.0, t_cross))
            if w_a > 0:
                t_lo, t_hi = 0.0, t_cross
            else:
                t_lo, t_hi = t_cross, 1.0
        n_pts = max(2, int(n_samples_per_edge * (t_hi - t_lo) + 0.5))
        for ki in range(n_pts):
            t = t_lo + (t_hi - t_lo) * (ki / (n_pts - 1)) if n_pts > 1 else t_lo
            Q = (1.0 - t) * A + t * B
            u_ideal, v_ideal, w = project_one(P, Q)
            if w <= 0:
                continue
            u_d, v_d = apply_distortion(u_ideal, v_ideal, K, k1, k2, k3, p1, p2)
            out.append([u_d, v_d])
    if len(out) < 3:
        return np.zeros((0, 2))
    return np.array(out)
