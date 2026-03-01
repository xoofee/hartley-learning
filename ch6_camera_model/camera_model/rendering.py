"""
Image-plane rendering: projected scene, vanishing points, world origin, rasterize.

Single responsibility: draw projected primitives and annotations on 2D image axes.
"""
from __future__ import annotations

import numpy as np

from . import pinhole
from . import distortion
from .distortion import polygon_3d_edges_to_distorted


def rasterize_scene_to_image(
    P: np.ndarray,
    square_pts: np.ndarray,
    triangle_pts: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Render scene to RGB image array (H, W, 3) by projecting and filling polygons."""
    img = np.ones((img_height, img_width, 3), dtype=np.float32)
    img[:, :, :] = 0.2

    def project(p: np.ndarray) -> np.ndarray:
        return pinhole.project_points(P, p.reshape(-1, 3))

    def clip_to_image(uv: np.ndarray) -> np.ndarray | None:
        u, v = uv[:, 0], uv[:, 1]
        if (
            np.any(u < -10)
            or np.any(u > img_width + 10)
            or np.any(v < -10)
            or np.any(v > img_height + 10)
        ):
            return None
        return np.column_stack([np.clip(u, 0, img_width - 1), np.clip(v, 0, img_height - 1)])

    def draw_filled_polygon(uv: np.ndarray, color: tuple) -> None:
        from matplotlib.path import Path
        u, v = uv[:, 0], uv[:, 1]
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return
        poly = Path(uv)
        for j in range(img_height):
            for i in range(img_width):
                if poly.contains_point([i, j]):
                    img[j, i, :] = color

    sq_uv = project(square_pts)
    tri_uv = project(triangle_pts)
    draw_filled_polygon(sq_uv, (0.2, 0.6, 0.2))
    draw_filled_polygon(tri_uv, (0.6, 0.2, 0.2))
    return img


def draw_projected_scene(
    ax,
    P: np.ndarray,
    square_pts: np.ndarray,
    triangle_pts: np.ndarray,
    rectangle_pts: np.ndarray,
    img_width: float,
    img_height: float,
    K: np.ndarray | None = None,
    dist: tuple[float, float, float, float, float] | None = None,
) -> None:
    """Draw projected square, triangle, and rectangle on axes.
    If K and dist are provided and non-zero, edges are sampled and distorted."""
    sq_uv = pinhole.project_points(P, square_pts)
    tri_uv = pinhole.project_points(P, triangle_pts)
    rect_uv = pinhole.project_points(P, rectangle_pts)
    if K is not None and dist is not None and distortion.distortion_params_nonzero(*dist):
        k1, k2, k3, p1, p2 = dist
        sq_uv = polygon_3d_edges_to_distorted(square_pts, P, K, k1, k2, k3, p1, p2)
        tri_uv = polygon_3d_edges_to_distorted(triangle_pts, P, K, k1, k2, k3, p1, p2)
        rect_uv = polygon_3d_edges_to_distorted(rectangle_pts, P, K, k1, k2, k3, p1, p2)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect("equal")
    from matplotlib.patches import Polygon
    if len(sq_uv) >= 3:
        ax.add_patch(Polygon(sq_uv, facecolor="green", edgecolor="darkgreen", alpha=0.8))
    if len(tri_uv) >= 3:
        ax.add_patch(Polygon(tri_uv, facecolor="red", edgecolor="darkred", alpha=0.8))
    if len(rect_uv) >= 3:
        ax.add_patch(Polygon(rect_uv, facecolor="blue", edgecolor="darkblue", alpha=0.8))


def vanishing_points_from_R_cw(
    K: np.ndarray, R_cw: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Vanishing points of world X, Y, Z axes. R_cw = R_world_to_cam; columns are world axes in camera frame.
    Returns (uv_X, uv_Y, uv_Z) each (2,) or None if not finite.
    """
    uv = []
    for i in range(3):
        d_cam = R_cw[:, i]
        x = K @ d_cam
        if abs(x[2]) <= 1e-6:
            uv.append(None)
            continue
        u, v = x[0] / x[2], x[1] / x[2]
        uv.append(np.array([u, v]))
    return uv[0], uv[1], uv[2]


def draw_vanishing_points(
    ax,
    K: np.ndarray,
    R_cw: np.ndarray,
    img_width: float,
    img_height: float,
    margin: float = 50.0,
    dist: tuple[float, float, float, float, float] | None = None,
) -> None:
    """Draw vanishing points for world X (R), Y (G), Z (B) on the image axes if finite and visible."""
    vp_x, vp_y, vp_z = vanishing_points_from_R_cw(K, R_cw)

    def in_bounds(u: float, v: float) -> bool:
        return (-margin <= u <= img_width + margin) and (-margin <= v <= img_height + margin)

    def maybe_distort(u: float, v: float):
        if dist is not None and distortion.distortion_params_nonzero(*dist):
            u, v = distortion.apply_distortion(u, v, K, *dist)
        return u, v

    for uv, color, label in [(vp_x, "red", "X"), (vp_y, "green", "Y"), (vp_z, "blue", "Z")]:
        if uv is None:
            continue
        u, v = maybe_distort(uv[0], uv[1])
        if not in_bounds(u, v):
            continue
        ax.scatter(u, v, c=color, s=80, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            label, (u, v), xytext=(5, 5), textcoords="offset points",
            color=color, fontsize=10, fontweight="bold",
        )


def world_origin_image_point(P: np.ndarray) -> np.ndarray | None:
    """Image (u, v) of world origin from P: P @ [0,0,0,1] = P[:, 3]. Returns (2,) uv or None if not finite."""
    p = P[:, 3]
    if abs(p[2]) <= 1e-6:
        return None
    u, v = p[0] / p[2], p[1] / p[2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    if p[2] <= 0:
        return None
    return np.array([u, v])


def draw_world_origin_on_image(
    ax,
    P: np.ndarray,
    img_width: float,
    img_height: float,
    margin: float = 50.0,
    K: np.ndarray | None = None,
    dist: tuple[float, float, float, float, float] | None = None,
) -> None:
    """Draw world origin image point in purple if finite and visible."""
    uv = world_origin_image_point(P)
    if uv is None:
        return
    u, v = uv[0], uv[1]
    if K is not None and dist is not None and distortion.distortion_params_nonzero(*dist):
        u, v = distortion.apply_distortion(u, v, K, *dist)
    if not ((-margin <= u <= img_width + margin) and (-margin <= v <= img_height + margin)):
        return
    ax.scatter(u, v, c="purple", s=80, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(
        "O", (u, v), xytext=(5, 5), textcoords="offset points",
        color="purple", fontsize=10, fontweight="bold",
    )
