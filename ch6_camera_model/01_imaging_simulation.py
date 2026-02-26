"""
A gui application to simulate a camera.

1 a scene with a horizontal square and a vertical triangle in the world coordinate system
2 a camera with a pinhole model
it have focal_length_mm, sensor (sensor_width_mm, sensor_height_mm), pixel pitch (pixel_size_x_mm, pixel_size_y_mm)
it have center C and orientation R
3 visualize the camera in the scene using a pyramid like shape. 
4 should two plots
left: 3d world coordinate system and the scene, and the camera
right: image
5 show the matrix P (3x4) A (3x3) R (3x3) t (3x1), P=A*[R|t]
6 let the user use mouse to move the camera in the right plot, like in 

Rotate / Orbit → Alt + Left Mouse Button drag
Pan / Track → Alt + Middle Mouse Button drag
(or sometimes Alt + Shift + LMB)
Zoom / Dolly → Alt + Right Mouse Button drag
(up = zoom in / down = zoom out)
OR Mouse wheel scroll (forward = zoom in, backward = zoom out)

dynamically update the image and the P R t

7 let user change the P or A R t and update the plots dynamically

note: we do not use any opengl or maya or blender or any 3d software to generate the image. we use pure python code to do it

coordinate (world: X right, Y forward, Z up, ENU-like):

    ↑ Z (up)
    |
    o --------→ X (right)
   /
  /
 Y (forward, into scene)


remove zoom pan rotate code
remove and forget about azimuth/elevation/roll
make the camera 6 dof free by letting user set the pitch yaw roll
still make the initial pose (pitch yaw roll) like the current (pitch 17 degrees looking to the (0,0,0)

the translation vector t should have spin to adjust

 ✅ Yaw (around Z axis, up)

Positive yaw = turn camera to the right

i.e. looking to the right side of the scene

✅ Pitch (around X axis, pointing right)

Positive pitch = look down

camera tilts downward

✅ Roll (around Y axis, forward)

Positive roll = clockwise rotation in the image

image rotates clockwise


let the user set camera center C in world coordinate. because t is not easy to understand or perceive

then make t only readable

# pitch

World is X right, Y forward, Z up. Camera default (zero angles) looks along -Y (into scene).
R_base aligns camera frame (optical axis, image Y down) with world; then pitch, yaw, roll are applied.
Pitch is relative to the world horizontal (xy) plane.

# vanishing point

the three column of R is the vanishing points of the three axes in the world coordinate system. show in the image plot if they are finite and visible in the image. use RGB colors for these three points corresponding to the X Y Z axes

# world origin
add a point in the 3d plot to represent the world origin (0,0,0). use purple color.
the last column of P is the world origin image point. show it in the image plot in purple if it is finite and visible in the image.

world origin relative to the camera center is -C and -RC is t which is the world origin in camera coordinate system

do not remove this comment.

"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QGroupBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

# ---------------------------------------------------------------------------
# Scene geometry (world coordinates)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pinhole camera model
# ---------------------------------------------------------------------------

import numpy as np

# convertion: R_wb: body to world, R_bw: world to body, R_cw: world to camera, R_wc: camera to world


def R_wc_from_yaw_pitch_roll_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build R_wb (3x3) which represents the orientation of the body in world coordinates.

    Note: the angles are eular angles describe the coordinate system rotation (describing the camera or aircraft orientation/pose), not vector rotation

    yaw pitch roll are never about vector rotation. they are about coordinate system rotation.

    camera axes:
        X: right
        Y: down
        Z: forward

    Rotations (right-hand rule, positive is CCW when looking along +axis):
        yaw   : about +Y
        pitch : about +X
        roll  : about +Z

    Composition (extrinsic rotations about world axes):
        R_wc = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    """

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Rotation about Z (yaw) coordinate, not vector, so the columns are the new x y z axis in the original world coordinate system
    Rz_roll = np.array([
        [ np.cos(roll), -np.sin(roll), 0.0],
        [ np.sin(roll),  np.cos(roll), 0.0],
        [ 0.0,          0.0,         1.0]
    ])

    # Rotation about X (pitch)
    Rx_pitch = np.array([
        [1.0, 0.0,           0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Rotation about Y (roll)
    Ry_yaw = np.array([
        [ np.cos(yaw), 0.0, np.sin(yaw)],
        [ 0.0,          1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)]
    ])

    R_wc = Ry_yaw @ Rx_pitch @ Rz_roll
    return R_wc

def R_wb_from_yaw_pitch_roll_world(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build R_wb (3x3) which represents the orientation of the body in world coordinates.

    Note: the angles are eular angles describe the coordinate system rotation (describing the camera or aircraft orientation/pose), not vector rotation

    yaw pitch roll are never about vector rotation. they are about coordinate system rotation.

    World axes:
        X: right
        Y: forward
        Z: up

    Rotations (right-hand rule, positive is CCW when looking along +axis):
        yaw   : about +Z
        pitch : about +X
        roll  : about +Y

    Composition (extrinsic rotations about world axes):
        R_wb = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    """

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Rotation about Z (yaw) coordinate, not vector, so the columns are the new x y z axis in the original world coordinate system
    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0.0],
        [ np.sin(yaw),  np.cos(yaw), 0.0],
        [ 0.0,          0.0,         1.0]
    ])

    # Rotation about X (pitch)
    Rx = np.array([
        [1.0, 0.0,           0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Rotation about Y (roll)
    Ry = np.array([
        [ np.cos(roll), 0.0, np.sin(roll)],
        [ 0.0,          1.0, 0.0],
        [-np.sin(roll), 0.0, np.cos(roll)]
    ])

    R_wb = Rz @ Rx @ Ry
    return R_wb

# Base rotation: camera default (yaw=pitch=roll=0) has optical axis along world -Y (forward into scene).
R_CW_BASE_CAM = R_wb_from_yaw_pitch_roll_world(0.0, -90.0, 0.0).T
# R_CW_BASE_CAM = np.eye(3)

def pitch_yaw_roll_from_R(R_world_to_cam: np.ndarray) -> tuple[float, float, float]:
    """
    Recover (yaw, pitch, roll) in degrees from R_world_to_cam.

    Uses the same convention as R_wc_from_yaw_pitch_roll_camera:
    camera axes X right, Y down, Z forward;
    R_wc = Ry(yaw) @ Rx(pitch) @ Rz(roll).
    We have R_bw = R_world_to_cam @ R_CW_BASE_CAM.T = R_wc.T, so R_wc = R_bw.T.
    Extract angles from R_wc (i.e. from R_bw).
    """
    # Remove base camera orientation -> body (camera) orientation world-to-body
    R_bw = R_world_to_cam @ R_CW_BASE_CAM.T
    # R_wc = R_bw.T = Ry(yaw) @ Rx(pitch) @ Rz(roll); extract from R_wc entries via R_bw[i,j] = R_wc[j,i]
    # pitch: R_wc[1,2] = -sin(pitch)
    sp = -R_bw[2, 1]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)

    if abs(cp) > 1e-6:
        # roll: R_wc[1,0] = cos(pitch)*sin(roll), R_wc[1,1] = cos(pitch)*cos(roll)
        roll = np.arctan2(R_bw[0, 1], R_bw[1, 1])
        # yaw: R_wc[0,2] = sin(yaw)*cos(pitch), R_wc[2,2] = cos(yaw)*cos(pitch)
        yaw = np.arctan2(R_bw[2, 0], R_bw[2, 2])
    else:
        # gimbal lock (pitch ~ ±90°): set roll=0, yaw from R_wc[0,0], R_wc[0,1]
        roll = 0.0
        yaw = np.arctan2(-R_bw[1, 0], R_bw[0, 0])

    return (
        np.rad2deg(pitch),
        np.rad2deg(yaw),
        np.rad2deg(roll),
    )

def build_intrinsic(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
    cx_px: float | None = None,
    cy_px: float | None = None,
) -> np.ndarray:
    """
    Build 3x3 intrinsic matrix K (camera matrix).
    Units: focal_length_mm (mm), sensor (mm), pixel pitch (mm/pixel).
    Focal length in pixels: fx_px = focal_length_mm / pixel_size_x_mm, etc.
    """
    image_width_px = sensor_width_mm / pixel_size_x_mm
    image_height_px = sensor_height_mm / pixel_size_y_mm
    fx_px = focal_length_mm / pixel_size_x_mm
    fy_px = focal_length_mm / pixel_size_y_mm
    cx = cx_px if cx_px is not None else image_width_px / 2.0
    cy = cy_px if cy_px is not None else image_height_px / 2.0
    return np.array([
        [fx_px, 0, cx],
        [0, fy_px, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def build_projection_matrix(
    K: np.ndarray,
    R_cw: np.ndarray,
    camera_center_world: np.ndarray,
) -> np.ndarray:
    """P = K [R_world_to_cam | t_world_to_cam]. R_wc = R_cam^T, t_wc = -R_wc @ camera_center_world. P is (3, 4)."""
    t_world_to_cam = (-R_cw @ camera_center_world).reshape(3, 1)
    Rt = np.hstack([R_cw, t_world_to_cam])
    return K @ Rt

def decompose_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose P (3,4) into K (3,3), R_world_to_cam (3,3), t_world_to_cam (3,1) with P = K [R_wc|t_wc]. Uses RQ on M = P[:,:3]."""
    from scipy.linalg import rq
    M = P[:, :3].copy()
    p = P[:, 3:4]
    # RQ: M = K_intrinsic @ R_world_to_cam
    K_intrinsic, R_world_to_cam = rq(M)
    D = np.diag(np.sign(np.diag(K_intrinsic)))
    D[np.diag(D) == 0] = 1
    K_intrinsic = K_intrinsic @ D
    R_world_to_cam = D.T @ R_world_to_cam
    if np.linalg.det(R_world_to_cam) < 0:
        R_world_to_cam = -R_world_to_cam
        K_intrinsic = -K_intrinsic
    t_world_to_cam = np.linalg.solve(K_intrinsic, p)
    return K_intrinsic, R_world_to_cam, t_world_to_cam


def project_points(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """points_world (N, 3) -> (N, 2) image coords (u, v)."""
    n = points_world.shape[0]
    ones = np.ones((n, 1))
    hom = np.hstack([points_world, ones])
    x = (P @ hom.T).T
    w = x[:, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    u = x[:, 0] / w
    v = x[:, 1] / w
    return np.column_stack([u, v])


def get_camera_pyramid(
    C: np.ndarray,
    R_cam: np.ndarray,
    scale: float = 0.5,
    depth: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_corners (4,3), apex C). Base is in front of camera at depth."""
    # In camera frame: base at z = depth, half-size scale
    b1 = np.array([-scale, -scale, depth])
    b2 = np.array([scale, -scale, depth])
    b3 = np.array([scale, scale, depth])
    b4 = np.array([-scale, scale, depth])
    pyramid_points_cam = np.array([b1, b2, b3, b4])

    R_wc = R_cam.T
    pyramid_points_cam_world = (R_wc @ pyramid_points_cam.T).T + C
    return pyramid_points_cam_world, C


# ---------------------------------------------------------------------------
# Image rendering (pure Python: project and draw in 2D)
# ---------------------------------------------------------------------------

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
        xy = project_points(P, p.reshape(-1, 3))
        return xy

    def clip_to_image(uv: np.ndarray) -> np.ndarray | None:
        u, v = uv[:, 0], uv[:, 1]
        if np.any(u < -10) or np.any(u > img_width + 10) or np.any(v < -10) or np.any(v > img_height + 10):
            return None
        return np.column_stack([np.clip(u, 0, img_width - 1), np.clip(v, 0, img_height - 1)])

    def draw_filled_polygon(uv: np.ndarray, color: tuple) -> None:
        from matplotlib.path import Path
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        u, v = uv[:, 0], uv[:, 1]
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return
        poly = Path(uv)
        for j in range(img_height):
            for i in range(img_width):
                if poly.contains_point([i, j]):
                    img[j, i, :] = color
        return

    # Use scan-line style: get projected polygons and fill via matplotlib Path
    sq_uv = project(square_pts)
    tri_uv = project(triangle_pts)
    # Image y is top-down; we use (u, v) with v = row (matplotlib convention: origin top-left for image)
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
) -> None:
    """Draw projected square, triangle, and rectangle on axes (for display). Uses projected vertices and fill."""
    sq_uv = project_points(P, square_pts)
    tri_uv = project_points(P, triangle_pts)
    rect_uv = project_points(P, rectangle_pts)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect("equal")
    
    # Draw filled polygons
    from matplotlib.patches import Polygon
    poly_sq = Polygon(sq_uv, facecolor="green", edgecolor="darkgreen", alpha=0.8)
    poly_tri = Polygon(tri_uv, facecolor="red", edgecolor="darkred", alpha=0.8)
    poly_rect = Polygon(rect_uv, facecolor="blue", edgecolor="darkblue", alpha=0.8)
    ax.add_patch(poly_sq)
    ax.add_patch(poly_tri)
    ax.add_patch(poly_rect)


def vanishing_points_from_R_cw(K: np.ndarray, R_cw: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Vanishing points of world X, Y, Z axes. R_cw = R_world_to_cam; columns are world axes in camera frame.
    Returns (uv_X, uv_Y, uv_Z) each (2,) or None if not finite (direction parallel to image plane or behind).
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
) -> None:
    """
    Draw vanishing points for world X (R), Y (G), Z (B) on the image axes if finite and visible.
    """
    vp_x, vp_y, vp_z = vanishing_points_from_R_cw(K, R_cw)
    def in_bounds(u: float, v: float) -> bool:
        return (-margin <= u <= img_width + margin) and (-margin <= v <= img_height + margin)
    for uv, color, label in [(vp_x, "red", "X"), (vp_y, "green", "Y"), (vp_z, "blue", "Z")]:
        if uv is None:
            continue
        u, v = uv[0], uv[1]
        if not in_bounds(u, v):
            continue
        ax.scatter(u, v, c=color, s=80, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(label, (u, v), xytext=(5, 5), textcoords="offset points", color=color, fontsize=10, fontweight="bold")


def world_origin_image_point(P: np.ndarray) -> np.ndarray | None:
    """
    Image (u, v) of world origin from P: P @ [0,0,0,1] = P[:, 3].
    Returns (2,) uv or None if not finite (e.g. behind camera).
    """
    p = P[:, 3]
    if abs(p[2]) <= 1e-6:
        return None
    u, v = p[0] / p[2], p[1] / p[2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    # In front of camera
    if p[2] <= 0:
        return None
    return np.array([u, v])


def draw_world_origin_on_image(
    ax,
    P: np.ndarray,
    img_width: float,
    img_height: float,
    margin: float = 50.0,
) -> None:
    """Draw world origin image point in purple if finite and visible."""
    uv = world_origin_image_point(P)
    if uv is None:
        return
    u, v = uv[0], uv[1]
    if not ((-margin <= u <= img_width + margin) and (-margin <= v <= img_height + margin)):
        return
    ax.scatter(u, v, c="purple", s=80, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate("O", (u, v), xytext=(5, 5), textcoords="offset points", color="purple", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# GUI state and interaction
# ---------------------------------------------------------------------------

class CameraState:
    def __init__(
        self,
        focal_length_mm: float = 50.0,
        sensor_width_mm: float = 36.0,
        sensor_height_mm: float = 24.0,
        pixel_size_x_mm: float = 0.01,
        pixel_size_y_mm: float = 0.01,
    ):
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.pixel_size_x_mm = pixel_size_x_mm
        self.pixel_size_y_mm = pixel_size_y_mm
        self.cx_px_override: float | None = None
        self.cy_px_override: float | None = None
        # 6-DOF pose: pitch/yaw/roll (degrees), camera center C in world (t = -R @ C)
        self.pitch_deg = -30.0
        self.yaw_deg = -90.0
        self.roll_deg = 0.0
        self.C_x = 4.0
        self.C_y = 0.0
        self.C_z = 2.0

    def get_R_cw(self) -> np.ndarray:
        return R_wc_from_yaw_pitch_roll_camera(self.yaw_deg, self.pitch_deg, self.roll_deg).T @ R_CW_BASE_CAM

    def get_camera_center_world(self) -> np.ndarray:
        return np.array([self.C_x, self.C_y, self.C_z])

    def get_K(self) -> np.ndarray:
        return build_intrinsic(
            self.focal_length_mm,
            self.sensor_width_mm,
            self.sensor_height_mm,
            self.pixel_size_x_mm,
            self.pixel_size_y_mm,
            cx_px=self.cx_px_override,
            cy_px=self.cy_px_override,
        )

    def get_P(self) -> np.ndarray:
        camera_center_world = self.get_camera_center_world()
        R_cam = self.get_R_cw()
        K = self.get_K()
        return build_projection_matrix(K, R_cam, camera_center_world)

    def get_R_and_t(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R_world_to_cam, t_world_to_cam). t = -R @ C."""
        R_cw = self.get_R_cw()
        C = np.array([[self.C_x], [self.C_y], [self.C_z]])
        t_world_to_cam = -R_cw @ C
        return R_cw, t_world_to_cam

    def set_from_K(self, K: np.ndarray) -> None:
        """Update focal_length_mm and principal point (px) from intrinsic matrix K. K[0,0]=fx_px, K[1,1]=fy_px."""
        self.focal_length_mm = float(
            0.5 * (K[0, 0] * self.pixel_size_x_mm + K[1, 1] * self.pixel_size_y_mm)
        )
        self.cx_px_override = float(K[0, 2])
        self.cy_px_override = float(K[1, 2])

    def set_from_C(self, C_world: np.ndarray) -> None:
        """Set camera center in world from 3-vector."""
        C = np.asarray(C_world).ravel()[:3]
        self.C_x, self.C_y, self.C_z = float(C[0]), float(C[1]), float(C[2])

    def set_from_P(self, P: np.ndarray) -> None:
        """Update state from full P: decompose to K, R, t; set pitch/yaw/roll and C (from t)."""
        K_intrinsic, R_world_to_cam, t_world_to_cam = decompose_P(P)
        self.set_from_K(K_intrinsic)
        self.pitch_deg, self.yaw_deg, self.roll_deg = pitch_yaw_roll_from_R(R_world_to_cam)
        C = (-R_world_to_cam.T @ t_world_to_cam).ravel()[:3]
        self.C_x, self.C_y, self.C_z = float(C[0]), float(C[1]), float(C[2])


# ---------------------------------------------------------------------------
# Qt widgets for matrix display and camera parameters (like 05_homography_research_gui)
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.4f}"


READONLY_BG = "background-color: #e0e0e0;"


class MatrixDisplayWidget(QWidget):
    """Read-only grid with gray background so user can see at a glance it is not editable."""

    def __init__(self, title: str, nrows: int, ncols: int):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.edits: list[list[QLineEdit]] = []
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout()
        grid = QGridLayout()
        for i in range(nrows):
            row = []
            for j in range(ncols):
                edit = QLineEdit()
                edit.setReadOnly(True)
                edit.setStyleSheet(READONLY_BG)
                edit.setMaximumWidth(72)
                edit.setAlignment(Qt.AlignRight)
                grid.addWidget(edit, i, j)
                row.append(edit)
            self.edits.append(row)
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def set_matrix(self, M: np.ndarray) -> None:
        for i in range(min(self.nrows, M.shape[0])):
            for j in range(min(self.ncols, M.shape[1])):
                self.edits[i][j].setText(_fmt(float(M[i, j])))


class MatrixEditWidget(QWidget):
    """Editable grid of QLineEdit; emits matrix when changed (for P, A, t)."""

    def __init__(self, title: str, nrows: int, ncols: int):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.edits: list[list[QLineEdit]] = []
        self.updating = False
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout()
        grid = QGridLayout()
        for i in range(nrows):
            row = []
            for j in range(ncols):
                edit = QLineEdit()
                edit.setMaximumWidth(72)
                edit.setAlignment(Qt.AlignRight)
                edit.editingFinished.connect(self._on_edit)
                grid.addWidget(edit, i, j)
                row.append(edit)
            self.edits.append(row)
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def _on_edit(self) -> None:
        if self.updating:
            return
        try:
            M = self.get_matrix()
            if M is not None:
                self.matrix_changed.emit(M)
        except (ValueError, TypeError):
            pass

    def get_matrix(self) -> np.ndarray | None:
        try:
            M = np.zeros((self.nrows, self.ncols))
            for i in range(self.nrows):
                for j in range(self.ncols):
                    M[i, j] = float(self.edits[i][j].text())
            return M
        except (ValueError, TypeError):
            return None

    def set_matrix(self, M: np.ndarray) -> None:
        self.updating = True
        for i in range(min(self.nrows, M.shape[0])):
            for j in range(min(self.ncols, M.shape[1])):
                self.edits[i][j].setText(_fmt(float(M[i, j])))
        self.updating = False

    matrix_changed = pyqtSignal(object)


class CameraParamsWidget(QWidget):
    """Editable intrinsics only (focal_length_mm, sensor_*, pixel_*)."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("Camera parameters")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout()
        grid = QGridLayout()
        row = 0
        grid.addWidget(QLabel("focal_length_mm:"), row, 0)
        self.spin_f = QDoubleSpinBox()
        self.spin_f.setRange(1.0, 5000.0)
        self.spin_f.setSuffix(" mm")
        self.spin_f.setValue(state.focal_length_mm)
        self.spin_f.setMaximumWidth(90)
        grid.addWidget(self.spin_f, row, 1)
        row += 1
        grid.addWidget(QLabel("sensor_width_mm:"), row, 0)
        self.spin_wphys = QDoubleSpinBox()
        self.spin_wphys.setRange(0.1, 200.0)
        self.spin_wphys.setDecimals(2)
        self.spin_wphys.setSuffix(" mm")
        self.spin_wphys.setValue(state.sensor_width_mm)
        self.spin_wphys.setMaximumWidth(90)
        grid.addWidget(self.spin_wphys, row, 1)
        row += 1
        grid.addWidget(QLabel("sensor_height_mm:"), row, 0)
        self.spin_hphys = QDoubleSpinBox()
        self.spin_hphys.setRange(0.1, 200.0)
        self.spin_hphys.setDecimals(2)
        self.spin_hphys.setSuffix(" mm")
        self.spin_hphys.setValue(state.sensor_height_mm)
        self.spin_hphys.setMaximumWidth(90)
        grid.addWidget(self.spin_hphys, row, 1)
        row += 1
        grid.addWidget(QLabel("pixel_size_x_mm:"), row, 0)
        self.spin_wpix = QDoubleSpinBox()
        self.spin_wpix.setRange(0.0001, 1.0)
        self.spin_wpix.setDecimals(4)
        self.spin_wpix.setSuffix(" mm")
        self.spin_wpix.setValue(state.pixel_size_x_mm)
        self.spin_wpix.setMaximumWidth(90)
        grid.addWidget(self.spin_wpix, row, 1)
        row += 1
        grid.addWidget(QLabel("pixel_size_y_mm:"), row, 0)
        self.spin_hpix = QDoubleSpinBox()
        self.spin_hpix.setRange(0.0001, 1.0)
        self.spin_hpix.setDecimals(4)
        self.spin_hpix.setSuffix(" mm")
        self.spin_hpix.setValue(state.pixel_size_y_mm)
        self.spin_hpix.setMaximumWidth(90)
        grid.addWidget(self.spin_hpix, row, 1)
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.focal_length_mm = self.spin_f.value()
        self.state.sensor_width_mm = self.spin_wphys.value()
        self.state.sensor_height_mm = self.spin_hphys.value()
        self.state.pixel_size_x_mm = self.spin_wpix.value()
        self.state.pixel_size_y_mm = self.spin_hpix.value()

    def sync_from_state(self) -> None:
        for spin in (self.spin_f, self.spin_wphys, self.spin_hphys, self.spin_wpix, self.spin_hpix):
            spin.blockSignals(True)
        self.spin_f.setValue(self.state.focal_length_mm)
        self.spin_wphys.setValue(self.state.sensor_width_mm)
        self.spin_hphys.setValue(self.state.sensor_height_mm)
        self.spin_wpix.setValue(self.state.pixel_size_x_mm)
        self.spin_hpix.setValue(self.state.pixel_size_y_mm)
        for spin in (self.spin_f, self.spin_wphys, self.spin_hphys, self.spin_wpix, self.spin_hpix):
            spin.blockSignals(False)


class RotationParamsWidget(QWidget):
    """Pitch, yaw, roll (°): one row per label+spin, spins in a column."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("Pitch / Yaw / Roll")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        grid = QGridLayout()
        grid.addWidget(QLabel("pitch:"), 0, 0)
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(-90.0, 90.0)
        self.spin_pitch.setDecimals(1)
        self.spin_pitch.setSuffix(" °")
        self.spin_pitch.setValue(state.pitch_deg)
        self.spin_pitch.setMaximumWidth(72)
        grid.addWidget(self.spin_pitch, 0, 1)
        grid.addWidget(QLabel("yaw:"), 1, 0)
        self.spin_yaw = QDoubleSpinBox()
        self.spin_yaw.setRange(-360.0, 360.0)
        self.spin_yaw.setDecimals(1)
        self.spin_yaw.setSuffix(" °")
        self.spin_yaw.setValue(state.yaw_deg)
        self.spin_yaw.setMaximumWidth(72)
        grid.addWidget(self.spin_yaw, 1, 1)
        grid.addWidget(QLabel("roll:"), 2, 0)
        self.spin_roll = QDoubleSpinBox()
        self.spin_roll.setRange(-180.0, 180.0)
        self.spin_roll.setDecimals(1)
        self.spin_roll.setSuffix(" °")
        self.spin_roll.setValue(state.roll_deg)
        self.spin_roll.setMaximumWidth(72)
        grid.addWidget(self.spin_roll, 2, 1)
        group.setLayout(grid)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.pitch_deg = self.spin_pitch.value()
        self.state.yaw_deg = self.spin_yaw.value()
        self.state.roll_deg = self.spin_roll.value()

    def sync_from_state(self) -> None:
        for spin in (self.spin_pitch, self.spin_yaw, self.spin_roll):
            spin.blockSignals(True)
        self.spin_pitch.setValue(self.state.pitch_deg)
        self.spin_yaw.setValue(self.state.yaw_deg)
        self.spin_roll.setValue(self.state.roll_deg)
        for spin in (self.spin_pitch, self.spin_yaw, self.spin_roll):
            spin.blockSignals(False)


class CameraCenterWidget(QWidget):
    """Camera center C in world: C_x, C_y, C_z as spinboxes for click adjustment."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("C (world)")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        grid = QGridLayout()
        grid.addWidget(QLabel("C_x:"), 0, 0)
        self.spin_Cx = QDoubleSpinBox()
        self.spin_Cx.setRange(-100.0, 100.0)
        self.spin_Cx.setDecimals(3)
        self.spin_Cx.setSingleStep(0.1)
        self.spin_Cx.setValue(state.C_x)
        self.spin_Cx.setMaximumWidth(72)
        grid.addWidget(self.spin_Cx, 0, 1)
        grid.addWidget(QLabel("C_y:"), 1, 0)
        self.spin_Cy = QDoubleSpinBox()
        self.spin_Cy.setRange(-100.0, 100.0)
        self.spin_Cy.setDecimals(3)
        self.spin_Cy.setSingleStep(0.1)
        self.spin_Cy.setValue(state.C_y)
        self.spin_Cy.setMaximumWidth(72)
        grid.addWidget(self.spin_Cy, 1, 1)
        grid.addWidget(QLabel("C_z:"), 2, 0)
        self.spin_Cz = QDoubleSpinBox()
        self.spin_Cz.setRange(-100.0, 100.0)
        self.spin_Cz.setDecimals(3)
        self.spin_Cz.setSingleStep(0.1)
        self.spin_Cz.setValue(state.C_z)
        self.spin_Cz.setMaximumWidth(72)
        grid.addWidget(self.spin_Cz, 2, 1)
        group.setLayout(grid)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.C_x = self.spin_Cx.value()
        self.state.C_y = self.spin_Cy.value()
        self.state.C_z = self.spin_Cz.value()

    def sync_from_state(self) -> None:
        for spin in (self.spin_Cx, self.spin_Cy, self.spin_Cz):
            spin.blockSignals(True)
        self.spin_Cx.setValue(self.state.C_x)
        self.spin_Cy.setValue(self.state.C_y)
        self.spin_Cz.setValue(self.state.C_z)
        for spin in (self.spin_Cx, self.spin_Cy, self.spin_Cz):
            spin.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.square_pts = get_scene_square(1.0, 0.0)
        self.triangle_pts = get_scene_triangle(0.8, 1.0, 1.5)
        self.rectangle_pts = get_scene_rectangle(0.4, 0.4, 1.5, y_center=0.8, z_center=0.4)
        self.state = CameraState()
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Camera simulation (P = K [R|t])")
        self.setGeometry(80, 80, 1400, 700)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        # Left: matplotlib figure with 3D and image (stretch to use space)
        self.fig = plt.figure(figsize=(8, 5))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_img = self.fig.add_subplot(122)
        self._set_3d_axes_limits_once()
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 1)

        # Right: scroll area with minimal width so plot expands
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)
        scroll_widget = QWidget()
        right_layout = QVBoxLayout()
        self.params_widget = CameraParamsWidget(self.state)
        for spin in (
            self.params_widget.spin_f,
            self.params_widget.spin_wphys,
            self.params_widget.spin_hphys,
            self.params_widget.spin_wpix,
            self.params_widget.spin_hpix,
        ):
            spin.valueChanged.connect(self._on_params_changed)
        right_layout.addWidget(self.params_widget)
        self.edit_P = MatrixEditWidget("P (3×4) editable", 3, 4)
        self.edit_P.matrix_changed.connect(self._on_P_changed)
        right_layout.addWidget(self.edit_P)
        self.edit_K = MatrixEditWidget("K intrinsic (3×3) editable", 3, 3)
        self.edit_K.matrix_changed.connect(self._on_K_changed)
        right_layout.addWidget(self.edit_K)
        self.display_R = MatrixDisplayWidget("R", 3, 3)
        right_layout.addWidget(self.display_R)
        pose_row = QWidget()
        pose_row_layout = QHBoxLayout()
        pose_row_layout.setContentsMargins(0, 0, 0, 0)
        self.rotation_widget = RotationParamsWidget(self.state)
        for spin in (
            self.rotation_widget.spin_pitch,
            self.rotation_widget.spin_yaw,
            self.rotation_widget.spin_roll,
        ):
            spin.valueChanged.connect(self._on_pose_changed)
        pose_row_layout.addWidget(self.rotation_widget)
        self.C_widget = CameraCenterWidget(self.state)
        for spin in (self.C_widget.spin_Cx, self.C_widget.spin_Cy, self.C_widget.spin_Cz):
            spin.valueChanged.connect(self._on_pose_changed)
        pose_row_layout.addWidget(self.C_widget)
        self.display_t = MatrixDisplayWidget("t = -R@C", 3, 1)
        pose_row_layout.addWidget(self.display_t)
        pose_row_layout.addStretch()
        pose_row.setLayout(pose_row_layout)
        right_layout.addWidget(pose_row)
        right_layout.addStretch()
        scroll_widget.setLayout(right_layout)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 0)
        central.setLayout(main_layout)

        self._update_matrix_displays()
        self._draw_all()

    def _set_3d_axes_limits_once(self) -> None:
        """Set 3D scene axis limits and equal box aspect once at init. User zoom/pan changes limits; we preserve them on redraw."""
        margin = 6.0
        self.ax3d.set_xlim(-margin, margin)
        self.ax3d.set_ylim(-margin, margin)
        self.ax3d.set_zlim(-0.5, margin)
        # Equal axis scale: box aspect = (x_range, y_range, z_range)
        # self.ax3d.set_box_aspect((2 * margin, 2 * margin, 2 * margin))

        xrange = margin * 2
        yrange = margin * 2
        zrange = margin + 0.5
        # self.ax3d.set_box_aspect((xrange, yrange, zrange))
        self.ax3d.set_box_aspect((1, 1, 2))

    def _on_params_changed(self) -> None:
        self.params_widget.apply_to_state()
        self._update_matrix_displays()
        self._draw_all()

    def _on_pose_changed(self) -> None:
        self.rotation_widget.apply_to_state()
        self.C_widget.apply_to_state()
        self._update_matrix_displays()
        self._draw_all()

    def _on_P_changed(self, P: np.ndarray) -> None:
        try:
            P = np.asarray(P).reshape(3, 4)
            if not np.allclose(P, self.state.get_P(), rtol=1e-9, atol=1e-12):
                self.state.set_from_P(P)
                self.params_widget.sync_from_state()
                self.rotation_widget.sync_from_state()
                self.C_widget.sync_from_state()
                self._update_matrix_displays()
                self._draw_all()
        except Exception:
            pass

    def _on_K_changed(self, K: np.ndarray) -> None:
        try:
            K = np.asarray(K).reshape(3, 3)
            if not np.allclose(K, self.state.get_K(), rtol=1e-9, atol=1e-12):
                self.state.set_from_K(K)
                self.params_widget.sync_from_state()
                self.rotation_widget.sync_from_state()
                self.C_widget.sync_from_state()
                self._update_matrix_displays()
                self._draw_all()
        except Exception:
            pass

    def _update_matrix_displays(self) -> None:
        P = self.state.get_P()
        K = self.state.get_K()
        R_world_to_cam, t_world_to_cam = self.state.get_R_and_t()
        self.edit_P.set_matrix(P)
        self.edit_K.set_matrix(K)
        self.display_R.set_matrix(R_world_to_cam)
        self.C_widget.sync_from_state()
        self.display_t.set_matrix(t_world_to_cam)

    def _draw_all(self) -> None:
        self.image_width_px = max(1, int(self.state.sensor_width_mm / self.state.pixel_size_x_mm))
        self.image_height_px = max(1, int(self.state.sensor_height_mm / self.state.pixel_size_y_mm))
        # Preserve 3D view limits (user may have zoomed/panned); do not change them when editing params
        xlim = self.ax3d.get_xlim()
        ylim = self.ax3d.get_ylim()
        zlim = self.ax3d.get_zlim()
        self.ax3d.cla()
        self.ax_img.cla()
        camera_center_world = self.state.get_camera_center_world()
        R_cam = self.state.get_R_cw()
        P = self.state.get_P()
        base_world, apex = get_camera_pyramid(camera_center_world, R_cam, scale=0.5, depth=0.3)
        verts_pyramid = [
            [apex, base_world[0], base_world[1]],
            [apex, base_world[1], base_world[2]],
            [apex, base_world[2], base_world[3]],
            [apex, base_world[3], base_world[0]],
            [base_world[0], base_world[1], base_world[2], base_world[3]],
        ]
        self.ax3d.add_collection3d(
            Poly3DCollection(verts_pyramid, facecolors="cyan", edgecolors="blue", alpha=0.4)
        )
        self.ax3d.scatter(
            self.square_pts[:, 0], self.square_pts[:, 1], self.square_pts[:, 2], c="green", s=20
        )
        self.ax3d.scatter(
            self.triangle_pts[:, 0], self.triangle_pts[:, 1], self.triangle_pts[:, 2], c="red", s=20
        )
        self.ax3d.scatter(
            self.rectangle_pts[:, 0], self.rectangle_pts[:, 1], self.rectangle_pts[:, 2], c="blue", s=20
        )
        # World origin
        self.ax3d.scatter([0], [0], [0], c="purple", s=80, zorder=5, edgecolors="white", linewidths=1.5)
        sq_edges = np.vstack([self.square_pts, self.square_pts[0:1]])
        tri_edges = np.vstack([self.triangle_pts, self.triangle_pts[0:1]])
        rect_edges = np.vstack([self.rectangle_pts, self.rectangle_pts[0:1]])
        self.ax3d.plot(sq_edges[:, 0], sq_edges[:, 1], sq_edges[:, 2], "g-", lw=2)
        self.ax3d.plot(tri_edges[:, 0], tri_edges[:, 1], tri_edges[:, 2], "r-", lw=2)
        self.ax3d.plot(rect_edges[:, 0], rect_edges[:, 1], rect_edges[:, 2], "b-", lw=2)
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("3D scene and camera")
        # Restore 3D limits and keep axis equal (only user zoom/pan changes limits)
        self.ax3d.set_xlim(xlim)
        self.ax3d.set_ylim(ylim)
        self.ax3d.set_zlim(zlim)
        rx, ry, rz = xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]
        self.ax3d.set_box_aspect((rx, ry, rz))
        draw_projected_scene(
            self.ax_img, P, self.square_pts, self.triangle_pts, self.rectangle_pts,
            self.image_width_px, self.image_height_px
        )
        K = self.state.get_K()
        draw_vanishing_points(
            self.ax_img, K, R_cam, self.image_width_px, self.image_height_px
        )
        draw_world_origin_on_image(
            self.ax_img, P, self.image_width_px, self.image_height_px
        )
        self.ax_img.set_title("Image")
        self.ax_img.set_xlabel("u (pixels)")
        self.ax_img.set_ylabel("v (pixels)")
        self._update_matrix_displays()
        self.canvas.draw_idle()


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
