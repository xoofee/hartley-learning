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


# ---------------------------------------------------------------------------
# Pinhole camera model
# ---------------------------------------------------------------------------

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


def camera_pose_from_target(
    camera_center_world: np.ndarray,
    target: np.ndarray,
    roll: float = 0.0,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """R_cam 3x3: columns are camera x, y, z axes in world. z points from camera toward target. roll (rad) rotates around view axis."""
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    d = target - camera_center_world
    d = d / (np.linalg.norm(d) + 1e-12)
    cam_z = d
    right0 = np.cross(world_up, cam_z)
    right0 = right0 / (np.linalg.norm(right0) + 1e-12)
    up0 = np.cross(cam_z, right0)
    up0 = up0 / (np.linalg.norm(up0) + 1e-12)
    cr, sr = np.cos(roll), np.sin(roll)
    right = cr * right0 - sr * up0
    up = sr * right0 + cr * up0
    R_cam = np.column_stack([right, up, cam_z])
    return R_cam


def build_projection_matrix(
    K: np.ndarray,
    R_cam: np.ndarray,
    camera_center_world: np.ndarray,
) -> np.ndarray:
    """P = K [R_world_to_cam | t_world_to_cam]. R_wc = R_cam^T, t_wc = -R_wc @ camera_center_world. P is (3, 4)."""
    R_world_to_cam = R_cam.T
    t_world_to_cam = (-R_world_to_cam @ camera_center_world).reshape(3, 1)
    Rt = np.hstack([R_world_to_cam, t_world_to_cam])
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


def rotation_to_angles(R_world_to_cam: np.ndarray) -> tuple[float, float, float]:
    """Extract azimuth, elevation, roll (radians) from R_world_to_cam. Camera axes in world = R_world_to_cam.T."""
    R_cam = R_world_to_cam.T
    view = R_cam[:, 2]  # view direction in world
    az = np.arctan2(view[1], view[0])
    el = np.arcsin(np.clip(view[2], -1.0, 1.0))
    world_up = np.array([0.0, 0.0, 1.0])
    right0 = np.cross(world_up, view)
    n = np.linalg.norm(right0)
    if n < 1e-10:
        roll = 0.0
    else:
        right0 = right0 / n
        up0 = np.cross(view, right0)
        right = R_cam[:, 0]
        roll = np.arctan2(np.dot(up0, right), np.dot(right0, right))
    return az, el, roll


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
    base_cam = np.array([b1, b2, b3, b4])
    base_world = (R_cam @ base_cam.T).T + C
    return base_world, C


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
    img_width: float,
    img_height: float,
) -> None:
    """Draw projected square and triangle on axes (for display). Uses projected vertices and fill."""
    sq_uv = project_points(P, square_pts)
    tri_uv = project_points(P, triangle_pts)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect("equal")
    # Draw filled polygons
    from matplotlib.patches import Polygon
    poly_sq = Polygon(sq_uv, facecolor="green", edgecolor="darkgreen", alpha=0.8)
    poly_tri = Polygon(tri_uv, facecolor="red", edgecolor="darkred", alpha=0.8)
    ax.add_patch(poly_sq)
    ax.add_patch(poly_tri)


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
        self.target = np.array([0.0, 0.0, 0.0])
        self.distance = 4.0
        self.azimuth = 0.0
        self.elevation = 0.3
        self.roll = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def get_camera_center_world(self) -> np.ndarray:
        r = self.distance
        el = self.elevation
        az = self.azimuth
        cx = self.target[0] + self.pan_x + r * np.cos(el) * np.cos(az)
        cy = self.target[1] + self.pan_y + r * np.cos(el) * np.sin(az)
        cz = self.target[2] + r * np.sin(el)
        return np.array([cx, cy, cz])

    def get_R_cam(self) -> np.ndarray:
        return camera_pose_from_target(
            self.get_camera_center_world(),
            self.target + np.array([self.pan_x, self.pan_y, 0]),
            roll=self.roll,
        )

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
        R_cam = self.get_R_cam()
        K = self.get_K()
        return build_projection_matrix(K, R_cam, camera_center_world)

    def get_R_and_t(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R_world_to_cam, t_world_to_cam)."""
        R_cam = self.get_R_cam()
        camera_center_world = self.get_camera_center_world()
        R_world_to_cam = R_cam.T
        t_world_to_cam = (-R_world_to_cam @ camera_center_world).reshape(3, 1)
        return R_world_to_cam, t_world_to_cam

    def orbit(self, d_az: float, d_el: float) -> None:
        self.azimuth += d_az
        self.elevation = np.clip(self.elevation + d_el, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)

    def pan(self, dx: float, dy: float) -> None:
        R_cam = self.get_R_cam()
        self.pan_x += dx * R_cam[0, 0] + dy * R_cam[0, 1]
        self.pan_y += dx * R_cam[1, 0] + dy * R_cam[1, 1]

    def zoom(self, delta: float) -> None:
        self.distance = np.clip(self.distance * (1.0 - delta), 0.5, 20.0)

    def set_from_K(self, K: np.ndarray) -> None:
        """Update focal_length_mm and principal point (px) from intrinsic matrix K. K[0,0]=fx_px, K[1,1]=fy_px."""
        self.focal_length_mm = float(
            0.5 * (K[0, 0] * self.pixel_size_x_mm + K[1, 1] * self.pixel_size_y_mm)
        )
        self.cx_px_override = float(K[0, 2])
        self.cy_px_override = float(K[1, 2])

    def set_from_t(self, t_world_to_cam: np.ndarray) -> None:
        """Update camera position (distance, azimuth, elevation) from t_world_to_cam; R from current angles."""
        R_world_to_cam = self.get_R_cam().T
        camera_center_world = -R_world_to_cam.T @ t_world_to_cam.ravel()
        C = camera_center_world
        eff_target = self.target + np.array([self.pan_x, self.pan_y, 0])
        d = eff_target - C
        self.distance = float(np.linalg.norm(d) + 1e-12)
        if self.distance > 1e-12:
            view = d / self.distance
            self.azimuth = float(np.arctan2(view[1], view[0]))
            self.elevation = float(np.arcsin(np.clip(view[2], -1.0, 1.0)))

    def set_from_P(self, P: np.ndarray) -> None:
        """Update state from full P: decompose to K, R_world_to_cam, t_world_to_cam; set intrinsics and pose from angles."""
        K_intrinsic, R_world_to_cam, t_world_to_cam = decompose_P(P)
        self.set_from_K(K_intrinsic)
        self.azimuth, self.elevation, self.roll = rotation_to_angles(R_world_to_cam)
        C = -R_world_to_cam.T @ t_world_to_cam.ravel()
        self.distance = float(np.linalg.norm(C) + 1e-12)
        self.pan_x, self.pan_y = 0.0, 0.0


# ---------------------------------------------------------------------------
# Qt widgets for matrix display and camera parameters (like 05_homography_research_gui)
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.4f}"


class MatrixDisplayWidget(QWidget):
    """Read-only grid of QLineEdit for displaying a matrix (e.g. P 3x4, A/R 3x3, t 3x1)."""

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
    """Editable f, physical/pixel size, distance, azimuth, elevation, roll with spinboxes."""

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
        row += 1
        grid.addWidget(QLabel("distance:"), row, 0)
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.5, 50.0)
        self.spin_dist.setValue(state.distance)
        self.spin_dist.setMaximumWidth(90)
        grid.addWidget(self.spin_dist, row, 1)
        row += 1
        grid.addWidget(QLabel("azimuth (°):"), row, 0)
        self.spin_az = QDoubleSpinBox()
        self.spin_az.setRange(-360.0, 360.0)
        self.spin_az.setDecimals(1)
        self.spin_az.setSuffix(" °")
        self.spin_az.setValue(np.degrees(state.azimuth))
        self.spin_az.setMaximumWidth(90)
        grid.addWidget(self.spin_az, row, 1)
        row += 1
        grid.addWidget(QLabel("elevation (°):"), row, 0)
        self.spin_el = QDoubleSpinBox()
        self.spin_el.setRange(-90.0, 90.0)
        self.spin_el.setDecimals(1)
        self.spin_el.setSuffix(" °")
        self.spin_el.setValue(np.degrees(state.elevation))
        self.spin_el.setMaximumWidth(90)
        grid.addWidget(self.spin_el, row, 1)
        row += 1
        grid.addWidget(QLabel("roll (°):"), row, 0)
        self.spin_roll = QDoubleSpinBox()
        self.spin_roll.setRange(-180.0, 180.0)
        self.spin_roll.setDecimals(1)
        self.spin_roll.setSuffix(" °")
        self.spin_roll.setValue(np.degrees(state.roll))
        self.spin_roll.setMaximumWidth(90)
        grid.addWidget(self.spin_roll, row, 1)
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
        self.state.distance = self.spin_dist.value()
        self.state.azimuth = np.radians(self.spin_az.value())
        self.state.elevation = np.radians(self.spin_el.value())
        self.state.roll = np.radians(self.spin_roll.value())

    def sync_from_state(self) -> None:
        self.spin_f.blockSignals(True)
        self.spin_wphys.blockSignals(True)
        self.spin_hphys.blockSignals(True)
        self.spin_wpix.blockSignals(True)
        self.spin_hpix.blockSignals(True)
        self.spin_dist.blockSignals(True)
        self.spin_az.blockSignals(True)
        self.spin_el.blockSignals(True)
        self.spin_roll.blockSignals(True)
        self.spin_f.setValue(self.state.focal_length_mm)
        self.spin_wphys.setValue(self.state.sensor_width_mm)
        self.spin_hphys.setValue(self.state.sensor_height_mm)
        self.spin_wpix.setValue(self.state.pixel_size_x_mm)
        self.spin_hpix.setValue(self.state.pixel_size_y_mm)
        self.spin_dist.setValue(self.state.distance)
        self.spin_az.setValue(np.degrees(self.state.azimuth))
        self.spin_el.setValue(np.degrees(self.state.elevation))
        self.spin_roll.setValue(np.degrees(self.state.roll))
        self.spin_f.blockSignals(False)
        self.spin_wphys.blockSignals(False)
        self.spin_hphys.blockSignals(False)
        self.spin_wpix.blockSignals(False)
        self.spin_hpix.blockSignals(False)
        self.spin_dist.blockSignals(False)
        self.spin_az.blockSignals(False)
        self.spin_el.blockSignals(False)
        self.spin_roll.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.square_pts = get_scene_square(1.0, 0.0)
        self.triangle_pts = get_scene_triangle(0.8, 1.0, 1.5)
        self.state = CameraState()
        self.last_xy = [0.0, 0.0]
        self.dragging = {"orbit": False, "pan": False, "zoom": False}
        self.modifier_alt = False
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Camera simulation (P = A [R|t])")
        self.setGeometry(80, 80, 1400, 700)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        # Left: matplotlib figure with 3D and image
        self.fig = plt.figure(figsize=(8, 5))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_img = self.fig.add_subplot(122)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 2)

        # Right: scroll area with group boxes (like 05_homography_research_gui)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        right_layout = QVBoxLayout()
        self.params_widget = CameraParamsWidget(self.state)
        for spin in (
            self.params_widget.spin_f,
            self.params_widget.spin_wphys,
            self.params_widget.spin_hphys,
            self.params_widget.spin_wpix,
            self.params_widget.spin_hpix,
            self.params_widget.spin_dist,
            self.params_widget.spin_az,
            self.params_widget.spin_el,
            self.params_widget.spin_roll,
        ):
            spin.valueChanged.connect(self._on_params_changed)
        right_layout.addWidget(self.params_widget)
        self.edit_P = MatrixEditWidget("P (3×4) editable", 3, 4)
        self.edit_P.matrix_changed.connect(self._on_P_changed)
        right_layout.addWidget(self.edit_P)
        self.edit_K = MatrixEditWidget("K intrinsic (3×3) editable", 3, 3)
        self.edit_K.matrix_changed.connect(self._on_K_changed)
        right_layout.addWidget(self.edit_K)
        self.display_R = MatrixDisplayWidget("R_world_to_cam (3×3) from angles", 3, 3)
        right_layout.addWidget(self.display_R)
        self.edit_t = MatrixEditWidget("t_world_to_cam (3×1) editable", 3, 1)
        self.edit_t.matrix_changed.connect(self._on_t_changed)
        right_layout.addWidget(self.edit_t)
        right_layout.addStretch()
        scroll_widget.setLayout(right_layout)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 1)
        central.setLayout(main_layout)

        # Modifier key tracking (matplotlib MouseEvent has no .alt)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._draw_all()

    def _on_key_press(self, event) -> None:
        if getattr(event, "key", None) == "alt":
            self.modifier_alt = True

    def _on_key_release(self, event) -> None:
        if getattr(event, "key", None) == "alt":
            self.modifier_alt = False
        self.dragging["orbit"] = False
        self.dragging["pan"] = False
        self.dragging["zoom"] = False

    def _on_mouse_press(self, event) -> None:
        if event.inaxes != self.ax_img:
            return
        if self.modifier_alt and event.button == 1:
            self.dragging["orbit"] = True
        elif (self.modifier_alt and event.button == 2) or (
            self.modifier_alt and event.button == 1 and getattr(event, "key", None) == "shift"
        ):
            self.dragging["pan"] = True
        elif self.modifier_alt and event.button == 3:
            self.dragging["zoom"] = True
        self.last_xy[0] = event.xdata or 0
        self.last_xy[1] = event.ydata or 0

    def _on_mouse_release(self, event) -> None:
        self.dragging["orbit"] = False
        self.dragging["pan"] = False
        self.dragging["zoom"] = False

    def _on_mouse_move(self, event) -> None:
        if event.inaxes != self.ax_img or event.xdata is None or event.ydata is None:
            return
        dx = (event.xdata - self.last_xy[0]) / self.image_width_px
        dy = (event.ydata - self.last_xy[1]) / self.image_height_px
        self.last_xy[0], self.last_xy[1] = event.xdata, event.ydata
        if self.dragging["orbit"]:
            self.state.orbit(-dx * 4, dy * 2)
            self.params_widget.sync_from_state()
            self._draw_all()
        elif self.dragging["pan"]:
            self.state.pan(-dx * 2, -dy * 2)
            self.params_widget.sync_from_state()
            self._draw_all()
        elif self.dragging["zoom"]:
            self.state.zoom(dy * 2)
            self.params_widget.sync_from_state()
            self._draw_all()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax_img:
            return
        self.state.zoom(0.15 if event.step > 0 else -0.15)
        self.params_widget.sync_from_state()
        self._draw_all()

    def _on_params_changed(self) -> None:
        self.params_widget.apply_to_state()
        self._draw_all()

    def _on_P_changed(self, P: np.ndarray) -> None:
        try:
            self.state.set_from_P(P.reshape(3, 4))
            self.params_widget.sync_from_state()
            self._update_matrix_displays()
            self._draw_all()
        except Exception:
            pass

    def _on_K_changed(self, K: np.ndarray) -> None:
        try:
            self.state.set_from_K(K.reshape(3, 3))
            self.params_widget.sync_from_state()
            self._update_matrix_displays()
            self._draw_all()
        except Exception:
            pass

    def _on_t_changed(self, t: np.ndarray) -> None:
        try:
            t_flat = np.asarray(t).ravel()
            self.state.set_from_t(t_flat[:3])
            self.params_widget.sync_from_state()
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
        self.edit_t.set_matrix(t_world_to_cam)

    def _draw_all(self) -> None:
        self.image_width_px = max(1, int(self.state.sensor_width_mm / self.state.pixel_size_x_mm))
        self.image_height_px = max(1, int(self.state.sensor_height_mm / self.state.pixel_size_y_mm))
        self.ax3d.cla()
        self.ax_img.cla()
        camera_center_world = self.state.get_camera_center_world()
        R_cam = self.state.get_R_cam()
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
        sq_edges = np.vstack([self.square_pts, self.square_pts[0:1]])
        tri_edges = np.vstack([self.triangle_pts, self.triangle_pts[0:1]])
        self.ax3d.plot(sq_edges[:, 0], sq_edges[:, 1], sq_edges[:, 2], "g-", lw=2)
        self.ax3d.plot(tri_edges[:, 0], tri_edges[:, 1], tri_edges[:, 2], "r-", lw=2)
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("3D scene and camera")
        margin = 2.0
        self.ax3d.set_xlim(-margin, margin)
        self.ax3d.set_ylim(-margin, margin)
        self.ax3d.set_zlim(-0.5, 2)
        draw_projected_scene(
            self.ax_img, P, self.square_pts, self.triangle_pts, self.image_width_px, self.image_height_px
        )
        self.ax_img.set_title("Image")
        self.ax_img.set_xlabel("u (pixels)")
        self.ax_img.set_ylabel("v (pixels)")
        self._update_matrix_displays()
        self.canvas.draw_idle()


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
