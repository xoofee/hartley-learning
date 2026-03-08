r"""
suppose the K matrix is known, and no distortion.
Let user select four corners of a rectangle (in the current image in centerwidget), say, A B C D, in the ground plane
so that the edges are AB BC CD DA
calculate the vanishing line l_inf

let the AB and CD intersect at X, let's define it the x axis vanishing point
Then the R could be inferred from the l_inf and the x axis vanishing point
(P = K[R|t])

let's define the world origin same as camera origin vertically so the ground plane is z=0 in world plane. and camera is at the x=0, y=0, and z = h

let user click two points on the image, say P and Q, and P is on the Ground, the PQ perpendicular to the ground plane

suppose the distance between P and Q is 1 (in world coordinates), then the world coordinates of P and Q could be inferred from the P and Q image coordinates

then let the user drag point of P, the point of Q will be dynamically updated so that the distance between P and Q is 1 (in world coordinates)
and PQ is perpendicular to the ground plane


Gemini refine:

## Perspective-Correct Height Measurement and Dynamic Interaction

### 1. Problem Statement

In a monocular camera setup (a single 2D image), estimating the physical world position and height of an object is an "ill-posed" problem because depth information is lost during projection.

To resolve this, we need to:

* **Establish a Coordinate System:** Define the ground plane ($Z=0$) and the camera's position relative to it (the height $h$).
* **Restore Perspective:** Use vanishing points from a ground-plane rectangle to determine the camera's orientation (Rotation matrix $R$).
* **Enforce Geometry:** Ensure that when a user selects a point $P$ on the ground and an arbitrary top point $Q$, the system can infer a 3D height of $1$ unit while maintaining the geometric constraint that $PQ$ is perpendicular to the ground, even if the user's initial mouse clicks are imprecise.

---

### 2. The Calculation Process

#### Step A: Establishing Ground Orientation ($R$)

1. **Define the Ground Plane:** The user selects four corners of a rectangle ($A, B, C, D$) on the ground.
2. **Locate Vanishing Points:** Calculate the intersection of the two sets of parallel edges:
* $v_x = (A \times B) \times (C \times D)$
* $v_y = (B \times C) \times (D \times A)$


3. **Compute Rotation ($R$):** Use the intrinsic matrix $K$ to map these to world space.

$$r_1 = \frac{K^{-1}v_x}{\|K^{-1}v_x\|}, \quad r_2 = \frac{K^{-1}v_y}{\|K^{-1}v_y\|}, \quad r_3 = r_1 \times r_2$$



The vertical vanishing point $v_z$ is then the intersection of the rays defined by the camera's optical axis and the ground normal.

#### Step B: Calibrating the Camera Height ($h$)

When the user clicks the base $P$ and top $Q$, we must solve for $h$ to establish the "scale" of the world:

1. **Back-project:** Transform pixel coordinates to world rays: $\mathbf{D}_p = R^\top K^{-1} p$ and $\mathbf{D}_q = R^\top K^{-1} q$.
2. **Solve for $h$:** We treat the initial clicks as a calibration step. By setting the horizontal alignment error to zero, we derive $h$:

$$h = \frac{D_{px} D_{qz}}{D_{px} D_{qz} - D_{pz} D_{qx}}$$



This $h$ represents the camera's distance from the ground plane.

#### Step C: Dynamic Interaction (The "Drag" Logic)

Once the camera is calibrated, we can dynamically update the position of $Q$ as the user drags $P$:

1. **Map $P$ to Ground:** As the user drags the mouse to $p_{new}$, calculate the new world position $P_{world}$ at the intersection of the ray and the plane $Z=0$:

$$P_{world} = C - \frac{h}{D_{p\_new,z}} \mathbf{D}_{p\_new}$$


2. **Calculate $Q$:** Because the object is fixed to height $1$:

$$Q_{world} = P_{world} + [0, 0, 1]^\top$$


3. **Re-project:** Map $Q_{world}$ back into the image:

$$q_{new} = K R (Q_{world} - C)$$



*Note: The calculated $q_{new}$ will automatically appear to "snap" onto the line connecting $p_{new}$ and the vertical vanishing point $v_z$, ensuring perfect geometric verticality regardless of the user's initial click tolerance.*

implement it. remember to make the code modular and flexible and have a good architecture. we may have other demos that require interactive like "click to select, drag a point on it". and some demo do not need this

do not remove this comment
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
)

from ..registry import Demo
from ...logging_ui import log as _app_log


# ---------- Geometry (homogeneous 2D) ----------

def _to_homogeneous(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 1.0], dtype=np.float64)


def _line_from_two_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Homogeneous line through two points (each 3-vec or (x,y))."""
    ah = np.asarray(a, dtype=np.float64).ravel()
    bh = np.asarray(b, dtype=np.float64).ravel()
    if ah.size == 2:
        ah = np.array([ah[0], ah[1], 1.0])
    if bh.size == 2:
        bh = np.array([bh[0], bh[1], 1.0])
    return np.cross(ah, bh)


def _intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Homogeneous point (3-vec) at intersection of two lines."""
    p = np.cross(np.asarray(l1, dtype=np.float64).ravel()[:3],
                 np.asarray(l2, dtype=np.float64).ravel()[:3])
    return p


def _vanishing_points_from_rectangle(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Corners in order A-B-C-D (edges AB, BC, CD, DA).
    Returns (v_x, v_y) where v_x = (AB)x(CD), v_y = (BC)x(DA), as 3-vec homogeneous.
    """
    ah = _to_homogeneous(A[0], A[1])
    bh = _to_homogeneous(B[0], B[1])
    ch = _to_homogeneous(C[0], C[1])
    dh = _to_homogeneous(D[0], D[1])
    line_ab = _line_from_two_points(ah, bh)
    line_cd = _line_from_two_points(ch, dh)
    line_bc = _line_from_two_points(bh, ch)
    line_da = _line_from_two_points(dh, ah)
    vx = _intersect_lines(line_ab, line_cd)
    vy = _intersect_lines(line_bc, line_da)
    return vx, vy


def _rotation_from_vanishing_points(
    K: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
) -> Optional[np.ndarray]:
    """
    R such that world X direction goes to v_x, world Y to v_y, Z up.
    r1 = K^{-1}v_x / |...|, r2 = K^{-1}v_y / |...|, r3 = r1 x r2.
    R is 3x3 with columns r1, r2, r3 (camera rotation from world to camera).
    """
    K = np.asarray(K, dtype=np.float64)
    v_x = np.asarray(v_x, dtype=np.float64).ravel()[:3]
    v_y = np.asarray(v_y, dtype=np.float64).ravel()[:3]
    K_inv = np.linalg.inv(K)
    r1 = K_inv @ v_x
    r2 = K_inv @ v_y
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None
    r1 = r1 / n1
    r2 = r2 / n2
    r3 = np.cross(r1, r2)
    n3 = np.linalg.norm(r3)
    if n3 < 1e-10:
        return None
    r3 = r3 / n3
    R = np.column_stack((r1, r2, r3))
    return R


def _vertical_vanishing_point(K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """v_z = K @ r3 (direction of world Z in image)."""
    r3 = R[:, 2]
    return (K @ r3).ravel()


def _back_project_direction(K: np.ndarray, R: np.ndarray, px: float, py: float) -> np.ndarray:
    """World-space direction (unnormalized) of ray through pixel (px, py). D = R^T K^{-1} p."""
    p = np.array([px, py, 1.0], dtype=np.float64)
    D = R.T @ (np.linalg.inv(K) @ p)
    return D


def _solve_camera_height(
    K: np.ndarray,
    R: np.ndarray,
    p_px: float, p_py: float,
    q_px: float, q_py: float,
) -> Optional[float]:
    """
    From base point p and top point q (image coords), solve h so that
    P is on ground Z=0 and Q = P + (0,0,1). Formula: h = D_px*D_qz / (D_px*D_qz - D_pz*D_qx).
    """
    D_p = _back_project_direction(K, R, p_px, p_py)
    D_q = _back_project_direction(K, R, q_px, q_py)
    denom = D_p[0] * D_q[2] - D_p[2] * D_q[0]
    if abs(denom) < 1e-12:
        return None
    h = (D_p[0] * D_q[2]) / denom
    if h <= 0 or not np.isfinite(h):
        return None
    return float(h)


def _ray_ground_intersection(
    K: np.ndarray,
    R: np.ndarray,
    h: float,
    px: float,
    py: float,
) -> Optional[np.ndarray]:
    """P_world on ground (Z=0) from camera at (0,0,h). P = C - (h/D_z) * D."""
    D = _back_project_direction(K, R, px, py)
    if abs(D[2]) < 1e-12:
        return None
    C = np.array([0.0, 0.0, h], dtype=np.float64)
    lam = -h / D[2]
    P = C + lam * D
    return P


def _world_to_image(
    K: np.ndarray,
    R: np.ndarray,
    h: float,
    P_world: np.ndarray,
) -> Tuple[float, float]:
    """Project world point to image. Q_cam = R (P_world - C), q = K @ Q_cam."""
    C = np.array([0.0, 0.0, h], dtype=np.float64)
    Q_cam = R @ (P_world - C)
    if Q_cam[2] <= 0:
        return float("nan"), float("nan")
    q = K @ Q_cam
    return float(q[0] / q[2]), float(q[1] / q[2])


# ---------- Demo state ----------

class _VanishingLineState:
    """Mutable state for the vanishing-line demo (per document)."""

    def __init__(self) -> None:
        self.corners: List[Tuple[float, float]] = []  # A, B, C, D image coords
        self.R: Optional[np.ndarray] = None
        self.v_x: Optional[np.ndarray] = None
        self.v_y: Optional[np.ndarray] = None
        self.v_z: Optional[np.ndarray] = None
        self.h: Optional[float] = None
        self.p_image: Optional[Tuple[float, float]] = None  # base on ground
        self.q_image: Optional[Tuple[float, float]] = None  # top (height 1)
        self.dragging_p: bool = False


# ---------- Demo class ----------

class VanishingLineDemo(Demo):
    def id(self) -> str:
        return "vanishing_line"

    def label(self) -> str:
        return "Vanishing line"

    def on_activated(self, context: dict) -> None:
        get_K = context.get("get_K")
        K = get_K() if callable(get_K) else None
        if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
            QMessageBox.warning(
                None,
                "Vanishing line",
                "Calibration matrix K is not available. Load or run calibration first.",
            )
            switch = context.get("switch_demo")
            if callable(switch):
                switch("none")
            return
        self._K = np.asarray(K, dtype=np.float64)
        self._context = context
        self._state = _VanishingLineState()
        self._current_document: Any = None
        self._log_to_widget: bool = True
        self._pane_state_label: Optional[QLabel] = None
        self._pane_command_label: Optional[QLabel] = None
        self._redraw()
        self._update_pane()

    def on_deactivated(self) -> None:
        self._restore_document_display()
        self._context = None
        self._state = None
        self._current_document = None

    def _log(self, message: str) -> None:
        """Log to app log widget only when demo pane 'Log to widget' is checked."""
        if getattr(self, "_log_to_widget", True):
            _app_log(f"[Vanishing line] {message}")

    def _update_pane(self) -> None:
        """Update state and command labels in the demo pane."""
        # Resolve labels from the pane widget so we always update the visible pane
        pane = getattr(self, "_pane_widget", None)
        if pane is None:
            return
        state_lbl = pane.findChild(QLabel, "vanishing_state")
        cmd_lbl = pane.findChild(QLabel, "vanishing_command")
        if state_lbl is None or cmd_lbl is None:
            return
        s = self._state
        if s is None:
            state_lbl.setText("State: —")
            cmd_lbl.setText("Hint: Click point A")
            return
        # State line
        if len(s.corners) < 4:
            state_str = " ".join("ABCD"[i] for i in range(len(s.corners)))
            if state_str:
                state_lbl.setText(f"State: {state_str}")
            else:
                state_lbl.setText("State: —")
        else:
            state_lbl.setText("State: A B C D set")
            if s.p_image is None:
                pass  # command set below
            elif s.q_image is None:
                state_lbl.setText("State: A B C D set, P set")
            else:
                state_lbl.setText("State: A B C D set, P and Q set")
        # Command / hint line (always show next action)
        if len(s.corners) < 4:
            next_letter = "ABCD"[len(s.corners)]
            cmd_lbl.setText(f"Hint: Click point {next_letter}")
        elif s.p_image is None:
            cmd_lbl.setText("Hint: Click point P (ground base)")
        elif s.q_image is None:
            cmd_lbl.setText("Hint: Click point Q (top of segment)")
        elif s.dragging_p:
            cmd_lbl.setText("Hint: Drag point P (release to update Q)")
        else:
            cmd_lbl.setText("Hint: Drag point P to move; Q follows (height = 1)")
        state_lbl.repaint()
        cmd_lbl.repaint()

    def _restore_document_display(self) -> None:
        if self._current_document is not None and hasattr(self._current_document, "image_view"):
            view = self._current_document.image_view()
            bgr = self._current_document.image_bgr() if hasattr(self._current_document, "image_bgr") else None
            if bgr is not None and bgr.size > 0:
                view.set_image(bgr)

    def _redraw(self) -> None:
        doc = self._context.get("get_current_document")() if self._context else None
        if doc is None or not hasattr(doc, "image_bgr") or not hasattr(doc, "image_view"):
            self._update_pane()
            return
        self._current_document = doc
        bgr = doc.image_bgr()
        if bgr is None or bgr.size == 0:
            self._update_pane()
            return
        disp = bgr.copy()
        s = self._state
        K = self._K

        # Draw corners (show instantly: 1–4 points) and rectangle edges
        if len(s.corners) > 0:
            pts = np.array([(c[0], c[1]) for c in s.corners], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(disp, [pts], False, (0, 255, 0), 2)
            for i, (x, y) in enumerate(s.corners):
                cv2.circle(disp, (int(x), int(y)), 6, (0, 255, 0), -1)
                cv2.putText(disp, "ABCD"[i], (int(x) + 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Vanishing points and line
        if s.R is not None and s.v_x is not None and s.v_y is not None:
            h_img, w_img = disp.shape[:2]
            for name, v in [("vx", s.v_x), ("vy", s.v_y)]:
                if v[2] != 0:
                    u, vv = int(v[0] / v[2]), int(v[1] / v[2])
                    if 0 <= u < w_img and 0 <= vv < h_img:
                        cv2.circle(disp, (u, vv), 8, (255, 0, 0), 2)
            if s.v_z is not None and np.isfinite(s.v_z).all():
                vz = s.v_z
                if abs(vz[2]) > 1e-10:
                    uz, vz_y = int(vz[0] / vz[2]), int(vz[1] / vz[2])
                    cv2.circle(disp, (uz, vz_y), 6, (0, 0, 255), 2)
                    cv2.putText(disp, "vz", (uz + 6, vz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # P and Q
        if s.p_image is not None:
            px, py = int(s.p_image[0]), int(s.p_image[1])
            cv2.circle(disp, (px, py), 8, (0, 255, 255), 2)
            cv2.putText(disp, "P", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255))
        if s.q_image is not None:
            qx, qy = int(s.q_image[0]), int(s.q_image[1])
            cv2.circle(disp, (qx, qy), 8, (255, 255, 0), 2)
            cv2.putText(disp, "Q", (qx + 10, qy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0))
        if s.p_image is not None and s.q_image is not None:
            cv2.line(
                disp,
                (int(s.p_image[0]), int(s.p_image[1])),
                (int(s.q_image[0]), int(s.q_image[1])),
                (200, 200, 200),
                2,
            )

        doc.image_view().set_image(disp)
        self._update_pane()

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        self._context = context
        widget = QWidget()
        self._pane_widget = widget
        layout = QVBoxLayout(widget)
        # State and command (updated on each redraw); objectName so _update_pane can find them
        state_label = QLabel("State: —")
        state_label.setObjectName("vanishing_state")
        state_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(state_label)
        command_label = QLabel("Click point A")
        command_label.setObjectName("vanishing_command")
        command_label.setWordWrap(True)
        command_label.setStyleSheet("color: #333;")
        layout.addWidget(command_label)
        # Log switch
        log_check = QCheckBox("Log to widget")
        log_check.setToolTip("When checked, demo actions and calculation output are written to the Log dock.")
        log_check.setChecked(getattr(self, "_log_to_widget", True))
        log_check.toggled.connect(lambda on: setattr(self, "_log_to_widget", on))
        layout.addWidget(log_check)
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Clear rectangle, P, Q and start over")
        reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(reset_btn)
        layout.addStretch()
        return widget

    def _on_reset(self) -> None:
        if self._state is not None:
            self._state.corners = []
            self._state.R = None
            self._state.v_x = None
            self._state.v_y = None
            self._state.v_z = None
            self._state.h = None
            self._state.p_image = None
            self._state.q_image = None
            self._state.dragging_p = False
        self._redraw()

    def handle_mouse_event(
        self,
        context: dict,
        document: Any,
        x: float,
        y: float,
        event_type: str,
    ) -> bool:
        if self._context is None or self._state is None or document is None:
            return False
        if not hasattr(document, "map_to_image_coords"):
            return False
        pt = document.map_to_image_coords(x, y)
        if pt is None:
            return False
        ix, iy = pt
        s = self._state
        K = self._K

        if event_type == "press":
            if s.dragging_p:
                s.dragging_p = False
                return True
            # Add corner or set P or Q
            if len(s.corners) < 4:
                letter = "ABCD"[len(s.corners)]
                s.corners.append((float(ix), float(iy)))
                self._log(f"Clicked {letter} at ({ix:.1f}, {iy:.1f})")
                if len(s.corners) == 4:
                    vx, vy = _vanishing_points_from_rectangle(
                        s.corners[0], s.corners[1], s.corners[2], s.corners[3]
                    )
                    if vx is None or vy is None:
                        s.corners.pop()
                        return True
                    s.R = _rotation_from_vanishing_points(K, vx, vy)
                    if s.R is not None:
                        s.v_x = vx
                        s.v_y = vy
                        s.v_z = _vertical_vanishing_point(K, s.R)
                        def _vp_str(v: np.ndarray) -> str:
                            if abs(v[2]) < 1e-10:
                                return "inf"
                            return f"({v[0]/v[2]:.1f}, {v[1]/v[2]:.1f})"
                        self._log(f"A B C D set. v_x={_vp_str(vx)}, v_y={_vp_str(vy)}, R computed, v_z={_vp_str(s.v_z)}")
                    else:
                        s.corners.pop()
                self._redraw()
                return True
            if s.R is None:
                return True
            if s.p_image is None:
                s.p_image = (float(ix), float(iy))
                self._log(f"Clicked P at ({ix:.1f}, {iy:.1f})")
                self._redraw()
                return True
            if s.q_image is None:
                s.q_image = (float(ix), float(iy))
                h = _solve_camera_height(K, s.R, s.p_image[0], s.p_image[1], ix, iy)
                if h is not None and h > 0:
                    s.h = h
                else:
                    s.h = 1.0  # fallback
                self._log(f"Clicked Q at ({ix:.1f}, {iy:.1f}); camera height h = {s.h:.4f}")
                self._redraw()
                return True
            # Check if click is near P -> start drag
            if s.p_image is not None:
                dx = ix - s.p_image[0]
                dy = iy - s.p_image[1]
                if dx * dx + dy * dy < 400:  # ~20px
                    s.dragging_p = True
                    return True
            return True

        if event_type == "release":
            if s.dragging_p and s.p_image is not None and s.h is not None and s.R is not None:
                P_world = _ray_ground_intersection(K, s.R, s.h, s.p_image[0], s.p_image[1])
                if P_world is not None:
                    Q_world = P_world + np.array([0.0, 0.0, 1.0])
                    self._log(
                        f"Drag ended: P_world=({P_world[0]:.3f}, {P_world[1]:.3f}, {P_world[2]:.3f}), "
                        f"Q_world=({Q_world[0]:.3f}, {Q_world[1]:.3f}, {Q_world[2]:.3f})"
                    )
            s.dragging_p = False
            self._redraw()
            return True

        if event_type == "move":
            if s.dragging_p and s.p_image is not None and s.h is not None and s.R is not None:
                # New P at (ix, iy); compute ground point and Q at height 1
                P_world = _ray_ground_intersection(K, s.R, s.h, ix, iy)
                if P_world is not None:
                    Q_world = P_world + np.array([0.0, 0.0, 1.0])
                    qx, qy = _world_to_image(K, s.R, s.h, Q_world)
                    if np.isfinite(qx) and np.isfinite(qy):
                        s.p_image = (float(ix), float(iy))
                        s.q_image = (qx, qy)
                self._redraw()
                return True
            return False

        return False
