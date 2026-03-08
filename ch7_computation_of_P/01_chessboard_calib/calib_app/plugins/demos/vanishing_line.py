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



let change a little bit, let P = [0, 0, 0] and Q=[0,0,1]

then get the camera position (x, y, z)

notice that Q may not be strictly conlinear with P and the z vanishing point, so you could

1 minimize a camera center that have minimum d(C, RayP) and d(C, RayQ)，or 

2 minimize camera center that have mimnimum reproj error of P and Q

3 cancel C in the equation and have a mininum algebraic error on the parameters/scalers



any one is ok, and choose the most feasible for code


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


def _line_through_two_points(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line (3-vec) through two points."""
    return np.cross(np.asarray(p, dtype=np.float64).ravel()[:3],
                    np.asarray(q, dtype=np.float64).ravel()[:3])


def _clip_line_to_rect(
    l: np.ndarray,
    w: int,
    h: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Clip infinite line l (ax+by+c=0) to rectangle [0,w] x [0,h]. Returns ((x1,y1),(x2,y2)) or None."""
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    pts: List[Tuple[float, float]] = []
    if abs(b) > 1e-10:
        for x in (0.0, float(w)):
            y = -(a * x + c) / b
            if 0 <= y <= h:
                pts.append((x, y))
    if abs(a) > 1e-10:
        for y in (0.0, float(h)):
            x = -(b * y + c) / a
            if 0 <= x <= w:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    # Deduplicate and take two furthest (or first two if only two)
    seen = set()
    unique = []
    for p in pts:
        key = (round(p[0], 2), round(p[1], 2))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if len(unique) < 2:
        return None
    if len(unique) == 2:
        return ((int(unique[0][0]), int(unique[0][1])), (int(unique[1][0]), int(unique[1][1])))
    # More than 2: pick two that span the visible segment (e.g. min and max x, or two extremes)
    p1 = min(unique, key=lambda p: p[0])
    p2 = max(unique, key=lambda p: p[0])
    return ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))


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
    R from world to camera: r1 from v_x, r2 orthogonalized from v_y (Gram-Schmidt), r3 = r1 x r2.
    Rays are forced to point forward (d[2] > 0); det(R) = 1 (right-handed).
    """
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))

    # 1. Back-project to rays in camera coordinates
    d1 = K_inv @ np.asarray(v_x, dtype=np.float64).ravel()[:3]
    d2 = K_inv @ np.asarray(v_y, dtype=np.float64).ravel()[:3]
    
    # 2. Force vectors to point "away" from camera (positive Z)
    # This prevents the R matrix from flipping the world behind the camera
    if d1[2] < 0: d1 = -d1
    if d2[2] < 0: d2 = -d2

    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None
    
    r1 = d1 / n1
    r2 = d2 / n2

    # 3. Gram-Schmidt Orthogonalization
    # User input is noisy; we must force r1 and r2 to be orthogonal.
    # We keep r1 (X-axis) as the primary direction and adjust r2.
    r3 = np.cross(r1, r2)
    n3 = np.linalg.norm(r3)
    if n3 < 1e-10:
        return None
    r3 = r3 / n3
    
    # Re-compute r2 to ensure it is perfectly orthogonal to r1 and r3
    r2 = np.cross(r3, r1)

    # 4. Final R assembly
    R = np.column_stack((r1, r2, r3))
    
    # 5. Coordinate System Check (Right-Handed)
    # Ensure determinant is 1, not -1 (prevents mirroring)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2] # Flip Z if necessary
        
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

def _solve_camera_position(
    K: np.ndarray,
    R: np.ndarray,
    p_img: Tuple[float, float],
    q_img: Tuple[float, float],
) -> Optional[np.ndarray]:
    """
    Find camera center C given P=[0,0,0], Q=[0,0,1] in world and image points p, q.
    Solves s2*dq - s1*dp = Q - P via least squares, then C = P - s1*dp = -s1*dp.
    Handles non-collinearity of p, q with the vertical vanishing point.
    """
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    dp = R.T @ (K_inv @ np.array([p_img[0], p_img[1], 1.0], dtype=np.float64))
    dq = R.T @ (K_inv @ np.array([q_img[0], q_img[1], 1.0], dtype=np.float64))

    # A @ [s1, s2]^T = B  with  -s1*dp + s2*dq = [0, 0, 1]
    A = np.column_stack((-dp, dq))
    B = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    try:
        scales, *_ = np.linalg.lstsq(A, B, rcond=None)
        s1, s2 = float(scales[0]), float(scales[1])
        C = -s1 * dp
        if not np.all(np.isfinite(C)) or C[2] <= 0:
            return None
        return C
    except (np.linalg.LinAlgError, TypeError):
        return None


def _ray_ground_intersection_with_C(
    K: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    px: float,
    py: float,
) -> Optional[np.ndarray]:
    """Point on ground plane Z=0 where ray from camera C through pixel (px, py) hits."""
    D = _back_project_direction(K, R, px, py)
    if abs(D[2]) < 1e-12:
        return None
    s = -C[2] / D[2]
    P = C + s * D
    return P


def _world_to_image_with_C(
    K: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    P_world: np.ndarray,
) -> Tuple[float, float]:
    """Project world point to image using explicit camera center C. p = K @ R @ (P_world - C)."""
    p_cam = R @ (np.asarray(P_world, dtype=np.float64).ravel()[:3] - C)
    if p_cam[2] <= 0:
        return float("nan"), float("nan")
    q = K @ p_cam
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
        self.C: Optional[np.ndarray] = None  # camera center (x, y, z), from P=[0,0,0], Q=[0,0,1]
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

        # Vanishing points and vanishing line (horizon)
        if s.R is not None and s.v_x is not None and s.v_y is not None:
            h_img, w_img = disp.shape[:2]
            for name, v in [("vx", s.v_x), ("vy", s.v_y)]:
                if v[2] != 0:
                    u, vv = int(v[0] / v[2]), int(v[1] / v[2])
                    if 0 <= u < w_img and 0 <= vv < h_img:
                        cv2.circle(disp, (u, vv), 8, (255, 0, 0), 2)
            # Vanishing line (horizon): line through v_x and v_y, clipped to image
            l_inf = _line_through_two_points(s.v_x, s.v_y)
            seg = _clip_line_to_rect(l_inf, w_img, h_img)
            if seg is not None:
                (x1, y1), (x2, y2) = seg
                cv2.line(disp, (x1, y1), (x2, y2), (255, 165, 0), 2)  # orange horizon
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
            self._state.C = None
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
                        self._log(f"\nR={s.R} \n K={K}")
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
                C = _solve_camera_position(K, s.R, s.p_image, (float(ix), float(iy)))
                if C is not None:
                    s.C = C
                    self._log(f"Clicked Q at ({ix:.1f}, {iy:.1f}); camera center C = ({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f})")
                else:
                    s.C = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # fallback
                    self._log(f"Clicked Q at ({ix:.1f}, {iy:.1f}); C solve failed, using fallback")
                self._redraw()
                return True
            # Check if click is near P -> start drag
            if s.p_image is not None:
                dx = ix - s.p_image[0]
                dy = iy - s.p_image[1]
                if dx * dx + dy * dy < 900:  # 30px radius
                    s.dragging_p = True
                    return True
            return True

        if event_type == "release":
            if s.dragging_p and s.p_image is not None and s.C is not None and s.R is not None:
                self._log(f"Drag ended: P_image=({s.p_image[0]:.3f}, {s.p_image[1]:.3f})")
                P_world = _ray_ground_intersection_with_C(K, s.R, s.C, s.p_image[0], s.p_image[1])
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
            if s.dragging_p and s.C is not None and s.R is not None:
                # New P at (ix, iy); ray from C hits ground Z=0 at P_world; Q_world = P_world + [0,0,1]
                P_world = _ray_ground_intersection_with_C(K, s.R, s.C, ix, iy)
                if P_world is not None:
                    Q_world = P_world + np.array([0.0, 0.0, 1.0])
                    qx, qy = _world_to_image_with_C(K, s.R, s.C, Q_world)
                    if np.isfinite(qx) and np.isfinite(qy):
                        s.p_image = (float(ix), float(iy))
                        s.q_image = (qx, qy)
                self._redraw()
                return True
            return False

        return False
