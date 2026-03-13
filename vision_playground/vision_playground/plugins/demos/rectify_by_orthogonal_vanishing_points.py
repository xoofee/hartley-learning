r"""
A demo to rectify a plane using two orthogonal vanishing points when the camera intrinsic matrix K is known.

let user click four points ABCD of a rectangle on a plane like in vanishing_line.py, and then calculate the rotation matrix R, and the homography matrix H, and apply the H to the image and show the rectified image.

may copy some logic from vanishing_line.py


To rectify a plane using two orthogonal vanishing points when the camera intrinsic matrix $K$ is known, you are essentially performing **metric rectification**. This process removes projective distortion, allowing you to measure true angles and ratios of lengths on that plane.

Since the vanishing points correspond to orthogonal directions in the world (e.g., the $x$ and $y$ axes of a rectangular floor), they provide enough information to determine the orientation of the plane relative to the camera.

---

### 1. Identify the Vanishing Line

The two vanishing points $\mathbf{v}_1$ and $\mathbf{v}_2$ (in homogeneous coordinates) lie on the vanishing line $\mathbf{l}_\infty$ of the plane. You can calculate this line using the cross product:


$$\mathbf{l}_\infty = \mathbf{v}_1 \times \mathbf{v}_2$$


In a rectified image, this line must be moved to infinity.

### 2. Determine the Plane Normal

With $K$ known, you can compute the "directions" of the vanishing points in 3D space. The direction vectors $\mathbf{d}_1$ and $\mathbf{d}_2$ are:


$$\mathbf{d}_1 = \frac{K^{-1}\mathbf{v}_1}{\|K^{-1}\mathbf{v}_1\|}, \quad \mathbf{d}_2 = \frac{K^{-1}\mathbf{v}_2}{\|K^{-1}\mathbf{v}_2\|}$$


Since these directions are orthogonal in the world, the normal to the plane $\mathbf{n}$ in the camera coordinate system is simply:


$$\mathbf{n} = \mathbf{d}_1 \times \mathbf{d}_2$$

### 3. Construct the Rotation Matrix ($R$)

To rectify the image, we want to rotate the camera so it looks directly at the plane (making the optical axis parallel to the plane normal). We can build a rotation matrix $R$ using our orthogonal directions:

* **Column 1:** $\mathbf{r}_1 = \mathbf{d}_1$
* **Column 2:** $\mathbf{r}_2 = \mathbf{d}_2$
* **Column 3:** $\mathbf{r}_3 = \mathbf{n}$

$$R = [\mathbf{r}_1 | \mathbf{r}_2 | \mathbf{r}_3]$$

### 4. Apply the Homography ($H$)

The rectification homography $H$ that transforms the original image coordinates to the rectified coordinates is given by:


$$H = K R K^{-1}$$


By applying $H$ to your image, the vanishing line $\mathbf{l}_\infty$ is mapped to the line at infinity $[0, 0, 1]^T$, and the two orthogonal directions will appear at $90^\circ$ to each other in the resulting image.

---

### Summary Table: Rectification Workflow

| Step | Operation | Result |
| --- | --- | --- |
| **Compute Directions** | $\mathbf{d}_i = K^{-1}\mathbf{v}_i$ | Rays in 3D camera space. |
| **Find Normal** | $\mathbf{n} = \mathbf{d}_1 \times \mathbf{d}_2$ | Orientation of the physical plane. |
| **Form Rotation** | $R = [\mathbf{d}_1, \mathbf{d}_2, \mathbf{n}]$ | Alignment of camera to plane. |
| **Final Warp** | $H = K R K^{-1}$ | The homography matrix to transform the pixels. |

> **Note:** If you only have the vanishing line $\mathbf{l}_\infty = [l_1, l_2, l_3]^T$ but not the specific vanishing points, you can still perform an affine rectification using $H = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ l_1 & l_2 & l_3 \end{bmatrix}$, but $K$ is required to achieve the full **metric** (orthogonal) correction.

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
from .vanishing_line import _vanishing_points_from_rectangle


# ---------- Rectification geometry ----------

def _homography_from_orthogonal_vanishing_points(
    K: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute rectification homography H = K R K^{-1} from two orthogonal vanishing points.
    R = [d1, d2, n] with d1 = K^{-1}v_x/|...|, d2 = K^{-1}v_y/|...|, n = d1 x d2.
    Ensures right-handed R (det=1) and directions pointing forward (d[2] > 0).
    """
    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    vx = np.asarray(v_x, dtype=np.float64).ravel()[:3]
    vy = np.asarray(v_y, dtype=np.float64).ravel()[:3]

    d1 = K_inv @ vx
    d2 = K_inv @ vy
    if d1[2] < 0:
        d1 = -d1
    if d2[2] < 0:
        d2 = -d2

    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None

    d1 = d1 / n1
    d2 = d2 / n2
    n = np.cross(d1, d2)
    n3 = np.linalg.norm(n)
    if n3 < 1e-10:
        return None
    n = n / n3

    R = np.column_stack((d1, d2, n))
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]

    H = K @ R @ K_inv
    return H


# ---------- Demo state ----------

class _RectifyState:
    def __init__(self) -> None:
        self.corners: List[Tuple[float, float]] = []  # A, B, C, D
        self.H: Optional[np.ndarray] = None
        self.show_rectified: bool = True


# ---------- Demo class ----------

class RectifyByOrthogonalVPDemo(Demo):
    def id(self) -> str:
        return "rectify_orthogonal_vp"

    def label(self) -> str:
        return "Rectify by orthogonal VPs"

    def on_activated(self, context: dict) -> None:
        get_K = context.get("get_K")
        K = get_K() if callable(get_K) else None
        if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
            QMessageBox.warning(
                None,
                "Rectify by orthogonal VPs",
                "Calibration matrix K is not available. Load or run calibration first.",
            )
            switch = context.get("switch_demo")
            if callable(switch):
                switch("none")
            return
        self._K = np.asarray(K, dtype=np.float64)
        self._context = context
        self._state = _RectifyState()
        self._current_document: Any = None
        self._pane_widget: Optional[QWidget] = None
        self._redraw()
        self._update_pane()

    def on_deactivated(self) -> None:
        self._restore_document_display()
        self._context = None
        self._state = None
        self._current_document = None

    def _log(self, message: str) -> None:
        if getattr(self, "_log_to_widget", True):
            _app_log(f"[Rectify] {message}")

    def _update_pane(self) -> None:
        pane = getattr(self, "_pane_widget", None)
        if pane is None:
            return
        state_lbl = pane.findChild(QLabel, "rectify_state")
        cmd_lbl = pane.findChild(QLabel, "rectify_command")
        if state_lbl is None or cmd_lbl is None:
            return
        s = self._state
        if s is None:
            state_lbl.setText("State: —")
            cmd_lbl.setText("Hint: Click point A")
            return
        if len(s.corners) < 4:
            state_str = " ".join("ABCD"[i] for i in range(len(s.corners)))
            state_lbl.setText(f"State: {state_str}" if state_str else "State: —")
            next_letter = "ABCD"[len(s.corners)]
            cmd_lbl.setText(f"Hint: Click point {next_letter}")
        else:
            state_lbl.setText("State: A B C D set (rectified)")
            cmd_lbl.setText("Hint: Reset to choose another rectangle")
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
        # Clear any homography from another demo (e.g. Rotate) so our overlay is shown correctly
        if hasattr(doc, "set_homography"):
            doc.set_homography(None)
        bgr = doc.image_bgr()
        if bgr is None or bgr.size == 0:
            self._update_pane()
            return

        s = self._state
        h_img, w_img = bgr.shape[:2]

        if len(s.corners) < 4 or s.H is None or not s.show_rectified:
            disp = bgr.copy()
            if len(s.corners) > 0:
                pts = np.array([(c[0], c[1]) for c in s.corners], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(disp, [pts], False, (0, 255, 255), 3)  # cyan, thick
                for i, (x, y) in enumerate(s.corners):
                    cx, cy = int(x), int(y)
                    # Dark outline so marker is visible on any background
                    cv2.circle(disp, (cx, cy), 14, (0, 0, 0), 4)
                    cv2.circle(disp, (cx, cy), 14, (0, 255, 255), 2)  # cyan fill
                    cv2.circle(disp, (cx, cy), 10, (0, 255, 255), -1)
                    cv2.putText(
                        disp, "ABCD"[i], (cx + 16, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3,
                    )
                    cv2.putText(
                        disp, "ABCD"[i], (cx + 16, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 1,
                    )
            view = doc.image_view()
            view.set_image(disp)
            view.update()
        else:
            rectified = cv2.warpPerspective(bgr, np.linalg.inv(s.H), (w_img, h_img), flags=cv2.INTER_LINEAR)
            view = doc.image_view()
            view.set_image(rectified)
            view.update()

        self._update_pane()

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        self._context = context
        widget = QWidget()
        self._pane_widget = widget
        layout = QVBoxLayout(widget)
        state_label = QLabel("State: —")
        state_label.setObjectName("rectify_state")
        state_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(state_label)
        command_label = QLabel("Click point A")
        command_label.setObjectName("rectify_command")
        command_label.setWordWrap(True)
        command_label.setStyleSheet("color: #333;")
        layout.addWidget(command_label)
        show_rect_check = QCheckBox("Show rectified")
        show_rect_check.setToolTip("When checked, display the rectified image after 4 points are set.")
        show_rect_check.setChecked(True)
        show_rect_check.toggled.connect(self._on_show_rectified_toggled)
        layout.addWidget(show_rect_check)
        log_check = QCheckBox("Log to widget")
        log_check.setToolTip("When checked, demo actions are written to the Log dock.")
        log_check.setChecked(getattr(self, "_log_to_widget", True))
        log_check.toggled.connect(lambda on: setattr(self, "_log_to_widget", on))
        layout.addWidget(log_check)
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Clear the four corners and start over")
        reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(reset_btn)
        layout.addStretch()
        return widget

    def _on_show_rectified_toggled(self, checked: bool) -> None:
        if self._state is not None:
            self._state.show_rectified = checked
        self._redraw()

    def _on_reset(self) -> None:
        if self._state is not None:
            self._state.corners = []
            self._state.H = None
            self._state.show_rectified = True
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
        if event_type != "press":
            return False
        if not hasattr(document, "map_to_image_coords"):
            return False
        pt = document.map_to_image_coords(x, y)
        if pt is None:
            return False
        ix, iy = pt
        s = self._state
        K = self._K

        if len(s.corners) >= 4:
            return True

        letter = "ABCD"[len(s.corners)]
        s.corners.append((float(ix), float(iy)))
        self._log(f"Clicked {letter} at ({ix:.1f}, {iy:.1f})")

        if len(s.corners) == 4:
            vx, vy = _vanishing_points_from_rectangle(
                s.corners[0], s.corners[1], s.corners[2], s.corners[3],
            )
            if vx is None or vy is None:
                s.corners.pop()
                self._log("Failed to compute vanishing points (parallel edges?).")
                self._redraw()
                return True
            H = _homography_from_orthogonal_vanishing_points(K, vx, vy)
            if H is not None:
                s.H = H
                self._log("A B C D set. Homography H = K inv(R) K^{-1} computed; showing rectified image.")
            else:
                s.corners.pop()
                self._log("Failed to compute homography (degenerate configuration?).")

        self._redraw()
        return True
