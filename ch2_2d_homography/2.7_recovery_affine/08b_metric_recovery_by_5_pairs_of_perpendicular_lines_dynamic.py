"""
same function as 08_metric_recovery_by_5_pairs_of_perpendicular_lines.py

but let user drag the 15 points like ch2\2.7_recovery_affine\05_homography_research_gui.py

and dynamicall update the transformed image
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
# to avoid on windows
# File "...\matplotlib\backends\backend_qt.py", line 166, in _may_clear_sock rsock.recv(1) OSError:

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


DEFAULT_IMAGE_PATH = os.path.join("ch2", "2.3_projective_transform", "building.jpg")
DEFAULT_POINTS_JSON = os.path.join(
    "ch2",
    "2.7_recovery_affine",
    "08_metric_recovery_by_5_pairs_of_perpendicular_lines.points.json",
)


def _try_load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    return img


def load_image_with_fallbacks(image_path: str) -> np.ndarray:
    candidates = [
        image_path,
        os.path.basename(image_path),
        DEFAULT_IMAGE_PATH,
    ]
    for p in candidates:
        img = _try_load_image(p)
        if img is not None:
            return img
    raise FileNotFoundError(
        f"Could not load image. Tried: {candidates}. "
        f"Working directory: {os.getcwd()}"
    )


def save_points_to_json(points: np.ndarray, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data = {
        "num_points": int(points.shape[0]),
        "points": points.astype(float).tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[cache] saved points -> {json_path}")


def load_points_from_json(json_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        points_list = data.get("points", [])
        num_points = int(data.get("num_points", len(points_list)))
        if num_points != 20 or len(points_list) != 20:
            print(f"[cache] invalid points count in {json_path}: {num_points}")
            return None
        pts = np.array(points_list, dtype=np.float64)
        if pts.shape != (20, 2):
            print(f"[cache] invalid points shape in {json_path}: {pts.shape}")
            return None
        print(f"[cache] loaded points <- {json_path}")
        return pts
    except Exception as e:
        print(f"[cache] failed to load {json_path}: {e}")
        return None


def point_to_homogeneous(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], p[1], 1.0], dtype=np.float64)


def line_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Line in homogeneous coordinates from two points: l = p1 x p2.

    We normalize by sqrt(l1^2 + l2^2) for numerical stability (scale-invariant).
    """
    l = np.cross(point_to_homogeneous(p1), point_to_homogeneous(p2)).astype(np.float64)
    n = np.hypot(l[0], l[1])
    if n > 1e-12:
        l = l / n
    return l


def solve_conic_from_perpendicular_line_pairs(line_pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Solve for symmetric 3x3 conic C from constraints l^T C m = 0.

    C = [[c11 c12 c13],
         [c12 c22 c23],
         [c13 c23 c33]]

    Each constraint contributes one row to A such that A @ c = 0 where:
      c = [c11, c12, c13, c22, c23, c33]^T
    """
    A = []
    for l, m in line_pairs:
        l1, l2, l3 = l
        m1, m2, m3 = m
        A.append(
            [
                l1 * m1,
                l1 * m2 + l2 * m1,
                l1 * m3 + l3 * m1,
                l2 * m2,
                l2 * m3 + l3 * m2,
                l3 * m3,
            ]
        )

    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] < 5:
        raise ValueError(f"Need at least 5 pairs, got {A.shape[0]}")

    _, _, Vt = np.linalg.svd(A)
    c = Vt[-1, :]
    c11, c12, c13, c22, c23, c33 = c

    C = np.array(
        [
            [c11, c12, c13],
            [c12, c22, c23],
            [c13, c23, c33],
        ],
        dtype=np.float64,
    )
    # enforce symmetry + stable scale (conic is up to scale anyway)
    C = (C + C.T) / 2.0
    n = np.linalg.norm(C, ord="fro")
    if n > 1e-12:
        C = C / n
    return C


def extract_v_and_KKT_from_C(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Based on the (intended) structure used in the chapter notes:
      C = [[KKT,      KKT*v],
           [v^T*KKT,  v^T*KKT*v]]

    So v solves: KKT * v = C[:2, 2].
    """
    C = (C + C.T) / 2.0
    KKT = (C[:2, :2] + C[:2, :2].T) / 2.0
    b = C[:2, 2]
    try:
        v = np.linalg.solve(KKT, b)
    except np.linalg.LinAlgError:
        v = np.linalg.pinv(KKT) @ b
    return v, KKT


def decompose_KKT_to_upper_K(KKT: np.ndarray) -> np.ndarray:
    """
    Find upper-triangular K such that K K^T ~= KKT.

    We first symmetrize, then make it PD by eigenvalue clipping if needed,
    then use Cholesky: KKT = L L^T => set K = L^T (upper).
    """
    KKT = (KKT + KKT.T) / 2.0

    # eigenvalue clip to ensure positive-definite
    w, V = np.linalg.eigh(KKT)
    w_clipped = np.maximum(w, 1e-10)
    KKT_pd = V @ np.diag(w_clipped) @ V.T
    KKT_pd = (KKT_pd + KKT_pd.T) / 2.0

    L = np.linalg.cholesky(KKT_pd)  # lower
    K = L.T  # upper
    return K


def construct_homography_from_K_and_v(K: np.ndarray, v: np.ndarray) -> np.ndarray:
    Ha = np.eye(3, dtype=np.float64)
    Ha[:2, :2] = K

    Hp = np.eye(3, dtype=np.float64)
    Hp[2, :2] = v.reshape(2)

    H = Hp @ Ha
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def compute_rectification(
    image_bgr: np.ndarray, points: np.ndarray, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns: (rectified_rgb, H_world_to_image)
    Rectified image is produced by warping with inv(H).
    """
    if points.shape != (20, 2):
        raise ValueError(f"Expected points shape (20,2), got {points.shape}")

    # build 5 perpendicular pairs: (AiBi) ⟂ (CiDi)
    line_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(5):
        A = points[i * 4 + 0]
        B = points[i * 4 + 1]
        C = points[i * 4 + 2]
        D = points[i * 4 + 3]
        l_ab = line_from_points(A, B)
        l_cd = line_from_points(C, D)
        line_pairs.append((l_ab, l_cd))

    C = solve_conic_from_perpendicular_line_pairs(line_pairs)
    v, KKT = extract_v_and_KKT_from_C(C)
    K = decompose_KKT_to_upper_K(KKT)
    H = construct_homography_from_K_and_v(K, v)

    H_inv = np.linalg.inv(H)
    if abs(H_inv[2, 2]) > 1e-12:
        H_inv = H_inv / H_inv[2, 2]

    h, w = image_bgr.shape[:2]
    Hs = np.diag([float(scale), float(scale), 1.0])
    rect_bgr = cv2.warpPerspective(image_bgr, Hs @ H_inv, (w, h))
    rect_rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
    return rect_rgb, H


def select_points_middle_click(image_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback point picker (same convention as 08):
    MIDDLE mouse button clicks 20 points: A1 B1 C1 D1 ... A5 B5 C5 D5
    """
    pts: list[list[float]] = []
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        "Click 20 points with MIDDLE mouse: A1 B1 C1 D1 ... A5 B5 C5 D5",
        fontsize=12,
    )
    ax.axis("off")

    labels = ["A", "B", "C", "D"]

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button != 2:
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(pts) >= 20:
            return

        x, y = float(event.xdata), float(event.ydata)
        pts.append([x, y])
        quad_i = (len(pts) - 1) // 4
        p_i = (len(pts) - 1) % 4
        label = f"{labels[p_i]}{quad_i + 1}"

        ax.plot([x], [y], marker="x", markersize=14, markeredgewidth=3, color="lime")
        ax.text(x + 10, y - 10, label, color="lime", fontsize=12, weight="bold")
        fig.canvas.draw_idle()

        print(f"{label}: ({x:.1f}, {y:.1f})")
        if len(pts) == 20:
            print("All 20 points selected. Close the window to continue...")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if len(pts) != 20:
        raise RuntimeError(f"Need 20 points, got {len(pts)}")
    return np.asarray(pts, dtype=np.float64)


@dataclass
class DragState:
    idx: Optional[int] = None
    is_dragging: bool = False


class MetricRecoveryGUI:
    def __init__(self, image_bgr: np.ndarray, points: np.ndarray, points_json_path: str):
        self.image_bgr = image_bgr
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.h, self.w = image_bgr.shape[:2]

        self.points = points.astype(np.float64).copy()
        self.points_json_path = points_json_path

        self.drag = DragState()
        self._last_update_ts = 0.0
        self._min_update_dt = 0.05  # seconds (throttle to keep UI responsive)
        self.scale = 0.6

        self.fig, (self.ax_src, self.ax_dst) = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.canvas.manager.set_window_title("08b metric recovery (drag points)")
        self.fig.subplots_adjust(bottom=0.16)

        # left: source image + draggable points + segments
        self.ax_src.imshow(self.image_rgb)
        self.ax_src.set_title("Source (drag points). Keys: s=save, r=reload, esc=quit", fontsize=11)
        self.ax_src.axis("off")

        # right: rectified image
        self.ax_dst.axis("off")
        self.ax_dst.set_title("Rectified (updates live)", fontsize=11)
        self.rect_im = self.ax_dst.imshow(np.zeros_like(self.image_rgb))
        self.err_text = self.ax_dst.text(
            0.02,
            0.98,
            "",
            transform=self.ax_dst.transAxes,
            va="top",
            ha="left",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.4, pad=6),
        )
        self.err_text.set_visible(False)

        # visuals for points
        self.scatter = self.ax_src.scatter(
            self.points[:, 0],
            self.points[:, 1],
            marker="x",
            s=220,
            linewidths=3,
            c="lime",
            picker=False,
        )

        self.labels_text = []
        lab = ["A", "B", "C", "D"]
        for i in range(5):
            for j in range(4):
                idx = i * 4 + j
                t = self.ax_src.text(
                    self.points[idx, 0] + 10,
                    self.points[idx, 1] - 10,
                    f"{lab[j]}{i + 1}",
                    color="lime",
                    fontsize=10,
                    weight="bold",
                )
                self.labels_text.append(t)

        # visuals for segments: AB and CD for each of 5 quads
        colors = ["lime", "cyan", "magenta", "yellow", "orange"]
        self.seg_lines = []
        for i in range(5):
            c = colors[i % len(colors)]
            (line_ab,) = self.ax_src.plot([0, 0], [0, 0], color=c, linewidth=3, alpha=0.8)
            (line_cd,) = self.ax_src.plot([0, 0], [0, 0], color=c, linewidth=3, alpha=0.8)
            self.seg_lines.append((line_ab, line_cd))

        self._update_segments()
        self._update_rectified(force=True)

        # slider: rectified view scale (zoom)
        ax_scale = self.fig.add_axes([0.25, 0.06, 0.5, 0.03])
        self.scale_slider = Slider(
            ax=ax_scale,
            label="Scale",
            valmin=0.1,
            valmax=2.0,
            valinit=self.scale,
            valstep=0.01,
        )
        self.scale_slider.on_changed(self._on_scale_changed)

        # events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_segments(self) -> None:
        for i in range(5):
            A = self.points[i * 4 + 0]
            B = self.points[i * 4 + 1]
            C = self.points[i * 4 + 2]
            D = self.points[i * 4 + 3]
            line_ab, line_cd = self.seg_lines[i]
            line_ab.set_data([A[0], B[0]], [A[1], B[1]])
            line_cd.set_data([C[0], D[0]], [C[1], D[1]])

    def _update_point_artists(self) -> None:
        self.scatter.set_offsets(self.points)
        for idx, t in enumerate(self.labels_text):
            t.set_position((self.points[idx, 0] + 10, self.points[idx, 1] - 10))
        self._update_segments()

    def _pick_point_index(self, event) -> Optional[int]:
        if event.inaxes != self.ax_src:
            return None
        if event.x is None or event.y is None:
            return None

        # pixel-distance picking
        pts_disp = self.ax_src.transData.transform(self.points)
        dx = pts_disp[:, 0] - event.x
        dy = pts_disp[:, 1] - event.y
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        if d2[idx] <= (14.0 * 14.0):
            return idx
        return None

    def _set_error(self, msg: str) -> None:
        if msg:
            self.err_text.set_text(msg)
            self.err_text.set_visible(True)
        else:
            self.err_text.set_visible(False)

    def _update_rectified(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_update_ts) < self._min_update_dt:
            return
        self._last_update_ts = now

        try:
            rect_rgb, H = compute_rectification(self.image_bgr, self.points, scale=self.scale)
            self.rect_im.set_data(rect_rgb)
            self.ax_dst.set_title(
                "Rectified (updates live)\n"
                f"scale={self.scale:.2f}  H[2,:2]={H[2,0]:+.3e}, {H[2,1]:+.3e}",
                fontsize=10,
            )
            self._set_error("")
        except Exception as e:
            self._set_error(f"rectification failed: {type(e).__name__}: {e}")

    def _on_scale_changed(self, val: float) -> None:
        self.scale = float(val)
        self._update_rectified(force=True)
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.button != 1:  # left click drag
            return
        idx = self._pick_point_index(event)
        if idx is None:
            return
        self.drag.idx = idx
        self.drag.is_dragging = True

    def _on_motion(self, event) -> None:
        if not self.drag.is_dragging or self.drag.idx is None:
            return
        if event.inaxes != self.ax_src:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.points[self.drag.idx, 0] = float(event.xdata)
        self.points[self.drag.idx, 1] = float(event.ydata)
        self._update_point_artists()
        self._update_rectified(force=False)
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if not self.drag.is_dragging:
            return
        self.drag.is_dragging = False
        self.drag.idx = None
        self._update_rectified(force=True)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ("escape", "q"):
            plt.close(self.fig)
            return

        if event.key == "s":
            try:
                save_points_to_json(self.points, self.points_json_path)
            except Exception as e:
                print(f"[cache] save failed: {e}")
            return

        if event.key == "r":
            pts = load_points_from_json(self.points_json_path)
            if pts is None:
                print("[cache] reload failed (file missing or invalid)")
                return
            self.points = pts.astype(np.float64).copy()
            self._update_point_artists()
            self._update_rectified(force=True)
            self.fig.canvas.draw_idle()
            return

    def show(self) -> None:
        plt.tight_layout()
        plt.show()


def main():
    image_bgr = load_image_with_fallbacks(DEFAULT_IMAGE_PATH)

    points_json = DEFAULT_POINTS_JSON
    pts = load_points_from_json(points_json)
    if pts is None:
        print("[cache] no valid cached points, falling back to point picking...")
        pts = select_points_middle_click(image_bgr)
        save_points_to_json(pts, points_json)

    gui = MetricRecoveryGUI(image_bgr=image_bgr, points=pts, points_json_path=points_json)
    gui.show()


if __name__ == "__main__":
    main()
