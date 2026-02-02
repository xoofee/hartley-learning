"""
load chessboard1.jpg and chessboard2.jpg
which is taken from different views of the same chessboard, that is an A4 9x6 30cm grid chessboard in landscape view (6x9 internal corners) shown on fullscreen of Dell 5001 laptop.

find the chessboard corners in both images, find the optimal homography and corresponding image points by iterative optimization.

output these errors:

Linear normalized
Gold Standard 
Linear unnormalized
Homogeneous scaling
Sampson
Error in 1 view

visualize the detected corners in both images.

reference from the book:

Algorithm 4.3. The Gold Standard algorithm and variations for estimating H from image correspondences.
The Gold Standard algorithm is preferred to the Sampson method for 2D homography computation.

---

OBJECTIVE

Given more than four image point correspondences
{x_i <-> x'_i},
determine the Maximum Likelihood (ML) estimate of the homography H between two images.

The ML estimation also solves for a set of corrected (latent) image points
{ x̂_i , x̂'_i }.

The goal is to minimize the total reprojection error:

```
sum over i of:
    d(x_i , x̂_i)^2 + d(x'_i , x̂'_i)^2
```

subject to the constraint:

```
x̂'_i = H * x̂_i
```

---

ALGORITHM

1. Initialization

Compute an initial estimate of the homography H.
This provides a starting point for geometric optimization.

Common choices:

* Normalized DLT algorithm
* RANSAC using four point correspondences

---

2. Geometric minimization

Two alternative approaches can be used.

---

Option A: Sampson error (approximate geometric error)

* Minimize the Sampson approximation of the geometric reprojection error.
* Use Newton’s method or Levenberg–Marquardt (LM) optimization.
* The homography H is parameterized directly by its 9 matrix entries.

This method is faster, but only approximates the true geometric error.

---

Option B: Gold Standard method (true ML solution)

Step 1: Initialize corrected points

* Use the measured points {x_i, x'_i}, or
* Preferably use Sampson-corrected points as the initial estimate.

Step 2: Joint optimization
Minimize the cost function:

```
sum over i of:
    d(x_i , x̂_i)^2 + d(x'_i , x̂'_i)^2
```

Optimization variables:

* The homography H (9 parameters)
* The corrected image points x̂_i (2 parameters per point)

Total number of variables:

* 2n parameters for corrected points
* 9 parameters for the homography

Step 3: Optimization method

* Use Levenberg–Marquardt.

Step 4: Large-scale case

* If the number of points is large, use a sparse optimization method.
* This is the recommended approach for efficiency.

Although this will be a single python file, it make it modular, extensible and clean

do not modify/remove the above comments.

"""

from __future__ import annotations

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import Tuple

# ---------------------------------------------------------------------------
# Data loading and corner detection
# ---------------------------------------------------------------------------

def load_images(path1: str, path2: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load two images from disk."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Cannot load images: {path1}, {path2}")
    return img1, img2


def find_chessboard_corners(
    img: np.ndarray,
    pattern_size: Tuple[int, int] = (6, 9),
    subpix_window: int = 5,
    flags: int | None = None,
) -> np.ndarray:
    """
    Detect chessboard corners and refine to subpixel accuracy.
    Returns (N, 2) array of corner coordinates in image (x, y).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if flags is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found and gray.size > 0:
        gray_eq = cv2.equalizeHist(gray)
        found_eq, corners_eq = cv2.findChessboardCorners(gray_eq, pattern_size, flags)
        if found_eq:
            found, corners, gray = True, corners_eq, gray_eq
    if not found:
        raise ValueError(
            f"Chessboard corners not found in image for pattern size {pattern_size}."
        )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (subpix_window, subpix_window),
        (-1, -1),
        criteria,
    )
    return corners.reshape(-1, 2)


_FLAG_OPTIONS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    0,
)


def find_chessboard_corners_auto(
    img: np.ndarray,
    pattern_candidates: Tuple[Tuple[int, int], ...] = ((5, 8)),
    subpix_window: int = 5,
) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """
    Detect chessboard corners trying multiple pattern sizes and flags.
    Returns (corners (N, 2), pattern_size_used, flags_used).
    """
    for size in pattern_candidates:
        for fl in _FLAG_OPTIONS:
            try:
                corners = find_chessboard_corners(img, size, subpix_window, flags=fl)
                return corners, size, fl
            except ValueError:
                continue
    raise ValueError(
        f"Chessboard corners not found for any of {pattern_candidates}."
    )


# ---------------------------------------------------------------------------
# Point normalization (Hartley)
# ---------------------------------------------------------------------------

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hartley normalization for 2D points.
    points: (N, 2)
    Returns:
        points_norm: (N, 2)
        T: (3, 3) normalization matrix
    """
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


# ---------------------------------------------------------------------------
# DLT homography
# ---------------------------------------------------------------------------

def _build_dlt_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Build design matrix A such that A h = 0 (h = vec(H)). src, dst: (N, 2)."""
    n = src.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src[i, 0], src[i, 1]
        u, v = dst[i, 0], dst[i, 1]
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]
    return A


def dlt_unnormalized(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Homography H from point correspondences using unnormalized DLT (SVD)."""
    A = _build_dlt_matrix(src_points, dst_points)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2] if np.abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    return H


def dlt_normalized(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Homography H using normalized DLT (Algorithm 4.2)."""
    src_n, T_src = normalize_points(src_points)
    dst_n, T_dst = normalize_points(dst_points)
    A = _build_dlt_matrix(src_n, dst_n)
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[2, 2] if np.abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    return H


# ---------------------------------------------------------------------------
# Algebraic residual and homogeneous scaling
# ---------------------------------------------------------------------------

def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N, 2) -> (N, 3) with last column 1."""
    return np.hstack([pts, np.ones((pts.shape[0], 1))])


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply H to (N, 2) points; return (N, 2) Euclidean."""
    pts_h = to_homogeneous(pts)
    out = (H @ pts_h.T).T
    w = out[:, 2]
    out = out[:, :2] / (np.abs(w)[:, np.newaxis] + 1e-12)
    return out


def algebraic_residual(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point algebraic residual x' × Hx as (N, 2)."""
    src_h = to_homogeneous(src)
    dst_h = to_homogeneous(dst)
    Hx = (H @ src_h.T).T  # (N, 3)
    # Cross product (x', y', 1) × (Hx, Hy, Hz): two components
    eps = np.zeros((src.shape[0], 2))
    eps[:, 0] = dst_h[:, 1] * Hx[:, 2] - dst_h[:, 2] * Hx[:, 1]  # v'*w - w'*Hy
    eps[:, 1] = dst_h[:, 2] * Hx[:, 0] - dst_h[:, 0] * Hx[:, 2]  # w'*Hx - u'*w
    return eps


def rms_algebraic(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """RMS of algebraic error (2 components per point)."""
    eps = algebraic_residual(H, src, dst)
    return np.sqrt(np.mean(eps ** 2))


# ---------------------------------------------------------------------------
# Reprojection and one-view errors
# ---------------------------------------------------------------------------

def reprojection_error_symmetric(
    H: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> float:
    """Total reprojection error: sum_i d(x_i, x̂_i)^2 + d(x'_i, x̂'_i)^2 with x̂'_i = H x̂_i.
    We use measured points as proxy for corrected (no iterative correction here).
    So this is sum_i d(x'_i, H x_i)^2 + d(x_i, H^{-1} x'_i)^2 (symmetric transfer error).
    """
    dst_pred = apply_homography(H, src)
    src_pred = apply_homography(np.linalg.inv(H), dst)
    e2 = np.sum((dst - dst_pred) ** 2) + np.sum((src - src_pred) ** 2)
    n = src.shape[0]
    return np.sqrt(e2 / (2 * n))


def reprojection_error_one_view(
    H: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    view: str = "second",
) -> float:
    """RMS reprojection error in one view only. view='second': ||x' - H*x||; 'first': ||x - H^{-1}*x'||."""
    if view == "second":
        dst_pred = apply_homography(H, src)
        diff = dst - dst_pred
    else:
        src_pred = apply_homography(np.linalg.inv(H), dst)
        diff = src - src_pred
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Sampson error (first-order approximation to geometric error)
# ---------------------------------------------------------------------------

def sampson_residual_per_point(
    H: np.ndarray,
    x: float,
    y: float,
    u: float,
    v: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Algebraic residual (2,) and Jacobian J (2,4) for one correspondence."""
    h11, h12, h13 = H[0, 0], H[0, 1], H[0, 2]
    h21, h22, h23 = H[1, 0], H[1, 1], H[1, 2]
    h31, h32, h33 = H[2, 0], H[2, 1], H[2, 2]
    wx = h31 * x + h32 * y + h33
    hx = h11 * x + h12 * y + h13
    hy = h21 * x + h22 * y + h23
    eps1 = v * wx - hy
    eps2 = hx - u * wx
    eps = np.array([eps1, eps2])
    J = np.array([
        [v * h31 - h21, v * h32 - h22, 0, wx],
        [h11 - u * h31, h12 - u * h32, -wx, 0],
    ])
    return eps, J


def sampson_error_rms(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """RMS Sampson (approximate geometric) error."""
    n = src.shape[0]
    total = 0.0
    for i in range(n):
        eps, J = sampson_residual_per_point(
            H, src[i, 0], src[i, 1], dst[i, 0], dst[i, 1]
        )
        JJT = J @ J.T
        total += eps @ np.linalg.solve(JJT, eps)
    return np.sqrt(total / n)


def sampson_corrected_points(
    H: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order corrected points (x̂, x̂') using Sampson step."""
    n = src.shape[0]
    src_c = src.copy()
    dst_c = dst.copy()
    for i in range(n):
        eps, J = sampson_residual_per_point(
            H, src[i, 0], src[i, 1], dst[i, 0], dst[i, 1]
        )
        JJT = J @ J.T
        delta = np.linalg.solve(JJT, eps)  # (2,)
        # J is 2x4: d(eps)/d(x,y,u,v). Delta in (eps1, eps2) space; we need delta in (x,y,u,v).
        # Minimizer of ||delta_param||^2 s.t. J @ delta_param = -eps is delta_param = -J^T (J J^T)^{-1} eps
        delta_param = -J.T @ np.linalg.solve(JJT, eps)
        src_c[i] += delta_param[:2]
        dst_c[i] += delta_param[2:]
    return src_c, dst_c


# ---------------------------------------------------------------------------
# Gold Standard: minimize sum of squared reprojection errors over H and x̂_i
# ---------------------------------------------------------------------------

def _gold_standard_residual(
    params: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
) -> np.ndarray:
    """Residual vector: [x_1 - x̂_1, y_1 - ŷ_1, x'_1 - x̂'_1, y'_1 - ŷ'_1, ...]."""
    H = params[:9].reshape(3, 3)
    x_hat = params[9:].reshape(n, 2)
    x_hat_prime = apply_homography(H, x_hat)
    r = np.zeros(4 * n)
    for i in range(n):
        r[4 * i] = src[i, 0] - x_hat[i, 0]
        r[4 * i + 1] = src[i, 1] - x_hat[i, 1]
        r[4 * i + 2] = dst[i, 0] - x_hat_prime[i, 0]
        r[4 * i + 3] = dst[i, 1] - x_hat_prime[i, 1]
    return r


def gold_standard_homography(
    src: np.ndarray,
    dst: np.ndarray,
    H_init: np.ndarray,
    x_hat_init: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gold Standard ML estimate: minimize sum_i d(x_i, x̂_i)^2 + d(x'_i, x̂'_i)^2
    subject to x̂'_i = H x̂_i. Returns (H_opt, x̂_opt).
    """
    n = src.shape[0]
    if x_hat_init is None:
        x_hat_init = src.copy()
    x0 = np.hstack([H_init.ravel(), x_hat_init.ravel()])

    def residual(p: np.ndarray) -> np.ndarray:
        return _gold_standard_residual(p, src, dst, n)

    res = least_squares(residual, x0, method="lm", max_nfev=2000)
    params = res.x
    H_opt = params[:9].reshape(3, 3)
    H_opt /= H_opt[2, 2]
    x_hat_opt = params[9:].reshape(n, 2)
    return H_opt, x_hat_opt


def gold_standard_error_rms(
    src: np.ndarray,
    dst: np.ndarray,
    x_hat: np.ndarray,
    x_hat_prime: np.ndarray,
) -> float:
    """RMS of sqrt( (x-x̂)^2 + (y-ŷ)^2 + (x'-x̂')^2 + (y'-ŷ')^2 ) over 2n dimensions -> per-point RMS."""
    d1 = np.sum((src - x_hat) ** 2, axis=1)
    d2 = np.sum((dst - x_hat_prime) ** 2, axis=1)
    return np.sqrt(np.mean(d1 + d2))


# ---------------------------------------------------------------------------
# Sampson optimization: minimize sum of Sampson squared errors over H
# ---------------------------------------------------------------------------

def _sampson_residual_vec(H_flat: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Residual = per-point Sampson contribution (sqrt of Sampson^2) for LM."""
    H = H_flat.reshape(3, 3)
    n = src.shape[0]
    r = []
    for i in range(n):
        eps, J = sampson_residual_per_point(
            H, src[i, 0], src[i, 1], dst[i, 0], dst[i, 1]
        )
        JJT = J @ J.T
        r.append(np.sqrt(eps @ np.linalg.solve(JJT, eps)))
    return np.array(r)


def homography_sampson_optimized(src: np.ndarray, dst: np.ndarray, H_init: np.ndarray) -> np.ndarray:
    """Refine H by minimizing sum of Sampson squared errors (LM)."""
    res = least_squares(
        lambda p: _sampson_residual_vec(p, src, dst),
        H_init.ravel(),
        method="lm",
        max_nfev=1000,
    )
    H = res.x.reshape(3, 3)
    H /= H[2, 2]
    return H


# ---------------------------------------------------------------------------
# Homogeneous scaling: H scaled so ||H||_F = 1, then algebraic RMS
# ---------------------------------------------------------------------------

def homogeneous_scaling_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """Algebraic RMS when H is scaled to unit Frobenius norm."""
    H_scaled = H / (np.linalg.norm(H) + 1e-12)
    return rms_algebraic(H_scaled, src, dst)


# ---------------------------------------------------------------------------
# Main: load, estimate, report errors, visualize
# ---------------------------------------------------------------------------

def run_pipeline(
    img1_path: str = "chessboard1.jpg",
    img2_path: str = "chessboard2.jpg",
    pattern_size: Tuple[int, int] = (9, 6),
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(script_dir, img1_path)
    path2 = os.path.join(script_dir, img2_path)

    img1, img2 = load_images(path1, path2)
    pattern_candidates = (
        pattern_size,
        (pattern_size[1], pattern_size[0]),
        (5, 8),
    )
    pts1, pattern_used, flags_used = find_chessboard_corners_auto(img1, pattern_candidates)
    # Use same pattern size; try both flag options for img2 in case detection differs
    pts2 = None
    for fl in _FLAG_OPTIONS:
        try:
            pts2 = find_chessboard_corners(img2, pattern_used, flags=fl)
            break
        except ValueError:
            continue
    if pts2 is None:
        raise ValueError(
            f"Chessboard corners not found in second image for pattern size {pattern_used}."
        )
    n = pts1.shape[0]
    assert pts2.shape[0] == n

    # Reference: image 1 = src (x), image 2 = dst (x')
    src, dst = pts1.astype(np.float64), pts2.astype(np.float64)

    # ----- Initial estimates -----
    H_unnorm = dlt_unnormalized(src, dst)
    H_norm = dlt_normalized(src, dst)

    # Linear normalized: H from normalized DLT, then reprojection error in original coords
    err_linear_normalized = reprojection_error_symmetric(H_norm, src, dst)

    # Linear unnormalized
    err_linear_unnormalized = reprojection_error_symmetric(H_unnorm, src, dst)

    # Homogeneous scaling: H scaled to unit norm, algebraic RMS
    err_homogeneous_scaling = homogeneous_scaling_error(H_norm, src, dst)

    # Sampson: refine H by minimizing Sampson error, then report Sampson RMS
    H_sampson = homography_sampson_optimized(src, dst, H_norm.copy())
    err_sampson = sampson_error_rms(H_sampson, src, dst)

    # Gold Standard: joint optimization over H and x̂_i
    src_sampson, dst_sampson = sampson_corrected_points(H_norm, src, dst)
    H_gold, x_hat = gold_standard_homography(src, dst, H_norm.copy(), src_sampson.copy())
    x_hat_prime = apply_homography(H_gold, x_hat)
    err_gold_standard = gold_standard_error_rms(src, dst, x_hat, x_hat_prime)

    # Error in 1 view (second image: x' vs H*x)
    err_one_view_second = reprojection_error_one_view(H_gold, src, dst, view="second")
    err_one_view_first = reprojection_error_one_view(H_gold, src, dst, view="first")

    # ----- Report -----
    print("Homography estimation from chessboard correspondences")
    print("=" * 60)
    print(f"  Linear normalized (reproj RMS):     {err_linear_normalized:.6f}")
    print(f"  Gold Standard (reproj RMS):        {err_gold_standard:.6f}")
    print(f"  Linear unnormalized (reproj RMS):   {err_linear_unnormalized:.6f}")
    print(f"  Homogeneous scaling (algebraic RMS): {err_homogeneous_scaling:.6f}")
    print(f"  Sampson (RMS):                      {err_sampson:.6f}")
    print(f"  Error in 1 view (second image):     {err_one_view_second:.6f}")
    print(f"  Error in 1 view (first image):      {err_one_view_first:.6f}")
    print("=" * 60)

    # ----- Visualize -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, pts, title in [
        (axes[0], img1, pts1, "Image 1 (corners)"),
        (axes[1], img2, pts2, "Image 2 (corners)"),
    ]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.scatter(pts[:, 0], pts[:, 1], c="lime", s=20, edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "corners_visualization.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_pipeline()