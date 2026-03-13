"""
Multi-view self-calibration: K from 3 or >=5 views (no calibration target).

Assumptions (all cases): same camera intrinsics, general motion, non-planar scene.

- 3 views: K = diag(f, f, 1) (square pixel, central principal point, no distortion).
- >=5 views: general K (fx, fy, cx, cy), no distortion.
- 4 views: not supported; call validate_view_count() to raise.

Pure functions, no Qt. Used by auto_calibration demo.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# View count
# ---------------------------------------------------------------------------


class UnsupportedViewCountError(ValueError):
    """Raised when exactly 4 views are used (unsupported)."""

    def __init__(self, n: int) -> None:
        self.n = n
        super().__init__(
            f"Auto calibration with exactly 4 views is not supported. "
            f"Use 3 views (simplified K) or 5 or more views (general K)."
        )


def validate_view_count(n: int) -> None:
    """Require n == 3 or n >= 5. Raise UnsupportedViewCountError for n == 4."""
    if n == 4:
        raise UnsupportedViewCountError(4)
    if n < 3:
        raise ValueError(f"At least 3 views required, got {n}.")
    if n not in (3,) and n < 5:
        raise ValueError(f"View count must be 3 or >= 5, got {n}.")


# ---------------------------------------------------------------------------
# Kruppa equations for IAC ω (symmetric 3x3). ω = K^{-T} K^{-1}.
# F ω F^T ∝ [e']_x ω [e']_x^T  =>  nonzero eigenvalues match up to scale.
# ---------------------------------------------------------------------------


def _epipole_right(F: np.ndarray) -> np.ndarray:
    """Right null of F: F^T e' = 0. Return (3,) normalized so e'[2] = 1 if possible."""
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    if abs(e[2]) > 1e-10:
        e = e / e[2]
    return e.ravel()


def _skew_sym(e: np.ndarray) -> np.ndarray:
    """Skew-symmetric [e]_x (3,3)."""
    return np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]], dtype=np.float64)


def _kruppa_eigenvalue_ratio(F: np.ndarray, omega: np.ndarray) -> float:
    """
    From F ω F^T and [e']_x ω [e']_x^T, the two nonzero eigenvalues should be
    in the same ratio (matrices are proportional). With ω = diag(a,a,1),
    returns the ratio (eig1/eig2) from F ω F^T (or the other matrix).
    """
    e = _epipole_right(F)
    ex = _skew_sym(e)
    A = F @ omega @ F.T
    B = ex @ omega @ ex.T
    # Both A and B are rank-2 symmetric; get two nonzero eigenvalues
    eigA = np.linalg.eigvalsh(A)
    eigB = np.linalg.eigvalsh(B)
    # Remove the zero eigenvalue (smallest in absolute value)
    eigA = np.sort(np.abs(eigA))[1:3]  # two nonzero
    eigB = np.sort(np.abs(eigB))[1:3]
    if eigA[1] < 1e-12 or eigB[1] < 1e-12:
        return np.nan
    rA = eigA[0] / eigA[1]
    rB = eigB[0] / eigB[1]
    # They should be equal; return their mean as the consistent ratio
    return (rA + rB) / 2.0


def _kruppa_residual_3view(f: float, F_list: list[np.ndarray]) -> float:
    """Residual for 3-view: ω = diag(1/f^2, 1/f^2, 1). One ratio per F; minimize variance."""
    if f <= 0:
        return 1e10
    a = 1.0 / (f * f)
    omega = np.diag([a, a, 1.0])
    ratios = []
    for F in F_list:
        r = _kruppa_eigenvalue_ratio(F, omega)
        if not np.isnan(r):
            ratios.append(r)
    if len(ratios) < 2:
        return 1e10
    return float(np.var(ratios))


# ---------------------------------------------------------------------------
# 3-view: K = diag(f, f, 1)
# ---------------------------------------------------------------------------


def calibrate_3views_from_F(
    F12: np.ndarray,
    F13: np.ndarray,
    f_guess: float = 500.0,
    f_min: float = 50.0,
    f_max: float = 10000.0,
) -> tuple[np.ndarray, float]:
    """
    Recover K = diag(f, f, 1) from two fundamental matrices (views 1-2 and 1-3).

    Returns:
        K: (3,3) with K[0,0]=K[1,1]=f, K[2,2]=1, principal point (0,0) in normalized coords
          (we use principal point at image center later if needed).
        f: recovered focal length.
    """
    F_list = [np.asarray(F12, dtype=np.float64), np.asarray(F13, dtype=np.float64)]
    # Minimize variance of Kruppa eigenvalue ratio over a grid / scalar search
    best_f = f_guess
    best_res = _kruppa_residual_3view(best_f, F_list)
    for _ in range(2):
        grid = np.linspace(max(f_min, best_f * 0.5), min(f_max, best_f * 2.0), 41)
        for f in grid:
            r = _kruppa_residual_3view(f, F_list)
            if r < best_res:
                best_res = r
                best_f = f
    f = best_f
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], dtype=np.float64)
    return K, float(f)


def calibrate_3views_from_correspondences(
    pts1: np.ndarray,
    pts2: np.ndarray,
    pts1_to_3: np.ndarray,
    pts3: np.ndarray,
    ransac_threshold: float = 1.0,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Compute F12, F13 from correspondences then recover K = diag(f,f,1).

    pts1 (N,2), pts2 (N,2): view 1 and 2.
    pts1_to_3 (M,2), pts3 (M,2): view 1 and 3 (same points as 1, different indices).
    So we need tracks (x1, x2, x3); this function takes (x1,x2) and (x1,x3) and
    runs RANSAC F estimation for both pairs.
    """
    import cv2

    F12, m12 = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=ransac_threshold, confidence=0.99
    )
    F13, m13 = cv2.findFundamentalMat(
        pts1_to_3, pts3, cv2.FM_RANSAC, ransacReprojThreshold=ransac_threshold, confidence=0.99
    )
    if F12 is None or F13 is None:
        return None, None
    K, f = calibrate_3views_from_F(F12, F13)
    return K, f


# ---------------------------------------------------------------------------
# 5+ views: general K (fx, fy, cx, cy). Use Kruppa with parameterized ω.
# ω = K^{-T} K^{-1}, K = [[fx,0,cx],[0,fy,cy],[0,0,1]] => 4 dof.
# ---------------------------------------------------------------------------


def _omega_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """IAC ω = K^{-T} K^{-1} from (fx, fy, cx, cy)."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    o = np.linalg.inv(K).T @ np.linalg.inv(K)
    return (o + o.T) / 2.0


def _kruppa_residual_general(
    params: np.ndarray,
    F_list: list[np.ndarray],
) -> np.ndarray:
    """Residuals for general K: params = (fx, fy, cx, cy). One ratio per F."""
    fx, fy, cx, cy = params
    if fx < 10 or fy < 10:
        return np.full(len(F_list), 1e10)
    omega = _omega_from_params(fx, fy, cx, cy)
    residuals = []
    for F in F_list:
        r = _kruppa_eigenvalue_ratio(F, omega)
        if np.isnan(r):
            residuals.append(0.0)  # skip
        else:
            residuals.append(r)
    if len(residuals) < 2:
        return np.array([1e10])
    r = np.array(residuals, dtype=np.float64)
    # Minimize variance of ratios (they should all be equal)
    mean_r = np.mean(r)
    return (r - mean_r).ravel()


def calibrate_general_k_from_F(
    F_list: list[np.ndarray],
    image_size: tuple[int, int],
    fx_fy_guess: float = 500.0,
) -> Optional[np.ndarray]:
    """
    Recover general K (no distortion) from multiple F matrices (>=5 views recommended).

    F_list: e.g. [F12, F13, F14, F23, F24, ...] (at least 2 for 4 dof).
    image_size: (width, height) to center principal point initial guess.
    """
    from scipy.optimize import least_squares

    w, h = image_size
    cx0, cy0 = w / 2.0, h / 2.0
    x0 = np.array([fx_fy_guess, fx_fy_guess, cx0, cy0])

    def residual(p: np.ndarray) -> np.ndarray:
        return _kruppa_residual_general(p, F_list)

    res = least_squares(residual, x0, bounds=([50, 50, 0, 0], [20000, 20000, w, h]), method="trf")
    if not res.success:
        return None
    fx, fy, cx, cy = res.x
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K

