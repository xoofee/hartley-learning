"""
Feature matching for two-view reconstruction: ORB and SIFT.
Pure functions, no Qt. Used by two_view_reconstruction demo.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ....logging_ui import log_debug, log_warning

# SIFT is in main OpenCV from 4.5.1; older or minimal builds may not have it
_SIFT_AVAILABLE: bool = False
try:
    _ = cv2.SIFT_create()
    _SIFT_AVAILABLE = True
except Exception:
    pass

FEATURE_TYPE_ORB = "orb"
FEATURE_TYPE_SIFT = "sift"


def _prepare_image(img: np.ndarray, use_clahe: bool) -> np.ndarray:
    """Convert to grayscale and optionally apply CLAHE for better feature detection."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def _apply_rootsift(des: np.ndarray) -> np.ndarray:
    """Apply rootSIFT to SIFT descriptors: L1-normalize then square root. In-place friendly."""
    if des is None or des.size == 0:
        return des
    des = np.asarray(des, dtype=np.float64)
    des /= des.sum(axis=1, keepdims=True) + 1e-7
    des = np.sqrt(des)
    return des.astype(np.float32)


def get_supported_feature_types() -> list[str]:
    """Return list of supported feature type ids (e.g. ['orb', 'sift'])."""
    out = [FEATURE_TYPE_ORB]
    if _SIFT_AVAILABLE:
        out.append(FEATURE_TYPE_SIFT)
    return out


def match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    feature_type: str,
    nfeatures: int = 5000,
    use_root_sift: bool = True,
    use_clahe: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect and match features between two images.
    Returns pts1 (N,2), pts2 (N,2), mask (N,) of good matches (all True for ratio-test passed).
    feature_type: 'orb' or 'sift' (if available).
    use_root_sift: if True (default), apply rootSIFT to SIFT descriptors for better matching.
    use_clahe: if True (default), apply CLAHE to grayscale images before feature detection.
    """
    log_debug(f"Matching features: {feature_type}, {nfeatures}, CLAHE={use_clahe}")
    g1 = _prepare_image(img1, use_clahe)
    g2 = _prepare_image(img2, use_clahe)
    empty = (
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.array([], dtype=bool),
    )

    if feature_type == FEATURE_TYPE_ORB:
        det = cv2.ORB_create(nfeatures=nfeatures)
        kp1, des1 = det.detectAndCompute(g1, None)
        kp2, des2 = det.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return empty
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(des1, des2, k=2)
    elif feature_type == FEATURE_TYPE_SIFT and _SIFT_AVAILABLE:
        det = cv2.SIFT_create(
            nfeatures=nfeatures,
            # contrastThreshold=0.01,  # Lower contrastThreshold (default is 0.04) to pick up "faint" features
            # edgeThreshold=5, 
            # sigma=1.6            
            )
        kp1, des1 = det.detectAndCompute(g1, None)
        kp2, des2 = det.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return empty
        if use_root_sift:
            des1 = _apply_rootsift(des1)
            des2 = _apply_rootsift(des2)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)
        log_debug(
            f"Matching features: {feature_type}, {nfeatures} (SIFT, rootSIFT={use_root_sift}) - matches: {len(matches)}"
        )
    else:
        # Fallback to ORB if SIFT requested but unavailable
        if feature_type == FEATURE_TYPE_SIFT:
            return match_features(img1, img2, FEATURE_TYPE_ORB, nfeatures, use_root_sift, use_clahe)
        return empty

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        log_warning(f"Matching features: {feature_type}, {nfeatures} - not enough matches: {len(good)}")
        return empty
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, np.ones(len(good), dtype=bool)
