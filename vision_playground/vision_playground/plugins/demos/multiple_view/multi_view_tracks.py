"""
Build multi-view feature tracks from pairwise matches.
Pure functions, no Qt. Used by auto_calibration demo.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# from ch7_computation_of_P.vision_playground.vision_playground.logging_ui import log_debug

from ....logging_ui import log_debug, log_warning


from .feature_matching import FEATURE_TYPE_SIFT, match_features


def build_tracks_3views(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    feature_type: str = FEATURE_TYPE_SIFT,
    nfeatures: int = 5000,
    merge_tol_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tracks (x1, x2, x3) by matching view 1-2 and view 1-3, then merging on view-1.

    Returns pts1 (N,2), pts2 (N,2), pts3 (N,2) for N tracks. View-1 is the reference.
    """
    log_debug(f"Building tracks for 3 views: {feature_type}, {nfeatures}, {merge_tol_px}")
    p1_12, p2_12, _ = match_features(img1, img2, feature_type, nfeatures)
    p1_13, p3_13, _ = match_features(img1, img3, feature_type, nfeatures)
    if p1_12.shape[0] < 8 or p1_13.shape[0] < 8:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    # Merge: same point in view 1 (within merge_tol_px) => one track (p1, p2, p3)
    tol2 = merge_tol_px * merge_tol_px
    tracks_p1, tracks_p2, tracks_p3 = [], [], []
    used_13 = set()
    for i in range(p1_12.shape[0]):
        x1, y1 = p1_12[i, 0], p1_12[i, 1]
        best_j = -1
        best_d2 = tol2 + 1
        for j in range(p1_13.shape[0]):
            if j in used_13:
                continue
            dx = p1_13[j, 0] - x1
            dy = p1_13[j, 1] - y1
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j >= 0:
            used_13.add(best_j)
            tracks_p1.append([x1, y1])
            tracks_p2.append([p2_12[i, 0], p2_12[i, 1]])
            tracks_p3.append([p3_13[best_j, 0], p3_13[best_j, 1]])
    if len(tracks_p1) < 8:
        log_warning(f"Building tracks for 3 views: not enough tracks: {len(tracks_p1)}")
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    pts1 = np.float32(tracks_p1)
    pts2 = np.float32(tracks_p2)
    pts3 = np.float32(tracks_p3)
    return pts1, pts2, pts3


def build_tracks_nviews(
    images: List[np.ndarray],
    feature_type: str = "orb",
    nfeatures: int = 5000,
    merge_tol_px: float = 2.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build tracks across n views by chaining matches 0-1, 1-2, ..., n-2-n-1 and merging.

    Returns list of n arrays, each (N,2) for N tracks; and track_mask (N,) bool (all True).
    """
    n = len(images)
    if n < 2:
        return [np.zeros((0, 2), dtype=np.float32)] * n, np.array([], dtype=bool)

    # Chain: match 0-1, then extend with 1-2 (merge on view 1), then 2-3 (merge on view 2), etc.
    pts_per_view: List[List[list]] = [[] for _ in range(n)]
    # Start with match 0-1
    p0, p1, _ = match_features(images[0], images[1], feature_type, nfeatures)
    if p0.shape[0] < 8:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(n)], np.array([], dtype=bool)
    for i in range(p0.shape[0]):
        pts_per_view[0].append(list(p0[i]))
        pts_per_view[1].append(list(p1[i]))

    tol2 = merge_tol_px * merge_tol_px
    for v in range(1, n - 1):
        pa, pb, _ = match_features(images[v], images[v + 1], feature_type, nfeatures)
        if pa.shape[0] < 8:
            continue
        # For each existing track we have pts_per_view[0..v] filled; we need to extend with view v+1.
        # Current tracks have length v+1; we have pts_per_view[v] as list of [x,y].
        existing_v = np.array(pts_per_view[v])  # (M, 2)
        # Match existing_v to pa: for each row in existing_v find closest in pa
        extended = 0
        for i in range(len(pts_per_view[0])):
            if len(pts_per_view[v + 1]) > i:
                continue  # already extended this track
            xv, yv = pts_per_view[v][i][0], pts_per_view[v][i][1]
            best_j = -1
            best_d2 = tol2 + 1
            for j in range(pa.shape[0]):
                dx = pa[j, 0] - xv
                dy = pa[j, 1] - yv
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
            if best_j >= 0:
                pts_per_view[v + 1].append([pb[best_j, 0], pb[best_j, 1]])
                extended += 1
        # Tracks that didn't extend get a placeholder so indices match; we'll drop incomplete later
        while len(pts_per_view[v + 1]) < len(pts_per_view[0]):
            pts_per_view[v + 1].append([np.nan, np.nan])

    # Drop tracks that don't have all views (nan in any view)
    valid = []
    for i in range(len(pts_per_view[0])):
        ok = True
        for v in range(n):
            if v < len(pts_per_view) and i < len(pts_per_view[v]):
                p = pts_per_view[v][i]
                if len(p) == 2 and (np.isnan(p[0]) or np.isnan(p[1])):
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            valid.append(i)
    if len(valid) < 8:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(n)], np.array([], dtype=bool)
    out = []
    for v in range(n):
        arr = np.float32([pts_per_view[v][i] for i in valid])
        out.append(arr)
    return out, np.ones(len(valid), dtype=bool)
