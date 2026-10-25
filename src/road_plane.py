"""RANSAC road-plane estimation.

Written from first principles rather than calling a library fit, because the
whole detector rests on this: an anomaly here is *defined* as a signed deviation
from the fitted plane, so a plane that is off by two centimetres produces two
centimetres of false pothole everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    """Plane ax + by + cz + d = 0 with a unit normal (a, b, c)."""

    normal: np.ndarray   # shape (3,), unit length
    d: float
    inlier_ratio: float
    n_inliers: int
    rmse: float          # metres, over inliers

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed perpendicular distance from each point to the plane.

        Sign convention: the normal is forced to point *up* in camera frame
        (i.e. -Y), so a point below the road surface — inside a pothole — gets a
        negative distance, and a bump gets a positive one.
        """
        return points @ self.normal + self.d

    def height_above(self, points: np.ndarray) -> np.ndarray:
        """Alias that reads correctly at the call site."""
        return self.signed_distance(points)


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Total least squares plane through >=3 points, via SVD on the centred set.

    The smallest right singular vector is the direction of least variance, which
    for a planar point set is the surface normal. This minimises perpendicular
    distance, unlike an ordinary least squares fit of z on (x, y), which
    minimises vertical distance and is biased when the plane is steep.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("degenerate point set: cannot determine a normal")
    normal = normal / norm
    d = float(-normal @ centroid)
    return normal, d


def _orient_upward(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    """Force the normal to point up in OpenCV camera frame (+Y is down)."""
    if normal[1] > 0:
        return -normal, -d
    return normal, d


def fit_road_plane(
    points: np.ndarray,
    distance_threshold_m: float = 0.03,
    max_iterations: int = 500,
    min_inlier_ratio: float = 0.30,
    confidence: float = 0.999,
    refine: bool = True,
    rng: np.random.Generator | None = None,
) -> Plane:
    """RANSAC plane fit over an (N, 3) array of finite points.

    distance_threshold_m defaults to 3 cm: tight enough that a typical 4-6 cm
    pothole falls outside the inlier band, loose enough to absorb stereo noise
    on a real road at 10-20 m. This is the parameter to tune first if the
    detector is over- or under-firing.

    The iteration count adapts: once a hypothesis with inlier ratio w is found,
    the number of trials needed for the given confidence is
    log(1 - confidence) / log(1 - w^3), so a clean road exits in a few dozen
    iterations instead of always running the full budget.

    Raises ValueError if fewer than 3 points are supplied or no plane reaches
    min_inlier_ratio — a silent bad plane is worse than a loud failure.
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = len(pts)
    if n < 3:
        raise ValueError(f"need at least 3 finite points to fit a plane, got {n}")
    if n < 50:
        normal, d = _orient_upward(*_fit_plane_svd(pts))
        residual = np.abs(pts @ normal + d)
        return Plane(normal, d, 1.0, n, float(np.sqrt((residual ** 2).mean())))

    rng = rng or np.random.default_rng(0)  # seeded: runs are reproducible
    best_inliers: np.ndarray | None = None
    best_count = 0
    trials = max_iterations
    i = 0

    while i < min(trials, max_iterations):
        i += 1
        idx = rng.choice(n, size=3, replace=False)
        sample = pts[idx]

        # Reject near-collinear samples: they define a pencil of planes, not one.
        v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-8:
            continue

        normal = cross / cross_norm
        d = float(-normal @ sample[0])
        inliers = np.abs(pts @ normal + d) < distance_threshold_m
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            w = count / n
            if w > 0:
                denom = np.log(max(1.0 - w ** 3, 1e-12))
                trials = int(np.ceil(np.log(1.0 - confidence) / denom))

    if best_inliers is None or best_count / n < min_inlier_ratio:
        ratio = (best_count / n) if n else 0.0
        raise ValueError(
            f"no plane reached min_inlier_ratio={min_inlier_ratio:.2f} "
            f"(best {ratio:.2f} over {n} points). The ROI probably is not road, "
            f"or distance_threshold_m={distance_threshold_m} is too tight."
        )

    # Refit on all inliers. The 3-point hypothesis only located the plane; this
    # is what actually makes it accurate.
    if refine:
        normal, d = _fit_plane_svd(pts[best_inliers])
        refined = np.abs(pts @ normal + d) < distance_threshold_m
        if int(refined.sum()) >= best_count:
            best_inliers = refined
            best_count = int(refined.sum())
    else:
        normal, d = _fit_plane_svd(pts[best_inliers])

    normal, d = _orient_upward(normal, d)
    residual = pts[best_inliers] @ normal + d
    return Plane(
        normal=normal,
        d=float(d),
        inlier_ratio=best_count / n,
        n_inliers=best_count,
        rmse=float(np.sqrt((residual ** 2).mean())),
    )


def camera_height(plane: Plane) -> float:
    """Perpendicular distance from the camera centre (origin) to the road.

    A useful sanity check: a car-mounted rig should come out around 1.2-1.8 m.
    Anything far outside that means the plane fit found the wrong surface.
    """
    return float(abs(plane.d) / np.linalg.norm(plane.normal))
