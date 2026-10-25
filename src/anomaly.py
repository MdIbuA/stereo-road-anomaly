"""Turn per-pixel height-above-road into discrete pothole detections."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from .road_plane import Plane


@dataclass(frozen=True)
class Detection:
    """One road anomaly, measured in metric units rather than pixels."""

    bbox: tuple[int, int, int, int]   # x, y, w, h in image pixels
    centroid_px: tuple[float, float]
    depth_m: float                    # mean Z of the region
    max_depth_below_road_m: float     # how deep the pothole is
    area_m2: float                    # metric footprint on the road plane
    pixel_area: int
    severity: str                     # low | medium | high

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def height_map(points: np.ndarray, plane: Plane) -> np.ndarray:
    """HxW signed height above the road plane, in metres. Invalid pixels NaN."""
    h, w = points.shape[:2]
    flat = points.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)

    heights = np.full(len(flat), np.nan, dtype=np.float32)
    if finite.any():
        heights[finite] = plane.signed_distance(flat[finite]).astype(np.float32)
    return heights.reshape(h, w)


def _metric_pixel_area(depth_m: float, fx: float, fy: float) -> float:
    """Ground area covered by one pixel at a given depth, in m^2.

    A pixel subtends depth/fx metres horizontally and depth/fy vertically, so
    area scales with depth squared. Without this, a distant pothole and a near
    one with the same pixel count would be reported as the same size.
    """
    return (depth_m / fx) * (depth_m / fy)


def depth_sigma(depth_m: np.ndarray | float, fb: float, disp_sigma_px: float) -> np.ndarray:
    """1-sigma depth uncertainty from disparity noise, in metres.

    dZ/dd = Z^2 / (f*B), so sigma_Z = sigma_d * Z^2 / (f*B). This is the single
    most important function in the detector. Everything downstream is a
    hypothesis test against this noise floor, because a fixed centimetre
    threshold is wrong by two orders of magnitude across the usable range:
    for KITTI geometry (f*B = 384 px.m) one pixel of disparity error is 6.5 cm
    at 5 m and 234 cm at 30 m.
    """
    z = np.asarray(depth_m, dtype=np.float64)
    return disp_sigma_px * (z ** 2) / fb


def max_reliable_range(
    fb: float,
    disp_sigma_px: float,
    min_pothole_depth_m: float,
    n_sigma: float,
    region_pixels: int,
    correlation_factor: float,
) -> float:
    """Range beyond which the shallowest pothole we care about is unmeasurable.

    Solves n_sigma * sigma_d * Z^2 / (f*B*sqrt(N_eff)) = min_pothole_depth_m
    for Z. Reporting this number, and refusing to emit detections past it, is
    more useful than silently emitting noise: a downstream consumer can plan a
    route or a second pass, which it cannot do with false positives.
    """
    n_eff = max(region_pixels / correlation_factor, 1.0)
    denom = n_sigma * disp_sigma_px / np.sqrt(n_eff)
    if denom <= 0:
        return float("inf")
    return float(np.sqrt(min_pothole_depth_m * fb / denom))


def detect_anomalies(
    points: np.ndarray,
    plane: Plane,
    roi_mask: np.ndarray,
    fb: float,
    min_depth_below_road_m: float = 0.025,
    min_area_m2: float = 0.01,
    max_area_m2: float = 4.0,
    morph_kernel: int = 5,
    fx: float = 721.5377,
    fy: float = 721.5377,
    disp_sigma_px: float = 0.25,
    candidate_n_sigma: float = 0.5,
    accept_n_sigma: float = 3.0,
    correlation_factor: float = 8.0,
    smooth_kernel: int = 5,
) -> tuple[list[Detection], np.ndarray, np.ndarray, dict]:
    """Find depressions in the road surface as a two-stage hypothesis test.

    Returns (detections, binary mask, height map, diagnostics).

    Stage 1 — per-pixel candidates at a *range-adaptive* threshold of
    candidate_n_sigma * sigma_Z(Z). A fixed threshold cannot work: sigma_Z grows
    as Z^2, so one constant is either blind at 8 m or hallucinating at 30 m. The
    earlier fixed-2.5 cm version of this function produced 25 detections on a
    4-pothole scene, every spurious one beyond 28 m.

    Stage 2 — region-level significance. Noise averages down over a region, so a
    blob of N pixels is accepted only if its *mean* depression exceeds
    accept_n_sigma * sigma_Z / sqrt(N_eff), where N_eff = N / correlation_factor.
    SGBM errors are spatially correlated, so treating all N pixels as independent
    would overstate confidence by roughly 3x; correlation_factor is the honest
    discount and is the parameter to re-measure on real data.

    Reported depth is the region *median*, not the minimum. The minimum is the
    single noisiest pixel in the blob and inflates shallow potholes badly.

    Sign convention: Plane.signed_distance is negative below the road.
    """
    heights = height_map(points, plane)
    depth = points[..., 2]

    # Spatial median: suppresses speckle without blurring the rim of a pothole
    # the way a Gaussian would. NaNs must be bridged first or they propagate.
    if smooth_kernel > 1:
        filled = np.where(np.isfinite(heights), heights, 0.0).astype(np.float32)
        smoothed = cv2.medianBlur(filled, smooth_kernel)
        heights_s = np.where(np.isfinite(heights), smoothed, np.nan)
    else:
        heights_s = heights

    valid = np.isfinite(heights_s) & np.isfinite(depth) & (depth > 0)
    sigma = np.full(depth.shape, np.inf, dtype=np.float64)
    sigma[valid] = depth_sigma(depth[valid], fb, disp_sigma_px)

    # Stage 1: loose, range-aware candidate threshold. The height map has
    # already been median-filtered, which cuts per-pixel noise by roughly
    # kernel/2 for spatially correlated error, so thresholding at the raw
    # per-pixel sigma would be double-counting the noise and would blind the
    # detector past ~8 m. Stage 1 exists for recall; stage 2 supplies precision.
    smooth_gain = max(1.0, smooth_kernel / 2.0)
    per_pixel_thr = np.maximum(
        min_depth_below_road_m * 0.4, candidate_n_sigma * sigma / smooth_gain
    )
    below = valid & (heights_s < -per_pixel_thr)
    candidate = (below & roi_mask).astype(np.uint8)

    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        # Open first to kill speckle, then close to heal a pothole split by a
        # specular highlight or a disparity dropout. Order matters.
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )

    detections: list[Detection] = []
    accepted = np.zeros_like(candidate)
    rejected = {"area": 0, "significance": 0, "empty": 0}

    for label in range(1, n_labels):  # 0 is background
        x, y, w, h, pixel_area = stats[label]
        region = labels == label

        region_depth = depth[region]
        region_depth = region_depth[np.isfinite(region_depth)]
        region_heights = heights_s[region]
        region_heights = region_heights[np.isfinite(region_heights)]
        if region_depth.size == 0 or region_heights.size == 0:
            rejected["empty"] += 1
            continue

        median_depth = float(np.median(region_depth))
        area_m2 = pixel_area * _metric_pixel_area(median_depth, fx, fy)
        if not (min_area_m2 <= area_m2 <= max_area_m2):
            rejected["area"] += 1
            continue

        # Stage 2: is the depression significant against the noise floor?
        # Test the pothole *floor*, not the whole blob. Blob extent and the
        # blob median are coupled: loosening stage 1 grows the shallow rim and
        # drags the median up, so a median test makes the detector sensitive to
        # a threshold it is supposed to be independent of. The deeper half of
        # the region is a stable, physically meaningful core.
        core = region_heights[region_heights <= np.median(region_heights)]
        if core.size == 0:
            rejected["empty"] += 1
            continue
        n_eff = max(core.size / correlation_factor, 1.0)
        sigma_region = depth_sigma(median_depth, fb, disp_sigma_px) / np.sqrt(n_eff)
        required = max(min_depth_below_road_m, accept_n_sigma * float(sigma_region))

        core_below = float(-core.mean())
        if core_below < required:
            rejected["significance"] += 1
            continue
        # Reported depth is a robust peak: a pothole is a bowl, so its mean is
        # shallower than its floor, but the raw minimum is the noisiest pixel.
        peak_below = float(-np.percentile(region_heights, 2))

        detections.append(
            Detection(
                bbox=(int(x), int(y), int(w), int(h)),
                centroid_px=(float(centroids[label][0]), float(centroids[label][1])),
                depth_m=round(median_depth, 3),
                max_depth_below_road_m=round(peak_below, 4),
                area_m2=round(area_m2, 4),
                pixel_area=int(pixel_area),
                severity=_severity(peak_below, area_m2),
            )
        )
        accepted[region] = 1

    detections.sort(key=lambda d: d.depth_m)  # nearest hazard first

    diagnostics = {
        "candidate_regions": int(n_labels - 1),
        "rejected": rejected,
        # Quoted for a 400-pixel blob, roughly a 0.3 m pothole at mid range.
        # A larger pothole stays detectable further out; a smaller one does not.
        "reliable_range_m": round(
            max_reliable_range(
                fb, disp_sigma_px, min_depth_below_road_m,
                accept_n_sigma, 400, correlation_factor,
            ), 2
        ),
        "disp_sigma_px": disp_sigma_px,
    }
    return detections, accepted.astype(bool), heights_s, diagnostics


def _severity(depth_below_m: float, area_m2: float) -> str:
    """Crude severity banding, loosely following road-maintenance practice:
    depth drives vehicle damage risk more than width does."""
    if depth_below_m >= 0.08 or area_m2 >= 0.5:
        return "high"
    if depth_below_m >= 0.04 or area_m2 >= 0.15:
        return "medium"
    return "low"


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay detections on the left image for visual inspection."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if mask is not None and mask.any():
        tint = np.zeros_like(out)
        tint[mask] = (0, 0, 255)
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)

    colours = {"low": (0, 200, 255), "medium": (0, 128, 255), "high": (0, 0, 255)}
    for det in detections:
        x, y, w, h = det.bbox
        colour = colours[det.severity]
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        label = f"{det.max_depth_below_road_m * 100:.0f}cm @ {det.depth_m:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y - 4, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), colour, -1)
        cv2.putText(out, label, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
