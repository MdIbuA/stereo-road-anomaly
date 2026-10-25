"""End-to-end orchestration: stereo pair in, metric detections out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .anomaly import Detection, detect_anomalies, draw_detections
from .calibration import StereoCalibration
from .disparity import SGBMParams, colorize_disparity, compute_disparity, disparity_coverage
from .reconstruct import reproject, road_roi_mask
from .road_plane import Plane, camera_height, fit_road_plane


@dataclass
class PipelineConfig:
    sgbm: SGBMParams | None = None
    use_wls: bool = True
    max_depth_m: float = 60.0
    roi_bottom_fraction: float = 0.45
    roi_side_margin: float = 0.12
    plane_threshold_m: float = 0.03
    plane_iterations: int = 500
    min_depth_below_road_m: float = 0.025
    min_area_m2: float = 0.01
    max_area_m2: float = 4.0
    morph_kernel: int = 5
    # Noise model. disp_sigma_px is the 1-sigma disparity error of the matcher;
    # measure it once against ground truth and keep it honest, because every
    # detection threshold is derived from it.
    disp_sigma_px: float = 0.25
    candidate_n_sigma: float = 0.5
    accept_n_sigma: float = 3.0
    correlation_factor: float = 8.0
    smooth_kernel: int = 5
    # Fitting every pixel is wasteful; RANSAC converges on a subsample and runs
    # an order of magnitude faster. 20k points is ample for a plane.
    plane_sample_limit: int = 20000


@dataclass
class PipelineResult:
    disparity: np.ndarray
    points: np.ndarray
    plane: Plane
    detections: list[Detection]
    mask: np.ndarray
    heights: np.ndarray
    roi: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0

    @property
    def camera_height_m(self) -> float:
        return camera_height(self.plane)

    def summary(self) -> dict[str, Any]:
        return {
            "detections": len(self.detections),
            "disparity_coverage": round(self.coverage, 4),
            "plane_inlier_ratio": round(self.plane.inlier_ratio, 4),
            "plane_rmse_m": round(self.plane.rmse, 5),
            "camera_height_m": round(self.camera_height_m, 3),
            "reliable_range_m": self.diagnostics.get("reliable_range_m"),
            "candidate_regions": self.diagnostics.get("candidate_regions"),
            "rejected": self.diagnostics.get("rejected"),
            "total_ms": round(sum(self.timings_ms.values()), 2),
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class StereoAnomalyPipeline:
    """Stateless per-frame pipeline. Safe to reuse across frames."""

    def __init__(self, calib: StereoCalibration, config: PipelineConfig | None = None):
        self.calib = calib
        self.config = config or PipelineConfig()
        if self.config.sgbm is None:
            self.config.sgbm = SGBMParams.for_scene(calib, min_depth_m=3.0)

    def run(self, left: np.ndarray, right: np.ndarray) -> PipelineResult:
        cfg = self.config
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        disparity = compute_disparity(left, right, cfg.sgbm, use_wls=cfg.use_wls)
        timings["disparity"] = (time.perf_counter() - t0) * 1000

        coverage = disparity_coverage(disparity)

        t0 = time.perf_counter()
        points = reproject(disparity, self.calib, max_depth_m=cfg.max_depth_m)
        timings["reproject"] = (time.perf_counter() - t0) * 1000

        roi = road_roi_mask(
            disparity.shape,
            bottom_fraction=cfg.roi_bottom_fraction,
            side_margin=cfg.roi_side_margin,
        )

        t0 = time.perf_counter()
        roi_points = points[roi]
        roi_points = roi_points[np.isfinite(roi_points).all(axis=1)]
        if len(roi_points) > cfg.plane_sample_limit:
            idx = np.random.default_rng(0).choice(
                len(roi_points), size=cfg.plane_sample_limit, replace=False
            )
            roi_points = roi_points[idx]
        plane = fit_road_plane(
            roi_points,
            distance_threshold_m=cfg.plane_threshold_m,
            max_iterations=cfg.plane_iterations,
        )
        timings["plane_fit"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        detections, mask, heights, diagnostics = detect_anomalies(
            points,
            plane,
            roi,
            fb=self.calib.fx * self.calib.baseline_m,
            min_depth_below_road_m=cfg.min_depth_below_road_m,
            min_area_m2=cfg.min_area_m2,
            max_area_m2=cfg.max_area_m2,
            morph_kernel=cfg.morph_kernel,
            fx=self.calib.fx,
            fy=self.calib.fy,
            disp_sigma_px=cfg.disp_sigma_px,
            candidate_n_sigma=cfg.candidate_n_sigma,
            accept_n_sigma=cfg.accept_n_sigma,
            correlation_factor=cfg.correlation_factor,
            smooth_kernel=cfg.smooth_kernel,
        )
        timings["detect"] = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            disparity=disparity,
            points=points,
            plane=plane,
            detections=detections,
            mask=mask,
            heights=heights,
            roi=roi,
            diagnostics=diagnostics,
            timings_ms=timings,
            coverage=coverage,
        )

    @staticmethod
    def visualise(left: np.ndarray, result: PipelineResult) -> dict[str, np.ndarray]:
        return {
            "detections": draw_detections(left, result.detections, result.mask),
            "disparity": colorize_disparity(result.disparity),
        }
