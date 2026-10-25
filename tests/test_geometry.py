"""Unit tests for the geometry, which is where silent errors hide."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anomaly import depth_sigma
from src.calibration import StereoCalibration, kitti_like
from src.disparity import SGBMParams
from src.reconstruct import reproject
from src.road_plane import camera_height, fit_road_plane
from src.synth import Pothole, RoadScene, render_stereo_pair


def test_q_matrix_roundtrip():
    """Q applied to (u, v, d) must return the depth the disparity implies."""
    c = kitti_like()
    Q = c.Q
    for z_true in (5.0, 12.0, 30.0):
        d = c.disparity_at(z_true)
        vec = Q @ np.array([c.cx, c.cy, d, 1.0])
        assert np.isclose(vec[2] / vec[3], z_true, rtol=1e-6)


def test_depth_from_disparity_matches_closed_form():
    c = kitti_like()
    disp = np.array([[10.0, 20.0, 40.0]])
    z = c.depth_from_disparity(disp)
    assert np.allclose(z, (c.fx * c.baseline_m) / disp)


def test_invalid_disparity_is_infinite_not_zero():
    c = kitti_like()
    z = c.depth_from_disparity(np.array([0.0, -3.0]))
    assert np.isinf(z).all()


def test_depth_error_grows_quadratically():
    """The core error model: halving range quarters the uncertainty."""
    c = kitti_like()
    assert np.isclose(c.depth_resolution_at(20) / c.depth_resolution_at(10), 4.0)
    assert np.isclose(depth_sigma(20, 384.4, 0.25) / depth_sigma(10, 384.4, 0.25), 4.0)


def test_sgbm_rejects_bad_params():
    with pytest.raises(ValueError):
        SGBMParams(num_disparities=100)   # not a multiple of 16
    with pytest.raises(ValueError):
        SGBMParams(block_size=4)          # even
    with pytest.raises(ValueError):
        SGBMParams(p1=500, p2=100)        # p2 must exceed p1


def test_num_disparities_covers_requested_depth():
    c = kitti_like()
    p = SGBMParams.for_scene(c, min_depth_m=3.0)
    assert p.num_disparities % 16 == 0
    assert p.min_measurable_depth(c) <= 3.0 + 1e-6


def test_negative_baseline_rejected():
    with pytest.raises(ValueError):
        StereoCalibration(700, 700, 600, 180, 600, -0.5, 1242, 375).Q


def test_plane_fit_recovers_known_plane():
    """A synthetic plane with 30% outliers must still be recovered exactly."""
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.uniform(-5, 5, n)
    z = rng.uniform(5, 30, n)
    y = 1.65 + 0.01 * z + rng.normal(0, 0.004, n)   # slight slope + noise
    pts = np.stack([x, y, z], axis=1)
    pts[: n // 3, 1] += rng.uniform(0.5, 3.0, n // 3)  # outliers above

    plane = fit_road_plane(pts, distance_threshold_m=0.03)
    assert plane.inlier_ratio > 0.6
    assert plane.rmse < 0.01
    assert np.isclose(camera_height(plane), 1.65, atol=0.05)
    assert plane.normal[1] < 0, "normal must point up (-Y) in camera frame"


def test_plane_fit_raises_on_garbage():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        fit_road_plane(rng.uniform(-10, 10, (2000, 3)), distance_threshold_m=0.01)


def test_plane_fit_needs_three_points():
    with pytest.raises(ValueError):
        fit_road_plane(np.zeros((2, 3)))


def test_synthetic_pair_disparity_is_physically_consistent():
    """The rendered pair must satisfy d = f*B/Z to sub-pixel accuracy, or every
    accuracy number measured against it is meaningless."""
    c = kitti_like()
    scene = RoadScene(potholes=[Pothole(0.0, 10.0, 0.08, 0.3)])
    _, _, depth, disp = render_stereo_pair(c, scene)
    valid = np.isfinite(depth) & np.isfinite(disp)
    expected = (c.fx * c.baseline_m) / depth[valid]
    assert np.allclose(disp[valid], expected, rtol=1e-5)


def test_reproject_drops_invalid_and_far_points():
    c = kitti_like(64, 48)
    disp = np.full((48, 64), np.nan, dtype=np.float32)
    disp[10, 10] = c.disparity_at(8.0)      # valid
    disp[11, 11] = c.disparity_at(500.0)    # beyond max_depth
    disp[12, 12] = -1.0                     # invalid
    pts = reproject(disp, c, max_depth_m=60.0)
    assert np.isfinite(pts[10, 10]).all()
    assert np.isclose(pts[10, 10, 2], 8.0, rtol=1e-4)
    assert np.isnan(pts[11, 11]).all()
    assert np.isnan(pts[12, 12]).all()
