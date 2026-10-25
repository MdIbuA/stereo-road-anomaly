"""Physically consistent synthetic stereo pairs with known ground truth.

Why this exists: KITTI is a 12 GB download and you cannot tune a stereo pipeline
you cannot run. This renders both views by ray-casting the *same* world surface
from two camera centres separated by the real baseline, so the resulting
disparity is correct by construction rather than faked with a pixel shift. That
means the pipeline's output can be checked against ground truth — if the
measured pothole depth does not match the depth that was rendered, something in
the calibration or plane fit is wrong, and you find out in seconds.

It is a development and unit-test harness. Results reported anywhere public must
come from real stereo data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import StereoCalibration


@dataclass
class Pothole:
    """A Gaussian depression in the road surface."""

    x_m: float        # lateral offset from the camera axis, metres
    z_m: float        # forward distance, metres
    depth_m: float    # how deep, metres
    radius_m: float   # 1-sigma radius, metres


@dataclass
class RoadScene:
    """A flat road with depressions, viewed by a forward-facing stereo rig."""

    camera_height_m: float = 1.65
    pitch_rad: float = 0.0          # positive = nose down
    potholes: list[Pothole] = field(default_factory=list)
    texture_scale: float = 24.0     # noise cells per metre; drives matchability
    seed: int = 7

    def surface_offset(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Downward displacement of the road surface at world (x, z), in metres."""
        offset = np.zeros_like(x, dtype=np.float64)
        for p in self.potholes:
            r2 = (x - p.x_m) ** 2 + (z - p.z_m) ** 2
            offset += p.depth_m * np.exp(-r2 / (2.0 * p.radius_m ** 2))
        return offset


def _value_noise(u: np.ndarray, v: np.ndarray, seed: int, cells: int = 512) -> np.ndarray:
    """Tileable bilinear value noise sampled at continuous world coordinates.

    Sampling in *world* space rather than image space is the point: both cameras
    look up the same texel for the same physical patch of road, which is exactly
    what makes the pair stereo-matchable.
    """
    rng = np.random.default_rng(seed)
    grid = rng.random((cells, cells))

    uu = np.mod(u, cells)
    vv = np.mod(v, cells)
    u0, v0 = np.floor(uu).astype(np.int64), np.floor(vv).astype(np.int64)
    fu, fv = uu - u0, vv - v0
    u1, v1 = (u0 + 1) % cells, (v0 + 1) % cells
    u0, v0 = u0 % cells, v0 % cells

    # Smoothstep keeps the noise C1 so SGBM sees gradients, not blocky steps.
    su = fu * fu * (3 - 2 * fu)
    sv = fv * fv * (3 - 2 * fv)

    a = grid[v0, u0] * (1 - su) + grid[v0, u1] * su
    b = grid[v1, u0] * (1 - su) + grid[v1, u1] * su
    return a * (1 - sv) + b * sv


def _road_texture(x: np.ndarray, z: np.ndarray, scene: RoadScene) -> np.ndarray:
    """Procedural asphalt in [0, 1], multi-octave so it matches at every scale."""
    s = scene.texture_scale
    tex = (
        0.55 * _value_noise(x * s, z * s, scene.seed)
        + 0.30 * _value_noise(x * s * 2.7, z * s * 2.7, scene.seed + 1)
        + 0.15 * _value_noise(x * s * 6.1, z * s * 6.1, scene.seed + 2)
    )
    # Faint lane markings give a strong horizontal-gradient feature, which is
    # what a block matcher keys on; without them the centre of the road is a
    # textureless hole in the disparity map.
    lane = np.exp(-((np.abs(np.abs(x) - 1.75)) ** 2) / (2 * 0.06 ** 2))
    return np.clip(tex * 0.75 + lane * 0.45, 0.0, 1.0)


def _render_view(
    calib: StereoCalibration,
    scene: RoadScene,
    camera_x: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast one view. Returns (BGR uint8 image, Z-depth float32 map)."""
    h, w = calib.height, calib.width
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    # Ray directions in camera frame (+X right, +Y down, +Z forward).
    dx = (us - calib.cx) / calib.fx
    dy = (vs - calib.cy) / calib.fy
    dz = np.ones_like(dx)

    # Rotate for pitch about the X axis.
    cp, sp = np.cos(scene.pitch_rad), np.sin(scene.pitch_rad)
    ry = cp * dy - sp * dz
    rz = sp * dy + cp * dz

    # Flat road: camera at Y = 0, surface at Y = +camera_height. Intersect where
    # t * ry = camera_height, so only rays pointing downward (ry > 0) hit it.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = scene.camera_height_m / ry
    hits = (ry > 1e-6) & np.isfinite(t) & (t > 0)

    X = np.where(hits, camera_x + t * dx, np.nan)
    Z = np.where(hits, t * rz, np.nan)

    # Depressions lower the surface, so the ray travels further before hitting.
    # Two fixed-point iterations are enough for centimetre-scale dips.
    for _ in range(2):
        valid = np.isfinite(X) & np.isfinite(Z)
        offset = np.zeros_like(X)
        offset[valid] = scene.surface_offset(X[valid], Z[valid])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (scene.camera_height_m + offset) / ry
        hits = (ry > 1e-6) & np.isfinite(t) & (t > 0)
        X = np.where(hits, camera_x + t * dx, np.nan)
        Z = np.where(hits, t * rz, np.nan)

    far = ~np.isfinite(Z) | (Z > max_depth_m) | (Z <= 0)
    Z = np.where(far, np.nan, Z)

    texture = np.zeros_like(X)
    valid = np.isfinite(X) & np.isfinite(Z)
    texture[valid] = _road_texture(X[valid], Z[valid], scene)

    # Sky above the horizon: a smooth vertical ramp. Deliberately low-texture,
    # which reproduces the real failure mode of no matches above the horizon.
    sky = np.clip(0.55 + 0.25 * (vs / h), 0, 1)
    gray = np.where(valid, texture, sky)

    # Mild range attenuation, as a real camera would see.
    atten = np.ones_like(gray)
    atten[valid] = np.clip(1.0 - 0.004 * Z[valid], 0.45, 1.0)
    gray = gray * atten

    img = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(scene.seed + int(camera_x * 1000))
    noise = rng.normal(0.0, 1.6, img.shape)  # sensor noise, else matching is unrealistically perfect
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    bgr = np.repeat(img[:, :, None], 3, axis=2)
    # Slight colour cast so the three channels are not perfectly redundant.
    bgr[..., 0] = np.clip(bgr[..., 0] * 0.96, 0, 255).astype(np.uint8)
    bgr[..., 2] = np.clip(bgr[..., 2] * 1.02, 0, 255).astype(np.uint8)

    return bgr, Z.astype(np.float32)


def render_stereo_pair(
    calib: StereoCalibration,
    scene: RoadScene,
    max_depth_m: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render a rectified stereo pair plus ground-truth depth and disparity.

    Returns (left_bgr, right_bgr, depth_left_m, disparity_left_px).
    """
    left, depth_left = _render_view(calib, scene, camera_x=0.0, max_depth_m=max_depth_m)
    right, _ = _render_view(calib, scene, camera_x=calib.baseline_m, max_depth_m=max_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = (calib.fx * calib.baseline_m) / depth_left
    disparity[~np.isfinite(depth_left)] = np.nan
    return left, right, depth_left, disparity.astype(np.float32)


def default_scene() -> RoadScene:
    """A road with four potholes spanning the useful depth range.

    Depths are chosen around the detector's 2.5 cm floor so the demo exercises
    both sides of the threshold: the 2 cm dip at 24 m should be *missed*, and
    that is correct behaviour, not a bug.
    """
    return RoadScene(
        camera_height_m=1.65,
        pitch_rad=0.012,
        potholes=[
            Pothole(x_m=-0.9, z_m=8.0, depth_m=0.09, radius_m=0.33),
            Pothole(x_m=0.7, z_m=12.5, depth_m=0.06, radius_m=0.28),
            Pothole(x_m=-0.2, z_m=17.0, depth_m=0.045, radius_m=0.24),
            Pothole(x_m=1.1, z_m=24.0, depth_m=0.02, radius_m=0.20),
        ],
    )
