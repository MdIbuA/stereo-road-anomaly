"""Disparity -> metric 3D point cloud, and the road-surface region of interest."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import StereoCalibration


def reproject(
    disparity: np.ndarray,
    calib: StereoCalibration,
    max_depth_m: float = 60.0,
) -> np.ndarray:
    """Return an HxWx3 array of camera-frame (X, Y, Z) in metres; invalid = NaN.

    Camera frame convention (OpenCV): +X right, +Y down, +Z forward. So the road
    surface sits at *positive* Y below the camera, which is why the plane fit
    later searches the lower image rows.

    max_depth_m exists because disparity error grows quadratically with range
    (see StereoCalibration.depth_resolution_at). Beyond ~60 m a KITTI-geometry
    rig cannot resolve a few-centimetre pothole at all, so those points are
    dropped rather than fed to the plane fit as noise.
    """
    disp = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 0)

    # reprojectImageTo3D has no NaN concept; feed it 0 and mask afterwards.
    safe = np.where(valid, disp, 0.0).astype(np.float32)
    points = cv2.reprojectImageTo3D(safe, calib.Q.astype(np.float32))

    z = points[..., 2]
    good = valid & np.isfinite(z) & (z > 0) & (z <= max_depth_m)
    points[~good] = np.nan
    return points


def road_roi_mask(
    shape: tuple[int, int],
    bottom_fraction: float = 0.45,
    side_margin: float = 0.12,
    horizon_fraction: float = 0.0,
) -> np.ndarray:
    """Boolean mask for the image region where road is plausible.

    This is a geometric prior, not a detector: for a forward-facing vehicle
    camera the road occupies the lower-centre of the frame. Seeding RANSAC
    inside this region rather than the whole image stops it from locking onto a
    building facade or a car bonnet, which is the usual failure mode.

    Pass horizon_fraction > 0 to additionally exclude rows above the horizon.
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)

    top = int(height * (1.0 - bottom_fraction))
    top = max(top, int(height * horizon_fraction))
    left = int(width * side_margin)
    right = int(width * (1.0 - side_margin))
    mask[top:, left:right] = True
    return mask


def points_to_ply(points: np.ndarray, colors: np.ndarray | None, path: str) -> int:
    """Write finite points to an ASCII PLY. Returns the number written.

    Useful for inspecting the cloud in MeshLab or CloudCompare — the fastest way
    to see that a Q matrix is wrong is to look at the cloud.
    """
    finite = np.isfinite(points).all(axis=-1)
    xyz = points[finite]
    if colors is not None:
        rgb = colors[finite]
        if rgb.ndim == 2 and rgb.shape[1] == 3:
            rgb = rgb[:, ::-1]  # BGR -> RGB
    else:
        rgb = np.full_like(xyz, 200, dtype=np.uint8)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(xyz)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, np.asarray(rgb, dtype=np.uint8)):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
    return int(len(xyz))
