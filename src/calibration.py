"""Stereo camera calibration: intrinsics, baseline, and the reprojection matrix Q.

The Q matrix is what turns a disparity map into metric 3D. Getting it wrong is the
single most common source of "my point cloud looks like a banana" bugs, so this
module keeps the derivation explicit rather than hiding it behind cv2.stereoRectify.

For a rectified stereo pair with equal focal lengths, OpenCV's convention is:

        [ 1  0   0        -cx      ]
    Q = [ 0  1   0        -cy      ]
        [ 0  0   0         f       ]
        [ 0  0  -1/Tx  (cx-cx')/Tx ]

where Tx is the signed baseline (negative for a right camera displaced in +x).
Then [X Y Z W]^T = Q . [u v d 1]^T and the metric point is (X/W, Y/W, Z/W),
which reduces to the familiar Z = f*B/d.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    """Intrinsics and extrinsics for a *rectified* stereo pair."""

    fx: float           # focal length in pixels (x)
    fy: float           # focal length in pixels (y)
    cx: float           # principal point x, left camera
    cy: float           # principal point y, left camera
    cx_right: float     # principal point x, right camera (equals cx when rectified)
    baseline_m: float   # positive baseline in metres
    width: int
    height: int

    @property
    def Q(self) -> np.ndarray:
        """OpenCV reprojection matrix mapping (u, v, disparity) -> homogeneous 3D."""
        if self.baseline_m <= 0:
            raise ValueError(f"baseline must be positive, got {self.baseline_m}")
        tx = -self.baseline_m  # right camera sits at +x, so Tx is negative
        return np.array(
            [
                [1.0, 0.0, 0.0, -self.cx],
                [0.0, 1.0, 0.0, -self.cy],
                [0.0, 0.0, 0.0, self.fx],
                [0.0, 0.0, -1.0 / tx, (self.cx - self.cx_right) / tx],
            ],
            dtype=np.float64,
        )

    def depth_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """Z = f*B/d, with non-positive disparity mapped to inf (unknown depth)."""
        d = np.asarray(disparity, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.fx * self.baseline_m) / d
        z[d <= 0] = np.inf
        return z

    def disparity_at(self, depth_m: float) -> float:
        """Inverse of the above. Useful for choosing num_disparities."""
        if depth_m <= 0:
            raise ValueError("depth must be positive")
        return (self.fx * self.baseline_m) / depth_m

    def depth_resolution_at(self, depth_m: float) -> float:
        """Metres of depth error per pixel of disparity error at a given range.

        dZ/dd = -f*B/d^2 = -Z^2/(f*B). Quadratic in Z: this is why stereo depth
        degrades fast with distance and why the useful pothole range is short.
        """
        return (depth_m ** 2) / (self.fx * self.baseline_m)


def _parse_kitti_calib(path: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI calib_cam_to_cam.txt into {key: float array}."""
    values: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, _, payload = line.partition(":")
            try:
                values[key.strip()] = np.array(
                    [float(tok) for tok in payload.split()], dtype=np.float64
                )
            except ValueError:
                continue  # non-numeric fields such as calib_time
    return values


def from_kitti(calib_path: str | Path) -> StereoCalibration:
    """Build a StereoCalibration from KITTI's calib_cam_to_cam.txt.

    KITTI stores rectified projection matrices P_rect_0x as 3x4 row-major. For the
    right colour camera, P_rect_03[0, 3] = -fx * baseline, so the baseline is
    recovered as -P3[0, 3] / fx. Cameras 02 (left colour) and 03 (right colour) are
    the standard pair; 00/01 are the greyscale pair.
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"KITTI calibration not found: {calib_path}")

    values = _parse_kitti_calib(calib_path)
    for key in ("P_rect_02", "P_rect_03", "S_rect_02"):
        if key not in values:
            raise KeyError(f"{calib_path} is missing {key}; is this calib_cam_to_cam.txt?")

    p2 = values["P_rect_02"].reshape(3, 4)
    p3 = values["P_rect_03"].reshape(3, 4)
    width, height = (int(round(v)) for v in values["S_rect_02"][:2])

    fx, fy = float(p2[0, 0]), float(p2[1, 1])
    baseline = float(-p3[0, 3] / fx)
    if baseline <= 0:
        raise ValueError(
            f"derived non-positive baseline ({baseline:.4f} m) — check P_rect_03"
        )

    return StereoCalibration(
        fx=fx,
        fy=fy,
        cx=float(p2[0, 2]),
        cy=float(p2[1, 2]),
        cx_right=float(p3[0, 2]),
        baseline_m=baseline,
        width=width,
        height=height,
    )


def kitti_like(width: int = 1242, height: int = 375) -> StereoCalibration:
    """Calibration with KITTI-typical values, for synthetic data and unit tests.

    fx ~ 721 px, baseline ~ 0.54 m are the standard KITTI colour-pair figures.
    """
    return StereoCalibration(
        fx=721.5377,
        fy=721.5377,
        cx=width / 2.0,
        cy=height / 2.0,
        cx_right=width / 2.0,
        baseline_m=0.5327,
        width=width,
        height=height,
    )
