"""Dense disparity estimation with Semi-Global Block Matching.

Why SGBM rather than a learned stereo network: it needs no training data, runs
deterministically, and gives a calibrated baseline to measure a learned model
against later. The parameters below are not magic numbers — each is derived from
the scene geometry, and `SGBMParams.for_scene` shows the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .calibration import StereoCalibration

# SGBM returns disparity as fixed-point with 4 fractional bits.
DISPARITY_SCALE = 16.0


@dataclass(frozen=True)
class SGBMParams:
    """Tuned SGBM configuration.

    min_disparity:    lowest disparity searched; 0 unless the scene has no far field.
    num_disparities:  search range, MUST be a multiple of 16. Sets the minimum
                      measurable depth: Z_min = f*B/(min_disp + num_disp).
    block_size:       matching window, odd. Small = sharper edges, noisier;
                      large = smoother, blurs depth discontinuities. 5-7 for road.
    p1, p2:           smoothness penalties for disparity change of 1 and >1.
                      OpenCV's documented scaling is 8*channels*block^2 and
                      32*channels*block^2; p2 must exceed p1.
    uniqueness_ratio: reject a match unless it beats the runner-up by this percent.
                      Higher = fewer but more trustworthy matches. 10-15 is sane.
    speckle_*:        remove small isolated disparity blobs (mismatches).
    disp12_max_diff:  left-right consistency check tolerance, in pixels.
    """

    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    p1: int = 8 * 3 * 5 * 5
    p2: int = 32 * 3 * 5 * 5
    uniqueness_ratio: int = 12
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    def __post_init__(self) -> None:
        if self.num_disparities % 16 != 0 or self.num_disparities <= 0:
            raise ValueError(
                f"num_disparities must be a positive multiple of 16, got {self.num_disparities}"
            )
        if self.block_size % 2 == 0 or self.block_size < 1:
            raise ValueError(f"block_size must be odd and >= 1, got {self.block_size}")
        if self.p2 <= self.p1:
            raise ValueError(f"p2 ({self.p2}) must exceed p1 ({self.p1})")

    @classmethod
    def for_scene(
        cls,
        calib: StereoCalibration,
        min_depth_m: float = 3.0,
        block_size: int = 5,
        channels: int = 3,
    ) -> "SGBMParams":
        """Derive the search range from the closest depth we need to resolve.

        A road-facing stereo rig does not need to measure the bonnet, so clamping
        min_depth_m to the nearest useful range keeps num_disparities — and
        therefore runtime, which is roughly linear in it — as small as possible.
        """
        needed = calib.disparity_at(min_depth_m)
        num_disp = int(np.ceil(needed / 16.0) * 16)
        num_disp = int(np.clip(num_disp, 16, 256))
        return cls(
            num_disparities=num_disp,
            block_size=block_size,
            p1=8 * channels * block_size * block_size,
            p2=32 * channels * block_size * block_size,
        )

    def with_(self, **kwargs) -> "SGBMParams":
        return replace(self, **kwargs)

    def min_measurable_depth(self, calib: StereoCalibration) -> float:
        return (calib.fx * calib.baseline_m) / (self.min_disparity + self.num_disparities)


def _build_matcher(params: SGBMParams) -> cv2.StereoSGBM:
    return cv2.StereoSGBM.create(
        minDisparity=params.min_disparity,
        numDisparities=params.num_disparities,
        blockSize=params.block_size,
        P1=params.p1,
        P2=params.p2,
        disp12MaxDiff=params.disp12_max_diff,
        uniquenessRatio=params.uniqueness_ratio,
        speckleWindowSize=params.speckle_window_size,
        speckleRange=params.speckle_range,
        preFilterCap=params.pre_filter_cap,
        mode=params.mode,
    )


def _has_wls() -> bool:
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createDisparityWLSFilter")


def compute_disparity(
    left: np.ndarray,
    right: np.ndarray,
    params: SGBMParams | None = None,
    use_wls: bool = True,
) -> np.ndarray:
    """Return a float32 disparity map in pixels; invalid pixels are NaN.

    Invalid is NaN rather than 0 deliberately: 0 is a legal disparity meaning
    "infinitely far", and silently conflating the two puts phantom points at
    infinity in the cloud. Every downstream consumer must handle NaN.

    Edge cases handled: mismatched shapes, colour or greyscale input, odd dtypes,
    and the WLS filter being absent from a base OpenCV build.
    """
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"shape mismatch: left {left.shape[:2]} vs right {right.shape[:2]}")

    params = params or SGBMParams()
    left_u8 = _as_uint8(left)
    right_u8 = _as_uint8(right)

    matcher = _build_matcher(params)
    raw_left = matcher.compute(left_u8, right_u8)

    if use_wls and _has_wls():
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        raw_right = right_matcher.compute(right_u8, left_u8)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        raw_left = wls.filter(raw_left, left_u8, disparity_map_right=raw_right)

    disparity = raw_left.astype(np.float32) / DISPARITY_SCALE

    # SGBM marks "no match" as (min_disparity - 1); everything at or below the
    # search floor is unreliable, so drop it rather than letting it propagate.
    disparity[disparity < params.min_disparity + 1e-6] = np.nan
    return disparity


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce to the 8-bit layout SGBM expects without changing the content."""
    if image.dtype == np.uint8:
        out = image
    elif np.issubdtype(image.dtype, np.floating):
        # Assume [0, 1] if it looks normalised, else already in [0, 255].
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 + 1e-6 else 1.0
        out = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        out = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def disparity_coverage(disparity: np.ndarray) -> float:
    """Fraction of pixels with a valid match. Below ~0.5 means bad tuning or
    a textureless scene, and the 3D output should not be trusted."""
    return float(np.isfinite(disparity).mean())


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    """BGR visualisation; invalid pixels rendered black."""
    valid = np.isfinite(disparity)
    out = np.zeros(disparity.shape, dtype=np.uint8)
    if valid.any():
        lo = float(np.percentile(disparity[valid], 2))
        hi = float(np.percentile(disparity[valid], 98))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        norm = np.clip(np.nan_to_num((disparity - lo) / (hi - lo), nan=0.0), 0, 1)
        out = (norm * 255).astype(np.uint8)
        out[~valid] = 0
    colour = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour
