#!/usr/bin/env python3
"""Run the full pipeline on synthetic stereo data with known ground truth.

This is the fastest way to confirm the install works and the geometry is right:
it renders a road with potholes of known depth, runs the pipeline, and reports
measured depth against rendered depth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import kitti_like
from src.pipeline import PipelineConfig, StereoAnomalyPipeline
from src.reconstruct import points_to_ply
from src.synth import default_scene, render_stereo_pair


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--width", type=int, default=1242)
    ap.add_argument("--height", type=int, default=375)
    ap.add_argument("--ply", action="store_true", help="also export the point cloud")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib = kitti_like(args.width, args.height)
    scene = default_scene()
    left, right, gt_depth, gt_disp = render_stereo_pair(calib, scene)

    print(f"calibration: fx={calib.fx:.1f}px  baseline={calib.baseline_m:.4f}m")
    print(f"depth resolution: {calib.depth_resolution_at(10)*100:.2f} cm/px @ 10m, "
          f"{calib.depth_resolution_at(30)*100:.2f} cm/px @ 30m")

    pipeline = StereoAnomalyPipeline(calib, PipelineConfig())
    print(f"SGBM num_disparities={pipeline.config.sgbm.num_disparities} "
          f"(min measurable depth {pipeline.config.sgbm.min_measurable_depth(calib):.2f} m)")

    result = pipeline.run(left, right)
    summary = result.summary()
    print("\n--- pipeline ---")
    print(json.dumps(summary, indent=2))

    # Accuracy against ground truth, which only synthetic data can give us.
    valid = np.isfinite(result.disparity) & np.isfinite(gt_disp)
    if valid.any():
        err = np.abs(result.disparity[valid] - gt_disp[valid])
        print(f"\ndisparity MAE vs ground truth: {err.mean():.3f} px")
        print(f"  fraction within 1 px: {(err < 1.0).mean():.3f}")
        print(f"  fraction within 3 px: {(err < 3.0).mean():.3f}")

    print(f"\ncamera height recovered: {result.camera_height_m:.3f} m "
          f"(rendered {scene.camera_height_m:.3f} m)")

    print(f"\n--- detections ({len(result.detections)}) ---")
    for det in result.detections:
        print(f"  z={det.depth_m:6.2f}m  depth={det.max_depth_below_road_m*100:5.1f}cm  "
              f"area={det.area_m2:.3f}m2  {det.severity}")

    print("\n--- ground truth potholes ---")
    for p in scene.potholes:
        print(f"  z={p.z_m:6.2f}m  depth={p.depth_m*100:5.1f}cm  r={p.radius_m:.2f}m")

    vis = StereoAnomalyPipeline.visualise(left, result)
    cv2.imwrite(str(out / "left.png"), left)
    cv2.imwrite(str(out / "right.png"), right)
    cv2.imwrite(str(out / "disparity.png"), vis["disparity"])
    cv2.imwrite(str(out / "detections.png"), vis["detections"])

    hm = result.heights.copy()
    finite = np.isfinite(hm)
    if finite.any():
        norm = np.zeros_like(hm, dtype=np.uint8)
        norm[finite] = np.clip((hm[finite] + 0.15) / 0.30 * 255, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_COOL)
        heat[~finite] = 0
        cv2.imwrite(str(out / "height_above_road.png"), heat)

    if args.ply:
        n = points_to_ply(result.points, left.reshape(-1, 3)[: result.points.size // 3],
                          str(out / "cloud.ply"))
        print(f"\nwrote {n} points to {out/'cloud.ply'}")

    (out / "summary.json").write_text(json.dumps({
        "summary": summary,
        "detections": [d.to_dict() for d in result.detections],
        "ground_truth": [vars(p) for p in scene.potholes],
    }, indent=2))

    print(f"\nwrote images and summary.json to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
