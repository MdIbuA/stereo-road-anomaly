#!/usr/bin/env python3
"""Run the pipeline on a real KITTI stereo pair.

Expected layout (KITTI raw or stereo-2015 synced+rectified):

    data/kitti/
      calib_cam_to_cam.txt
      image_02/data/0000000000.png     <- left colour, rectified
      image_03/data/0000000000.png     <- right colour, rectified

Download: https://www.cvlibs.net/datasets/kitti/
Use the *rectified* (synced+rectified) archives. The pipeline assumes rectified
input and does not run stereoRectify itself; feeding it unrectified images
produces a plausible-looking but meaningless disparity map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import from_kitti
from src.pipeline import PipelineConfig, StereoAnomalyPipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, help="left rectified image")
    ap.add_argument("--right", required=True, help="right rectified image")
    ap.add_argument("--calib", required=True, help="calib_cam_to_cam.txt")
    ap.add_argument("--out", default="outputs/kitti")
    ap.add_argument("--disp-sigma", type=float, default=0.25,
                    help="1-sigma disparity noise in px; re-measure on real data")
    args = ap.parse_args()

    left, right = cv2.imread(args.left), cv2.imread(args.right)
    if left is None:
        raise SystemExit(f"could not read {args.left}")
    if right is None:
        raise SystemExit(f"could not read {args.right}")

    calib = from_kitti(args.calib)
    print(f"calib: fx={calib.fx:.2f}px baseline={calib.baseline_m:.4f}m "
          f"{calib.width}x{calib.height}")

    if (left.shape[1], left.shape[0]) != (calib.width, calib.height):
        print(f"WARNING: image is {left.shape[1]}x{left.shape[0]} but calibration "
              f"says {calib.width}x{calib.height}. KITTI crops vary between "
              f"sequences; cx/cy are now wrong and depths will be biased.")

    cfg = PipelineConfig(disp_sigma_px=args.disp_sigma)
    result = StereoAnomalyPipeline(calib, cfg).run(left, right)

    print(json.dumps(result.summary(), indent=2))
    if result.coverage < 0.4:
        print(f"\nWARNING: only {result.coverage:.1%} of pixels matched. Check "
              f"rectification and raise num_disparities.")

    for det in result.detections:
        print(f"  z={det.depth_m:6.2f}m depth={det.max_depth_below_road_m*100:5.1f}cm "
              f"area={det.area_m2:.3f}m2 {det.severity}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vis = StereoAnomalyPipeline.visualise(left, result)
    cv2.imwrite(str(out / "detections.png"), vis["detections"])
    cv2.imwrite(str(out / "disparity.png"), vis["disparity"])
    (out / "summary.json").write_text(json.dumps({
        "summary": result.summary(),
        "detections": [d.to_dict() for d in result.detections],
    }, indent=2))
    print(f"\nwrote results to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
