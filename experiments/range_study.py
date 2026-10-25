#!/usr/bin/env python3
"""Measure the detector's actual operating range, and chart it.

This is the experiment that characterises the system. It renders potholes of
several depths across a range of distances and records whether each is detected,
producing the detectability envelope implied by the stereo error model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import kitti_like
from src.pipeline import PipelineConfig, StereoAnomalyPipeline
from src.synth import Pothole, RoadScene, render_stereo_pair

DEPTHS_CM = [3, 5, 8, 12]
RANGES_M = [5, 7, 9, 11, 14, 18, 22]


def main() -> int:
    out = Path("outputs"); out.mkdir(exist_ok=True)
    calib = kitti_like()
    pipeline = StereoAnomalyPipeline(calib, PipelineConfig())
    fb = calib.fx * calib.baseline_m

    results: dict[str, dict[str, bool]] = {}
    print(f"f*B = {fb:.1f} px.m   (one pixel of disparity error = Z^2/{fb:.0f} metres)\n")
    header = "depth \\ range " + "".join(f"{r:>7}m" for r in RANGES_M)
    print(header)

    for depth_cm in DEPTHS_CM:
        row = {}
        line = f"{depth_cm:>9} cm   "
        for z in RANGES_M:
            scene = RoadScene(
                camera_height_m=1.65,
                pitch_rad=0.012,
                potholes=[Pothole(x_m=0.0, z_m=float(z), depth_m=depth_cm / 100.0,
                                  radius_m=0.30)],
            )
            left, right, _, _ = render_stereo_pair(calib, scene)
            try:
                res = pipeline.run(left, right)
                hit = any(abs(d.depth_m - z) < max(1.5, 0.15 * z) for d in res.detections)
            except ValueError:
                hit = False
            row[str(z)] = bool(hit)
            line += f"{'  HIT  ' if hit else '   .   '}"
        results[str(depth_cm)] = row
        print(line)

    print("\nHIT = detected at the correct range. '.' = below the noise floor.")
    (out / "range_study.json").write_text(json.dumps(results, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.array([[results[str(d)][str(r)] for r in RANGES_M] for d in DEPTHS_CM])
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(RANGES_M)), [f"{r} m" for r in RANGES_M])
        ax.set_yticks(range(len(DEPTHS_CM)), [f"{d} cm" for d in DEPTHS_CM])
        ax.set_xlabel("Distance from camera")
        ax.set_ylabel("Pothole depth")
        ax.set_title("Detectability envelope (KITTI geometry, SGBM)")

        zs = np.linspace(RANGES_M[0], RANGES_M[-1], 200)
        # 3-sigma floor for a 400-pixel region, converted to the y-axis scale.
        floor_cm = 3.0 * 0.25 * zs ** 2 / fb / np.sqrt(400 / 8) * 100
        xs = np.interp(zs, RANGES_M, range(len(RANGES_M)))
        ys = np.interp(floor_cm, DEPTHS_CM, range(len(DEPTHS_CM)))
        ax.plot(xs, ys, "k--", lw=2, label="predicted 3-sigma noise floor")
        ax.legend(loc="upper left", fontsize=8)

        for i in range(len(DEPTHS_CM)):
            for j in range(len(RANGES_M)):
                ax.text(j, i, "HIT" if grid[i, j] else "", ha="center", va="center",
                        fontsize=8, weight="bold")
        fig.tight_layout()
        fig.savefig(out / "range_study.png", dpi=150)
        print(f"wrote {out/'range_study.png'}")
    except ImportError:
        print("matplotlib not installed; skipped the chart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
