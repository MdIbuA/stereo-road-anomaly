"""Latency benchmarking and the ONNX inference path.

The lab's stack is TensorRT/ONNX on Jetson Orin. No Jetson is available here, so
this measures the ONNX Runtime path on CPU and states plainly what it does and
does not predict about Jetson performance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    runs: int

    def __str__(self) -> str:
        return (f"{self.name:<34} mean {self.mean_ms:7.2f} ms  "
                f"p50 {self.p50_ms:7.2f}  p95 {self.p95_ms:7.2f}  "
                f"{self.fps:6.1f} FPS")


def benchmark(fn: Callable[[], object], name: str, runs: int = 20,
              warmup: int = 3) -> BenchResult:
    """Time a callable. Warm-up runs are discarded: the first call pays for
    lazy allocation and cache population and is not representative."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return BenchResult(
        name=name,
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        fps=1000.0 / float(arr.mean()),
        runs=runs,
    )


class OnnxDetector:
    """Thin ONNX Runtime wrapper for a learned pothole detector.

    The classical pipeline localises anomalies geometrically. A learned detector
    is complementary: it recognises potholes from appearance where stereo has no
    disparity (wet, textureless, or beyond range). Train one separately, export
    to ONNX with scripts/export_onnx.py, and point this at the file.
    """

    def __init__(self, model_path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Dynamic axes come back as strings; fall back to a common default.
        self.input_shape = tuple(d if isinstance(d, int) else 640 for d in shape)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2

        _, _, h, w = self.input_shape
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        chw = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        return np.ascontiguousarray(chw[None], dtype=np.float32) / 255.0

    def __call__(self, image: np.ndarray):
        return self.session.run(None, {self.input_name: self.preprocess(image)})
