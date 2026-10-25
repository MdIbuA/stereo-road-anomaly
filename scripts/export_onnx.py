#!/usr/bin/env python3
"""Export a trained PyTorch detector to ONNX, then verify the export.

Run this on a machine with torch installed. The verification step is the point:
an export that silently changes numerics is worse than no export, and
`torch.onnx.export` succeeding tells you nothing about whether outputs match.

    python scripts/export_onnx.py --weights runs/best.pt --out models/detector.onnx
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="models/detector.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError:
        print("torch is required for export. pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu", file=sys.stderr)
        return 1

    # Ultralytics exports in one line; this path is for a plain nn.Module.
    model = torch.load(args.weights, map_location="cpu", weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    model.eval().float()

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["images"], output_names=["output"],
        opset_version=args.opset,
        # Dynamic batch only. Dynamic spatial dims stop TensorRT from picking
        # optimal kernels, which costs more than the flexibility is worth here.
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"exported -> {args.out}")

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"images": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy)
    torch_out = (torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    print(f"max |torch - onnx| = {max_diff:.3e}")
    if max_diff > 1e-3:
        print("FAIL: outputs diverge. Check unsupported ops or opset.", file=sys.stderr)
        return 1
    print("verified: ONNX matches PyTorch within tolerance")
    print("\nJetson Orin next step:")
    print("  trtexec --onnx=%s --saveEngine=detector.int8.engine \\" % args.out)
    print("          --int8 --calib=calib.cache --workspace=4096")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
