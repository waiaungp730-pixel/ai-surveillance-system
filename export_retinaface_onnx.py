"""
Export RetinaFace to ONNX for faster inference via onnxruntime.

  python export_retinaface_onnx.py [--output retinaface.onnx] [--opset 17]

The serengil/retina-face package ships a Keras model; this script exports it
via tf2onnx when available, or falls back to a PyTorch ONNX export when a
compatible PyTorch RetinaFace build is installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _export_keras(opset: int, output_path: Path) -> None:
    import tensorflow as tf
    from retinaface.model import retinaface_model

    try:
        import tf2onnx
    except ImportError as exc:
        raise RuntimeError(
            "tf2onnx is required to export the Keras RetinaFace model.\n"
            "  pip install tf2onnx"
        ) from exc

    keras_model = retinaface_model.build_model()
    input_sig = [tf.TensorSpec((None, None, None, 3), tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_sig, opset=opset)

    with open(output_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())


def _export_pytorch(opset: int, output_path: Path) -> None:
    import torch

    try:
        from retinaface.model.retinaface_model import RetinaFaceModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch RetinaFaceModel not found; use the default Keras export."
        ) from exc

    model = RetinaFaceModel()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("RetinaFace PyTorch export: .to('cuda')")
    model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}},
    )


def _verify(output_path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    print("ONNX inputs :", inputs)
    print("ONNX outputs:", outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RetinaFace to ONNX")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("retinaface.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--backend",
        choices=("keras", "pytorch", "auto"),
        default="auto",
        help="Export backend (auto tries keras/tf2onnx first)",
    )
    args = parser.parse_args()
    output_path = args.output.resolve()

    if args.backend in ("auto", "keras"):
        try:
            _export_keras(args.opset, output_path)
        except Exception as keras_exc:
            if args.backend == "keras":
                raise
            print(f"[INFO] Keras export failed ({keras_exc}); trying PyTorch…")
            _export_pytorch(args.opset, output_path)
    else:
        _export_pytorch(args.opset, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.2f} MiB)")
    _verify(output_path)
    print("Export complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
