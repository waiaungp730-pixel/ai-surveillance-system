#!/usr/bin/env python3
"""
Dedicated real-time license plate capture (separate from face surveillance).

ANPR two-step: YOLOv8 plate detector + PaddleOCR via ``license_plate_ocr.py``.
Saves plate crops under ``plate_captures/`` and prints recognized text.

Install (if needed):
    pip install opencv-python ultralytics paddleocr paddlepaddle numpy

Usage:
    py -3.12 plate_capture.py --all                 # all cameras in cameras.json
    py -3.12 plate_capture.py --gates               # CAM-28, 29, 30
    py -3.12 plate_capture.py --cam 3               # one camera
    py -3.12 plate_capture.py --cam 28,29,30        # selected cameras
    py -3.12 plate_capture.py --webcam 0            # local webcam
    py -3.12 plate_capture.py image.jpg             # single image file

Press Ctrl+C to stop live capture.
"""

from __future__ import annotations

import numpy_compat  # noqa: F401  — NumPy 1.x / 2.x import bridge

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from license_plate_ocr import (
    GATE_CAMERA_IDS,
    PLATE_SAVE_DIR,
    detect_and_read_plate,
    load_all_cctv_cameras,
    parse_cctv_ids,
    run_all_cctv_realtime,
    run_gate_cctv_realtime,
    run_multi_cctv_realtime,
    run_realtime,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture license plates from CCTV (separate plate-capture tool).",
    )
    p.add_argument(
        "image",
        nargs="?",
        default="",
        help="Optional still image path (offline OCR).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Capture from every camera in cameras.json.",
    )
    p.add_argument(
        "--gates",
        action="store_true",
        help="Capture from gate cameras (default LICENSE_PLATE_CAMERAS: 3,28,29,30).",
    )
    p.add_argument(
        "--cam",
        metavar="IDS",
        default="",
        help="Camera id(s) from cameras.json, e.g. 3 or 28,29,30.",
    )
    p.add_argument(
        "--webcam",
        metavar="INDEX",
        type=int,
        default=None,
        help="Local webcam index (usually 0).",
    )
    p.add_argument(
        "--every",
        type=int,
        default=0,
        help="OCR every N frames (default: 15 for --all, 8 for fewer cams).",
    )
    p.add_argument(
        "--save-dir",
        default="",
        help=f"Folder for plate JPEGs (default: {PLATE_SAVE_DIR.name}/).",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window (single-cam / webcam).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    save_dir = Path(args.save_dir) if args.save_dir else PLATE_SAVE_DIR

    # --- Still image ---
    if args.image:
        result = detect_and_read_plate(args.image)
        if result.get("error"):
            print(result["error"], file=sys.stderr)
        for i, plate in enumerate(result.get("plates") or [], start=1):
            print(
                f"Plate {i}: {plate.get('text')!r} "
                f"(det={plate.get('confidence', 0):.0%} "
                f"ocr={plate.get('ocr_confidence', 0):.0%})"
            )
        if result.get("annotated_image") is not None:
            out = save_dir / f"annotated_{Path(args.image).stem}.jpg"
            save_dir.mkdir(parents=True, exist_ok=True)
            import cv2

            cv2.imwrite(str(out), result["annotated_image"])
            print(f"Saved -> {out}")
        return 0 if result.get("success") else 1

    # --- Live: all cameras ---
    if args.all:
        every = args.every if args.every > 0 else 15
        print(f"[PLATE-CAPTURE] ALL cameras | every={every} | saves -> {save_dir}")
        run_all_cctv_realtime(
            ocr_every=every,
            save_dir=save_dir,
        )
        return 0

    # --- Live: gate cameras ---
    if args.gates:
        every = args.every if args.every > 0 else 8
        print(
            f"[PLATE-CAPTURE] Gates {', '.join(GATE_CAMERA_IDS)} "
            f"| every={every} | saves -> {save_dir}"
        )
        run_gate_cctv_realtime(
            ocr_every=every,
            save_dir=save_dir,
        )
        return 0

    # --- Live: selected cameras ---
    if args.cam:
        ids = parse_cctv_ids(args.cam)
        cameras = load_all_cctv_cameras(ids)
        if not cameras:
            print(f"No cameras matched: {args.cam}", file=sys.stderr)
            return 2
        every = args.every if args.every > 0 else 8
        names = ", ".join(f"CAM-{c['id']}" for c in cameras)
        print(f"[PLATE-CAPTURE] {names} | every={every} | saves -> {save_dir}")
        if len(cameras) == 1:
            cam = cameras[0]
            run_realtime(
                cam["source"],
                display=not args.no_display,
                ocr_every=every,
                save_dir=save_dir,
                camera_id=cam["id"],
            )
        else:
            run_multi_cctv_realtime(
                cameras,
                label="PLATE-CAPTURE",
                ocr_every=every,
                save_dir=save_dir,
            )
        return 0

    # --- Live: webcam ---
    if args.webcam is not None:
        every = args.every if args.every > 0 else 5
        print(f"[PLATE-CAPTURE] Webcam {args.webcam} | every={every}")
        run_realtime(
            args.webcam,
            display=not args.no_display,
            ocr_every=every,
            save_dir=save_dir,
            camera_id=f"webcam{args.webcam}",
        )
        return 0

    _build_parser().print_help()
    print(
        "\nExamples:\n"
        "  py -3.12 plate_capture.py --all\n"
        "  py -3.12 plate_capture.py --gates\n"
        "  py -3.12 plate_capture.py --cam 28,29,30\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
