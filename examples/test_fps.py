#!/usr/bin/env python3
"""
FPS and stability test for the C++ CUDA AprilTag detector via the Jetson wrapper.

This script:
- Runs the opencv_cuda_demo binary through CppApriltagDetector
- Enables -verbose so each processed frame prints a \"tag #:\" line
- Counts detections and estimates effective FPS over a given duration
- Reports whether the detector crashed (segfault) or exited normally via timeout
"""

import argparse
import re
from typing import Tuple, Optional

from jetson_apriltag import ApriltagDetector


def parse_camera_fps(stdout: str) -> Optional[float]:
    """
    Parse the FPS that OpenCV reports for the camera, e.g.:
      \"  1280.000x720.000 @120.000FPS\"
    """
    for raw in stdout.splitlines():
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        m = re.search(r"@([0-9]+(?:\\.[0-9]+)?)FPS", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def estimate_detection_fps(stdout: str, duration_sec: float) -> Tuple[int, float]:
    """
    Estimate FPS by counting 'tag #:' lines.

    Note: assumes roughly one detection block per frame. If multiple tags per
    frame, this will slightly over-estimate FPS, but it's still a useful lower
    bound on effective processing rate.
    """
    count = 0
    for raw in stdout.splitlines():
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        if "tag #:" in line:
            count += 1

    fps = count / duration_sec if duration_sec > 0 else 0.0
    return count, fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test AprilTag CUDA detector FPS via C++ binary wrapper"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--family",
        type=str,
        default="tag36h11",
        help="AprilTag family (e.g. tag36h11)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Run duration in seconds (timeout for the C++ binary)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Enable GUI display (-show); requires X11 / display",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Jetson AprilTag CUDA FPS Test (C++ binary via Python wrapper)")
    print("=" * 70)
    print()
    print(f"Camera:   {args.camera}")
    print(f"Family:   {args.family}")
    print(f"Duration: {args.duration:.1f} s")
    print(f"Show GUI: {args.show}")
    print()

    detector = ApriltagDetector(camera=args.camera, family=args.family, show=args.show)

    print("Running detector (verbose mode, using -verbose for per-frame logs)...")
    res = detector.run(
        duration_sec=args.duration,
        quiet=False,   # we want stdout
        verbose=True,  # enable per-detection logs
    )

    print()
    print("-" * 70)
    print(f"Exit code: {res.exit_code}")
    print(f"Crashed?:  {res.crashed}")
    print(f"Runtime:   {res.duration_sec:.2f} s")

    cam_fps = parse_camera_fps(res.stdout)
    det_count, det_fps = estimate_detection_fps(res.stdout, res.duration_sec)

    if cam_fps is not None:
        print(f"Camera-reported FPS (OpenCV CAP_PROP_FPS): {cam_fps:.1f}")
    else:
        print("Camera-reported FPS: (not found in output)")

    print(f"Detections counted via 'tag #:' lines: {det_count}")
    print(f"Estimated detection FPS (with verbose logging on): {det_fps:.1f}")
    print("-" * 70)

    # Show last few lines of output for sanity
    lines = res.stdout.splitlines()
    tail = lines[-30:] if len(lines) > 30 else lines
    print("Tail of detector output:")
    for raw in tail:
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        print(line)


if __name__ == "__main__":
    main()


