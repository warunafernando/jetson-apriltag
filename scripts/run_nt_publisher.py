#!/usr/bin/env python3
"""
Jetson AprilTag → NetworkTables publisher, using the subprocess-based CUDA wrapper.

Usage (one process per camera, e.g. 4 cameras on the same Jetson):

  cd /home/nav/jetson_apriltag
  python3 run_nt_publisher.py \
      --camera 0 \
      --camera-name front \
      --server roborio-9202-frc.local

  python3 run_nt_publisher.py --camera 1 --camera-name back   --server roborio-9202-frc.local
  python3 run_nt_publisher.py --camera 2 --camera-name left   --server roborio-9202-frc.local
  python3 run_nt_publisher.py --camera 3 --camera-name right  --server roborio-9202-frc.local

Each instance:
- Runs the Team 971/766 CUDA AprilTag C++ binary via subprocess for short bursts
- Parses detections (id, hamming, margin, center, pose rows) from stdout
- Publishes results into NetworkTables under: /JetsonAprilTag/<cameraName>/*
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List

try:
    from ntcore import NetworkTableInstance
except ImportError:
    print("❌ ntcore Python module not found. Install WPILib's ntcore for Python.", file=sys.stderr)
    sys.exit(1)

from jetson_apriltag import ApriltagDetector


def parse_detections(stdout: str) -> List[Dict[str, Any]]:
    """
    Parse detections from opencv_cuda_demo verbose output.

    Expected block format (repeated):
        tag #: 7
        hamming: 0
        margin: 10.313
        center: 854.913,444.192

        -66.611 -90.596 854.913
        95.038 -48.135 444.192
        -0.014 0.018 1.000
    """
    dets: List[Dict[str, Any]] = []

    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "tag #:" in line:
            det: Dict[str, Any] = {}

            # tag id
            try:
                det["id"] = int(line.split("#:")[1].strip())
            except Exception:
                det["id"] = -1

            # next expected lines: hamming, margin, center
            if i + 1 < len(lines) and "hamming:" in lines[i + 1]:
                try:
                    det["hamming"] = int(lines[i + 1].split("hamming:")[1].strip())
                except Exception:
                    det["hamming"] = -1
            if i + 2 < len(lines) and "margin:" in lines[i + 2]:
                try:
                    det["margin"] = float(lines[i + 2].split("margin:")[1].strip())
                except Exception:
                    det["margin"] = 0.0
            if i + 3 < len(lines) and "center:" in lines[i + 3]:
                center_str = lines[i + 3].split("center:")[1].strip()
                try:
                    cx_str, cy_str = center_str.split(",")
                    det["center_x"] = float(cx_str)
                    det["center_y"] = float(cy_str)
                except Exception:
                    det["center_x"] = 0.0
                    det["center_y"] = 0.0

            # pose rows (3 lines after a blank line, if present)
            pose_rows: List[List[float]] = []
            # Skip optional blank line
            j = i + 4
            if j < len(lines) and not lines[j].strip():
                j += 1
            # Try to read up to 3 numeric rows
            for _ in range(3):
                if j < len(lines):
                    parts = lines[j].strip().split()
                    try:
                        row = [float(p) for p in parts]
                    except Exception:
                        row = []
                    if row:
                        pose_rows.append(row)
                    j += 1
            if pose_rows:
                det["pose_rows"] = pose_rows

            dets.append(det)
            i = j
        else:
            i += 1

    return dets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish Jetson AprilTag CUDA detections to NetworkTables"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--camera-name",
        type=str,
        default="cam0",
        help="Logical camera name (used in NT path, e.g. front/back/left/right)",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="tag36h11",
        help="AprilTag family name for the C++ detector",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="roborio-9202-frc.local",
        help="NetworkTables server (roboRIO hostname or IP)",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.5,
        help="Seconds of detector runtime per cycle (timeout duration_sec)",
    )

    args = parser.parse_args()

    print("============================================================")
    print("Jetson AprilTag NT Publisher")
    print("============================================================")
    print(f"Camera index : {args.camera}")
    print(f"Camera name  : {args.camera_name}")
    print(f"Family       : {args.family}")
    print(f"Server       : {args.server}")
    print(f"Period       : {args.period} s")
    print("============================================================")

    cam_name = args.camera_name  # easier alias

    # Set up NetworkTables client
    inst = NetworkTableInstance.getDefault()
    inst.startClient4(f"jetson_apriltag_{cam_name}")
    inst.setServer(args.server)

    table = inst.getTable(f"/JetsonAprilTag/{cam_name}")

    # Detector
    det = ApriltagDetector(camera=args.camera, family=args.family, show=False)

    try:
        while True:
            start_loop = time.time()
            res = det.run(duration_sec=args.period, quiet=False, verbose=True)
            loop_dur = time.time() - start_loop

            dets = parse_detections(res.stdout)

            # Basic summaries
            ids = [d.get("id", -1) for d in dets]
            centers_x = [d.get("center_x", 0.0) for d in dets]
            centers_y = [d.get("center_y", 0.0) for d in dets]

            # Publish to NT
            table.putString("cameraName", cam_name)
            table.putNumber("lastRunSec", res.duration_sec)
            table.putBoolean("crashed", res.crashed)
            table.putNumber("detectionCount", len(dets))
            table.putNumberArray("ids", ids)
            table.putNumberArray("centerX", centers_x)
            table.putNumberArray("centerY", centers_y)
            table.putString("detectionsJson", json.dumps(dets))

            # Flush and sleep a bit before next cycle
            inst.flush()
            # Aim for roughly args.period + small overhead per cycle
            sleep_time = max(0.0, args.period - (time.time() - start_loop))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        inst.stopClient()


if __name__ == "__main__":
    main()


