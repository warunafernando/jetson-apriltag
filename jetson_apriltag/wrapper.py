#!/usr/bin/env python3
"""Core subprocess-based wrapper around the 971/766 CUDA AprilTag C++ binary.

This is intentionally simple and robust:
- No in-process CUDA, no pybind11, no direct GPU pointers in Python.
- All heavy lifting happens in the `opencv_cuda_demo` C++ binary.
- Python spawns that binary via `subprocess.run`, with a timeout.

The public API is `ApriltagDetector`, exported from `jetson_apriltag.__init__`.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


DEFAULT_BINARY = os.path.expanduser("~/apriltags_cuda/build/opencv_cuda_demo")


@dataclass
class DetectorResult:
    """Result of running the C++ detector for a fixed duration.

    Attributes:
        exit_code: Process return code; <0 usually means the OS delivered a
            signal (e.g. -15 = SIGTERM from timeout, -11 = SIGSEGV).
        duration_sec: Wall-clock duration of the run, in seconds.
        stdout: Raw stdout text from the C++ binary.
        stderr: Raw stderr text from the C++ binary.
    """

    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str

    @property
    def crashed(self) -> bool:
        """Return True if the detector appears to have crashed.

        Heuristic:
        - 0   : clean exit (not expected; binary usually runs forever)
        - -15 : SIGTERM from our timeout (normal for this wrapper)
        - -11 : SIGSEGV (real crash)
        """

        if self.exit_code in (0, -15):
            return False
        return self.exit_code < 0


class ApriltagDetector:
    """High-level detector interface, similar in spirit to 971/766 wrappers.

    This does **not** expose a per-frame `detect(image)` like their in-process
    CUDA Python bindings; instead, it runs the continuous-loop C++ binary for a
    fixed duration and gives you the raw output to analyze FPS and detections.

    Parameters
    ----------
    camera: int
        Camera index (e.g. 0).
    family: str
        AprilTag family, e.g. "tag36h11".
    binary_path: str, optional
        Path to `opencv_cuda_demo`. Defaults to `DEFAULT_BINARY`.
    show: bool
        If True, passes `-show` to enable the OpenCV GUI window
        (requires X11 / a display).
    """

    def __init__(
        self,
        camera: int = 0,
        family: str = "tag36h11",
        binary_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        if binary_path is None:
            binary_path = DEFAULT_BINARY

        self.binary_path = binary_path
        self.camera = camera
        self.family = family
        self.show = show

        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

    def run(
        self,
        duration_sec: float = 5.0,
        quiet: bool = True,
        verbose: bool = False,
    ) -> DetectorResult:
        """Run the detector binary for a fixed duration.

        Parameters
        ----------
        duration_sec: float
            Timeout in seconds; the subprocess will be killed after this.
        quiet: bool
            If True, pass `-quiet` to reduce C++ output.
        verbose: bool
            If True, pass `-verbose` to enable per-detection logging.

        Returns
        -------
        DetectorResult
            Contains exit code, runtime, and raw stdout/stderr.
        """

        cmd = [
            self.binary_path,
            "-camera",
            str(self.camera),
            "-family",
            self.family,
        ]

        if quiet:
            cmd.append("-quiet")
        if verbose:
            cmd.append("-verbose")
        if self.show:
            cmd.append("-show")

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                timeout=duration_sec,
                capture_output=True,
                text=True,
            )
            end = time.time()
            return DetectorResult(
                exit_code=proc.returncode,
                duration_sec=end - start,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as e:
            # Process was killed by timeout; treat as non-crash.
            end = time.time()
            return DetectorResult(
                exit_code=-15,
                duration_sec=end - start,
                stdout=(e.stdout or ""),
                stderr=(e.stderr or ""),
            )


if __name__ == "__main__":
    # Simple manual test when running this module directly.
    det = ApriltagDetector(camera=0, family="tag36h11", show=False)
    res = det.run(duration_sec=5.0, quiet=True, verbose=False)
    print("Exit code:", res.exit_code)
    print("Duration:", f"{res.duration_sec:.2f}s")
    print("Crashed?:", res.crashed)

