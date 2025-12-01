"""jetson_apriltag: Python wrapper around Team 971/766 CUDA AprilTag detector.

This package provides a high-level API similar in spirit to the FRC 971/766
wrappers, but uses a **subprocess-based** approach for robustness on Jetson:

- The actual CUDA AprilTag detector runs in the existing C++ binary
  `opencv_cuda_demo` from the `apriltags_cuda` repo.
- Python talks to this binary via `subprocess.run`, so any CUDA crashes or
  driver issues cannot take down your Python process.

Typical usage:

```python
from jetson_apriltag import ApriltagDetector

# Create detector on camera 0, family tag36h11
det = ApriltagDetector(camera=0, family="tag36h11", show=False)

# Run for 5 seconds, with verbose logging for FPS/detection analysis
res = det.run(duration_sec=5.0, quiet=False, verbose=True)

print("exit:", res.exit_code, "crashed:", res.crashed)
print("runtime:", res.duration_sec)
print(res.stdout)
```

You must have a working `/home/nav/apriltags_cuda/build/opencv_cuda_demo`
binary built from the Team 766/971 `apriltags_cuda` project.
"""

from .wrapper import ApriltagDetector, DetectorResult

__all__ = [
    "ApriltagDetector",
    "DetectorResult",
]
