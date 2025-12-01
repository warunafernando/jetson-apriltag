# jetson-apriltag

High-performance CUDA-accelerated AprilTag detection for NVIDIA Jetson, using Team 971/766's detector via a robust subprocess-based Python wrapper.

## Features

- ✅ **CUDA-accelerated** AprilTag detection (Team 971/766 implementation)
- ✅ **Subprocess isolation** - No segfaults in your Python process
- ✅ **Multi-camera support** - Run multiple detectors simultaneously
- ✅ **NetworkTables integration** - Publish detections to roboRIO
- ✅ **ArduCam utilities** - Helper scripts for ArduCam camera setup
- ✅ **Production-ready** - Tested on Jetson Orin with 4 cameras

## Requirements

- NVIDIA Jetson (Orin recommended)
- CUDA toolkit installed
- Python 3.8+
- Built `apriltags_cuda` binary from [Team 766's repo](https://github.com/Team766/apriltags_cuda)
- NetworkTables (for robot integration)

## Installation

### 1. Build the CUDA AprilTag Library

First, build the Team 766/971 CUDA AprilTag detector:

```bash
cd ~
git clone https://github.com/Team766/apriltags_cuda.git
cd apriltags_cuda
sudo ./install_deps.sh
cmake -B build -DCMAKE_CUDA_COMPILER=clang++-17 -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build
```

Verify the binary exists:
```bash
ls -lh ~/apriltags_cuda/build/opencv_cuda_demo
```

### 2. Install jetson-apriltag

```bash
cd ~/jetson_apriltag
pip3 install -e .
```

Or install dependencies manually:
```bash
pip3 install ntcore
```

## Quick Start

### Basic Usage

```python
from jetson_apriltag import ApriltagDetector

# Create detector for camera 0
det = ApriltagDetector(camera=0, family="tag36h11", show=False)

# Run for 5 seconds
res = det.run(duration_sec=5.0, quiet=True, verbose=False)

print(f"Exit code: {res.exit_code}")
print(f"Crashed: {res.crashed}")
print(f"Duration: {res.duration_sec:.2f}s")
```

### FPS Testing

```bash
cd ~/jetson_apriltag
python3 examples/test_fps.py --camera 0 --duration 10
```

### NetworkTables Publisher (for Robot Integration)

Run one instance per camera:

```bash
# Front camera
python3 scripts/run_nt_publisher.py \
  --camera 0 \
  --camera-name front \
  --server roborio-9202-frc.local &

# Back camera
python3 scripts/run_nt_publisher.py \
  --camera 1 \
  --camera-name back \
  --server roborio-9202-frc.local &

# Left camera
python3 scripts/run_nt_publisher.py \
  --camera 2 \
  --camera-name left \
  --server roborio-9202-frc.local &

# Right camera
python3 scripts/run_nt_publisher.py \
  --camera 3 \
  --camera-name right \
  --server roborio-9202-frc.local &
```

## ArduCam Setup

The repository includes ArduCam utilities for camera configuration:

```bash
# Check camera status
bash scripts/arducam_setup.sh

# Use ArduCam utilities
python3 -m jetson_apriltag.arducam.<module>
```

## API Reference

### `ApriltagDetector`

Main detector class that wraps the C++ binary.

```python
det = ApriltagDetector(
    camera=0,                    # Camera index
    family="tag36h11",           # AprilTag family
    binary_path=None,            # Path to opencv_cuda_demo (auto-detected)
    show=False,                  # Enable OpenCV GUI window
)

res = det.run(
    duration_sec=5.0,            # How long to run
    quiet=True,                  # Reduce C++ output
    verbose=False,               # Enable per-detection logging
)
```

### `DetectorResult`

Result object returned by `det.run()`:

- `exit_code: int` - Process return code (-15 = timeout, -11 = segfault)
- `duration_sec: float` - Wall-clock runtime
- `stdout: str` - Raw stdout from C++ binary
- `stderr: str` - Raw stderr
- `crashed: bool` - True if real crash detected

## NetworkTables Schema

When using `run_nt_publisher.py`, detections are published under:

```
/JetsonAprilTag/<cameraName>/
  ├── cameraName: string
  ├── lastRunSec: float
  ├── crashed: boolean
  ├── detectionCount: int
  ├── ids: number[]
  ├── centerX: number[]
  ├── centerY: number[]
  └── detectionsJson: string (JSON array)
```

Example robot code (Java):

```java
NetworkTable table = NetworkTableInstance.getDefault()
    .getTable("/JetsonAprilTag/front");

long[] ids = table.getEntry("ids").getNumberArray(new long[0]);
double[] centerX = table.getEntry("centerX").getNumberArray(new double[0]);
double[] centerY = table.getEntry("centerY").getNumberArray(new double[0]);
```

## Project Structure

```
jetson_apriltag/
├── jetson_apriltag/          # Python package
│   ├── __init__.py
│   ├── wrapper.py            # Core detector wrapper
│   └── arducam/              # ArduCam utilities
├── scripts/                  # Standalone scripts
│   ├── run_nt_publisher.py   # NetworkTables publisher
│   └── arducam_setup.sh      # ArduCam setup script
├── examples/                  # Example scripts
│   └── test_fps.py           # FPS testing
├── docs/                      # Documentation
├── README.md
├── setup.py
├── requirements.txt
└── pyproject.toml
```

## Performance

- **Resolution**: 1280x720 @ 120 FPS (camera-dependent)
- **Detection FPS**: ~11-30 FPS (depends on verbose logging)
- **Latency**: <50ms per detection cycle
- **Stability**: No segfaults (subprocess isolation)

## Troubleshooting

### Binary Not Found

Ensure `~/apriltags_cuda/build/opencv_cuda_demo` exists:

```bash
ls -lh ~/apriltags_cuda/build/opencv_cuda_demo
```

If missing, rebuild the `apriltags_cuda` project.

### Camera Not Detected

Check camera devices:

```bash
ls -la /dev/video*
v4l2-ctl --list-devices
```

### NetworkTables Connection Issues

Verify roboRIO connectivity:

```bash
ping roborio-9202-frc.local
```

Or use IP address:

```bash
python3 scripts/run_nt_publisher.py --server 10.92.2.2 ...
```

## License

MIT License - See LICENSE file for details.

## Credits

- **Team 971** - Original CUDA AprilTag implementation
- **Team 766** - Standalone Jetson port ([apriltags_cuda](https://github.com/Team766/apriltags_cuda))
- **FRC Team 9202** - Python wrapper and integration

## Contributing

Contributions welcome! Please open an issue or pull request.
