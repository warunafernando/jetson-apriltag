# Installation Guide

Complete installation guide for jetson-apriltag on NVIDIA Jetson.

## Prerequisites

- NVIDIA Jetson device (Orin recommended)
- Ubuntu 20.04+ / JetPack 5.0+
- Internet connection
- sudo access

## Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    v4l-utils \
    libopencv-dev
```

## Step 2: Install CUDA Toolkit

If not already installed:

```bash
# Check CUDA version
nvcc --version

# If missing, install CUDA 11.8+ (adjust for your JetPack version)
# Follow NVIDIA's JetPack installation guide
```

## Step 3: Build Team 766/971 CUDA AprilTag Library

```bash
cd ~
git clone https://github.com/Team766/apriltags_cuda.git
cd apriltags_cuda

# Install dependencies
sudo ./install_deps.sh

# Determine your GPU compute capability
# Jetson Orin: 8.7
# Jetson Xavier: 7.2
# Jetson Nano: 5.3

# Build (adjust CMAKE_CUDA_ARCHITECTURES for your device)
cmake -B build \
    -DCMAKE_CUDA_COMPILER=clang++-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DNUM_PROCESSORS=2

cmake --build build -j$(nproc)

# Verify binary
ls -lh build/opencv_cuda_demo
```

## Step 4: Install jetson-apriltag

```bash
cd ~
git clone <YOUR_GITHUB_REPO_URL> jetson_apriltag
cd jetson_apriltag

# Install Python package
pip3 install -e .

# Or install dependencies manually
pip3 install -r requirements.txt
```

## Step 5: Verify Installation

```bash
# Test basic import
python3 -c "from jetson_apriltag import ApriltagDetector; print('OK')"

# Test detector (short run)
python3 -c "
from jetson_apriltag import ApriltagDetector
det = ApriltagDetector(camera=0, family='tag36h11')
res = det.run(duration_sec=2.0, quiet=True)
print(f'Exit: {res.exit_code}, Crashed: {res.crashed}')
"
```

## Step 6: Configure Cameras

### USB Cameras

```bash
# List cameras
ls -la /dev/video*
v4l2-ctl --list-devices

# Test camera
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### ArduCam Setup

```bash
cd ~/jetson_apriltag
bash scripts/arducam_setup.sh
```

## Step 7: Test FPS Performance

```bash
cd ~/jetson_apriltag
python3 examples/test_fps.py --camera 0 --duration 10
```

Expected output:
- Exit code: -15 (timeout, normal)
- Crashed: False
- Detection FPS: 10-30 (depends on verbose mode)

## Troubleshooting

### Binary Not Found

```bash
# Check if binary exists
ls -lh ~/apriltags_cuda/build/opencv_cuda_demo

# If missing, rebuild
cd ~/apriltags_cuda
cmake --build build
```

### CUDA Not Found

```bash
# Check CUDA
nvcc --version
nvidia-smi

# If missing, reinstall JetPack or CUDA toolkit
```

### Camera Access Denied

```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

### NetworkTables Issues

```bash
# Install ntcore
pip3 install ntcore

# Test connection
python3 -c "import ntcore; print('OK')"
```

## Next Steps

- See [README.md](README.md) for usage examples
- Check [docs/](docs/) for detailed documentation
- Run `examples/test_fps.py` to benchmark performance

