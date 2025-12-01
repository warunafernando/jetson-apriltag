#!/bin/bash
# Arducam Setup and Driver Installation Script for Jetson

set -e

echo "=========================================="
echo "Arducam Setup for Jetson Orin"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "Step 1: Checking current camera configuration..."
echo ""

# Check current camera overlay
CURRENT_OVERLAY=$(grep "OVERLAYS" /boot/extlinux/extlinux.conf | awk '{print $2}' | xargs basename)
echo "Current camera overlay: $CURRENT_OVERLAY"

# Check for video devices
VIDEO_DEVICES=$(ls /dev/video* 2>/dev/null | wc -l)
if [ "$VIDEO_DEVICES" -gt 0 ]; then
    print_status "Found $VIDEO_DEVICES video device(s)"
    ls -la /dev/video*
else
    print_warning "No video devices found (/dev/video*)"
fi

echo ""
echo "Step 2: Checking for Arducam installation..."
echo ""

# Check if Arducam drivers are installed
ARDUCAM_INSTALLED=false

if [ -d "/opt/arducam" ] || [ -f "/usr/local/bin/arducam_displayer" ]; then
    ARDUCAM_INSTALLED=true
    print_status "Arducam directory/tools found"
else
    print_warning "Arducam drivers not found"
fi

# Check for Arducam kernel modules
if lsmod | grep -q "arducam"; then
    print_status "Arducam kernel modules loaded"
    lsmod | grep arducam
else
    print_warning "Arducam kernel modules not loaded"
fi

echo ""
echo "Step 3: Checking I2C buses for camera devices..."
echo ""

# Check I2C buses
for i2c_bus in 0 1 2 4 5; do
    if [ -e "/dev/i2c-$i2c_bus" ]; then
        echo "I2C-$i2c_bus devices:"
        i2cdetect -y $i2c_bus 2>/dev/null | grep -E "UU|:[0-9a-f]{2}" || echo "  No devices detected"
        echo ""
    fi
done

echo ""
echo "Step 4: Camera detection test..."
echo ""

# Test with OpenCV
python3 << 'PYTHON_SCRIPT'
import cv2
import sys

print("Testing camera access...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Camera {i}: Working (resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
        else:
            print(f"  ⚠ Camera {i}: Opened but cannot read frames")
        cap.release()
    else:
        if i == 0:
            print(f"  ✗ Camera {i}: Not accessible")
        break
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "INSTALLATION OPTIONS"
echo "=========================================="
echo ""
echo "If Arducam drivers are not installed, you can:"
echo ""
echo "Option 1: Install Arducam MIPI Camera Driver"
echo "  cd ~"
echo "  wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh"
echo "  chmod +x install_full.sh"
echo "  ./install_full.sh -m arducam"
echo ""
echo "Option 2: Install Arducam USB Camera Driver (if USB-based)"
echo "  # USB cameras typically work with UVC drivers"
echo "  sudo apt install -y v4l-utils"
echo ""
echo "Option 3: Use Jetson-IO to configure camera overlay"
echo "  sudo /opt/nvidia/jetson-io/jetson-io.py"
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Physically connect your Arducam camera"
echo "2. Check if it's USB or MIPI CSI"
echo "3. Install appropriate drivers"
echo "4. Configure device tree overlay if needed"
echo "5. Reboot if driver installation requires it"
echo ""

