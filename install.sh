#!/bin/bash
# Installation script for jetson-apriltag

set -e

echo "=========================================="
echo "jetson-apriltag Installation"
echo "=========================================="
echo ""

# Check if apriltags_cuda binary exists
BINARY_PATH="$HOME/apriltags_cuda/build/opencv_cuda_demo"
if [ ! -f "$BINARY_PATH" ]; then
    echo "⚠️  WARNING: CUDA binary not found at $BINARY_PATH"
    echo "   Please build apriltags_cuda first (see INSTALL.md)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install package in development mode
echo "Installing jetson-apriltag package..."
pip3 install -e .

echo ""
echo "✅ Installation complete!"
echo ""
echo "Quick test:"
echo "  python3 -c 'from jetson_apriltag import ApriltagDetector; print(\"OK\")'"
echo ""
