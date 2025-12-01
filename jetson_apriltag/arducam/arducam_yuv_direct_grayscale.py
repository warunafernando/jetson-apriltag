#!/usr/bin/env python3
"""
Arducam Grayscale using direct YUV Y-channel extraction
Camera outputs YUYV (already YUV format) - extract Y channel directly
Uses GPU/CUDA acceleration when available, falls back to CPU
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time
import glob

def extract_y_from_yuyv(frame, gpu_frame=None, use_gpu=True):
    """
    Extract Y (luminance) channel from YUYV format frame
    Uses GPU/CUDA acceleration when available
    
    Args:
        frame: Input BGR frame (numpy array)
        gpu_frame: Pre-allocated GPU memory for frame (cv2.cuda_GpuMat)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Grayscale frame (Y channel only)
        If GPU used, returns GPU GpuMat; otherwise returns numpy array
    """
    if len(frame.shape) == 2:
        # Already grayscale
        return frame
    
    # Check if CUDA is available
    cuda_available = use_gpu and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    if cuda_available and gpu_frame is not None:
        # GPU path: all operations on GPU
        gpu_frame.upload(frame)
        gpu_yuv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2YUV)
        channels = cv2.cuda.split(gpu_yuv)
        return channels[0]  # Return Y channel on GPU
    else:
        # CPU path: convert BGR to YUV and extract Y channel
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 0]  # Y channel is the first channel

def find_available_camera(max_tries=5):
    """
    Find first available camera by trying multiple indices and backends
    Returns (camera_index, backend, device_path) or None if not found
    """
    print("Searching for available camera...")
    
    # Check for /dev/video devices first
    video_devices = sorted(glob.glob('/dev/video*'))
    if video_devices:
        print(f"Found video devices: {', '.join(video_devices)}")
    
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "ANY")
    ]
    
    for camera_idx in range(max_tries):
        for backend_code, backend_name in backends:
            try:
                cap = cv2.VideoCapture(camera_idx, backend_code)
                if cap.isOpened():
                    # Try to read a frame to verify it works
                    ret, _ = cap.read()
                    if ret:
                        cap.release()
                        device = f"/dev/video{camera_idx}" if camera_idx < len(video_devices) else "unknown"
                        print(f"  ✓ Found camera at index {camera_idx} using {backend_name} ({device})")
                        return (camera_idx, backend_code, device)
                    cap.release()
            except:
                pass
    
    print("  ✗ No working camera found")
    return None

def test_direct_yuv_grayscale(camera_index=0, width=1280, height=720, fps=120,
                               duration=10, use_v4l2=True, display=False, display_host=None):
    """
    Test direct YUV Y-channel extraction (most efficient method)
    """
    print("="*70)
    print("Direct YUV Y-Channel Grayscale Extraction")
    print("="*70)
    print()
    
    if display_host:
        os.environ['DISPLAY'] = display_host
    
    # Check CUDA availability
    cuda_available = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Acceleration: {'GPU (CUDA)' if cuda_available else 'CPU'}")
    print(f"  Method: YUV → Y channel (luminance) extraction")
    print(f"  Camera format: YUYV (already YUV, extracting Y values)")
    print()
    
    # Try to find camera if the specified one doesn't work
    backend = cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    
    if not cap.isOpened() or not cap.read()[0]:
        print(f"Camera {camera_index} not available, searching...")
        cap.release()
        found = find_available_camera()
        if found:
            camera_index, backend, device_path = found
            cap = cv2.VideoCapture(camera_index, backend)
        else:
            print(f"\nError: No camera found!")
            print(f"  • Check if camera is connected")
            print(f"  • Check USB connection (for USB3 Arducam)")
            print(f"  • Run: lsusb | grep -i camera")
            print(f"  • Run: ls -la /dev/video*")
            return None
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera {camera_index}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened:")
    print(f"  Resolution: {actual_w}x{actual_h}")
    print(f"  FPS: {actual_fps}")
    print(f"  Backend: V4L2" if use_v4l2 else "  Backend: Default")
    print()
    
    # Initialize GPU resources if available
    gpu_frame = None
    if cuda_available:
        print("Initializing GPU resources...")
        ret, test_frame = cap.read()
        if ret:
            gpu_frame = cv2.cuda_GpuMat()
            print("✓ GPU initialized")
        print()
    
    # Warm up
    print("Warming up...")
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            _ = extract_y_from_yuyv(frame, gpu_frame, cuda_available)
    print()
    
    # Test FPS
    print(f"Measuring FPS for {duration} seconds...")
    print()
    
    frame_count = 0
    start_time = time.time()
    conversion_times = []
    
    window_name = "YUV Y-Channel Grayscale"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract Y channel (this is the key operation)
            conv_start = time.time()
            gray = extract_y_from_yuyv(frame, gpu_frame, cuda_available)
            conv_time = (time.time() - conv_start) * 1000
            conversion_times.append(conv_time)
            
            frame_count += 1
            
            # Display every 10th frame to avoid X11 bottleneck (120 FPS / 10 = 12 FPS display)
            # This allows capture to run at full speed while display updates smoothly
            if display and frame_count % 10 == 0:
                # Download from GPU if needed
                if cuda_available and isinstance(gray, cv2.cuda_GpuMat):
                    display_frame = gray.download()
                else:
                    display_frame = gray
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Progress update
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                avg_conv = sum(conversion_times[-60:]) / min(60, len(conversion_times))
                display_rate = current_fps / 10 if display else 0
                if display:
                    print(f"Frames: {frame_count} | Capture FPS: {current_fps:.2f} | Display FPS: ~{display_rate:.1f} | Y-extract: {avg_conv:.2f}ms", end='\r')
                else:
                    print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Y-extract: {avg_conv:.2f}ms", end='\r')
            
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        # Clean up GPU resources
        if gpu_frame is not None:
            gpu_frame.release()
    
    # Statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_conv = sum(conversion_times) / len(conversion_times) if conversion_times else 0
    
    print(f"\n\nResults:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average Y-extraction time: {avg_conv:.2f}ms")
    print(f"  Processing: {'GPU (CUDA)' if cuda_available else 'CPU'}")
    print(f"  Target FPS: {fps}")
    print(f"  Performance: {(avg_fps/fps*100):.1f}% of target")
    print()
    
    return {
        'fps': avg_fps,
        'frames': frame_count,
        'conversion_time_ms': avg_conv,
        'performance_pct': (avg_fps/fps*100) if fps > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Direct YUV Y-channel Grayscale')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    parser.add_argument('--height', type=int, default=720, help='Height')
    parser.add_argument('--fps', type=int, default=120, help='FPS')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--display-host', type=str, default='192.168.68.31:0.0',
                       help='X display host')
    parser.add_argument('--no-v4l2', action='store_true', help='Don\'t use V4L2')
    
    args = parser.parse_args()
    
    result = test_direct_yuv_grayscale(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        use_v4l2=not args.no_v4l2,
        display=args.display,
        display_host=args.display_host if args.display else None
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()

