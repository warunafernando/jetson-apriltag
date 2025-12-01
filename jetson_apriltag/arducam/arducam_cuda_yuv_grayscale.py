#!/usr/bin/env python3
"""
Arducam Grayscale using CUDA-accelerated YUV conversion
Uses NVIDIA CUDA/GPU to convert BGR to YUV and extract Y channel
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time

def extract_y_channel_cuda(frame, use_cuda=True):
    """
    Extract Y channel from frame using CUDA acceleration if available
    """
    if len(frame.shape) == 2:
        return frame
    
    if not use_cuda or not hasattr(cv2, 'cuda') or cv2.cuda.getCudaEnabledDeviceCount() == 0:
        # Fallback to CPU
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 0]
    
    try:
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Convert BGR to YUV on GPU
        gpu_yuv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2YUV)
        
        # Extract Y channel (first channel)
        # Split into separate channels
        channels = cv2.cuda.split(gpu_yuv)
        gpu_y = channels[0]  # Y channel
        
        # Download back to CPU
        y_channel = gpu_y.download()
        return y_channel
        
    except Exception as e:
        # Fallback to CPU if CUDA fails
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 0]

def test_cuda_yuv_grayscale(camera_index=0, width=1280, height=720, fps=120,
                            duration=10, use_cuda=True, use_v4l2=True,
                            display=False, display_host=None):
    """
    Test CUDA-accelerated YUV to grayscale conversion
    """
    print("="*70)
    print("CUDA-Accelerated YUV Y-Channel Grayscale")
    print("="*70)
    print()
    
    if display_host:
        os.environ['DISPLAY'] = display_host
    
    # Check CUDA availability
    cuda_available = False
    if use_cuda and hasattr(cv2, 'cuda'):
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            cuda_available = True
            print(f"✓ CUDA available: {cuda_devices} device(s)")
        else:
            print("⚠ CUDA not available, using CPU")
    else:
        print("⚠ CUDA not available, using CPU")
    
    print()
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Acceleration: {'CUDA' if cuda_available else 'CPU'}")
    print(f"  Method: BGR → YUV → Y channel extraction")
    print()
    
    # Open camera
    backend = cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    
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
    print()
    
    # Warm up
    print("Warming up (including CUDA initialization)...")
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            _ = extract_y_channel_cuda(frame, use_cuda and cuda_available)
    print()
    
    # Test FPS
    print(f"Measuring FPS for {duration} seconds...")
    print()
    
    frame_count = 0
    start_time = time.time()
    conversion_times = []
    
    window_name = "CUDA YUV Grayscale"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract Y channel
            conv_start = time.time()
            gray = extract_y_channel_cuda(frame, use_cuda and cuda_available)
            conv_time = (time.time() - conv_start) * 1000
            conversion_times.append(conv_time)
            
            frame_count += 1
            
            if display:
                cv2.imshow(window_name, gray)
            
            # Progress
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                avg_conv = sum(conversion_times[-60:]) / min(60, len(conversion_times))
                accel = "CUDA" if cuda_available else "CPU"
                print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Y-extract: {avg_conv:.2f}ms ({accel})", end='\r')
            
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    # Statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_conv = sum(conversion_times) / len(conversion_times) if conversion_times else 0
    
    print(f"\n\nResults:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average Y-extraction time: {avg_conv:.2f}ms")
    print(f"  Acceleration: {'CUDA' if cuda_available else 'CPU'}")
    print()
    
    # Compare with BGR2GRAY
    print("Comparison with cv2.cvtColor(BGR2GRAY):")
    cap2 = cv2.VideoCapture(camera_index, backend)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, test_frame = cap2.read()
    if ret:
        start = time.time()
        for _ in range(100):
            _ = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        cpu_time = (time.time() - start) / 100 * 1000
        
        start = time.time()
        for _ in range(100):
            _ = extract_y_channel_cuda(test_frame, cuda_available)
        yuv_time = (time.time() - start) / 100 * 1000
        
        print(f"  BGR2GRAY: {cpu_time:.2f}ms")
        print(f"  YUV Y-extract: {yuv_time:.2f}ms")
        if yuv_time < cpu_time:
            print(f"  Improvement: {(1 - yuv_time/cpu_time)*100:.1f}% faster")
        else:
            print(f"  Overhead: {(yuv_time/cpu_time - 1)*100:.1f}% slower")
    cap2.release()
    
    return {
        'fps': avg_fps,
        'frames': frame_count,
        'conversion_time_ms': avg_conv,
        'cuda_used': cuda_available
    }

def main():
    parser = argparse.ArgumentParser(description='CUDA YUV Grayscale Conversion')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    parser.add_argument('--height', type=int, default=720, help='Height')
    parser.add_argument('--fps', type=int, default=120, help='FPS')
    parser.add_argument('--duration', type=int, default=10, help='Test duration')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--display-host', type=str, default='192.168.68.31:0.0',
                       help='X display host')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no-v4l2', action='store_true', help='Don\'t use V4L2')
    
    args = parser.parse_args()
    
    result = test_cuda_yuv_grayscale(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        use_cuda=not args.no_cuda,
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


