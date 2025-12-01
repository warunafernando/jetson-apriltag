#!/usr/bin/env python3
"""
Arducam Grayscale using NVIDIA ISP YUV conversion
Uses GStreamer pipeline with hardware acceleration to convert to YUV
and extract Y channel (luma) for grayscale output
"""

import cv2
import numpy as np
import argparse
import sys
import os

def create_gstreamer_pipeline_yuv(camera_index=0, width=1280, height=720, fps=120):
    """
    Create GStreamer pipeline for YUV conversion using NVIDIA hardware acceleration
    Pipeline: v4l2src -> nvvidconv (YUV) -> extract Y channel -> grayscale output
    """
    # Option 1: Use nvvidconv for hardware-accelerated YUV conversion
    pipeline = (
        f"v4l2src device=/dev/video{camera_index} ! "
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
        "nvvidconv ! "  # NVIDIA hardware-accelerated video converter
        "video/x-raw(memory:NVMM),format=NV12 ! "  # Convert to NV12 (YUV format with hardware)
        "nvvidconv ! "  # Convert back to system memory
        "video/x-raw,format=I420 ! "  # YUV 4:2:0 format (has Y channel)
        "videoconvert ! "  # Convert to grayscale (extract Y channel)
        "video/x-raw,format=GRAY8 ! "  # Grayscale output (Y channel only)
        "appsink drop=1 max-buffers=1"
    )
    
    return pipeline

def capture_with_nvidia_isp_yuv(camera_index=0, width=1280, height=720, fps=120, 
                                 use_gstreamer=True, backend=cv2.CAP_GSTREAMER):
    """
    Capture frames using NVIDIA ISP for YUV conversion
    """
    if use_gstreamer:
        pipeline = create_gstreamer_pipeline_yuv(camera_index, width, height, fps)
        print(f"GStreamer pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, backend)
    else:
        # Fallback: Use V4L2 and convert YUV manually
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    
    return cap

def extract_y_channel_from_yuv(frame):
    """
    Extract Y (luminance) channel from YUV frame
    If frame is already grayscale, return as-is
    If frame is BGR, convert to YUV and extract Y
    """
    if len(frame.shape) == 2:
        # Already grayscale
        return frame
    
    if len(frame.shape) == 3:
        if frame.shape[2] == 3:
            # BGR frame - convert to YUV and extract Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]  # Extract Y (luminance) channel
            return y_channel
    
    return frame

def test_nvidia_isp_grayscale(camera_index=0, width=1280, height=720, fps=120, 
                               duration=5, display=False, display_host=None):
    """
    Test NVIDIA ISP grayscale conversion
    """
    print("="*70)
    print("NVIDIA ISP Grayscale Conversion Test")
    print("="*70)
    print()
    
    # Set display if provided
    if display_host:
        os.environ['DISPLAY'] = display_host
        print(f"Display set to: {display_host}")
    
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Method: NVIDIA ISP YUV → Y channel extraction")
    print()
    
    # Try GStreamer pipeline first
    print("Attempting GStreamer pipeline with NVIDIA acceleration...")
    try:
        cap = capture_with_nvidia_isp_yuv(camera_index, width, height, fps, 
                                          use_gstreamer=True)
        if cap.isOpened():
            print("✓ GStreamer pipeline opened")
            method = "GStreamer"
        else:
            raise Exception("GStreamer pipeline failed")
    except Exception as e:
        print(f"⚠ GStreamer pipeline failed: {e}")
        print("Falling back to V4L2 with YUV extraction...")
        cap = capture_with_nvidia_isp_yuv(camera_index, width, height, fps, 
                                          use_gstreamer=False)
        method = "V4L2+YUV"
    
    if not cap.isOpened():
        print("Error: Failed to open camera")
        return None
    
    # Get actual settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened:")
    print(f"  Resolution: {actual_w}x{actual_h}")
    print(f"  FPS: {actual_fps}")
    print(f"  Method: {method}")
    print()
    
    # Test capture and conversion
    print(f"Testing for {duration} seconds...")
    print()
    
    frame_count = 0
    start_time = cv2.getTickCount()
    conversion_times = []
    y_channel_only = False
    
    window_name = "NVIDIA ISP Grayscale (YUV Y-channel)"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                continue
            
            # Check if frame is already grayscale (from GStreamer pipeline)
            if len(frame.shape) == 2:
                gray = frame
                y_channel_only = True
                conv_time = 0
            else:
                # Extract Y channel from YUV
                conv_start = cv2.getTickCount()
                gray = extract_y_channel_from_yuv(frame)
                conv_time = (cv2.getTickCount() - conv_start) / cv2.getTickFrequency() * 1000
                conversion_times.append(conv_time)
            
            frame_count += 1
            
            if display:
                cv2.imshow(window_name, gray)
            
            # Calculate FPS
            if frame_count % 60 == 0:
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                current_fps = frame_count / elapsed
                if conversion_times:
                    avg_conv = sum(conversion_times[-60:]) / min(60, len(conversion_times))
                    print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Conv: {avg_conv:.2f}ms", end='\r')
                else:
                    print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Hardware grayscale", end='\r')
            
            # Check duration
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed >= duration:
                break
            
            # Check for quit
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
    total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_conv = sum(conversion_times) / len(conversion_times) if conversion_times else 0
    
    print(f"\n\nResults:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    if y_channel_only:
        print(f"  Conversion: Hardware (GStreamer pipeline)")
    else:
        print(f"  Average conversion time: {avg_conv:.2f}ms")
    print(f"  Method: {method}")
    print()
    
    return {
        'fps': avg_fps,
        'frames': frame_count,
        'method': method,
        'hardware_accelerated': y_channel_only
    }

def main():
    parser = argparse.ArgumentParser(description='NVIDIA ISP Grayscale Conversion')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    parser.add_argument('--height', type=int, default=720, help='Height')
    parser.add_argument('--fps', type=int, default=120, help='FPS')
    parser.add_argument('--duration', type=int, default=5, help='Test duration')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--display-host', type=str, default='192.168.68.31:0.0',
                       help='X display host')
    
    args = parser.parse_args()
    
    result = test_nvidia_isp_grayscale(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        display=args.display,
        display_host=args.display_host if args.display else None
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()

