#!/usr/bin/env python3
"""
Arducam Grayscale using YUV Y-channel extraction
Uses hardware-accelerated YUV conversion when possible, then extracts Y channel
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time

def extract_y_from_yuv(frame):
    """
    Extract Y (luminance) channel from frame
    Handles multiple input formats: BGR, YUV, already grayscale
    """
    if len(frame.shape) == 2:
        # Already grayscale
        return frame
    
    if len(frame.shape) == 3:
        if frame.shape[2] == 3:
            # Convert BGR to YUV and extract Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            return yuv[:, :, 0]  # Y channel (luminance)
        elif frame.shape[2] == 2:
            # Already YUV-like format
            return frame[:, :, 0]  # First channel is Y
    
    return frame

def create_gstreamer_yuv_pipeline(camera_index=0, width=1280, height=720, fps=120):
    """
    Create GStreamer pipeline optimized for YUV Y-channel extraction
    """
    # Try hardware-accelerated pipeline first
    pipeline = (
        f"v4l2src device=/dev/video{camera_index} ! "
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
        "nvvidconv ! "  # NVIDIA hardware-accelerated converter
        "video/x-raw(memory:NVMM),format=NV12 ! "  # Hardware YUV format
        "nvvidconv ! "  # Convert back to system memory  
        "video/x-raw,format=I420 ! "  # YUV 4:2:0 (Y plane is first)
        "videoconvert ! "  # Extract Y channel
        "video/x-raw,format=GRAY8 ! "  # Grayscale (Y channel)
        "appsink drop=1 max-buffers=1"
    )
    return pipeline

def test_yuv_grayscale(camera_index=0, width=1280, height=720, fps=120, 
                       duration=10, use_hw_accel=True, display=False, display_host=None):
    """
    Test grayscale using YUV Y-channel extraction
    """
    print("="*70)
    print("YUV Y-Channel Grayscale Conversion")
    print("="*70)
    print()
    
    if display_host:
        os.environ['DISPLAY'] = display_host
        print(f"Display: {display_host}")
    
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Method: YUV → Y channel (luminance) extraction")
    print()
    
    # Try GStreamer hardware-accelerated pipeline
    if use_hw_accel:
        print("Attempting GStreamer pipeline with NVIDIA acceleration...")
        try:
            pipeline = create_gstreamer_yuv_pipeline(camera_index, width, height, fps)
            print(f"Pipeline: {pipeline[:100]}...")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                # Test read to see if we get grayscale directly
                ret, test_frame = cap.read()
                if ret:
                    if len(test_frame.shape) == 2:
                        print("✓ GStreamer pipeline providing grayscale directly!")
                        method = "GStreamer Hardware (direct grayscale)"
                        direct_grayscale = True
                    else:
                        print("⚠ GStreamer pipeline returned color, extracting Y channel...")
                        method = "GStreamer Hardware + Y extraction"
                        direct_grayscale = False
            else:
                raise Exception("Pipeline failed to open")
                
        except Exception as e:
            print(f"⚠ GStreamer pipeline failed: {e}")
            print("Falling back to V4L2 with software Y extraction...")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            method = "V4L2 + Software Y extraction"
            direct_grayscale = False
    else:
        # Use V4L2 directly
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        method = "V4L2 + Software Y extraction"
        direct_grayscale = False
    
    if not cap.isOpened():
        print("Error: Failed to open camera")
        return None
    
    # Get actual settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nCamera opened:")
    print(f"  Resolution: {actual_w}x{actual_h}")
    print(f"  FPS: {actual_fps}")
    print(f"  Method: {method}")
    print()
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            break
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
                print("Warning: Failed to capture frame")
                continue
            
            # Extract Y channel if needed
            if direct_grayscale:
                gray = frame  # Already grayscale from pipeline
                conv_time = 0
            else:
                conv_start = time.time()
                gray = extract_y_from_yuv(frame)
                conv_time = (time.time() - conv_start) * 1000
                conversion_times.append(conv_time)
            
            frame_count += 1
            
            if display:
                cv2.imshow(window_name, gray)
            
            # Progress update
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                if conversion_times:
                    avg_conv = sum(conversion_times[-60:]) / min(60, len(conversion_times))
                    print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Y-extract: {avg_conv:.2f}ms", end='\r')
                else:
                    print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Hardware", end='\r')
            
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
    if direct_grayscale:
        print(f"  Conversion: Hardware-accelerated (GStreamer)")
    else:
        print(f"  Average Y-extraction time: {avg_conv:.2f}ms")
    print(f"  Method: {method}")
    print()
    
    return {
        'fps': avg_fps,
        'frames': frame_count,
        'method': method,
        'hardware_accelerated': direct_grayscale
    }

def main():
    parser = argparse.ArgumentParser(description='YUV Y-channel Grayscale Extraction')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    parser.add_argument('--height', type=int, default=720, help='Height')
    parser.add_argument('--fps', type=int, default=120, help='FPS')
    parser.add_argument('--duration', type=int, default=10, help='Test duration')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--display-host', type=str, default='192.168.68.31:0.0',
                       help='X display host')
    parser.add_argument('--no-hw', action='store_true', 
                       help='Disable hardware acceleration')
    
    args = parser.parse_args()
    
    result = test_yuv_grayscale(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        use_hw_accel=not args.no_hw,
        display=args.display,
        display_host=args.display_host if args.display else None
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()


