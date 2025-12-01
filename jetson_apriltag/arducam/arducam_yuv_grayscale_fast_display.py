#!/usr/bin/env python3
"""
Arducam Grayscale with Fast Display
Captures at 120 FPS, displays every Nth frame to avoid X11 bottleneck
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time

def extract_y_from_yuyv(frame):
    """Extract Y channel from YUV"""
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 0]
    return frame

def test_fast_display(camera_index=0, width=1280, height=720, fps=120,
                      duration=10, display=False, display_host=None,
                      display_skip=4):
    """
    Fast display: Capture all frames, display every Nth frame
    """
    print("="*70)
    print("Fast Display YUV Grayscale")
    print("="*70)
    print()
    
    if display_host:
        os.environ['DISPLAY'] = display_host
    
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Capture FPS: {fps}")
    if display:
        print(f"  Display: Every {display_skip} frames (~{fps//display_skip} FPS display)")
    else:
        print(f"  Display: Disabled")
    print()
    
    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Failed to open camera {camera_index}")
        return None
    
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
    print("Warming up...")
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            _ = extract_y_from_yuyv(frame)
    print()
    
    # Test
    print(f"Measuring for {duration} seconds...")
    print()
    
    frame_count = 0
    display_count = 0
    start_time = time.time()
    conversion_times = []
    
    window_name = f"YUV Grayscale (capturing at {fps} FPS)"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract Y channel
            conv_start = time.time()
            gray = extract_y_from_yuyv(frame)
            conv_time = (time.time() - conv_start) * 1000
            conversion_times.append(conv_time)
            
            frame_count += 1
            
            # Display every Nth frame only
            if display and frame_count % display_skip == 0:
                cv2.imshow(window_name, gray)
                display_count += 1
                # Non-blocking wait
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Progress
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                capture_fps = frame_count / elapsed
                if display:
                    display_fps = display_count / elapsed
                    print(f"Capture: {capture_fps:.1f} FPS | Display: {display_fps:.1f} FPS | Frames: {frame_count}", end='\r')
                else:
                    print(f"Capture: {capture_fps:.1f} FPS | Frames: {frame_count}", end='\r')
            
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    # Statistics
    total_time = time.time() - start_time
    capture_fps = frame_count / total_time if total_time > 0 else 0
    display_fps = display_count / total_time if total_time > 0 and display else 0
    avg_conv = sum(conversion_times) / len(conversion_times) if conversion_times else 0
    
    print(f"\n\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Frames captured: {frame_count}")
    print(f"  Capture FPS: {capture_fps:.2f}")
    if display:
        print(f"  Frames displayed: {display_count}")
        print(f"  Display FPS: {display_fps:.2f}")
        print(f"  Display skip: Every {display_skip} frames")
    print(f"  Y-extraction: {avg_conv:.2f}ms avg")
    print()
    
    if capture_fps >= fps * 0.9:
        print("✓ SUCCESS: Achieving target capture FPS!")
    else:
        print(f"⚠ Capture FPS below target ({capture_fps:.1f} < {fps})")
    
    return {
        'capture_fps': capture_fps,
        'display_fps': display_fps,
        'frames': frame_count
    }

def main():
    parser = argparse.ArgumentParser(description='Fast Display YUV Grayscale')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    parser.add_argument('--height', type=int, default=720, help='Height')
    parser.add_argument('--fps', type=int, default=120, help='Capture FPS')
    parser.add_argument('--duration', type=int, default=10, help='Test duration')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--display-host', type=str, default='192.168.68.31:0.0',
                       help='X display host')
    parser.add_argument('--display-skip', type=int, default=4,
                       help='Display every Nth frame (default: 4 = ~30 FPS display)')
    
    args = parser.parse_args()
    
    result = test_fast_display(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        display=args.display,
        display_host=args.display_host if args.display else None,
        display_skip=args.display_skip
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()


