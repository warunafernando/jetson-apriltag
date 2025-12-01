#!/usr/bin/env python3
"""
Arducam Grayscale with Optimized Display
Captures at 120 FPS but displays at lower rate to avoid X11 bottleneck
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time
import threading
from collections import deque

class FrameBuffer:
    """Thread-safe frame buffer"""
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.latest_frame = None
    
    def put(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            if len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(frame.copy())
    
    def get(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.latest_frame = None

def capture_thread(cap, frame_buffer, fps_target=120, stop_event=None):
    """Capture frames at high rate"""
    frame_count = 0
    start_time = time.time()
    
    while not (stop_event and stop_event.is_set()):
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to grayscale (YUV Y-channel)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        gray = yuv[:, :, 0]
        
        # Put in buffer (overwrites old frames)
        frame_buffer.put(gray)
        frame_count += 1
        
        # Maintain target FPS
        elapsed = time.time() - start_time
        expected_frames = fps_target * elapsed
        if frame_count > expected_frames + 1:
            time.sleep(0.001)  # Small delay if ahead
    
    return frame_count

def display_thread(frame_buffer, display_fps=30, stop_event=None):
    """Display frames at lower rate"""
    frame_time = 1.0 / display_fps if display_fps > 0 else 0.033
    last_display_time = time.time()
    
    while not (stop_event and stop_event.is_set()):
        current_time = time.time()
        
        # Check if it's time to display
        if current_time - last_display_time >= frame_time:
            frame = frame_buffer.get()
            if frame is not None:
                cv2.imshow("YUV Grayscale (120 FPS capture, 30 FPS display)", frame)
                last_display_time = current_time
            
            # Process events (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if stop_event:
                    stop_event.set()
                break
        else:
            # Sleep until next display time
            sleep_time = frame_time - (current_time - last_display_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

def test_optimized_grayscale(camera_index=0, width=1280, height=720, fps=120,
                              capture_fps=120, display_fps=30, duration=10,
                              display=False, display_host=None):
    """
    Optimized grayscale: capture at high FPS, display at lower FPS
    """
    print("="*70)
    print("Optimized YUV Grayscale (Decoupled Capture/Display)")
    print("="*70)
    print()
    
    if display_host:
        os.environ['DISPLAY'] = display_host
    
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Capture FPS: {capture_fps}")
    print(f"  Display FPS: {display_fps} (if displaying)")
    print(f"  Method: YUV → Y channel extraction")
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
    
    # Frame buffer for capture/display decoupling
    frame_buffer = FrameBuffer(max_size=2)
    stop_event = threading.Event()
    
    # Start capture thread
    print("Starting capture thread...")
    capture_count = [0]  # Use list for mutable reference
    
    def capture_wrapper():
        nonlocal capture_count
        count = 0
        start = time.time()
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            gray = yuv[:, :, 0]
            
            frame_buffer.put(gray)
            count += 1
            
            # Progress
            if count % 120 == 0:
                elapsed = time.time() - start
                current_fps = count / elapsed if elapsed > 0 else 0
                print(f"  Captured: {count} frames | Capture FPS: {current_fps:.2f}", end='\r')
        
        capture_count[0] = count
    
    capture_th = threading.Thread(target=capture_wrapper, daemon=True)
    capture_th.start()
    
    # Start display thread if needed
    display_th = None
    if display:
        print("Starting display thread...")
        display_th = threading.Thread(target=display_thread, 
                                      args=(frame_buffer, display_fps, stop_event),
                                      daemon=True)
        display_th.start()
    
    print()
    print(f"Running for {duration} seconds...")
    print("Press 'q' in display window to quit")
    print()
    
    # Run for specified duration
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.5)  # Let threads finish
    
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # Statistics
    total_time = time.time() - start_time
    capture_fps_result = capture_count[0] / total_time if total_time > 0 else 0
    
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Frames captured: {capture_count[0]}")
    print(f"Capture FPS: {capture_fps_result:.2f}")
    print(f"Target capture FPS: {capture_fps}")
    print(f"Display FPS: {display_fps} (display thread)")
    print(f"Performance: {(capture_fps_result/capture_fps*100):.1f}% of target")
    print()
    
    if capture_fps_result >= capture_fps * 0.9:
        print("✓ SUCCESS: Achieving target capture FPS!")
    else:
        print("⚠ WARNING: Not achieving target capture FPS")
    
    return {
        'capture_fps': capture_fps_result,
        'frames': capture_count[0],
        'display_fps': display_fps
    }

def main():
    parser = argparse.ArgumentParser(description='Optimized YUV Grayscale')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    parser.add_argument('--height', type=int, default=720, help='Height')
    parser.add_argument('--capture-fps', type=int, default=120, help='Capture FPS')
    parser.add_argument('--display-fps', type=int, default=30, 
                       help='Display FPS (default: 30, lower for X11)')
    parser.add_argument('--duration', type=int, default=10, help='Test duration')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--display-host', type=str, default='192.168.68.31:0.0',
                       help='X display host')
    
    args = parser.parse_args()
    
    result = test_optimized_grayscale(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.capture_fps,
        capture_fps=args.capture_fps,
        display_fps=args.display_fps,
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


