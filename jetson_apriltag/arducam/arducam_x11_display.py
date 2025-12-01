#!/usr/bin/env python3
"""
Arducam Video Display with X11 Forwarding
Displays OpenCV video window on Windows X server
"""

import cv2
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Arducam Video Display via X11')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1920,
                       help='Frame width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Frame height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--display', type=str, default='192.168.68.31:0.0',
                       help='X display (default: 192.168.68.31:0.0)')
    
    args = parser.parse_args()
    
    # Set DISPLAY environment variable
    os.environ['DISPLAY'] = args.display
    print(f"Display set to: {args.display}")
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera {args.camera}")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened successfully!")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps}")
    print(f"  Display: {args.display}")
    print("\nPress 'q' to quit\n")
    
    window_name = "Arducam Video Stream"
    
    try:
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                continue
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f}", end='\r')
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Window closed.")


if __name__ == '__main__':
    main()

