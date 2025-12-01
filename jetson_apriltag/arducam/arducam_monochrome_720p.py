#!/usr/bin/env python3
"""
Arducam Monochrome Display at 720p @ 60fps
Displays grayscale video via X11 forwarding
"""

import cv2
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Arducam Monochrome Display at 720p')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height (default: 720)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Target FPS (default: 60)')
    parser.add_argument('--display', type=str, default='192.168.68.31:0.0',
                       help='X display (default: 192.168.68.31:0.0)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no display)')
    
    args = parser.parse_args()
    
    # Set DISPLAY environment variable
    if not args.headless:
        os.environ['DISPLAY'] = args.display
        print(f"Display set to: {args.display}")
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera {args.camera}")
        sys.exit(1)
    
    # Set camera properties - force 720p
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
    print(f"  Mode: Monochrome (grayscale)")
    if not args.headless:
        print(f"  Display: {args.display}")
    else:
        print(f"  Mode: Headless (no display)")
    print("\nPress 'q' to quit\n")
    
    window_name = "Arducam Monochrome (720p @ 60fps)"
    
    try:
        frame_count = 0
        start_time = cv2.getTickCount()
        conversion_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                continue
            
            # Convert to grayscale
            conv_start = cv2.getTickCount()
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            conv_time = (cv2.getTickCount() - conv_start) / cv2.getTickFrequency() * 1000
            conversion_times.append(conv_time)
            
            # Display frame (if not headless)
            if not args.headless:
                cv2.imshow(window_name, gray)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 60 == 0:
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                fps = frame_count / elapsed
                avg_conv = sum(conversion_times[-60:]) / min(60, len(conversion_times))
                print(f"FPS: {fps:.2f} | Conversion: {avg_conv:.2f}ms | Frames: {frame_count}", end='\r')
            
            # Check for quit
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
            else:
                # In headless mode, run for a bit then exit or use Ctrl+C
                if frame_count >= 600:  # 10 seconds at 60fps
                    break
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # Final stats
        total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_conv = sum(conversion_times) / len(conversion_times) if conversion_times else 0
        
        print(f"\n\nFinal Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Average conversion time: {avg_conv:.2f}ms")
        print("Camera released.")


if __name__ == '__main__':
    main()


