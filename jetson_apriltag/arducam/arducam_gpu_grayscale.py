#!/usr/bin/env python3
"""
Arducam Grayscale - Fully GPU Accelerated
All operations run on GPU using CUDA, minimal CPU usage
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time
import glob

def check_cuda_availability():
    """Check if CUDA is available"""
    if not hasattr(cv2, 'cuda'):
        return False, 0
    device_count = cv2.cuda.getCudaEnabledDeviceCount()
    return device_count > 0, device_count

def extract_y_gpu(gpu_frame):
    """
    Extract Y channel directly on GPU using CUDA
    Input: cv2.cuda_GpuMat (BGR format)
    Output: cv2.cuda_GpuMat (grayscale Y channel)
    """
    # Convert BGR to YUV on GPU (all on GPU, no CPU)
    gpu_yuv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2YUV)
    
    # Split into channels on GPU
    channels = cv2.cuda.split(gpu_yuv)
    return channels[0]  # Return Y channel (grayscale) - stays on GPU

def find_available_camera(max_tries=5):
    """Find first available camera"""
    print("Searching for available camera...")
    
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

def test_gpu_grayscale(camera_index=0, width=1280, height=720, fps=120,
                       duration=60, use_v4l2=True, display=False, display_host=None):
    """
    Fully GPU-accelerated grayscale conversion
    All processing happens on GPU with minimal CPU-GPU transfers
    """
    print("="*70)
    print("Fully GPU-Accelerated Grayscale Processing")
    print("="*70)
    print()
    
    # Check CUDA availability
    cuda_available, device_count = check_cuda_availability()
    if not cuda_available:
        print("⚠ ERROR: CUDA not available!")
        print("  • OpenCV was not built with CUDA support")
        print("  • Or no CUDA-capable GPU detected")
        print("  • Falling back to CPU mode...")
        print()
        use_gpu = False
    else:
        print(f"✓ CUDA available: {device_count} GPU device(s)")
        use_gpu = True
        print()
    
    if display_host:
        os.environ['DISPLAY'] = display_host
    
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration}s")
    print(f"  Processing: {'GPU (CUDA)' if use_gpu else 'CPU (fallback)'}")
    print(f"  Method: BGR → YUV → Y channel (all on GPU)")
    print()
    
    # Open camera
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
    
    # Initialize GPU resources (warm up)
    print("Initializing GPU...")
    gpu_frame = None
    gpu_gray = None
    
    if use_gpu:
        # Pre-allocate GPU memory
        ret, test_frame = cap.read()
        if ret:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(test_frame)
            gpu_gray = extract_y_gpu(gpu_frame)
            print("✓ GPU initialized and ready")
        else:
            print("⚠ Failed to initialize GPU, falling back to CPU")
            use_gpu = False
    
    print()
    
    # Warm up
    print("Warming up...")
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            continue
        
        if use_gpu:
            gpu_frame.upload(frame)
            gpu_gray = extract_y_gpu(gpu_frame)
        else:
            # CPU fallback
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            _ = yuv[:, :, 0]
    print()
    
    # Main processing loop
    print(f"Processing for {duration} seconds...")
    print()
    
    frame_count = 0
    start_time = time.time()
    conversion_times = []
    upload_times = []
    download_times = []
    
    window_name = "GPU Grayscale Processing"
    
    # Re-allocate GPU memory if needed
    if use_gpu and gpu_frame is None:
        ret, test_frame = cap.read()
        if ret:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_gray = cv2.cuda_GpuMat()
    
    try:
        while True:
            # Read frame from camera (CPU operation - unavoidable)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            if use_gpu:
                # GPU processing path
                # Upload to GPU (only CPU-GPU transfer needed)
                upload_start = time.time()
                gpu_frame.upload(frame)
                upload_time = (time.time() - upload_start) * 1000
                upload_times.append(upload_time)
                
                # Process on GPU (all operations on GPU)
                conv_start = time.time()
                gpu_gray = extract_y_gpu(gpu_frame)
                conv_time = (time.time() - conv_start) * 1000
                conversion_times.append(conv_time)
                
                # Display: need to download only when displaying
                if display and frame_count % 10 == 0:
                    download_start = time.time()
                    gray = gpu_gray.download()  # Only download when displaying
                    download_time = (time.time() - download_start) * 1000
                    download_times.append(download_time)
                    cv2.imshow(window_name, gray)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            else:
                # CPU fallback path
                conv_start = time.time()
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                gray = yuv[:, :, 0]
                conv_time = (time.time() - conv_start) * 1000
                conversion_times.append(conv_time)
                
                if display and frame_count % 10 == 0:
                    cv2.imshow(window_name, gray)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
            # Progress update
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                avg_conv = sum(conversion_times[-60:]) / min(60, len(conversion_times))
                display_rate = current_fps / 10 if display else 0
                
                if use_gpu and upload_times:
                    avg_upload = sum(upload_times[-60:]) / min(60, len(upload_times))
                    avg_download = sum(download_times[-60:]) / min(60, len(download_times)) if download_times else 0
                    if display:
                        print(f"Frames: {frame_count} | Capture FPS: {current_fps:.2f} | Display FPS: ~{display_rate:.1f} | Upload: {avg_upload:.2f}ms | GPU Process: {avg_conv:.2f}ms | Download: {avg_download:.2f}ms", end='\r')
                    else:
                        print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Upload: {avg_upload:.2f}ms | GPU Process: {avg_conv:.2f}ms", end='\r')
                else:
                    if display:
                        print(f"Frames: {frame_count} | Capture FPS: {current_fps:.2f} | Display FPS: ~{display_rate:.1f} | Process: {avg_conv:.2f}ms", end='\r')
                    else:
                        print(f"Frames: {frame_count} | FPS: {current_fps:.2f} | Process: {avg_conv:.2f}ms", end='\r')
            
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
        if use_gpu:
            if gpu_frame is not None:
                gpu_frame.release()
            if gpu_gray is not None:
                gpu_gray.release()
    
    # Statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_conv = sum(conversion_times) / len(conversion_times) if conversion_times else 0
    
    print(f"\n\nResults:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Processing: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    
    if use_gpu:
        avg_upload = sum(upload_times) / len(upload_times) if upload_times else 0
        avg_download = sum(download_times) / len(download_times) if download_times else 0
        print(f"  Average GPU processing time: {avg_conv:.2f}ms")
        print(f"  Average upload time: {avg_upload:.2f}ms")
        if download_times:
            print(f"  Average download time: {avg_download:.2f}ms")
        print(f"  Total GPU overhead: {avg_upload + avg_conv:.2f}ms per frame")
    else:
        print(f"  Average CPU processing time: {avg_conv:.2f}ms")
    
    print(f"  Target FPS: {fps}")
    print(f"  Performance: {(avg_fps/fps*100):.1f}% of target")
    print()
    
    return {
        'fps': avg_fps,
        'frames': frame_count,
        'processing_time_ms': avg_conv,
        'gpu_used': use_gpu,
        'performance_pct': (avg_fps/fps*100) if fps > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Fully GPU-Accelerated Grayscale Processing')
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
    
    result = test_gpu_grayscale(
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


