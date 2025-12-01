#!/usr/bin/env python3
"""
Arducam Video Stream Server
Streams video from Arducam USB3 camera via HTTP (MJPEG)
Accessible via SSH tunnel from Windows machine
"""

import cv2
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import argparse

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming"""
    
    def do_GET(self):
        if self.path == '/':
            # Serve HTML page with video stream
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Arducam Stream</title>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        background: #1a1a1a;
                        color: white;
                        font-family: Arial, sans-serif;
                    }
                    h1 { text-align: center; }
                    #video {
                        display: block;
                        margin: 20px auto;
                        max-width: 100%;
                        border: 2px solid #333;
                        border-radius: 8px;
                    }
                    .info {
                        text-align: center;
                        margin: 10px;
                        color: #aaa;
                    }
                </style>
            </head>
            <body>
                <h1>Arducam USB3 Camera Stream</h1>
                <div class="info">Resolution: 3840x2160 (4K) | FPS: ~20</div>
                <img id="video" src="/stream.mjpg" alt="Video Stream">
                <div class="info">Press Ctrl+C in terminal to stop</div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/stream.mjpg':
            # Stream MJPEG video
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            
            try:
                while True:
                    if hasattr(self.server, 'frame'):
                        frame = self.server.frame
                        if frame is not None:
                            # Encode frame as JPEG
                            ret, buffer = cv2.imencode('.jpg', frame, 
                                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if ret:
                                frame_bytes = buffer.tobytes()
                                self.wfile.write(b'--FRAME\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', len(frame_bytes))
                                self.end_headers()
                                self.wfile.write(frame_bytes)
                                self.wfile.write(b'\r\n')
                    time.sleep(0.05)  # ~20 FPS
            except Exception as e:
                print(f"Streaming error: {e}")
        else:
            self.send_error(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threading HTTP server"""
    daemon_threads = True
    frame = None


class ArducamStreamer:
    """Main streaming class"""
    
    def __init__(self, camera_index=0, width=1920, height=1080, fps=30, port=8080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.cap = None
        self.server = None
        self.running = False
        
    def start_camera(self):
        """Initialize camera"""
        print(f"Opening camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened successfully!")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps}")
        
        return True
    
    def capture_loop(self):
        """Capture frames from camera"""
        frame_time = 1.0 / self.fps if self.fps > 0 else 0.033
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Resize if needed for streaming (optional)
                # frame = cv2.resize(frame, (1920, 1080))
                self.server.frame = frame
            else:
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
            
            time.sleep(frame_time)
    
    def start_server(self):
        """Start HTTP server"""
        self.server = ThreadingHTTPServer(('0.0.0.0', self.port), StreamingHandler)
        print(f"HTTP server started on port {self.port}")
        print(f"Access at: http://localhost:{self.port}")
        print(f"Or via SSH tunnel: http://localhost:<tunnel_port>")
        
    def run(self):
        """Start streaming"""
        try:
            # Start camera
            self.start_camera()
            
            # Start server
            self.start_server()
            
            # Start capture thread
            self.running = True
            capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            capture_thread.start()
            
            print("\n" + "="*60)
            print("Arducam Stream Server Running")
            print("="*60)
            print(f"Local URL: http://localhost:{self.port}")
            print(f"SSH Tunnel URL: http://localhost:<your_tunnel_port>")
            print("\nPress Ctrl+C to stop")
            print("="*60 + "\n")
            
            # Run server
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            print("\n\nStopping server...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.server:
            self.server.shutdown()
        print("Server stopped. Camera released.")


def main():
    parser = argparse.ArgumentParser(description='Arducam Video Stream Server')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1920,
                       help='Frame width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Frame height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port (default: 8080)')
    
    args = parser.parse_args()
    
    streamer = ArducamStreamer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port
    )
    
    streamer.run()


if __name__ == '__main__':
    main()


