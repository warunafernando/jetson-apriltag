#!/usr/bin/env python3
"""
Arducam Video Server
Sends video frames over network to Windows client
Client displays using OpenCV imshow()
"""

import cv2
import socket
import struct
import pickle
import threading
import time
import argparse

class ArducamServer:
    def __init__(self, camera_index=0, width=1920, height=1080, fps=30, port=5000):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.cap = None
        self.server_socket = None
        self.running = False
        self.client_socket = None
        
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
    
    def send_frame(self, frame):
        """Send frame to client"""
        try:
            # Serialize frame
            data = pickle.dumps(frame)
            size = len(data)
            
            # Send frame size first
            self.client_socket.sendall(struct.pack("L", size))
            # Send frame data
            self.client_socket.sendall(data)
            return True
        except Exception as e:
            print(f"Error sending frame: {e}")
            return False
    
    def handle_client(self, client_socket, addr):
        """Handle client connection"""
        print(f"Client connected from {addr}")
        self.client_socket = client_socket
        
        frame_time = 1.0 / self.fps if self.fps > 0 else 0.033
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    if not self.send_frame(frame):
                        break
                else:
                    print("Warning: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                time.sleep(frame_time)
        except Exception as e:
            print(f"Client connection error: {e}")
        finally:
            print(f"Client {addr} disconnected")
            self.client_socket = None
    
    def start_server(self):
        """Start TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        
        print(f"\n{'='*60}")
        print("Arducam Video Server Started")
        print(f"{'='*60}")
        print(f"Listening on port {self.port}")
        print(f"Waiting for Windows client to connect...")
        print(f"\nOn Windows, run:")
        print(f"  python arducam_client.py --host 192.168.68.202 --port {self.port}")
        print(f"{'='*60}\n")
    
    def run(self):
        """Run server"""
        try:
            # Start camera
            self.start_camera()
            
            # Start server
            self.start_server()
            
            self.running = True
            
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    # Handle client in main thread (or use threading for multiple clients)
                    self.handle_client(client_socket, addr)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Server error: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nStopping server...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        if self.cap:
            self.cap.release()
        print("Server stopped. Camera released.")


def main():
    parser = argparse.ArgumentParser(description='Arducam Video Server')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1920,
                       help='Frame width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Frame height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port (default: 5000)')
    
    args = parser.parse_args()
    
    server = ArducamServer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port
    )
    
    server.run()


if __name__ == '__main__':
    main()

