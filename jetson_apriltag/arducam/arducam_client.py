#!/usr/bin/env python3
"""
Arducam Video Client (Windows)
Receives video frames from Jetson server and displays using OpenCV imshow()
Run this on your Windows machine
"""

import cv2
import socket
import struct
import pickle
import argparse
import sys

class ArducamClient:
    def __init__(self, host='192.168.68.202', port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
    def connect(self):
        """Connect to server"""
        try:
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.host, self.port))
            print("Connected successfully!")
            return True
        except socket.timeout:
            print("Connection timeout. Is the server running?")
            return False
        except ConnectionRefusedError:
            print("Connection refused. Check if server is running and port is correct.")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def receive_frame(self):
        """Receive frame from server"""
        try:
            # Receive frame size
            size_data = b""
            while len(size_data) < 4:
                chunk = self.socket.recv(4 - len(size_data))
                if not chunk:
                    return None
                size_data += chunk
            
            size = struct.unpack("L", size_data)[0]
            
            # Receive frame data
            data = b""
            while len(data) < size:
                chunk = self.socket.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            
            # Deserialize frame
            frame = pickle.loads(data)
            return frame
        except Exception as e:
            print(f"Error receiving frame: {e}")
            return None
    
    def run(self):
        """Run client"""
        if not self.connect():
            return
        
        self.running = True
        window_name = "Arducam Video Stream"
        
        print("\n" + "="*60)
        print("Displaying video stream...")
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        try:
            frame_count = 0
            start_time = cv2.getTickCount()
            
            while self.running:
                frame = self.receive_frame()
                if frame is None:
                    print("Connection lost or no frame received")
                    break
                
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
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.socket:
            self.socket.close()
        cv2.destroyAllWindows()
        print("Client closed.")


def main():
    parser = argparse.ArgumentParser(description='Arducam Video Client (Windows)')
    parser.add_argument('--host', type=str, default='192.168.68.202',
                       help='Server IP address (default: 192.168.68.202)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port (default: 5000)')
    
    args = parser.parse_args()
    
    # Check if OpenCV is available
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Error: OpenCV not installed!")
        print("Install with: pip install opencv-python")
        sys.exit(1)
    
    client = ArducamClient(host=args.host, port=args.port)
    client.run()


if __name__ == '__main__':
    main()

