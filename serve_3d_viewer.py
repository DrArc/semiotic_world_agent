"""
Simple HTTP server to serve the 3D viewer and mesh files
This fixes CORS issues when loading local GLB files
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def start_server():
    """Start a local HTTP server to serve the 3D viewer"""
    
    # Set up the server
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🚀 Starting 3D viewer server on http://localhost:{PORT}")
            print(f"📁 Serving files from: {os.getcwd()}")
            print(f"🔗 3D Viewer URL: http://localhost:{PORT}/3d_viewer_real.html")
            print(f"🧪 Test URL: http://localhost:{PORT}/test_real_3d_viewer.html")
            print("\n" + "="*60)
            print("✅ Server started successfully!")
            print("   The 3D viewer should now work properly.")
            print("   Press Ctrl+C to stop the server.")
            print("="*60 + "\n")
            
            # Open the 3D viewer in browser
            viewer_url = f"http://localhost:{PORT}/3d_viewer_real.html"
            webbrowser.open(viewer_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 10048:  # Port already in use
            print(f"❌ Port {PORT} is already in use. Trying port {PORT + 1}...")
            start_server_on_port(PORT + 1)
        else:
            print(f"❌ Error starting server: {e}")

def start_server_on_port(port):
    """Start server on a specific port"""
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"🚀 Starting 3D viewer server on http://localhost:{port}")
            print(f"📁 Serving files from: {os.getcwd()}")
            print(f"🔗 3D Viewer URL: http://localhost:{port}/3d_viewer_real.html")
            print(f"🧪 Test URL: http://localhost:{port}/test_real_3d_viewer.html")
            print("\n" + "="*60)
            print("✅ Server started successfully!")
            print("   The 3D viewer should now work properly.")
            print("   Press Ctrl+C to stop the server.")
            print("="*60 + "\n")
            
            # Open the 3D viewer in browser
            viewer_url = f"http://localhost:{port}/3d_viewer_real.html"
            webbrowser.open(viewer_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server on port {port}: {e}")

if __name__ == "__main__":
    print("🔧 Starting 3D Viewer Server...")
    print("   This fixes CORS issues when loading local GLB files")
    print()
    start_server()


