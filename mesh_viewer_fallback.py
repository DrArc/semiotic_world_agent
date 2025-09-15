#!/usr/bin/env python3
"""
Fallback mesh viewer using Open3D or basic file operations
"""
import os
import sys
from pathlib import Path

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class MeshViewerFallback(QWidget):
    def __init__(self):
        super().__init__()
        self.mesh_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.label = QLabel("3D mesh will appear here after generation")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedHeight(200)
        self.label.setStyleSheet("border: 2px solid #555555; background-color: #404040; border-radius: 5px;")
        layout.addWidget(self.label)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Mesh information will appear here...")
        layout.addWidget(self.info_text)
        
        button_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Load 3D Model")
        self.load_btn.clicked.connect(self.load_mesh)
        button_layout.addWidget(self.load_btn)
        
        self.open_btn = QPushButton("Open in External Viewer")
        self.open_btn.clicked.connect(self.open_external)
        self.open_btn.setEnabled(False)
        button_layout.addWidget(self.open_btn)
        
        layout.addLayout(button_layout)
    
    def set_mesh_path(self, mesh_path):
        """Set the mesh path and update display"""
        self.mesh_path = mesh_path
        if mesh_path and os.path.exists(mesh_path):
            self.label.setText(f"3D Model: {os.path.basename(mesh_path)}\nClick 'Load 3D Model' to view")
            self.load_btn.setEnabled(True)
            self.open_btn.setEnabled(True)
            self.analyze_mesh()
        else:
            self.label.setText("No valid mesh file")
            self.load_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
    
    def analyze_mesh(self):
        """Analyze the mesh file and show information"""
        if not self.mesh_path or not os.path.exists(self.mesh_path):
            return
        
        try:
            file_size = os.path.getsize(self.mesh_path)
            file_size_mb = file_size / (1024 * 1024)
            
            info = f"File: {os.path.basename(self.mesh_path)}\n"
            info += f"Size: {file_size_mb:.2f} MB\n"
            info += f"Path: {self.mesh_path}\n"
            
            if OPEN3D_AVAILABLE:
                try:
                    mesh = o3d.io.read_triangle_mesh(self.mesh_path)
                    if len(mesh.vertices) > 0:
                        info += f"Vertices: {len(mesh.vertices):,}\n"
                        info += f"Triangles: {len(mesh.triangles):,}\n"
                        if mesh.has_vertex_colors():
                            info += "Has vertex colors: Yes\n"
                        if mesh.has_vertex_normals():
                            info += "Has vertex normals: Yes\n"
                        if mesh.has_textures():
                            info += "Has textures: Yes\n"
                    else:
                        info += "Status: Empty mesh\n"
                except Exception as e:
                    info += f"Open3D Error: {str(e)}\n"
            else:
                info += "Open3D not available for detailed analysis\n"
            
            self.info_text.setText(info)
            
        except Exception as e:
            self.info_text.setText(f"Error analyzing mesh: {str(e)}")
    
    def load_mesh(self):
        """Load and display the 3D mesh"""
        if not self.mesh_path or not os.path.exists(self.mesh_path):
            self.label.setText("No valid mesh file to load")
            return
        
        if OPEN3D_AVAILABLE:
            try:
                mesh = o3d.io.read_triangle_mesh(self.mesh_path)
                if len(mesh.vertices) > 0:
                    self.label.setText(f"3D Model loaded successfully!\n{len(mesh.vertices):,} vertices\n{len(mesh.triangles):,} triangles")
                    
                    # Try to show the mesh in Open3D viewer
                    try:
                        o3d.visualization.draw_geometries([mesh], window_name="3D Model Viewer")
                    except Exception as e:
                        self.label.setText(f"3D Model loaded but viewer failed:\n{str(e)}")
                else:
                    self.label.setText("3D Model file is empty or invalid")
            except Exception as e:
                self.label.setText(f"Error loading mesh: {str(e)}")
        else:
            self.label.setText(f"3D Model file ready: {os.path.basename(self.mesh_path)}\n(Install Open3D for 3D viewing)")
    
    def open_external(self):
        """Open the mesh file in external application"""
        if self.mesh_path and os.path.exists(self.mesh_path):
            try:
                os.startfile(self.mesh_path)
            except Exception as e:
                self.label.setText(f"Error opening file: {str(e)}")
        else:
            self.label.setText("No valid mesh file to open")
