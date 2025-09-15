#!/usr/bin/env python3
"""
Generation Tracker - CSV logging system for pipeline runs
"""

import csv
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

class GenerationTracker:
    """Tracks all pipeline generations in CSV format"""
    
    def __init__(self, csv_path: str = "generation_log.csv"):
        self.csv_path = Path(csv_path)
        self.fieldnames = [
            'timestamp',
            'session_id', 
            'prompt',
            'step1_image',           # Generated image
            'step2_clean_image',     # Background removed
            'step3_mesh_shape',      # 3D mesh (shape only)
            'step3_mesh_textured',   # 3D mesh (with texture)
            'step3_mesh_final',      # Final mesh used
            'pipeline_duration',     # Total time in seconds
            'status',                # success/error/partial
            'error_message',         # If error occurred
            'texture_enabled',       # Whether texturing was enabled
            'depth_threshold',       # Depth settings
            'depth_max',
            'notes'                  # Additional notes
        ]
        
        # Initialize CSV file if it doesn't exist
        self._init_csv()
        
        # Current session tracking
        self.current_session = None
        self.session_start_time = None
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def start_session(self, prompt: str, texture_enabled: bool = True, 
                     depth_threshold: int = 100, depth_max: int = 200) -> str:
        """Start a new generation session"""
        self.current_session = {
            'timestamp': datetime.now().isoformat(),
            'session_id': f"session_{int(time.time())}",
            'prompt': prompt,
            'step1_image': '',
            'step2_clean_image': '',
            'step3_mesh_shape': '',
            'step3_mesh_textured': '',
            'step3_mesh_final': '',
            'pipeline_duration': '',
            'status': 'in_progress',
            'error_message': '',
            'texture_enabled': texture_enabled,
            'depth_threshold': depth_threshold,
            'depth_max': depth_max,
            'notes': ''
        }
        self.session_start_time = time.time()
        return self.current_session['session_id']
    
    def log_step1(self, image_path: str):
        """Log step 1 (image generation)"""
        if self.current_session:
            self.current_session['step1_image'] = str(image_path)
            self._log("Step 1 completed: Image generated")
    
    def log_step2(self, clean_image_path: str):
        """Log step 2 (background removal)"""
        if self.current_session:
            self.current_session['step2_clean_image'] = str(clean_image_path)
            self._log("Step 2 completed: Background removed")
    
    def log_step3(self, mesh_shape_path: Optional[str] = None, 
                  mesh_textured_path: Optional[str] = None,
                  final_mesh_path: Optional[str] = None):
        """Log step 3 (3D generation)"""
        if self.current_session:
            if mesh_shape_path:
                self.current_session['step3_mesh_shape'] = str(mesh_shape_path)
            if mesh_textured_path:
                self.current_session['step3_mesh_textured'] = str(mesh_textured_path)
            if final_mesh_path:
                self.current_session['step3_mesh_final'] = str(final_mesh_path)
            self._log("Step 3 completed: 3D mesh generated")
    
    def complete_session(self, status: str = 'success', error_message: str = '', notes: str = ''):
        """Complete the current session"""
        if self.current_session:
            if self.session_start_time:
                duration = time.time() - self.session_start_time
                self.current_session['pipeline_duration'] = f"{duration:.2f}"
            
            self.current_session['status'] = status
            self.current_session['error_message'] = error_message
            self.current_session['notes'] = notes
            
            # Write to CSV
            self._write_session()
            self._log(f"Session completed: {status}")
            
            # Reset session
            self.current_session = None
            self.session_start_time = None
    
    def _write_session(self):
        """Write current session to CSV"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(self.current_session)
    
    def _log(self, message: str):
        """Internal logging"""
        print(f"[Tracker] {message}")
    
    def get_recent_sessions(self, limit: int = 10) -> list:
        """Get recent sessions from CSV"""
        sessions = []
        if self.csv_path.exists():
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                sessions = list(reader)
        return sessions[-limit:] if sessions else []
    
    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get specific session by ID"""
        sessions = self.get_recent_sessions(limit=1000)  # Get all sessions
        for session in sessions:
            if session['session_id'] == session_id:
                return session
        return None
    
    def export_session_files(self, session_id: str, export_dir: str = "exported_sessions"):
        """Export all files from a session to a directory"""
        session = self.get_session_by_id(session_id)
        if not session:
            print(f"Session {session_id} not found")
            return
        
        export_path = Path(export_dir) / session_id
        export_path.mkdir(parents=True, exist_ok=True)
        
        files_to_copy = [
            ('step1_image', '01_generated.png'),
            ('step2_clean_image', '02_cleaned.png'),
            ('step3_mesh_shape', '03_mesh_shape.glb'),
            ('step3_mesh_textured', '03_mesh_textured.glb'),
            ('step3_mesh_final', '03_mesh_final.glb')
        ]
        
        copied_files = []
        for field, filename in files_to_copy:
            file_path = session.get(field, '')
            if file_path and os.path.exists(file_path):
                import shutil
                dest_path = export_path / filename
                shutil.copy2(file_path, dest_path)
                copied_files.append(str(dest_path))
        
        # Save session info
        info_path = export_path / "session_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Timestamp: {session['timestamp']}\n")
            f.write(f"Prompt: {session['prompt']}\n")
            f.write(f"Status: {session['status']}\n")
            f.write(f"Duration: {session['pipeline_duration']}s\n")
            f.write(f"Files copied: {len(copied_files)}\n")
            for file in copied_files:
                f.write(f"  - {file}\n")
        
        print(f"Exported session {session_id} to {export_path}")
        return str(export_path)

# Global tracker instance
tracker = GenerationTracker()

# Convenience functions for easy integration
def start_generation(prompt: str, **kwargs) -> str:
    """Start tracking a new generation"""
    return tracker.start_session(prompt, **kwargs)

def log_step1(image_path: str):
    """Log step 1 completion"""
    tracker.log_step1(image_path)

def log_step2(clean_image_path: str):
    """Log step 2 completion"""
    tracker.log_step2(clean_image_path)

def log_step3(mesh_shape_path: str = None, mesh_textured_path: str = None, final_mesh_path: str = None):
    """Log step 3 completion"""
    tracker.log_step3(mesh_shape_path, mesh_textured_path, final_mesh_path)

def complete_generation(status: str = 'success', **kwargs):
    """Complete the current generation"""
    tracker.complete_session(status, **kwargs)

def get_recent_generations(limit: int = 10):
    """Get recent generations"""
    return tracker.get_recent_sessions(limit)

# Example usage and testing
if __name__ == "__main__":
    # Test the tracker
    print("Testing Generation Tracker...")
    
    # Start a test session
    session_id = start_generation(
        prompt="A beautiful mountain landscape",
        texture_enabled=True,
        depth_threshold=100,
        depth_max=200
    )
    print(f"Started session: {session_id}")
    
    # Simulate steps
    log_step1("output/images/test_gen.png")
    log_step2("output/images/test_clean.png")
    log_step3(
        mesh_shape_path="output/meshes/test_shape.glb",
        mesh_textured_path="output/meshes/test_textured.glb",
        final_mesh_path="output/meshes/test_textured.glb"
    )
    
    # Complete session
    complete_generation(status='success', notes='Test generation')
    
    # Show recent sessions
    recent = get_recent_generations(5)
    print(f"Recent sessions: {len(recent)}")
    for session in recent:
        print(f"  {session['session_id']}: {session['prompt'][:50]}... ({session['status']})")

