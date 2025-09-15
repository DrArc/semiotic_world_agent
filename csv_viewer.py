#!/usr/bin/env python3
"""
CSV Viewer - Simple utility to view generation logs
"""

import csv
import os
from pathlib import Path
from generation_tracker import GenerationTracker

def view_csv(csv_path: str = "generation_log.csv"):
    """View the generation log CSV in a readable format"""
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    tracker = GenerationTracker(csv_path)
    recent = tracker.get_recent_sessions(50)  # Get last 50 sessions
    
    if not recent:
        print("No generations found in log.")
        return
    
    print("="*80)
    print("GENERATION LOG VIEWER")
    print("="*80)
    print(f"Total sessions: {len(recent)}")
    print(f"CSV file: {csv_path}")
    print("="*80)
    
    for i, session in enumerate(recent[-20:], 1):  # Show last 20
        print(f"\n{i}. SESSION: {session['session_id']}")
        print(f"   Time: {session['timestamp']}")
        print(f"   Prompt: {session['prompt'][:80]}...")
        print(f"   Status: {session['status']}")
        print(f"   Duration: {session['pipeline_duration']}s")
        print(f"   Texture: {session['texture_enabled']}")
        print(f"   Depth: {session['depth_threshold']}-{session['depth_max']}")
        
        # File paths
        files = []
        if session['step1_image']:
            files.append(f"Step1: {os.path.basename(session['step1_image'])}")
        if session['step2_clean_image']:
            files.append(f"Step2: {os.path.basename(session['step2_clean_image'])}")
        if session['step3_mesh_shape']:
            files.append(f"Step3: {os.path.basename(session['step3_mesh_shape'])}")
        if session['step3_mesh_textured']:
            files.append(f"Textured: {os.path.basename(session['step3_mesh_textured'])}")
        
        if files:
            print(f"   Files: {' | '.join(files)}")
        
        if session['error_message']:
            print(f"   Error: {session['error_message']}")
        
        if session['notes']:
            print(f"   Notes: {session['notes']}")
        
        print("-" * 80)

def export_session(session_id: str, csv_path: str = "generation_log.csv"):
    """Export a specific session's files"""
    tracker = GenerationTracker(csv_path)
    export_path = tracker.export_session_files(session_id)
    print(f"Exported session {session_id} to: {export_path}")

def search_prompts(search_term: str, csv_path: str = "generation_log.csv"):
    """Search for sessions by prompt content"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    tracker = GenerationTracker(csv_path)
    all_sessions = tracker.get_recent_sessions(1000)  # Get all sessions
    
    matches = []
    for session in all_sessions:
        if search_term.lower() in session['prompt'].lower():
            matches.append(session)
    
    if not matches:
        print(f"No sessions found matching: {search_term}")
        return
    
    print(f"Found {len(matches)} sessions matching '{search_term}':")
    print("="*60)
    
    for i, session in enumerate(matches, 1):
        print(f"{i}. {session['session_id']} - {session['prompt'][:60]}...")
        print(f"   Status: {session['status']} | Duration: {session['pipeline_duration']}s")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "view":
            csv_file = sys.argv[2] if len(sys.argv) > 2 else "generation_log.csv"
            view_csv(csv_file)
        
        elif command == "export":
            if len(sys.argv) < 3:
                print("Usage: python csv_viewer.py export <session_id>")
            else:
                session_id = sys.argv[2]
                csv_file = sys.argv[3] if len(sys.argv) > 3 else "generation_log.csv"
                export_session(session_id, csv_file)
        
        elif command == "search":
            if len(sys.argv) < 3:
                print("Usage: python csv_viewer.py search <search_term>")
            else:
                search_term = sys.argv[2]
                csv_file = sys.argv[3] if len(sys.argv) > 3 else "generation_log.csv"
                search_prompts(search_term, csv_file)
        
        else:
            print("Unknown command. Use: view, export, or search")
    
    else:
        # Default: view recent generations
        view_csv()

