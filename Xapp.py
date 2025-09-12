#!/usr/bin/env python3
"""
Main application file for Indian Speech-to-Text Models
Complete Guide to Free Open-Source Speech-to-Text Models for Indian Languages
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ui.gradio_app import GradioSpeechToTextApp
from utils.config import load_config

def main():
    """Main entry point for the application."""
    print("🎤 Indian Speech-to-Text Models - Starting Application...")
    
    # Load configuration
    config = load_config()
    
    print(f"📍 Environment: {config['APP_ENV']}")
    print(f"🔧 Debug Mode: {config['DEBUG']}")
    print(f"🎯 Default Model: {config['DEFAULT_MODEL']}")
    print(f"🌐 Server: {config['GRADIO_SERVER_NAME']}:{config['GRADIO_SERVER_PORT']}")
    
    # Create and launch the application
    app = GradioSpeechToTextApp()
    app.launch()

if __name__ == "__main__":
    main()
