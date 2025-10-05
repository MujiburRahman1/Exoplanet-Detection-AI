#!/usr/bin/env python3
"""
WSGI configuration for Exoplanet Detection AI
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the Flask app
from src.web.app import app

# For production deployment
application = app

if __name__ == "__main__":
    application.run()
