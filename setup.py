#!/usr/bin/env python3
"""
Setup script for the Exoplanet Detection AI project.
This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'models',
        'uploads',
        'static/css',
        'static/js',
        'templates',
        'notebooks',
        'src/data',
        'src/models',
        'src/web',
        'src/utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    return True

def create_sample_files():
    """Create sample configuration files."""
    # Create .env file
    env_content = """# Exoplanet Detection AI Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✓ Created .env file")
    
    # Create data directory README
    data_readme = """# Data Directory

Place your NASA exoplanet datasets in this directory:

- kepler_objects_of_interest.csv (Kepler Objects of Interest)
- tess_objects_of_interest.csv (TESS Objects of Interest)  
- k2_planets_and_candidates.csv (K2 Planets and Candidates)

You can download these datasets from NASA's exoplanet archive:
https://exoplanetarchive.ipac.caltech.edu/

The sample_data.csv file is provided for testing purposes.
"""
    
    with open('data/README.md', 'w') as f:
        f.write(data_readme)
    print("✓ Created data/README.md")

def main():
    """Main setup function."""
    print("🚀 Setting up Exoplanet Detection AI Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install. You may need to install them manually.")
    
    # Create sample files
    print("\n📄 Creating configuration files...")
    create_sample_files()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download NASA datasets and place them in the data/ directory")
    print("2. Run the web application: python src/web/app.py")
    print("3. Open your browser to http://localhost:5000")
    print("4. Or train a model from command line: python train_model.py --help")
    
    print("\nFor more information, see the README.md file.")

if __name__ == '__main__':
    main()
