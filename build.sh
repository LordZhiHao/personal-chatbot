#!/bin/bash
# build.sh - Custom build script for Render deployment

# Update system packages
echo "Installing system dependencies..."
apt-get update
apt-get install -y portaudio19-dev python3-pyaudio python3-dev libportaudio2 libasound-dev

# Install Python requirements (without PyAudio)
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Now try to install PyAudio with system dependencies in place
echo "Installing PyAudio..."
pip install PyAudio
pip install --only-binary :all: PyAudio

# Verify installation
echo "Installed packages:"
pip list

# Run a quick test to verify PyAudio import
echo "Testing PyAudio import..."
python -c "import pyaudio; print(f'PyAudio version: {pyaudio.__version__}')" || echo "PyAudio import test failed"