#!/bin/bash
# build.sh - Custom build script for Render deployment

# Update system packages
apt-get update 

# Install base dependencies first
pip install --upgrade pip
pip install -r requirements.txt

# Install PyAudio using only-binary option
echo "Installing PyAudio using only-binary option..."
pip install --only-binary :all: PyAudio==0.2.14

# If that fails, try using alternative wheel sources
if ! pip list | grep -q "PyAudio"; then
  echo "First method failed, trying alternative PyAudio installation method..."
  pip install --find-links https://wheel-index.linuxserver.io/ubuntu/ PyAudio==0.2.14
fi

# Verify installation
echo "Installed packages:"
pip list

# Run a quick test to verify PyAudio import works
python -c "import pyaudio; print(f'PyAudio version: {pyaudio.__version__}')" || echo "PyAudio import test failed"