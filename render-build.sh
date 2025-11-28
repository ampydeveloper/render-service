#!/bin/bash
# Render Build Script for JamKey Audio Analysis Service

set -o errexit  # Exit on error

echo "[BUILD] Starting build process..."
echo "[BUILD] Note: System dependencies (ffmpeg, libsndfile1) will be installed via aptfile"

echo "[BUILD] Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "[BUILD] ✓ ffmpeg is available"
    ffmpeg -version | head -n 1
else
    echo "[BUILD] ⚠ ffmpeg not found yet (should be installed via aptfile)"
fi

echo "[BUILD] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "[BUILD] Verifying critical imports..."
python -c "
import sys
print(f'Python version: {sys.version}')
import librosa
import numpy as np
import soundfile as sf
print(f'✓ librosa {librosa.__version__}')
print(f'✓ numpy {np.__version__}')
print(f'✓ soundfile imported')
"

echo "[BUILD] Build completed successfully!"
