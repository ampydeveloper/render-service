#!/bin/bash

# JamKey Audio Analysis Service Startup Script
echo "[STARTUP] Starting JamKey Audio Analysis Service..."
echo "[STARTUP] Python version: $(python --version)"
echo "[STARTUP] Current directory: $(pwd)"
echo "[STARTUP] Available files: $(ls -la)"

# Set Python environment variables for better compatibility
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Install/upgrade pip and setuptools first
echo "[STARTUP] Updating pip and setuptools..."
python -m pip install --upgrade pip setuptools>=68.0.0 wheel

# Install requirements with retry logic
echo "[STARTUP] Installing Python dependencies..."
for i in {1..3}; do
    echo "[STARTUP] Installation attempt $i/3"
    if pip install -r requirements.txt --no-cache-dir; then
        echo "[STARTUP] Dependencies installed successfully"
        break
    else
        echo "[STARTUP] Installation attempt $i failed, retrying..."
        sleep 5
    fi
done

# Verify critical imports with comprehensive testing
echo "[STARTUP] Verifying Python imports..."
python -c "
print('[STARTUP] Testing core imports...')
try:
    import sys
    from types import ModuleType
    print('[STARTUP] Python version:', sys.version)

    # Inject a minimal aifc mock BEFORE importing anything that may require it
    if 'aifc' not in sys.modules:
        aifc_module = ModuleType('aifc')
        class MockAifcError(Exception):
            pass
        class MockAifcFile:
            def __init__(self, *args, **kwargs):
                pass
            def close(self):
                pass
        def _open(*args, **kwargs):
            return MockAifcFile()
        aifc_module.open = _open
        aifc_module.openfp = _open
        aifc_module.Error = MockAifcError
        sys.modules['aifc'] = aifc_module
        print('[STARTUP] Injected minimal aifc mock for startup test')

    # Test aifc import
    print('[STARTUP] Testing aifc module...')
    import aifc
    print('[STARTUP] aifc module available (mocked or real)')

    # Test core libraries
    import librosa
    import numpy
    import soundfile
    import flask

    print('[STARTUP] Core libraries imported successfully')
    print('[STARTUP] Librosa version:', librosa.__version__)
    print('[STARTUP] NumPy version:', numpy.__version__)

    # Test librosa functionality
    print('[STARTUP] Testing librosa functionality...')
    test_signal = numpy.sin(2 * numpy.pi * 440 * numpy.linspace(0, 1, 22050))
    chroma_test = librosa.feature.chroma_stft(y=test_signal, sr=22050)
    print('[STARTUP] Librosa test successful, chroma shape:', chroma_test.shape)

except Exception as e:
    print('[STARTUP] Import/test error:', e)
    import traceback
    traceback.print_exc()
    exit(1)

print('[STARTUP] All critical imports and tests verified')
"

# Start the application with better error handling
echo "[STARTUP] Starting Flask application..."
if [ "$PORT" ]; then
    echo "[STARTUP] Using PORT from environment: $PORT"
    exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --preload --log-level info --max-requests 1000 --max-requests-jitter 100 app:app
else
    echo "[STARTUP] Using default port 5000"
    exec gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 300 --preload --log-level info --max-requests 1000 --max-requests-jitter 100 app:app
fi