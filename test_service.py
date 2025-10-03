#!/usr/bin/env python3
"""
Test script for JamKey Audio Analysis Service
This script tests the service functionality and aifc module mock
"""

import requests
import json
import sys
import os
import tempfile

def test_imports():
    """Test critical imports including aifc mock"""
    print("Testing critical imports...")
    
    try:
        # Test aifc import first (this is what was failing)
        print("  Testing aifc module...")
        import aifc
        print(f"  ✓ aifc module imported: {hasattr(aifc, 'open')}")
        
        # Test core libraries
        print("  Testing core libraries...")
        import librosa
        import numpy
        import soundfile
        import flask
        
        print(f"  ✓ librosa version: {librosa.__version__}")
        print(f"  ✓ numpy version: {numpy.__version__}")
        print("  ✓ soundfile imported")
        print("  ✓ flask imported")
        
        # Test librosa functionality
        print("  Testing librosa functionality...")
        test_signal = numpy.sin(2 * numpy.pi * 440 * numpy.linspace(0, 1, 22050))
        chroma_test = librosa.feature.chroma_stft(y=test_signal, sr=22050)
        print(f"  ✓ librosa test successful, chroma shape: {chroma_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_audio():
    """Create a simple test audio file"""
    try:
        import numpy as np
        import soundfile as sf
        
        # Generate a 3-second test tone (C major chord)
        duration = 3.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple chord (C major: C, E, G)
        c_freq = 261.63  # C4
        e_freq = 329.63  # E4
        g_freq = 392.00  # G4
        
        signal = (np.sin(2 * np.pi * c_freq * t) + 
                  np.sin(2 * np.pi * e_freq * t) + 
                  np.sin(2 * np.pi * g_freq * t)) / 3
        
        # Add some envelope to make it more realistic
        envelope = np.exp(-t * 0.5)  # Exponential decay
        signal = signal * envelope
        
        return signal, sample_rate
    except Exception as e:
        print(f"Failed to create test audio: {e}")
        return None, None

def test_service_health(base_url):
    """Test if the service is responding to health checks"""
    print(f"Testing service health at {base_url}...")
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"Root endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"Root response: {response.json()}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_service_analysis(base_url):
    """Test the analysis endpoint with a real audio file"""
    print(f"Testing analysis endpoint at {base_url}...")
    
    try:
        # Create test audio
        signal, sample_rate = create_test_audio()
        if signal is None:
            print("Skipping analysis test - could not create test audio")
            return False
        
        # Save to temporary file
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, signal, sample_rate)
            temp_filename = temp_file.name
        
        try:
            # Send to service
            with open(temp_filename, 'rb') as audio_file:
                files = {'file': ('test.wav', audio_file, 'audio/wav')}
                response = requests.post(f"{base_url}/analyze", files=files, timeout=60)
            
            print(f"Analysis endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Analysis successful:")
                print(f"  Key: {result.get('key', 'Unknown')}")
                print(f"  Confidence: {result.get('confidence', 0):.3f}")
                print(f"  BPM: {result.get('bpm', 'Unknown')}")
                print(f"  Is Minor: {result.get('is_minor', False)}")
                return True
            else:
                print(f"✗ Analysis failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('error', 'Unknown error')}")
                    if 'details' in error_data:
                        print(f"  Details: {error_data['details']}")
                except:
                    print(f"  Raw response: {response.text[:200]}")
                return False
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
                
    except Exception as e:
        print(f"Analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(f"Testing JamKey Audio Analysis Service")
    print("=" * 60)
    
    # Test imports first (this is what was failing)
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import tests failed")
        return 1
    
    print()
    
    # Test service if URL provided
    service_url = None
    if len(sys.argv) > 1:
        service_url = sys.argv[1]
    else:
        service_url = os.environ.get('SERVICE_URL', "https://rork-jamkey-9-7-25-385-1.onrender.com")
    
    if service_url:
        print(f"Testing service at: {service_url}")
        print("-" * 40)
        
        # Test health
        health_ok = test_service_health(service_url)
        print()
        
        # Test analysis endpoint
        analysis_ok = test_service_analysis(service_url)
        print()
        
        # Summary
        print("=" * 60)
        print("Test Summary:")
        print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
        print(f"Health check: {'PASS' if health_ok else 'FAIL'}")
        print(f"Analysis endpoint: {'PASS' if analysis_ok else 'FAIL'}")
        
        if imports_ok and health_ok and analysis_ok:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed.")
            return 1
    else:
        print("No service URL provided, only testing imports")
        print("\n✅ Import tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())