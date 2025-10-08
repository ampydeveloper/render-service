#!/usr/bin/env python3
"""
JamKey Audio Analysis Service
A Flask service for music key detection using librosa
"""

import sys
import os
from types import ModuleType

# CRITICAL: Mock aifc module BEFORE ANY OTHER IMPORTS
# Python 3.13+ removed aifc module which librosa depends on
# This must happen before importing logging or any other module

print("[STARTUP] Initializing comprehensive aifc module mock...")

class MockAifcError(Exception):
    """Mock AIFC error class"""
    pass

class MockAifcFile:
    """Mock AIFC file class with comprehensive interface"""
    def __init__(self, *args, **kwargs):
        self._closed = False
        self._nchannels = 1
        self._sampwidth = 2
        self._framerate = 44100
        self._nframes = 0
        self._comptype = 'NONE'
        self._compname = 'not compressed'
        self._position = 0
        self._markers = []

    def close(self):
        self._closed = True

    def getnchannels(self):
        return self._nchannels

    def getsampwidth(self):
        return self._sampwidth

    def getframerate(self):
        return self._framerate

    def getnframes(self):
        return self._nframes

    def readframes(self, n):
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return b'\x00' * (n * self._nchannels * self._sampwidth)

    def writeframes(self, data):
        if self._closed:
            raise ValueError("I/O operation on closed file")
        pass

    def setparams(self, params):
        if len(params) >= 6:
            self._nchannels, self._sampwidth, self._framerate, self._nframes, self._comptype, self._compname = params[:6]

    def getparams(self):
        return (self._nchannels, self._sampwidth, self._framerate, self._nframes, self._comptype, self._compname)

    def tell(self):
        return self._position

    def rewind(self):
        self._position = 0

    def setpos(self, pos):
        self._position = pos

    def getmark(self, id):
        for mark in self._markers:
            if mark[0] == id:
                return mark
        return None

    def getmarkers(self):
        return self._markers[:]

    def mark(self, id, pos, name):
        self._markers.append((id, pos, name))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Additional methods that might be called
    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        return self._compname

    def setnchannels(self, nchannels):
        self._nchannels = nchannels

    def setsampwidth(self, sampwidth):
        self._sampwidth = sampwidth

    def setframerate(self, framerate):
        self._framerate = framerate

    def setnframes(self, nframes):
        self._nframes = nframes

    def setcomptype(self, comptype, compname):
        self._comptype = comptype
        self._compname = compname

class MockAifc:
    """Mock AIFC module class"""
    Error = MockAifcError

    def open(self, file, mode='rb'):
        """Open an AIFC file and return a file object"""
        return MockAifcFile(file, mode)

    def openfp(self, fp, mode='rb'):
        """Open an AIFC file from a file pointer"""
        return MockAifcFile(fp, mode)

# Create comprehensive aifc module mock with all necessary attributes
if 'aifc' not in sys.modules:
    print("[STARTUP] Creating comprehensive aifc module mock...")
    aifc_module = ModuleType('aifc')
    mock_aifc = MockAifc()

    # Core functions
    aifc_module.open = mock_aifc.open
    aifc_module.openfp = mock_aifc.openfp

    # Exception classes
    aifc_module.Error = MockAifcError

    # File classes
    aifc_module.Aifc_read = MockAifcFile
    aifc_module.Aifc_write = MockAifcFile

    # Constants that might be referenced by librosa or audioread
    aifc_module.AIFC_VERSION = 1
    aifc_module._AIFC_version = 0xA2805140

    # Compression types
    aifc_module.COMPRESSION_TYPES = ['NONE', 'ULAW', 'ALAW']

    # Add module-level attributes that might be accessed
    aifc_module.__version__ = '1.0.0'
    aifc_module.__all__ = ['open', 'openfp', 'Error', 'Aifc_read', 'Aifc_write']

    # Add to sys.modules before any other imports that might need it
    sys.modules['aifc'] = aifc_module
    print("[STARTUP] Comprehensive aifc module mock created successfully")
else:
    print("[STARTUP] aifc module already exists in sys.modules")
    # Verify the existing module has the necessary attributes
    existing_aifc = sys.modules['aifc']
    if not hasattr(existing_aifc, 'open'):
        print("[STARTUP] WARNING: Existing aifc module missing 'open' function")
    if not hasattr(existing_aifc, 'Error'):
        print("[STARTUP] WARNING: Existing aifc module missing 'Error' class")

# Now import logging and set it up
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("[STARTUP] Pre-emptively created aifc mock module")
logger.info(f"[STARTUP] Python version: {sys.version}")
logger.info(f"[STARTUP] Available modules: {list(sys.modules.keys())[:10]}...")

# Now import Flask and other libraries
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import librosa and numpy with error handling
try:
    import librosa
    import numpy as np
    import soundfile as sf
    import audioread
    logger.info("Successfully imported librosa, numpy, soundfile, and audioread")
    logger.info(f"Librosa version: {librosa.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    raise

# Import standard library modules that might be missing
try:
    import wave
    import audioop
    logger.info("Successfully imported basic audio codec modules")
except ImportError as e:
    logger.warning(f"Some basic audio codec modules not available: {e}")

# Handle other potentially missing audio codec modules
try:
    import sndhdr
    logger.info("Successfully imported sndhdr module")
except ImportError as e:
    logger.warning(f"sndhdr module not available: {e}")
    # Create a minimal sndhdr mock
    import sys
    from types import ModuleType
    
    class MockSndhdr:
        def what(self, filename):
            # Return None to indicate unknown format
            return None
        
        def whathdr(self, filename):
            return None
    
    sndhdr_module = ModuleType('sndhdr')
    mock_sndhdr = MockSndhdr()
    sndhdr_module.what = mock_sndhdr.what
    sndhdr_module.whathdr = mock_sndhdr.whathdr
    
    sys.modules['sndhdr'] = sndhdr_module
    logger.info("Created sndhdr mock module")

# Handle ossaudiodev module (Unix-specific)
try:
    import ossaudiodev
    logger.info("Successfully imported ossaudiodev module")
except ImportError as e:
    logger.warning(f"ossaudiodev module not available (expected on non-Unix systems): {e}")
    # This is expected on non-Unix systems, no need to mock

# Handle pkg_resources import with comprehensive fallbacks
try:
    # First try the standard pkg_resources
    import pkg_resources
    logging.info("pkg_resources imported successfully")
except ImportError:
    try:
        # Try setuptools.pkg_resources first (most common)
        from setuptools import pkg_resources
        logging.info("Using setuptools.pkg_resources")
    except ImportError:
        try:
            # Try importlib.metadata (Python 3.8+)
            import importlib.metadata as pkg_resources
            logging.info("Using importlib.metadata as pkg_resources fallback")
        except ImportError:
            try:
                # Try importlib_metadata (backport)
                import importlib_metadata as pkg_resources
                logging.info("Using importlib_metadata as pkg_resources fallback")
            except ImportError:
                try:
                    # Try distutils for older systems
                    from distutils import version
                    import sys
                    
                    class MockPkgResources:
                        @staticmethod
                        def get_distribution(name):
                            class MockDistribution:
                                version = '1.0.0'
                                def __init__(self):
                                    pass
                            return MockDistribution()
                        
                        @staticmethod
                        def resource_filename(package, resource):
                            return f"/tmp/{package}_{resource}"
                        
                        @staticmethod
                        def resource_string(package, resource):
                            return b"mock_resource"
                        
                        @staticmethod
                        def resource_exists(package, resource):
                            return True
                        
                        @staticmethod
                        def working_set():
                            return []
                    
                    pkg_resources = MockPkgResources()
                    logging.info("Using enhanced mock pkg_resources with distutils fallback")
                except ImportError:
                    # Final fallback - create minimal mock
                    class MinimalPkgResources:
                        @staticmethod
                        def get_distribution(name):
                            class MockDistribution:
                                version = '1.0.0'
                            return MockDistribution()
                        
                        @staticmethod
                        def resource_filename(package, resource):
                            return f"/tmp/{package}_{resource}"
                        
                        @staticmethod
                        def resource_string(package, resource):
                            return b"mock_resource"
                        
                        @staticmethod
                        def resource_exists(package, resource):
                            return True
                        
                        @staticmethod
                        def working_set():
                            return []
                    
                    pkg_resources = MinimalPkgResources()
                    logging.info("Using minimal mock pkg_resources - final fallback")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Test librosa functionality on startup
try:
    logger.info("Testing librosa functionality...")
    # Create a simple test signal
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 1 second of 440Hz tone
    # Test basic librosa functions
    chroma_test = librosa.feature.chroma_cqt(y=test_signal, sr=22050)
    tempo_test, _ = librosa.beat.beat_track(y=test_signal, sr=22050)
    logger.info(f"Librosa test successful - tempo: {tempo_test}, chroma shape: {chroma_test.shape}")
except Exception as test_error:
    logger.error(f"Librosa test failed: {test_error}")
    logger.error("Service may not function properly")

# Map of pitch class to key names
PITCH_CLASS_MAP = {
    0: ['C Major', 'A Minor'],
    1: ['C# Major', 'A# Minor'],
    2: ['D Major', 'B Minor'],
    3: ['Eb Major', 'C Minor'],
    4: ['E Major', 'C# Minor'],
    5: ['F Major', 'D Minor'],
    6: ['F# Major', 'D# Minor'],
    7: ['G Major', 'E Minor'],
    8: ['Ab Major', 'F Minor'],
    9: ['A Major', 'F# Minor'],
    10: ['Bb Major', 'G Minor'],
    11: ['B Major', 'G# Minor']
}

@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "ok", "service": "jamkey-audio-analysis", "version": "1.0"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "service": "jamkey-audio-analysis"})

@app.route('/healthz', methods=['GET'])
def health_check_z():
    return jsonify({"status": "ok", "service": "jamkey-audio-analysis"})

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        if 'file' not in request.files:
            logging.error("No file in request.files. Available keys: %s", list(request.files.keys()))
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['file']
        
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        logging.info("Received file: %s", audio_file.filename)
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_filename = temp_file.name
        
        try:
            logging.info("Starting audio analysis with librosa...")
            
            # Load the audio file with librosa
            logging.info("Loading audio file...")
            try:
                # Try loading with librosa first (handles most formats)
                # Limit file size and use memory-efficient loading
                file_size = os.path.getsize(temp_filename)
                if file_size > 50 * 1024 * 1024:  # 50MB limit
                    raise Exception("Audio file too large. Maximum size is 50MB.")
                
                y, sr = librosa.load(temp_filename, sr=22050, mono=True, dtype=np.float32)
                logging.info(f"Audio loaded with librosa: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz, memory_usage={y.nbytes / 1024 / 1024:.1f}MB")
            except Exception as load_error:
                logging.warning(f"Librosa load failed: {load_error}")
                try:
                    # Fallback to soundfile
                    y, sr = sf.read(temp_filename)
                    logging.info(f"Audio loaded with soundfile: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
                except Exception as sf_error:
                    logging.error(f"Both librosa and soundfile failed: librosa={load_error}, soundfile={sf_error}")
                    raise Exception(f"Unable to load audio file. Librosa error: {load_error}. Soundfile error: {sf_error}")
            
            # Extract harmonic component for key detection
            logging.info("Extracting harmonic component...")
            y_harmonic = librosa.effects.harmonic(y)
            
            # Compute chromagram
            logging.info("Computing chromagram...")
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            
            # Compute BPM using beat tracking
            logging.info("Computing BPM...")
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)
            logging.info(f"BPM detected: {bpm}")
            
            # Compute key using Krumhansl-Schmuckler key-finding algorithm
            logging.info("Computing key using Krumhansl-Schmuckler algorithm...")
            key_corrs = []
            for i in range(12):  # 12 major keys
                major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
                major_template = np.roll(major_template, i)
                major_corr = np.corrcoef(np.mean(chroma, axis=1), major_template)[0, 1]
                key_corrs.append(major_corr)
            
            for i in range(12):  # 12 minor keys
                minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
                minor_template = np.roll(minor_template, i)
                minor_corr = np.corrcoef(np.mean(chroma, axis=1), minor_template)[0, 1]
                key_corrs.append(minor_corr)
            
            # Find key with maximum correlation
            key_index = np.argmax(key_corrs)
            is_minor = key_index >= 12
            pitch_class = key_index % 12
            
            # Determine confidence based on correlation value
            confidence = float((key_corrs[key_index] + 1) / 2)  # Normalize from [-1,1] to [0,1]
            
            # Get key name
            if is_minor:
                key_name = PITCH_CLASS_MAP[pitch_class][1]  # Minor key
            else:
                key_name = PITCH_CLASS_MAP[pitch_class][0]  # Major key
            
            logging.info(f"Key analysis complete: {key_name} (confidence: {confidence:.3f})")
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
            result = {
                "key": key_name,
                "confidence": confidence,
                "pitch_class": int(pitch_class),
                "is_minor": bool(is_minor),
                "bpm": round(bpm, 1)
            }
            
            logging.info("Analysis result: %s", result)
            return jsonify(result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Error analyzing audio: {str(e)}")
            logging.error(f"Full traceback: {error_details}")
            
            # Clean up the temporary file
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.unlink(temp_filename)
            
            # Provide more specific error messages
            error_msg = str(e)
            if "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                error_msg = "Audio file too large. Please try a shorter audio file."
            elif "format" in error_msg.lower() or "codec" in error_msg.lower() or "unsupported" in error_msg.lower():
                error_msg = "Unsupported audio format. Please use WAV, MP3, or M4A files."
            elif "timeout" in error_msg.lower():
                error_msg = "Audio analysis timed out. Please try a shorter audio file."
            elif "librosa" in error_msg.lower():
                error_msg = f"Audio processing error: {error_msg}"
            elif "numpy" in error_msg.lower():
                error_msg = f"Numerical computation error: {error_msg}"
            else:
                error_msg = "Unable to process audio. Please try again."
            
            return jsonify({"error": error_msg, "details": str(e)}), 500
            
    except Exception as e:
        logging.error(f"Error in analyze_audio: {str(e)}")
        return jsonify({"error": f"Request processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting JamKey Audio Analysis Service...")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Librosa version: {librosa.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)