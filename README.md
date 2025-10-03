# JamKey Audio Analysis Service

This is the Python backend service for JamKey that provides audio analysis using librosa.

## Features

- Audio key detection using Krumhansl-Schmuckler algorithm
- BPM detection using beat tracking
- Support for various audio formats (WAV, MP3, M4A)
- CORS enabled for web client access
- Health check endpoints

## Dependencies

- Python 3.8+
- librosa 0.10.1
- numpy
- flask
- flask-cors
- gunicorn

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python app.py
```

3. Test the service:
```bash
python test_service.py
```

## Deployment

### Render.com

1. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Select the `render-service` directory as the root directory

2. **Configure Build Settings**
   - Build Command: `chmod +x start.sh && ./start.sh`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload --log-level info app:app`
   - Environment: `Python 3`

3. **Environment Variables**
   - `PORT`: Automatically set by Render
   - `PYTHON_VERSION`: `3.11.0` (recommended)

4. **Service Configuration**
   - Instance Type: `Starter` (512MB RAM should be sufficient)
   - Auto-Deploy: `Yes` (deploy on every push)

### Troubleshooting Deployment

**Common Issues:**

1. **pkg_resources errors**: The service includes comprehensive fallbacks for this common Python packaging issue

2. **Memory issues**: Use single worker (`--workers 1`) to manage memory usage

3. **Timeout issues**: The service uses 120-second timeout for audio processing

4. **Cold starts**: Render free tier services sleep after 15 minutes of inactivity. First request after sleep may take 30-60 seconds

**Monitoring:**
- Check Render logs for detailed error messages
- Use `/health` endpoint to verify service status
- Test with `python test_service.py https://your-service.onrender.com`

### Environment Variables

- `PORT`: Port to run the service on (automatically set by Render)
- `PYTHON_VERSION`: Python version to use (recommended: 3.11.0)

## API Endpoints

### Health Check
- `GET /health` - Returns service status
- `GET /healthz` - Returns service status (Kubernetes style)
- `GET /` - Returns service info

### Audio Analysis
- `POST /analyze` - Analyze audio file
  - Body: multipart/form-data with 'file' field
  - Returns: JSON with key, confidence, pitch_class, is_minor, bpm

## Error Handling

The service provides detailed error messages and logging for debugging. Common issues:

- Missing `pkg_resources`: Service will attempt to install setuptools
- Audio format issues: Service supports WAV, MP3, M4A formats
- Memory issues: Service uses single worker to manage memory usage

## Troubleshooting

### Local Development Issues

1. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **pkg_resources issues**: The app includes automatic fallbacks, but you can manually install with `pip install setuptools`
3. **Audio format issues**: Ensure test files are in WAV, MP3, or M4A format

### Production Issues

1. **Service unavailable**: Check Render logs and ensure the service is deployed correctly
2. **Cold start delays**: First request after inactivity may take 30-60 seconds on free tier
3. **Memory issues**: Monitor memory usage in Render dashboard
4. **Analysis failures**: Check that audio files are valid and not corrupted

### Testing

```bash
# Test local service
python test_service.py

# Test deployed service
python test_service.py https://your-service.onrender.com

# Test with curl
curl https://your-service.onrender.com/health
```