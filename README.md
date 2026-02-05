
# LatticeVAD AI Voice Detection Backend

This is the FastAPI implementation of the AI Voice Detection API for LatticeVAD.

## Prerequisites
- Python 3.9+
- FFmpeg (required by librosa for MP3 decoding)

## Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

3. **API Access**:
   The API will be available at `http://localhost:8000/api/voice-detection`.

## Testing the API (cURL)

```bash
curl -X POST http://localhost:8000/api/voice-detection \
-H "Content-Type: application/json" \
-H "x-api-key: sk_test_123456789" \
-d '{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}'
```

## Implementation Notes
- **Authentication**: Uses `x-api-key` header validation.
- **Forensics**: Uses `librosa` to analyze spectral variance as a proxy for human vs synthetic vocal jitter.
- **Constraints**: Strictly enforces the 5 supported languages and MP3 format.
