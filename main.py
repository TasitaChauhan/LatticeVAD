
import base64
import io
import os
import time
import numpy as np
import librosa
from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional
from enum import Enum
from fastapi.responses import JSONResponse

# --- Configuration ---
API_KEY_NAME = "x-api-key"
# Allow configuring valid API keys via environment variable (comma-separated). Falls back to defaults for local testing.
VALID_API_KEYS = set(os.environ.get('VALID_API_KEYS', 'sk_test_123456789,LATTICE_PROD_9921').split(','))

# CORS origins can be provided via env (comma-separated) or default to allowing all origins for ease of public testing.
_cors_origins = os.environ.get('CORS_ORIGINS', '*')
if _cors_origins == '*':
    CORS_ORIGINS = ['*']
else:
    CORS_ORIGINS = [o.strip() for o in _cors_origins.split(',') if o.strip()]

# ========== STABILITY CONFIGURATION ==========
# Enforce strict Base64 payload size limit (3MB)
MAX_BASE64_LENGTH = int(os.environ.get('MAX_BASE64_LENGTH', '3000000'))

# Enforce maximum audio duration processing (30 seconds)
MAX_AUDIO_SECONDS = int(os.environ.get('MAX_AUDIO_SECONDS', '30'))
TARGET_SAMPLE_RATE = int(os.environ.get('TARGET_SAMPLE_RATE', '16000'))

# Processing time budget per request (5 seconds)
PROCESSING_TIME_BUDGET = float(os.environ.get('PROCESSING_TIME_BUDGET', '5.0'))

# ========== FORENSIC THRESHOLDS (Language-Aware) ==========
THRESHOLDS = {
    'English': {
        'pitch_vibrato_strength': 0.015,
        'pitch_contour_entropy': 1.2,
        'spectral_entropy': 4.5,
        'mfcc_variance': 0.8,
        'formant_bandwidth': 80,
        'breath_presence': 0.05,
        'harmonicity': 0.75,
        'energy_envelope_smoothness': 0.92,
    },
    'Tamil': {
        'pitch_vibrato_strength': 0.02,
        'pitch_contour_entropy': 1.3,
        'spectral_entropy': 4.6,
        'mfcc_variance': 0.85,
        'formant_bandwidth': 90,
        'breath_presence': 0.06,
        'harmonicity': 0.72,
        'energy_envelope_smoothness': 0.90,
    },
    'Malayalam': {
        'pitch_vibrato_strength': 0.02,
        'pitch_contour_entropy': 1.35,
        'spectral_entropy': 4.7,
        'mfcc_variance': 0.87,
        'formant_bandwidth': 95,
        'breath_presence': 0.07,
        'harmonicity': 0.70,
        'energy_envelope_smoothness': 0.88,
    },
    'Hindi': {
        'pitch_vibrato_strength': 0.018,
        'pitch_contour_entropy': 1.25,
        'spectral_entropy': 4.55,
        'mfcc_variance': 0.82,
        'formant_bandwidth': 85,
        'breath_presence': 0.055,
        'harmonicity': 0.73,
        'energy_envelope_smoothness': 0.91,
    },
    'Telugu': {
        'pitch_vibrato_strength': 0.02,
        'pitch_contour_entropy': 1.3,
        'spectral_entropy': 4.6,
        'mfcc_variance': 0.85,
        'formant_bandwidth': 90,
        'breath_presence': 0.06,
        'harmonicity': 0.72,
        'energy_envelope_smoothness': 0.90,
    }
}

class Language(str, Enum):
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"

class Classification(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

# --- Models ---
class DetectionRequest(BaseModel):
    language: Language
    audioFormat: str
    audioBase64: str

    @validator('audioFormat')
    def format_must_be_mp3(cls, v):
        if v.lower() != 'mp3':
            raise ValueError('audioFormat must be mp3')
        return v.lower()

class DetectionResponse(BaseModel):
    status: str
    language: str
    classification: Classification
    confidenceScore: float
    explanation: str

class ErrorResponse(BaseModel):
    status: str
    message: str

app = FastAPI(title="LatticeVAD AI Voice Detection API")

# Configure CORS so this API can be called from public tester UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "LatticeVAD API running", "endpoint": "/api/voice-detection"}


@app.post("/api/voice-detection/debug", tags=["debug"])
async def voice_detection_debug(
    request: DetectionRequest,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    """Debug endpoint: returns raw feature values without classification."""
    provided_key = None
    if x_api_key:
        provided_key = x_api_key.strip()
    elif authorization:
        auth = authorization.strip()
        if auth.lower().startswith('bearer '):
            provided_key = auth.split(' ', 1)[1].strip()
        else:
            provided_key = auth

    if not provided_key or provided_key not in VALID_API_KEYS:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # Quick protect: reject excessively large uploads to avoid timeouts
    if not request.audioBase64 or len(request.audioBase64) > MAX_BASE64_LENGTH:
        return JSONResponse(
            status_code=413,
            content={"status": "error", "message": "Audio payload too large or empty. Maximum 3MB base64 allowed."}
        )

    try:
        audio_data = base64.b64decode(request.audioBase64)
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid base64 payload: {str(e)}"})

    # Enforce processing time budget
    request_start = time.time()
    try:
        features = extract_forensic_features(audio_data, start_time=request_start, time_budget=PROCESSING_TIME_BUDGET)
        thresholds = THRESHOLDS.get(request.language.value, THRESHOLDS['English'])
        return {
            "status": "success",
            "language": request.language,
            "features": features,
            "thresholds": thresholds
        }
    except TimeoutError:
        return JSONResponse(status_code=408, content={"status": "error", "message": "Processing time exceeded budget (timeout)"})
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Debug analysis failed: {str(e)}"}
        )

# --- Helper Functions ---
def extract_forensic_features(audio_bytes: bytes, start_time=None, time_budget=None):
    """
    Extracts forensic-grade features for detecting AI-generated vs human voices.
    Analyzes multiple independent forensic markers using lightweight librosa features.
    """
    # Initialize timing for request-level time budget enforcement
    if start_time is None:
        start_time = time.time()
    if time_budget is None:
        time_budget = PROCESSING_TIME_BUDGET

    def _check_time():
        """Helper to enforce processing time budget; raises TimeoutError if exceeded."""
        if time.time() - start_time > time_budget:
            raise TimeoutError(f"Processing exceeded time budget of {time_budget}s")

    # Load audio with strict limits: resample to 16kHz, limit to 30 seconds
    with io.BytesIO(audio_bytes) as audio_file:
        y, sr = librosa.load(audio_file, sr=TARGET_SAMPLE_RATE, mono=True, duration=MAX_AUDIO_SECONDS)

    # Prevent processing extremely short audio
    if len(y) < int(0.25 * sr):
        raise ValueError("Audio too short (minimum 0.25 second required)")

    # ====== FORENSIC MARKER 1: PITCH VARIANCE ======
    # Human voices show natural pitch variance; AI often unnaturally flat.
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, trough_threshold=0.1)
    except Exception:
        f0 = np.zeros(max(1, int(len(y) / 512)))
    
    f0_voiced = f0[f0 > 0]
    if len(f0_voiced) > 2:
        pitch_variance = float(np.var(f0_voiced))
        pitch_diff_variance = float(np.var(np.diff(f0_voiced)))
    else:
        pitch_variance = 0.0
        pitch_diff_variance = 0.0
    _check_time()

    # ====== FORENSIC MARKER 2: SPECTRAL ENTROPY ======
    # Human speech has rich spectral content; AI typically simpler.
    duration_seconds = len(y) / float(sr)
    if duration_seconds > 30:
        n_mels = 32
        n_mfcc = 8
    elif duration_seconds > 10:
        n_mels = 48
        n_mfcc = 13
    else:
        n_mels = 64
        n_mfcc = 13

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = np.abs(S_db)
    S_norm = S_norm / (np.sum(S_norm) + 1e-10)
    spectral_entropy = -np.sum(S_norm * np.log(S_norm + 1e-10)) / np.log(S_norm.size)
    _check_time()

    # ====== FORENSIC MARKER 3: MFCC MEAN & VARIANCE ======
    # AI: overly consistent MFCCs; Human: high natural variation.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))
    mfcc_frame_diff = np.abs(np.diff(mfcc, axis=1)) if mfcc.shape[1] > 1 else np.zeros((mfcc.shape[0], 1))
    mfcc_delta_variance = float(np.mean(np.var(mfcc_frame_diff, axis=1)))
    _check_time()

    # ====== FORENSIC MARKER 4: SPECTRAL CENTROID & ROLLOFF ======
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=512)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, hop_length=512)[0]
    formant_bandwidth = np.mean(spectral_rolloff - spectral_centroid)
    formant_bandwidth_hz = formant_bandwidth * sr / 2
    spectral_centroid_var = float(np.var(spectral_centroid))
    _check_time()

    # ====== FORENSIC MARKER 5: BREATH & SILENCE PATTERNS ======
    # Human speech has natural pauses; AI often unnaturally continuous.
    rms_energy = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    rms_threshold = np.percentile(rms_energy, 20)
    silence_frames = np.sum(rms_energy < rms_threshold)
    silence_ratio = silence_frames / len(rms_energy)

    silent_segments = []
    in_silence = False
    segment_length = 0
    for energy in rms_energy:
        if energy < rms_threshold:
            if not in_silence:
                in_silence = True
                segment_length = 1
            else:
                segment_length += 1
        else:
            if in_silence and 5 < segment_length < 50:
                silent_segments.append(segment_length)
            in_silence = False
            segment_length = 0

    breath_presence = len(silent_segments) / (len(rms_energy) / 512 + 1e-6)
    _check_time()

    # ====== FORENSIC MARKER 6: HARMONICITY ======
    # AI: overly pure/harmonic; Human: noisier, less harmonic.
    autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    fundamental_region = autocorr[int(sr/300):int(sr/50)]
    harmonicity = float(np.max(fundamental_region)) if len(fundamental_region) > 0 else 0.5
    _check_time()

    # ====== FORENSIC MARKER 7: ENERGY ENVELOPE SMOOTHNESS ======
    # AI: unnaturally smooth; Human: natural variability.
    if np.max(rms_energy) - np.min(rms_energy) > 1e-9:
        rms_normalized = (rms_energy - np.min(rms_energy)) / (np.max(rms_energy) - np.min(rms_energy) + 1e-10)
    else:
        rms_normalized = rms_energy * 0.0
    rms_diff = np.abs(np.diff(rms_normalized))
    rms_smoothness = np.mean(rms_diff)
    energy_envelope_smoothness = 1.0 - rms_smoothness
    _check_time()

    # ====== ADDITIONAL LIGHTWEIGHT FEATURES ======
    try:
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=512)))
    except Exception:
        spectral_flatness = 0.0

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)[0]
    zcr_variance = float(np.var(zcr))
    _check_time()

    # Return all forensic features
    return {
        'pitch_variance': pitch_variance,
        'pitch_diff_variance': pitch_diff_variance,
        'spectral_entropy': float(spectral_entropy),
        'spectral_flatness': spectral_flatness,
        'spectral_centroid_variance': spectral_centroid_var,
        'formant_bandwidth_hz': float(formant_bandwidth_hz),
        'mfcc_mean': [float(x) for x in mfcc_mean],
        'mfcc_variance': mfcc_variance,
        'mfcc_delta_variance': mfcc_delta_variance,
        'silence_ratio': float(silence_ratio),
        'breath_presence': float(breath_presence),
        'harmonicity': harmonicity,
        'energy_envelope_smoothness': float(energy_envelope_smoothness),
        'zcr_variance': zcr_variance,
        'duration': float(len(y) / sr),
        'sample_rate': int(sr)
    }

# --- Endpoint ---
@app.post("/api/voice-detection", response_model=DetectionResponse)
async def voice_detection(
    request: DetectionRequest,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    """Main AI-generated voice detection endpoint."""
    # 1. API Key Validation
    provided_key = None
    if x_api_key:
        provided_key = x_api_key.strip()
    elif authorization:
        auth = authorization.strip()
        if auth.lower().startswith('bearer '):
            provided_key = auth.split(' ', 1)[1].strip()
        else:
            provided_key = auth

    if not provided_key or provided_key not in VALID_API_KEYS:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key or malformed request"}
        )

    # 2. Payload size check
    if not request.audioBase64 or len(request.audioBase64) > MAX_BASE64_LENGTH:
        return JSONResponse(
            status_code=413,
            content={"status": "error", "message": "Audio payload too large or empty. Maximum 3MB base64 allowed."}
        )

    try:
        # 3. Base64 Decoding
        try:
            audio_data = base64.b64decode(request.audioBase64)
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid base64 payload: {str(e)}"})

        # 4. Feature Extraction with time budget
        request_start = time.time()
        try:
            features = extract_forensic_features(audio_data, start_time=request_start, time_budget=PROCESSING_TIME_BUDGET)
        except TimeoutError:
            return JSONResponse(status_code=408, content={"status": "error", "message": "Processing time exceeded budget (timeout)"})

        # 5. Get language-specific thresholds
        lang_code = request.language.value
        thresholds = THRESHOLDS.get(lang_code, THRESHOLDS['English'])

        # ====== FORENSIC-GRADE CLASSIFICATION ======
        lang = lang_code
        lang_factor = 1.0
        if lang in ('Tamil', 'Malayalam', 'Hindi', 'Telugu'):
            lang_factor = 1.15

        def score_low(value, ref, scale):
            return float(max(0.0, min(1.0, (ref - value) / (scale + 1e-9))))

        def score_high(value, ref, scale):
            return float(max(0.0, min(1.0, (value - ref) / (scale + 1e-9))))

        f = features
        refs = {
            'pitch_diff_var': 20.0 / lang_factor,
            'mfcc_var': 0.6 * lang_factor,
            'mfcc_delta_var': 0.5 * lang_factor,
            'spectral_centroid_var': 300.0 / lang_factor,
            'spectral_flatness': 0.04 * lang_factor,
            'energy_smooth': 0.90 * lang_factor,
            'zcr_var': 0.001 * lang_factor,
            'breath_presence': 0.03 * lang_factor,
            'harmonicity': 0.80 * lang_factor
        }

        # Score each forensic marker (1.0 = strong AI indicator)
        s_pitch = score_low(f.get('pitch_diff_variance', 0.0), refs['pitch_diff_var'], refs['pitch_diff_var'] * 1.2)
        s_mfcc = score_low(f.get('mfcc_variance', 0.0), refs['mfcc_var'], refs['mfcc_var'] * 1.2)
        s_mfcc_delta = score_low(f.get('mfcc_delta_variance', 0.0), refs['mfcc_delta_var'], refs['mfcc_delta_var'] * 1.2)
        s_scent = score_low(f.get('spectral_centroid_variance', 0.0), refs['spectral_centroid_var'], refs['spectral_centroid_var'] * 1.2)
        s_flat = score_low(f.get('spectral_flatness', 0.0), refs['spectral_flatness'], refs['spectral_flatness'] * 1.2)
        s_energy = score_high(f.get('energy_envelope_smoothness', 0.0), refs['energy_smooth'], 0.08)
        s_zcr = score_low(f.get('zcr_variance', 0.0), refs['zcr_var'], refs['zcr_var'] * 2.0)
        s_breath = score_low(f.get('breath_presence', 0.0), refs['breath_presence'], refs['breath_presence'] * 1.5)
        s_harm = score_high(f.get('harmonicity', 0.0), refs['harmonicity'], 0.15)

        # Weighted aggregation (weights sum to 1.0)
        weights = {
            'pitch': 0.18,
            'mfcc': 0.18,
            'mfcc_delta': 0.12,
            'scent': 0.12,
            'flat': 0.12,
            'energy': 0.12,
            'zcr': 0.04,
            'breath': 0.04,
            'harm': 0.08
        }

        weighted_score = (
            weights['pitch'] * s_pitch
            + weights['mfcc'] * s_mfcc
            + weights['mfcc_delta'] * s_mfcc_delta
            + weights['scent'] * s_scent
            + weights['flat'] * s_flat
            + weights['energy'] * s_energy
            + weights['zcr'] * s_zcr
            + weights['breath'] * s_breath
            + weights['harm'] * s_harm
        )

        # Confidence mapping: [0.0-1.0] -> [0.5-0.95]
        confidence = 0.5 + (0.45 * float(max(0.0, min(1.0, weighted_score))))
        confidence = min(0.95, confidence)

        # Classification
        if weighted_score > 0.45:
            classification = Classification.AI_GENERATED
        elif weighted_score < 0.25:
            classification = Classification.HUMAN
        else:
            # Uncertain zone: prefer AI if any strong signal
            classification = Classification.AI_GENERATED if max(s_pitch, s_mfcc, s_energy, s_flat) > 0.5 else Classification.HUMAN

        # Build explanations from strong signals
        cues = []
        if s_pitch > 0.35:
            cues.append("Unnaturally consistent pitch contour detected")
        if s_flat > 0.35:
            cues.append("Low spectral flatness indicative of synthesis")
        if s_scent > 0.35:
            cues.append("Stable spectral centroid detected (low variance)")
        if s_energy > 0.35:
            cues.append("Excessively smooth energy envelope detected")
        if s_breath > 0.35:
            cues.append("Absence of micro-pauses and breath artifacts")
        if s_mfcc > 0.35:
            cues.append("Abnormally consistent MFCC/timbre detected")

        explanation = ", ".join(cues) if cues else "No strong forensic indicators detected"

        # Reduce human confidence if strong AI signal exists
        if classification == Classification.HUMAN and max(s_pitch, s_mfcc, s_energy, s_flat) > 0.4:
            confidence = min(confidence, 0.8)

        confidence = round(float(confidence), 2)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Processing failed: {str(e)}"}
        )
