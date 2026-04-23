"""
FastAPI Service for Deepfake Voice Detection
Run: uvicorn api:app --reload
"""

import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
import shutil
import os

# ===============================
# CONFIG
# ===============================

MODEL_PATH = "model/fast_model.keras"
SAMPLE_RATE = 16000
DURATION = 2
N_MELS = 64

CLASSES = ["Human Voice", "Trusted Automated", "Deepfake"]

# ===============================
# LOAD MODEL
# ===============================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ===============================
# FAST FEATURE EXTRACTION
# ===============================

def extract_feature(file_path):

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    target_len = SAMPLE_RATE * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    mel_db = mel_db[..., np.newaxis]
    mel_db = np.expand_dims(mel_db, axis=0)

    return mel_db

# ===============================
# FASTAPI APP
# ===============================

app = FastAPI(title="Deepfake Voice Detection API")

@app.get("/")
def root():
    return {"message": "Deepfake Voice Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_file = "temp_audio.wav"

    # Save uploaded file
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract features
    features = extract_feature(temp_file)

    # Predict
    prediction = model.predict(features)
    label = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Cleanup
    os.remove(temp_file)

    return {
        "prediction": CLASSES[label],
        "confidence": round(confidence, 4)
    }
