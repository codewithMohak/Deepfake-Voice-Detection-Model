"""
FAST TRAINING - Simplified Deepfake Voice Detection
Trains much faster for development/testing
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import librosa
from tqdm import tqdm

# ==============================
# BASIC CONFIG (Hardcoded)
# ==============================

DATA_DIR = "data/my_dataset"  # Change to your dataset path
CLASSES = ["human_voice", "automated_trusted", "deepfake"]

SAMPLE_RATE = 16000
DURATION = 2              # Reduced from 3 → Faster
N_MELS = 64               # Reduced feature size
EPOCHS = 8                # Reduced epochs
BATCH_SIZE = 16

MODEL_SAVE_PATH = "model/fast_model.keras"


# ==============================
# FEATURE EXTRACTION (FASTER)
# ==============================

def extract_feature(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        target_len = SAMPLE_RATE * DURATION

        # Fix audio length
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64,
            n_fft=1024,
            hop_length=256
        )

        mel_db = librosa.power_to_db(mel)

        # ---------- FORCE SHAPE ----------
        mel_db = librosa.util.fix_length(mel_db, size=64, axis=0)
        mel_db = librosa.util.fix_length(mel_db, size=63, axis=1)
        # ---------------------------------

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        mel_db = mel_db[..., np.newaxis]   # (64,63,1)

        return mel_db.astype(np.float32)

    except Exception as e:
        print("Bad file skipped:", file_path)
        return None



# ==============================
# LOAD DATASET
# ==============================

print("\nLoading dataset...")

X, y = [], []

base_path = Path(DATA_DIR)
print("Dataset path →", base_path.resolve())

for label, cls in enumerate(CLASSES):

    folder = base_path / cls

    if not folder.exists():
        print(f"❌ Missing folder → {folder}")
        continue

    # Load audio recursively from subfolders like hindi/, marathi/
    files = (
        list(folder.rglob("*.wav")) +
        list(folder.rglob("*.mp3")) +
        list(folder.rglob("*.flac")) +
        list(folder.rglob("*.m4a")) +
        list(folder.rglob("*.ogg"))
    )

    print(f"{cls} → Found {len(files)} audio files")

    # Debug: show first few files
    for f in files[:5]:
        print("   ", f)

    for file in tqdm(files, desc=f"Loading {cls}"):
        feat = extract_feature(str(file))

        # Skip broken files safely
        if feat is None:
            print(f"⚠ Skipping bad file → {file}")
            continue

        X.append(feat)
        y.append(label)

print("Number of extracted samples:", len(X))
print("Number of labels:", len(y))

if len(X) == 0:
    raise ValueError("❌ No valid audio features extracted. Check folder structure or audio preprocessing.")

X = np.stack(X)
y = np.array(y)

print("Dataset shape →", X.shape)
print("\nTotal samples loaded →", len(X))


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert labels → one-hot (VERY IMPORTANT)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test  = tf.keras.utils.to_categorical(y_test,  num_classes=3)


# ==============================
# LIGHTWEIGHT MODEL (FASTER)
# ==============================

print("\nBuilding lightweight model...")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==============================
# TRAIN (FAST)
# ==============================

print("\nTraining (FAST MODE)...")

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ==============================
# SAVE MODEL
# ==============================

Path("model").mkdir(exist_ok=True)
model.save(MODEL_SAVE_PATH)

print("\nModel saved →", MODEL_SAVE_PATH)


# ==============================
# EVALUATION
# ==============================

print("\nEvaluating...")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred_classes,
    labels=[0,1,2],              # Force all 3 classes
    target_names=CLASSES,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes, labels=[0,1,2]))
