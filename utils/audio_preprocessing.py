"""
FAST AUDIO PREPROCESSOR (FIXED SHAPE)
Deepfake Voice Detection
Ensures output = (64, 63, 1) for ALL files
"""

# =========================
# IMPORTS
# =========================

import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# =====================================================
# AUDIO PREPROCESSOR CLASS
# =====================================================

class AudioPreprocessor:

    def __init__(self,
                 sample_rate=16000,
                 duration=2,
                 n_mels=64,
                 n_fft=1024,
                 hop_length=256):

        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = sample_rate * duration

    # =====================================================
    # LOAD AUDIO
    # =====================================================

    def load_audio(self, file_path):

        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=self.duration
            )

            return self._fix_length(audio)

        except:
            return np.zeros(self.target_length, dtype=np.float32)

    # =====================================================
    # FIX AUDIO LENGTH
    # =====================================================

    def _fix_length(self, audio):

        if len(audio) < self.target_length:
            return np.pad(audio, (0, self.target_length - len(audio)))

        return audio[:self.target_length]

    # =====================================================
    # MEL SPECTROGRAM (FORCED SHAPE)
    # =====================================================

    def extract_features(self, audio):

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # -------- FORCE WIDTH = 63 --------
        if mel_db.shape[1] < 63:
            pad_width = 63 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))
        else:
            mel_db = mel_db[:, :63]
        # ---------------------------------

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # Add channel dimension → (64, 63, 1)
        mel_db = mel_db[..., np.newaxis]

        return mel_db.astype(np.float32)


# =====================================================
# COMPLETE PIPELINE
# =====================================================

def process_audio_file(file_path, preprocessor):

    audio = preprocessor.load_audio(file_path)
    features = preprocessor.extract_features(audio)

    return features


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    preprocessor = AudioPreprocessor()

    dummy = np.zeros(16000 * 2)
    feat = preprocessor.extract_features(dummy)

    print("AudioPreprocessor READY")
    print("Output shape:", feat.shape)   # Must be (64, 63, 1)
