"""
SIMPLIFIED + DEBUG DATA LOADER
Deepfake Voice Detection
"""

# =========================
# IMPORTS
# =========================

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pickle


# =========================
# DATASET LOADER CLASS
# =========================

class DatasetLoader:

    def __init__(self, data_dir: str, sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate

        self.data_info = {
            'file_paths': [],
            'labels': []
        }

    # =====================================================
    # LOAD CUSTOM DATASET
    # =====================================================

    def load_custom_dataset(self, dataset_path: str, label_mapping: Dict = None) -> Dict:

        if label_mapping is None:
            label_mapping = {
                'human_voice': 0,
                'automated_trusted': 1,
                'deepfake': 2
            }

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"\n❌ Dataset path NOT FOUND → {dataset_path.resolve()}")

        file_paths = []
        labels = []

        print("\n📂 Scanning dataset...")

        for label_name, label_id in label_mapping.items():

            label_dir = dataset_path / label_name

            if not label_dir.exists():
                print(f"⚠ Missing folder: {label_dir}")
                continue

            audio_files = (
                list(label_dir.glob("*.wav")) +
                list(label_dir.glob("*.flac")) +
                list(label_dir.glob("*.mp3"))
            )

            print(f"{label_name} → {len(audio_files)} files")

            for audio_file in audio_files:
                file_paths.append(str(audio_file))
                labels.append(label_id)

        if len(file_paths) == 0:
            raise ValueError(
                "\n❌ NO AUDIO FILES FOUND!\n"
                "Check:\n"
                "1. Folder structure\n"
                "2. File format (.wav/.flac/.mp3)\n"
                "3. Correct dataset path\n"
            )

        print(f"\n✅ Total files loaded: {len(file_paths)}")

        return {'file_paths': file_paths, 'labels': labels}

    # =====================================================
    # COMBINE DATASETS
    # =====================================================

    def combine_datasets(self, datasets: List[Dict]) -> None:

        for dataset in datasets:
            self.data_info['file_paths'].extend(dataset.get('file_paths', []))
            self.data_info['labels'].extend(dataset.get('labels', []))

        print(f"\n📊 Combined dataset size: {len(self.data_info['file_paths'])}")

    # =====================================================
    # DEBUG DATASET
    # =====================================================

    def debug_dataset(self):

        print("\n====== DATASET DEBUG ======")
        print("Total files:", len(self.data_info['file_paths']))

        if len(self.data_info['file_paths']) > 0:
            print("Sample file:", self.data_info['file_paths'][0])
            print("Sample label:", self.data_info['labels'][0])
        print("===========================\n")

    # =====================================================
    # TRAIN / VAL / TEST SPLIT
    # =====================================================

    def get_train_val_test_split(self,
                                 train_size=0.7,
                                 val_size=0.15,
                                 test_size=0.15,
                                 random_state=42) -> Tuple:

        X = np.array(self.data_info['file_paths'])
        y = np.array(self.data_info['labels'])

        if len(X) == 0:
            raise ValueError("❌ Dataset EMPTY — No files loaded.")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        val_ratio = val_size / (train_size + val_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )

        print("\n📦 Data Split:")
        print("Train:", len(X_train))
        print("Val:", len(X_val))
        print("Test:", len(X_test))

        return X_train, X_val, X_test, y_train, y_val, y_test

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    def compute_class_weights(self) -> Dict:

        labels = np.array(self.data_info['labels'])
        classes = np.unique(labels)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )

        return {int(c): float(w) for c, w in zip(classes, weights)}

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    def get_dataset_statistics(self):

        labels = np.array(self.data_info['labels'])
        unique, counts = np.unique(labels, return_counts=True)

        class_names = {
            0: "Human Voice",
            1: "Automated Trusted",
            2: "Deepfake"
        }

        print("\n📊 Dataset Statistics")

        total = len(labels)

        for u, c in zip(unique, counts):
            print(f"{class_names[u]} → {c} ({(c/total)*100:.2f}%)")

        print("Total samples:", total)

    # =====================================================
    # SAVE / LOAD DATASET INFO
    # =====================================================

    def save_dataset_info(self, path="dataset_info.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.data_info, f)
        print("Dataset info saved.")

    def load_dataset_info(self, path="dataset_info.pkl"):
        with open(path, "rb") as f:
            self.data_info = pickle.load(f)
        print("Dataset info loaded.")


# =====================================================
# QUICK TEST (RUN THIS FILE DIRECTLY)
# =====================================================

if __name__ == "__main__":

    loader = DatasetLoader(data_dir="data")

    dataset = loader.load_custom_dataset("data/my_dataset")
    loader.combine_datasets([dataset])

    loader.debug_dataset()
    loader.get_dataset_statistics()
