"""
FAST & LIGHTWEIGHT MODELS
Deepfake Voice Detection
Optimized for Faster Training on GTX 1650 / CPU
"""

# =========================
# IMPORTS
# =========================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC


# =====================================================
# FAST CNN MODEL (RECOMMENDED)
# =====================================================

def build_fast_cnn(input_shape, num_classes=3, dropout_rate=0.25):
    """
    Lightweight CNN → Fast training + Good accuracy
    """

    model = models.Sequential([

        # Conv Block 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten
        layers.Flatten(),

        # Dense Layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),

        # Output
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# =====================================================
# OPTIONAL: SMALL CNN + LSTM (If you want temporal info)
# =====================================================

def build_fast_cnn_lstm(input_shape, num_classes=3, dropout_rate=0.25):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for LSTM
    x = layers.Reshape((-1, x.shape[-1]))(x)

    x = layers.LSTM(64)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model


# =====================================================
# MODEL SELECTOR
# =====================================================

def get_model(model_type="fast_cnn",
              input_shape=(64, 63, 1),
              num_classes=3,
              dropout_rate=0.25):

    if model_type == "fast_cnn":
        return build_fast_cnn(input_shape, num_classes, dropout_rate)

    elif model_type == "fast_cnn_lstm":
        return build_fast_cnn_lstm(input_shape, num_classes, dropout_rate)

    else:
        raise ValueError("Unknown model_type → use 'fast_cnn' or 'fast_cnn_lstm'")


# =====================================================
# COMPILE MODEL
# =====================================================

def compile_model(model, learning_rate=0.001):

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc")
        ]
    )

    return model


# =====================================================
# TEST MODEL (RUN FILE DIRECTLY)
# =====================================================

if __name__ == "__main__":

    print("Building FAST CNN model...")

    model = get_model("fast_cnn", input_shape=(64, 63, 1))
    model = compile_model(model)

    model.summary()

    print("\nTotal Parameters:", model.count_params())
