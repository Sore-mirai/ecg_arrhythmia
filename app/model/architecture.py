"""
CNN-LSTM Hybrid Model Architecture
====================================
1D-CNN extracts local morphological features from ECG waveforms.
LSTM captures long-term temporal dependencies in cardiac rhythms.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from app.config import WINDOW_SIZE, NUM_CLASSES, LEARNING_RATE


def build_cnn_lstm_model(input_length=WINDOW_SIZE, num_classes=NUM_CLASSES):
    """
    Build a hybrid 1D-CNN + LSTM model for ECG arrhythmia classification.

    Architecture:
        Input → Conv1D blocks (feature extraction) → LSTM (temporal modelling)
        → Dense layers → Softmax output

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_length, 1), name="ecg_input")

    # ── CNN Feature Extraction Block 1 ──
    x = layers.Conv1D(64, kernel_size=7, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # ── CNN Feature Extraction Block 2 ──
    x = layers.Conv1D(128, kernel_size=5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(0.2, name="drop2")(x)

    # ── CNN Feature Extraction Block 3 ──
    x = layers.Conv1D(256, kernel_size=3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # ── LSTM Temporal Modelling ──
    x = layers.LSTM(128, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(0.3, name="drop_lstm1")(x)
    x = layers.LSTM(64, return_sequences=False, name="lstm2")(x)
    x = layers.Dropout(0.3, name="drop_lstm2")(x)

    # ── Classification Head ──
    x = layers.Dense(128, activation="relu", name="fc1",
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ECG_CNN_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model):
    """Return model summary as string."""
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    return "\n".join(summary_lines)


if __name__ == "__main__":
    model = build_cnn_lstm_model()
    model.summary()
