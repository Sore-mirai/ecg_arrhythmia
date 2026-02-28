"""
Training Script for CNN-LSTM Arrhythmia Detection Model
=========================================================
Loads the MIT-BIH Arrhythmia Database, preprocesses signals, trains the
hybrid CNN-LSTM model, and saves it for inference.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from app.config import (
    MODEL_PATH, WINDOW_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS,
    CLASS_NAMES, CLASS_SHORT, MODEL_DIR
)
from app.model.preprocessing import prepare_dataset
from app.model.architecture import build_cnn_lstm_model


def train():
    print("=" * 60)
    print("  ECG Arrhythmia Detection — Model Training")
    print("=" * 60)

    # ── Step 1: Load and preprocess data ──
    print("\n[1/5] Loading and preprocessing MIT-BIH records...")
    X, y = prepare_dataset()

    print(f"\n  Dataset shape: X={X.shape}, y={y.shape}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y == i)
        print(f"    {name}: {count} samples ({100*count/len(y):.1f}%)")

    # ── Step 2: Reshape for model input ──
    X = X.reshape(-1, WINDOW_SIZE, 1).astype(np.float32)

    # ── Step 3: Train/val/test split ──
    print("\n[2/5] Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # ── Step 4: Compute class weights for imbalanced data ──
    print("\n[3/5] Computing class weights...")
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"  Class weights: {class_weight_dict}")

    # ── Step 5: Build and train model ──
    print("\n[4/5] Building CNN-LSTM model...")
    model = build_cnn_lstm_model(input_length=WINDOW_SIZE, num_classes=NUM_CLASSES)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("\n  Training started...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 6: Evaluate on test set ──
    print("\n[5/5] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_SHORT, zero_division=0))

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save training history
    np.save(os.path.join(MODEL_DIR, "training_history.npy"), history.history)

    print(f"\n✓ Model saved to: {MODEL_PATH}")
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
