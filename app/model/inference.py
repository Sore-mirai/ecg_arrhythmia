"""
Inference Engine
=================
Loads trained model and provides prediction + Grad-CAM for ECG segments.
"""

import numpy as np
import tensorflow as tf
import os

from app.config import (
    MODEL_PATH, WINDOW_SIZE, NUM_CLASSES, CLASS_NAMES, CLASS_SHORT,
    CONFIDENCE_THRESHOLD
)
from app.model.architecture import build_cnn_lstm_model
from app.model.gradcam import compute_gradcam, get_gradcam_explanation
from app.model.preprocessing import (
    preprocess_signal, normalize_signal, add_noise,
    sliding_window_segments, load_record, get_normal_template
)


class ECGInferenceEngine:
    """Handles model loading and ECG arrhythmia prediction."""

    def __init__(self):
        self.model = None
        self.normal_template = None
        self._loaded = False

    def load(self):
        """Load trained model from disk."""
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self._loaded = True
            print(f"[INFO] Model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] No trained model found at {MODEL_PATH}. Train the model first.")
            self._loaded = False

    def is_loaded(self):
        return self._loaded

    def predict_segment(self, segment, noise_level=0.0):
        """
        Predict arrhythmia class for a single ECG segment.

        Parameters
        ----------
        segment : np.ndarray
            ECG segment of shape (window_size,).
        noise_level : float
            Gaussian noise level to inject (0.0 = no noise).

        Returns
        -------
        result : dict
            Prediction result including class, confidence, heatmap, alert level.
        """
        if not self._loaded:
            return {"error": "Model not loaded"}

        # Optionally inject noise
        original_segment = segment.copy()
        if noise_level > 0:
            segment = add_noise(segment, noise_level)

        # Prepare input
        x = segment.reshape(1, -1, 1).astype(np.float32)

        # Predict
        probs = self.model.predict(x, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Compute Grad-CAM
        heatmap = compute_gradcam(self.model, segment, pred_class)

        # Determine alert level
        if confidence < CONFIDENCE_THRESHOLD:
            alert_level = "uncertain"
            alert_message = "⚠ Uncertain Prediction – Human Review Recommended"
        elif pred_class == 0:
            alert_level = "normal"
            alert_message = "✓ Normal Sinus Rhythm Detected"
        else:
            alert_level = "arrhythmia"
            alert_message = f"⚡ Arrhythmia Detected: {CLASS_NAMES[pred_class]}"

        # Generate explanation
        explanation = get_gradcam_explanation(heatmap, CLASS_NAMES[pred_class])

        return {
            "predicted_class": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "class_short": CLASS_SHORT[pred_class],
            "confidence": confidence,
            "probabilities": {CLASS_SHORT[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            "heatmap": heatmap.tolist(),
            "signal": segment.tolist(),
            "original_signal": original_segment.tolist(),
            "alert_level": alert_level,
            "alert_message": alert_message,
            "explanation": explanation,
            "noise_level": noise_level,
        }

    def predict_record_stream(self, record_id, noise_level=0.0, window_size=WINDOW_SIZE, stride=180):
        """
        Simulate real-time streaming inference on a full record.
        Yields predictions for each sliding window.
        """
        if not self._loaded:
            yield {"error": "Model not loaded"}
            return

        from app.config import DATA_DIR, LEAD_INDEX
        record, annotation = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        signal = preprocess_signal(signal)
        signal = normalize_signal(signal)

        segments, starts = sliding_window_segments(signal, window_size, stride)

        results = []
        for i, (seg, start) in enumerate(zip(segments, starts)):
            result = self.predict_segment(seg, noise_level)
            result["window_index"] = i
            result["start_sample"] = int(start)
            result["end_sample"] = int(start + window_size)
            results.append(result)

        return results

    def get_normal_template(self):
        """Get a normal ECG beat template for comparison view."""
        if self.normal_template is None:
            from app.config import DATA_DIR
            self.normal_template = get_normal_template(DATA_DIR).tolist()
        return self.normal_template


# Global engine instance
engine = ECGInferenceEngine()
