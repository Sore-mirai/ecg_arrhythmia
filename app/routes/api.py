"""
Flask API Routes
=================
REST API endpoints for ECG arrhythmia detection, streaming, and reporting.
"""

import os
import json
import numpy as np
from flask import Blueprint, request, jsonify, send_file

from app.model.inference import engine
from app.model.preprocessing import (
    load_record, preprocess_signal, normalize_signal,
    sliding_window_segments, get_record_list, add_noise
)
from app.routes.report import generate_pdf_report
from app.config import (
    DATA_DIR, LEAD_INDEX, WINDOW_SIZE, WINDOW_STRIDE,
    SAMPLING_RATE, CLASS_NAMES, CLASS_SHORT, CONFIDENCE_THRESHOLD
)

api = Blueprint("api", __name__, url_prefix="/api")


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@api.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": engine.is_loaded(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    })


# ──────────────────────────────────────────────
# List available records
# ──────────────────────────────────────────────

@api.route("/records", methods=["GET"])
def list_records():
    records = get_record_list(DATA_DIR)
    return jsonify({"records": records})


# ──────────────────────────────────────────────
# Load a full record's signal for display
# ──────────────────────────────────────────────

@api.route("/record/<record_id>", methods=["GET"])
def get_record(record_id):
    try:
        record, annotation = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        raw_signal = signal.tolist()

        # Preprocess
        processed = preprocess_signal(signal)
        processed = normalize_signal(processed)

        # Downsample for display (every 4th sample to keep it manageable)
        step = 4
        time_axis = [i / SAMPLING_RATE for i in range(0, len(processed), step)]

        return jsonify({
            "record_id": record_id,
            "signal": processed[::step].tolist(),
            "time": time_axis,
            "sampling_rate": SAMPLING_RATE,
            "total_samples": len(processed),
            "duration_sec": len(processed) / SAMPLING_RATE,
            "annotations": {
                "samples": annotation.sample.tolist(),
                "symbols": annotation.symbol,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────
# Predict a specific window from a record
# ──────────────────────────────────────────────

@api.route("/predict", methods=["POST"])
def predict():
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.json
    record_id = data.get("record_id", "100")
    start_sample = data.get("start_sample", 0)
    noise_level = data.get("noise_level", 0.0)

    try:
        record, _ = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        signal = preprocess_signal(signal)
        signal = normalize_signal(signal)

        end_sample = start_sample + WINDOW_SIZE
        if end_sample > len(signal):
            start_sample = len(signal) - WINDOW_SIZE
            end_sample = len(signal)

        segment = signal[start_sample:end_sample]
        result = engine.predict_segment(segment, noise_level)
        result["record_id"] = record_id
        result["start_sample"] = start_sample
        result["end_sample"] = end_sample
        result["time_start"] = start_sample / SAMPLING_RATE
        result["time_end"] = end_sample / SAMPLING_RATE

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────
# Simulated real-time streaming inference
# ──────────────────────────────────────────────

@api.route("/stream", methods=["POST"])
def stream_predict():
    """Return predictions for multiple sliding windows (simulated streaming)."""
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    data = request.json
    record_id = data.get("record_id", "100")
    noise_level = data.get("noise_level", 0.0)
    num_windows = data.get("num_windows", 20)
    start_offset = data.get("start_offset", 0)

    try:
        record, _ = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        signal = preprocess_signal(signal)
        signal = normalize_signal(signal)

        segments, starts = sliding_window_segments(
            signal[start_offset:], WINDOW_SIZE, WINDOW_STRIDE
        )

        # Limit to requested number of windows
        segments = segments[:num_windows]
        starts = starts[:num_windows]

        results = []
        for i, (seg, start) in enumerate(zip(segments, starts)):
            result = engine.predict_segment(seg, noise_level)
            result["window_index"] = i
            result["start_sample"] = int(start + start_offset)
            result["end_sample"] = int(start + start_offset + WINDOW_SIZE)
            result["time_start"] = (start + start_offset) / SAMPLING_RATE
            result["time_end"] = (start + start_offset + WINDOW_SIZE) / SAMPLING_RATE
            # Remove heavy data for stream responses (signal/heatmap sent separately)
            result.pop("signal", None)
            result.pop("original_signal", None)
            result.pop("heatmap", None)
            results.append(result)

        return jsonify({
            "record_id": record_id,
            "num_windows": len(results),
            "predictions": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────
# Get normal template for comparison view
# ──────────────────────────────────────────────

@api.route("/normal-template", methods=["GET"])
def normal_template():
    try:
        template = engine.get_normal_template()
        time_axis = [i / SAMPLING_RATE for i in range(len(template))]
        return jsonify({
            "signal": template,
            "time": time_axis,
            "label": "Normal Sinus Rhythm (Reference)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────
# Generate PDF Report
# ──────────────────────────────────────────────

@api.route("/report", methods=["POST"])
def generate_report():
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    data = request.json
    record_id = data.get("record_id", "100")
    start_sample = data.get("start_sample", 0)
    noise_level = data.get("noise_level", 0.0)

    try:
        record, _ = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        signal = preprocess_signal(signal)
        signal = normalize_signal(signal)

        end_sample = start_sample + WINDOW_SIZE
        if end_sample > len(signal):
            start_sample = len(signal) - WINDOW_SIZE
            end_sample = len(signal)

        segment = signal[start_sample:end_sample]
        result = engine.predict_segment(segment, noise_level)
        result["record_id"] = record_id

        pdf_path = generate_pdf_report(result, record_id)

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────
# Update confidence threshold
# ──────────────────────────────────────────────

@api.route("/threshold", methods=["POST"])
def update_threshold():
    import app.config as cfg
    data = request.json
    new_threshold = data.get("threshold", 0.60)
    cfg.CONFIDENCE_THRESHOLD = float(new_threshold)
    return jsonify({"threshold": cfg.CONFIDENCE_THRESHOLD})
