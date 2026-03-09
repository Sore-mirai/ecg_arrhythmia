"""
Flask API Routes
=================
REST API endpoints for ECG arrhythmia detection, streaming, file upload, and reporting.
"""

import os
import time
import uuid
import tempfile
import numpy as np
import wfdb
from flask import Blueprint, request, jsonify, send_file

from app.model.inference import engine
from app.model.preprocessing import (
    load_record, preprocess_signal, normalize_signal,
    sliding_window_segments, get_record_list
)
from app.routes.report import generate_pdf_report
from app.config import (
    DATA_DIR, LEAD_INDEX, WINDOW_SIZE, WINDOW_STRIDE,
    SAMPLING_RATE, CLASS_NAMES, CONFIDENCE_THRESHOLD
)

api = Blueprint("api", __name__, url_prefix="/api")

# In-memory uploaded ECG cache.
# NOTE: This is process-local and reset on server restart.
UPLOADED_SIGNALS = {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _extract_single_lead(signal_array, lead_index=0):
    """Return a 1D lead from uploaded signal (supports 1D or 2D input)."""
    arr = np.asarray(signal_array, dtype=np.float32)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        # Handle both (samples, leads) and (leads, samples).
        if arr.shape[0] >= arr.shape[1]:
            idx = min(max(lead_index, 0), arr.shape[1] - 1)
            return arr[:, idx]
        idx = min(max(lead_index, 0), arr.shape[0] - 1)
        return arr[idx, :]

    raise ValueError("Unsupported ECG shape. Expected 1D or 2D data.")


def _load_signal_from_csv(file_storage, lead_index=0):
    """Load ECG from CSV/TXT (single or multi-column numeric file)."""
    raw = file_storage.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")

    # Decode with UTF-8 fallback to latin-1 to tolerate local exports.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    # Try comma first, then whitespace-delimited fallback.
    try:
        data = np.genfromtxt(text.splitlines(), delimiter=",", dtype=np.float32)
    except Exception:
        data = np.genfromtxt(text.splitlines(), dtype=np.float32)

    if data.size == 0 or np.all(np.isnan(data)):
        raise ValueError("Could not parse numeric ECG values from file.")

    if data.ndim == 0:
        data = np.array([float(data)], dtype=np.float32)

    # Clean NaNs/infs that can appear in partially malformed rows.
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return _extract_single_lead(data, lead_index)


def _load_signal_from_wfdb(files, lead_index=0):
    """
    Load WFDB record from uploaded files.
    Requires at least matching .dat and .hea files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        basenames = {}
        for f in files:
            filename = os.path.basename(f.filename)
            if not filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in {".dat", ".hea", ".atr"}:
                continue
            file_path = os.path.join(tmpdir, filename)
            f.save(file_path)
            base = os.path.splitext(filename)[0]
            basenames.setdefault(base, set()).add(ext)

        valid_bases = [b for b, exts in basenames.items() if ".dat" in exts and ".hea" in exts]
        if not valid_bases:
            raise ValueError("WFDB upload requires matching .dat and .hea files.")

        base = valid_bases[0]
        record = wfdb.rdrecord(os.path.join(tmpdir, base))
        signal = record.p_signal
        return _extract_single_lead(signal, lead_index), base


def _get_uploaded_session(session_id):
    session = UPLOADED_SIGNALS.get(session_id)
    if session is None:
        raise ValueError("Upload session expired or invalid. Please upload ECG again.")
    return session


def _predict_from_signal(signal, start_sample, noise_level, record_id, sampling_rate=SAMPLING_RATE):
    end_sample = start_sample + WINDOW_SIZE
    if len(signal) < WINDOW_SIZE:
        raise ValueError(f"Signal too short ({len(signal)} samples). Need at least {WINDOW_SIZE}.")

    if end_sample > len(signal):
        start_sample = len(signal) - WINDOW_SIZE
        end_sample = len(signal)

    segment = signal[start_sample:end_sample]
    result = engine.predict_segment(segment, noise_level)
    result["record_id"] = record_id
    result["start_sample"] = int(start_sample)
    result["end_sample"] = int(end_sample)
    result["time_start"] = float(start_sample / sampling_rate)
    result["time_end"] = float(end_sample / sampling_rate)
    return result


def _stream_from_signal(signal, noise_level, num_windows, start_offset, record_id, sampling_rate=SAMPLING_RATE):
    if len(signal) < WINDOW_SIZE:
        raise ValueError(f"Signal too short ({len(signal)} samples). Need at least {WINDOW_SIZE}.")

    segments, starts = sliding_window_segments(signal[start_offset:], WINDOW_SIZE, WINDOW_STRIDE)
    segments = segments[:num_windows]
    starts = starts[:num_windows]

    results = []
    for i, (seg, start) in enumerate(zip(segments, starts)):
        result = engine.predict_segment(seg, noise_level)
        result["window_index"] = int(i)
        result["start_sample"] = int(start + start_offset)
        result["end_sample"] = int(start + start_offset + WINDOW_SIZE)
        result["time_start"] = float((start + start_offset) / sampling_rate)
        result["time_end"] = float((start + start_offset + WINDOW_SIZE) / sampling_rate)
        result.pop("signal", None)
        result.pop("original_signal", None)
        result.pop("heatmap", None)
        results.append(result)

    return {
        "record_id": record_id,
        "num_windows": len(results),
        "predictions": results,
    }


@api.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": engine.is_loaded(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    })


@api.route("/records", methods=["GET"])
def list_records():
    records = get_record_list(DATA_DIR)
    return jsonify({"records": records})


@api.route("/record/<record_id>", methods=["GET"])
def get_record(record_id):
    try:
        record, annotation = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]

        processed = preprocess_signal(signal)
        processed = normalize_signal(processed)

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


@api.route("/upload-signal", methods=["POST"])
def upload_signal():
    """
    Upload ECG data and cache preprocessed signal for prediction/streaming.

    Supported:
    - CSV/TXT numeric file (single or multi-column)
    - WFDB pair (.dat + .hea, optional .atr)
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    lead_index = _safe_int(request.form.get("lead_index", LEAD_INDEX), LEAD_INDEX)
    sampling_rate = _safe_float(request.form.get("sampling_rate", SAMPLING_RATE), SAMPLING_RATE)

    try:
        names = [f.filename for f in files if f and f.filename]
        lower_names = [n.lower() for n in names]

        if any(n.endswith(".csv") or n.endswith(".txt") for n in lower_names):
            csv_file = next(f for f in files if f.filename and f.filename.lower().endswith((".csv", ".txt")))
            raw_signal = _load_signal_from_csv(csv_file, lead_index=lead_index)
            source_name = os.path.basename(csv_file.filename)
        elif any(n.endswith(".dat") for n in lower_names):
            raw_signal, wfdb_base = _load_signal_from_wfdb(files, lead_index=lead_index)
            source_name = f"{wfdb_base}.dat"
        else:
            return jsonify({"error": "Unsupported file format. Upload CSV/TXT or WFDB .dat + .hea files."}), 400

        processed = preprocess_signal(raw_signal, fs=sampling_rate)
        processed = normalize_signal(processed).astype(np.float32)

        if len(processed) < WINDOW_SIZE:
            return jsonify({"error": f"Signal too short ({len(processed)} samples). Need at least {WINDOW_SIZE}."}), 400

        session_id = uuid.uuid4().hex
        UPLOADED_SIGNALS[session_id] = {
            "signal": processed,
            "sampling_rate": float(sampling_rate),
            "source_name": source_name,
            "uploaded_at": time.time(),
        }

        return jsonify({
            "upload_session_id": session_id,
            "source_name": source_name,
            "sampling_rate": float(sampling_rate),
            "total_samples": int(len(processed)),
            "duration_sec": float(len(processed) / sampling_rate),
            "message": "ECG uploaded and preprocessed successfully.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api.route("/predict", methods=["POST"])
def predict():
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.json or {}
    record_id = data.get("record_id", "100")
    start_sample = _safe_int(data.get("start_sample", 0), 0)
    noise_level = _safe_float(data.get("noise_level", 0.0), 0.0)

    try:
        record, _ = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        signal = preprocess_signal(signal)
        signal = normalize_signal(signal)

        result = _predict_from_signal(
            signal=signal,
            start_sample=start_sample,
            noise_level=noise_level,
            record_id=record_id,
            sampling_rate=SAMPLING_RATE,
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api.route("/upload/predict", methods=["POST"])
def predict_uploaded():
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.json or {}
    session_id = data.get("upload_session_id")
    start_sample = _safe_int(data.get("start_sample", 0), 0)
    noise_level = _safe_float(data.get("noise_level", 0.0), 0.0)

    try:
        if not session_id:
            return jsonify({"error": "upload_session_id is required."}), 400

        session = _get_uploaded_session(session_id)
        signal = session["signal"]
        sampling_rate = session["sampling_rate"]
        source_name = session["source_name"]

        result = _predict_from_signal(
            signal=signal,
            start_sample=start_sample,
            noise_level=noise_level,
            record_id=f"uploaded:{source_name}",
            sampling_rate=sampling_rate,
        )
        result["upload_session_id"] = session_id
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api.route("/stream", methods=["POST"])
def stream_predict():
    """Return predictions for multiple sliding windows (simulated streaming)."""
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    data = request.json or {}
    record_id = data.get("record_id", "100")
    noise_level = _safe_float(data.get("noise_level", 0.0), 0.0)
    num_windows = _safe_int(data.get("num_windows", 20), 20)
    start_offset = _safe_int(data.get("start_offset", 0), 0)

    try:
        record, _ = load_record(record_id, DATA_DIR)
        signal = record.p_signal[:, LEAD_INDEX]
        signal = preprocess_signal(signal)
        signal = normalize_signal(signal)

        response = _stream_from_signal(
            signal=signal,
            noise_level=noise_level,
            num_windows=num_windows,
            start_offset=start_offset,
            record_id=record_id,
            sampling_rate=SAMPLING_RATE,
        )
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api.route("/upload/stream", methods=["POST"])
def stream_predict_uploaded():
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    data = request.json or {}
    session_id = data.get("upload_session_id")
    noise_level = _safe_float(data.get("noise_level", 0.0), 0.0)
    num_windows = _safe_int(data.get("num_windows", 20), 20)
    start_offset = _safe_int(data.get("start_offset", 0), 0)

    try:
        if not session_id:
            return jsonify({"error": "upload_session_id is required."}), 400

        session = _get_uploaded_session(session_id)
        signal = session["signal"]
        sampling_rate = session["sampling_rate"]
        source_name = session["source_name"]

        response = _stream_from_signal(
            signal=signal,
            noise_level=noise_level,
            num_windows=num_windows,
            start_offset=start_offset,
            record_id=f"uploaded:{source_name}",
            sampling_rate=sampling_rate,
        )
        response["upload_session_id"] = session_id
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


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



@api.route("/upload/report", methods=["POST"])
def generate_upload_report():
    if not engine.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    data = request.json or {}
    session_id = data.get("upload_session_id")
    start_sample = _safe_int(data.get("start_sample", 0), 0)
    noise_level = _safe_float(data.get("noise_level", 0.0), 0.0)

    try:
        if not session_id:
            return jsonify({"error": "upload_session_id is required."}), 400

        session = _get_uploaded_session(session_id)
        signal = session["signal"]
        sampling_rate = session["sampling_rate"]
        source_name = session["source_name"]

        display_record_id = f"uploaded:{source_name}"
        result = _predict_from_signal(
            signal=signal,
            start_sample=start_sample,
            noise_level=noise_level,
            record_id=display_record_id,
            sampling_rate=sampling_rate,
        )

        pdf_path = generate_pdf_report(result, f"uploaded_{source_name}")

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api.route("/threshold", methods=["POST"])
def update_threshold():
    import app.config as cfg
    data = request.json
    new_threshold = data.get("threshold", 0.60)
    cfg.CONFIDENCE_THRESHOLD = float(new_threshold)
    return jsonify({"threshold": cfg.CONFIDENCE_THRESHOLD})
