"""
ECG Signal Preprocessing Module
================================
Handles loading, filtering, normalization, and segmentation of MIT-BIH
Arrhythmia Database records using the WFDB toolkit.
"""

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import median_filter
import os
import pickle

from app.config import (
    DATA_DIR, SAMPLING_RATE, WINDOW_SIZE, WINDOW_STRIDE,
    LEAD_INDEX, SYMBOL_TO_AAMI, AAMI_CLASSES, SCALER_PATH
)


# ──────────────────────────────────────────────
# Signal Filtering
# ──────────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=SAMPLING_RATE, order=4):
    """Apply Butterworth bandpass filter for noise removal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def remove_baseline_wander(signal, fs=SAMPLING_RATE):
    """
    Remove baseline wander using two-stage median filtering.
    First median filter with 200ms window, second with 600ms window.
    """
    # Convert ms to samples (must be odd)
    win1 = int(0.2 * fs)
    win1 = win1 if win1 % 2 == 1 else win1 + 1
    win2 = int(0.6 * fs)
    win2 = win2 if win2 % 2 == 1 else win2 + 1

    baseline1 = medfilt(signal, kernel_size=win1)
    baseline2 = medfilt(baseline1, kernel_size=win2)
    return signal - baseline2


def preprocess_signal(signal, fs=SAMPLING_RATE):
    """Full preprocessing pipeline: baseline removal + bandpass filter + normalization."""
    # Step 1: Remove baseline wander
    signal = remove_baseline_wander(signal, fs)
    # Step 2: Bandpass filter (0.5 – 45 Hz)
    signal = bandpass_filter(signal, fs=fs)
    return signal


def normalize_signal(signal):
    """Z-score normalization."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std


def add_noise(signal, noise_level=0.05):
    """Inject Gaussian noise at a specified level (fraction of signal std)."""
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    return signal + noise


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_record(record_id, data_dir=DATA_DIR):
    """Load a single WFDB record and its annotations."""
    record_path = os.path.join(data_dir, str(record_id))
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")
    return record, annotation


def get_record_list(data_dir=DATA_DIR):
    """Read list of available records."""
    records_file = os.path.join(data_dir, "RECORDS")
    with open(records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]
    return records


# ──────────────────────────────────────────────
# Beat Segmentation (for training)
# ──────────────────────────────────────────────

def segment_beats(signal, annotation, window_size=WINDOW_SIZE):
    """
    Segment ECG signal into fixed-length windows centered around R-peaks.
    Returns arrays of segments and their AAMI labels.
    """
    segments = []
    labels = []
    half_win = window_size // 2

    for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        # Skip non-beat annotations
        if symbol not in SYMBOL_TO_AAMI:
            continue

        aami_class = SYMBOL_TO_AAMI[symbol]
        label = AAMI_CLASSES[aami_class]

        start = sample - half_win
        end = sample + half_win

        if start < 0 or end > len(signal):
            continue

        segment = signal[start:end]
        segments.append(segment)
        labels.append(label)

    return np.array(segments), np.array(labels)


# ──────────────────────────────────────────────
# Sliding Window Segmentation (for real-time inference)
# ──────────────────────────────────────────────

def sliding_window_segments(signal, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    Create overlapping sliding windows from a continuous signal.
    Used for simulated real-time streaming inference.
    """
    segments = []
    starts = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segments.append(signal[start:start + window_size])
        starts.append(start)
    return np.array(segments), np.array(starts)


# ──────────────────────────────────────────────
# Full Dataset Preparation
# ──────────────────────────────────────────────

def prepare_dataset(record_ids=None, data_dir=DATA_DIR, window_size=WINDOW_SIZE):
    """
    Load and preprocess all records, segment beats, return X and y arrays.
    """
    if record_ids is None:
        record_ids = get_record_list(data_dir)

    all_segments = []
    all_labels = []

    for rid in record_ids:
        try:
            record, annotation = load_record(rid, data_dir)
            # Use MLII lead (channel 0)
            signal = record.p_signal[:, LEAD_INDEX]
            # Preprocess
            signal = preprocess_signal(signal)
            signal = normalize_signal(signal)
            # Segment beats
            segments, labels = segment_beats(signal, annotation, window_size)
            all_segments.append(segments)
            all_labels.append(labels)
            print(f"  Record {rid}: {len(segments)} beats extracted")
        except Exception as e:
            print(f"  Record {rid}: SKIPPED ({e})")

    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\nTotal: {len(X)} beats | Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def get_normal_template(data_dir=DATA_DIR, window_size=WINDOW_SIZE):
    """
    Extract a clean 'Normal' beat template for the comparison view.
    Uses Record 100 which has predominantly normal beats.
    """
    record, annotation = load_record("100", data_dir)
    signal = record.p_signal[:, LEAD_INDEX]
    signal = preprocess_signal(signal)
    signal = normalize_signal(signal)

    half_win = window_size // 2
    # Find first normal beat with enough padding
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol == "N" and sample - half_win >= 0 and sample + half_win <= len(signal):
            return signal[sample - half_win: sample + half_win]

    return np.zeros(window_size)
