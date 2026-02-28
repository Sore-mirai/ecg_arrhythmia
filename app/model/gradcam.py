"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for 1D ECG Signals
==========================================================================
Highlights salient regions of the ECG waveform that contribute most to the
model's arrhythmia prediction, providing clinically meaningful explanations.
"""

import numpy as np
import tensorflow as tf


def compute_gradcam(model, input_signal, predicted_class=None, layer_name="conv3"):
    """
    Compute Grad-CAM heatmap for a 1D ECG signal.

    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN-LSTM model.
    input_signal : np.ndarray
        Single ECG segment of shape (window_size,) or (1, window_size, 1).
    predicted_class : int or None
        Class index to explain. If None, uses the predicted class.
    layer_name : str
        Name of the convolutional layer to use for Grad-CAM.

    Returns
    -------
    heatmap : np.ndarray
        1D heatmap of shape (window_size,) with values in [0, 1].
    """
    # Prepare input
    if input_signal.ndim == 1:
        input_tensor = tf.convert_to_tensor(
            input_signal.reshape(1, -1, 1), dtype=tf.float32
        )
    elif input_signal.ndim == 2:
        input_tensor = tf.convert_to_tensor(
            input_signal.reshape(1, input_signal.shape[0], 1), dtype=tf.float32
        )
    else:
        input_tensor = tf.convert_to_tensor(input_signal, dtype=tf.float32)

    # Build sub-model that outputs conv layer activations + final predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    # Forward pass with gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        if predicted_class is None:
            predicted_class = tf.argmax(predictions[0]).numpy()
        class_output = predictions[:, predicted_class]

    # Compute gradients of class output w.r.t. conv layer
    grads = tape.gradient(class_output, conv_outputs)

    # Global average pooling of gradients → channel weights
    weights = tf.reduce_mean(grads, axis=1)  # shape: (1, num_filters)

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]  # (time_steps, num_filters)
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)  # (time_steps,)

    # ReLU — only positive contributions
    cam = tf.nn.relu(cam).numpy()

    # Interpolate to original signal length
    original_length = input_signal.shape[0] if input_signal.ndim == 1 else input_signal.shape[1]
    cam_resized = np.interp(
        np.linspace(0, len(cam) - 1, original_length),
        np.arange(len(cam)),
        cam,
    )

    # Normalize to [0, 1]
    cam_max = cam_resized.max()
    if cam_max > 0:
        cam_resized = cam_resized / cam_max

    return cam_resized


def get_gradcam_explanation(heatmap, class_name, threshold=0.5):
    """
    Generate a detailed, layperson-friendly explanation of Grad-CAM regions.

    Parameters
    ----------
    heatmap : np.ndarray
        1D Grad-CAM heatmap (values 0–1).
    class_name : str
        Predicted class name.
    threshold : float
        Attention threshold for "high contribution" regions.

    Returns
    -------
    explanation : dict
        Contains 'summary', 'regions', and 'clinical_note'.
    """
    sampling_rate = 360
    window_size = len(heatmap)
    high_attention = np.where(heatmap >= threshold)[0]

    # ── ECG waveform component mapping (approximate for a 2-second, 720-sample window) ──
    # A typical heartbeat cycle at 360 Hz is ~250–400 samples (0.7–1.1s).
    # We estimate based on typical positions within a beat-centered window.
    def classify_ecg_region(sample_idx):
        """Map a sample position to an approximate ECG waveform component."""
        # Normalize position to fraction of window
        frac = sample_idx / window_size
        if frac < 0.15:
            return "P-wave region", "the electrical impulse that triggers atrial contraction (upper chambers of the heart)"
        elif frac < 0.25:
            return "PR interval", "the delay between atrial and ventricular activation, reflecting conduction through the AV node"
        elif frac < 0.45:
            return "QRS complex", "the main electrical impulse that triggers ventricular contraction (the heart’s main pumping chambers)"
        elif frac < 0.55:
            return "ST segment", "the period between ventricular contraction and recovery — abnormalities here can indicate ischemia or injury"
        elif frac < 0.75:
            return "T-wave region", "the electrical recovery (repolarization) of the ventricles after each heartbeat"
        else:
            return "U-wave / baseline", "the resting phase of the heart between beats"

    # ── Class-specific clinical context ──
    class_context = {
        "Normal (N)": {
            "what_it_means": "The heart is beating in a normal, regular rhythm originating from the sinus node (the natural pacemaker).",
            "what_model_looks_for": "regular R-R intervals, a normal upright P-wave before each QRS, and a narrow QRS complex (~60–100 ms)",
            "risk_level": "None — this is a healthy rhythm.",
        },
        "Supraventricular (S)": {
            "what_it_means": "An abnormal beat originating above the ventricles (in the atria or AV junction). This includes premature atrial contractions (PACs) and junctional beats.",
            "what_model_looks_for": "an unusually shaped or absent P-wave, a premature beat arriving earlier than expected, or an inverted P-wave",
            "risk_level": "Usually benign but frequent episodes may require monitoring. Can feel like a skipped beat or flutter.",
        },
        "Ventricular (V)": {
            "what_it_means": "An abnormal beat originating from the ventricles (lower chambers). This includes premature ventricular contractions (PVCs) — the ventricle fires before the normal signal arrives.",
            "what_model_looks_for": "a wide, bizarre-looking QRS complex (>120 ms) without a preceding P-wave, and an abnormal T-wave in the opposite direction",
            "risk_level": "Occasional PVCs are common, but frequent or repetitive patterns require clinical evaluation as they may indicate heart disease.",
        },
        "Fusion (F)": {
            "what_it_means": "A ‘hybrid’ beat where a normal sinus impulse and a ventricular impulse collide and merge, producing a beat with mixed characteristics.",
            "what_model_looks_for": "a QRS shape that is intermediate between normal and ventricular — not as wide as a PVC but not as narrow as a normal beat",
            "risk_level": "May indicate co-existing ventricular ectopy. Warrants review especially if frequent.",
        },
        "Unknown/Paced (Q)": {
            "what_it_means": "A beat from an artificial pacemaker, or a beat that could not be confidently classified. Pacemaker spikes are electrical pulses from an implanted device.",
            "what_model_looks_for": "a sharp pacemaker spike artifact before the QRS, or an unusual morphology that doesn’t fit other categories",
            "risk_level": "Paced beats are expected in patients with pacemakers. Unclassifiable beats should be reviewed by a clinician.",
        },
    }

    context = class_context.get(class_name, {
        "what_it_means": f"This is classified as {class_name}.",
        "what_model_looks_for": "characteristic waveform features for this class",
        "risk_level": "Consult a healthcare professional for interpretation.",
    })

    if len(high_attention) == 0:
        return {
            "summary": f"The model shows diffuse, spread-out attention across the entire ECG segment for its \u2018{class_name}\u2019 prediction. This means no single region dominated — the overall shape of the waveform was used.",
            "regions": [],
            "clinical_note": context,
        }

    # ── Find contiguous attention regions ──
    regions = []
    start = high_attention[0]
    for i in range(1, len(high_attention)):
        if high_attention[i] - high_attention[i - 1] > 5:
            regions.append((start, high_attention[i - 1]))
            start = high_attention[i]
    regions.append((start, high_attention[-1]))

    # ── Build rich region descriptions ──
    region_details = []
    for s, e in regions:
        mid = (s + e) // 2
        component, component_desc = classify_ecg_region(mid)
        time_start = s / sampling_rate
        time_end = e / sampling_rate
        peak_attention = float(np.max(heatmap[s:e + 1]))
        region_details.append({
            "samples": f"{s}–{e}",
            "time": f"{time_start:.2f}–{time_end:.2f}s",
            "component": component,
            "component_description": component_desc,
            "peak_attention": round(peak_attention * 100, 1),
        })

    # ── Build summary sentence ──
    focused_components = list(set(r["component"] for r in region_details))
    summary = (
        f"The model focused most on the <strong>{', '.join(focused_components)}</strong> "
        f"of the ECG waveform to reach its \u2018{class_name}\u2019 prediction. "
    )

    if class_name != "Normal (N)":
        summary += (
            f"This means the AI detected something unusual in {'this area' if len(focused_components) == 1 else 'these areas'} "
            f"compared to what a normal heartbeat looks like. "
            f"Specifically, it\'s looking for: {context['what_model_looks_for']}."
        )
    else:
        summary += (
            f"The AI confirmed normal features in {'this area' if len(focused_components) == 1 else 'these areas'}: "
            f"{context['what_model_looks_for']}."
        )

    return {
        "summary": summary,
        "regions": region_details,
        "clinical_note": context,
    }
