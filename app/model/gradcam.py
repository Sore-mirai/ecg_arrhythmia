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
    Generate a human-readable explanation of Grad-CAM regions.

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
    explanation : str
    """
    high_attention = np.where(heatmap >= threshold)[0]

    if len(high_attention) == 0:
        return f"The model shows diffuse attention across the ECG segment for '{class_name}' prediction."

    # Find contiguous regions
    regions = []
    start = high_attention[0]
    for i in range(1, len(high_attention)):
        if high_attention[i] - high_attention[i - 1] > 5:
            regions.append((start, high_attention[i - 1]))
            start = high_attention[i]
    regions.append((start, high_attention[-1]))

    region_strs = [f"samples {s}–{e}" for s, e in regions]
    return (
        f"Red/warm regions indicate high contribution to the '{class_name}' prediction. "
        f"Key attention zones: {', '.join(region_strs)}. "
        f"These regions likely correspond to morphological features characteristic of {class_name}."
    )
