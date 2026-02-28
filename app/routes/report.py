"""
PDF Report Generator
=====================
Generates professional clinical-style PDF reports for ECG arrhythmia detections.
"""

import io
import os
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from app.config import REPORT_DIR, SAMPLING_RATE, CLASS_NAMES


# Color scheme
PRIMARY = HexColor("#1a237e")
ACCENT = HexColor("#c62828")
SUCCESS = HexColor("#2e7d32")
WARNING = HexColor("#f57f17")
LIGHT_BG = HexColor("#f5f5f5")


def _create_ecg_plot(signal, heatmap, title, sampling_rate=SAMPLING_RATE):
    """Generate ECG plot with Grad-CAM overlay as bytes."""
    time_axis = np.arange(len(signal)) / sampling_rate

    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=150)
    fig.patch.set_facecolor("white")

    # Plot ECG signal
    ax.plot(time_axis, signal, color="#1a237e", linewidth=0.8, label="ECG Signal")

    # Overlay Grad-CAM heatmap
    if heatmap is not None:
        heatmap_arr = np.array(heatmap)
        ax.fill_between(
            time_axis, min(signal), max(signal),
            where=heatmap_arr > 0.3,
            alpha=0.3, color="red", label="High Attention (Grad-CAM)"
        )
        # Color-coded scatter for high attention
        high_idx = heatmap_arr > 0.5
        if np.any(high_idx):
            ax.scatter(
                time_axis[high_idx], np.array(signal)[high_idx],
                c=heatmap_arr[high_idx], cmap="YlOrRd", s=2, zorder=5
            )

    ax.set_xlabel("Time (seconds)", fontsize=8)
    ax.set_ylabel("Amplitude (mV)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1a237e")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, color="#e0e0e0")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(prediction_result, record_id="Unknown"):
    """
    Generate a professional PDF report for an ECG arrhythmia detection.

    Parameters
    ----------
    prediction_result : dict
        Output from ECGInferenceEngine.predict_segment().
    record_id : str
        Patient/record identifier.

    Returns
    -------
    pdf_path : str
        Path to the generated PDF file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ECG_Report_{record_id}_{timestamp}.pdf"
    pdf_path = os.path.join(REPORT_DIR, filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=22, textColor=PRIMARY, spaceAfter=5,
        fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#666666"), spaceAfter=15,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=13, textColor=PRIMARY, spaceBefore=15, spaceAfter=8,
        fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        "BodyText", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6
    )
    alert_normal = ParagraphStyle(
        "AlertNormal", parent=styles["Normal"],
        fontSize=12, textColor=SUCCESS, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceBefore=5, spaceAfter=5
    )
    alert_arrhythmia = ParagraphStyle(
        "AlertArrhythmia", parent=styles["Normal"],
        fontSize=12, textColor=ACCENT, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceBefore=5, spaceAfter=5
    )
    alert_uncertain = ParagraphStyle(
        "AlertUncertain", parent=styles["Normal"],
        fontSize=12, textColor=WARNING, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceBefore=5, spaceAfter=5
    )

    elements = []

    # ── Header ──
    elements.append(Paragraph("Cardiac Arrhythmia Detection Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | "
        f"AI-Assisted ECG Analysis System",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    elements.append(Spacer(1, 10))

    # ── Patient / Record Info ──
    elements.append(Paragraph("Patient Information", heading_style))
    info_data = [
        ["Record ID:", str(record_id)],
        ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Signal Duration:", "2.0 seconds (720 samples @ 360 Hz)"],
        ["Lead:", "MLII (Modified Limb Lead II)"],
    ]
    info_table = Table(info_data, colWidths=[120, 350])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 10))

    # ── Prediction Result ──
    elements.append(Paragraph("Prediction Result", heading_style))

    alert_level = prediction_result.get("alert_level", "normal")
    alert_msg = prediction_result.get("alert_message", "")
    if alert_level == "normal":
        elements.append(Paragraph(alert_msg, alert_normal))
    elif alert_level == "arrhythmia":
        elements.append(Paragraph(alert_msg, alert_arrhythmia))
    else:
        elements.append(Paragraph(alert_msg, alert_uncertain))

    pred_data = [
        ["Predicted Class:", prediction_result.get("class_name", "N/A")],
        ["Confidence:", f"{prediction_result.get('confidence', 0)*100:.1f}%"],
        ["Noise Level:", f"{prediction_result.get('noise_level', 0)*100:.1f}%"],
    ]
    pred_table = Table(pred_data, colWidths=[120, 350])
    pred_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 5))

    # ── Class Probabilities ──
    elements.append(Paragraph("Class Probabilities", heading_style))
    probs = prediction_result.get("probabilities", {})
    prob_data = [["Class", "Probability"]]
    for cls, prob in probs.items():
        prob_data.append([cls, f"{prob*100:.2f}%"])

    prob_table = Table(prob_data, colWidths=[120, 120])
    prob_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 10))

    # ── ECG Signal with Grad-CAM ──
    elements.append(Paragraph("ECG Signal with XAI Heatmap", heading_style))
    elements.append(Paragraph(
        "<i>Red/warm regions indicate high contribution to the prediction. "
        "These highlighted areas correspond to morphological features that the AI model "
        "considers most important for its classification decision.</i>",
        body_style
    ))

    signal = prediction_result.get("signal", [])
    heatmap = prediction_result.get("heatmap", None)
    class_name = prediction_result.get("class_name", "Unknown")

    ecg_buf = _create_ecg_plot(
        signal, heatmap,
        f"ECG Segment — Predicted: {class_name} ({prediction_result.get('confidence', 0)*100:.1f}%)"
    )
    ecg_img = Image(ecg_buf, width=170 * mm, height=55 * mm)
    elements.append(ecg_img)
    elements.append(Spacer(1, 10))

    # ── XAI Explanation ──
    elements.append(Paragraph("Model Explanation (Grad-CAM)", heading_style))
    explanation = prediction_result.get("explanation", "No explanation available.")
    elements.append(Paragraph(explanation, body_style))
    elements.append(Spacer(1, 15))

    # ── Disclaimer ──
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc")))
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#999999"), leading=10,
        spaceBefore=10, alignment=TA_CENTER
    )
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI-assisted academic prototype system "
        "and is NOT intended for direct clinical use. All predictions should be reviewed by a "
        "qualified healthcare professional. This system is designed for research and educational purposes only.",
        disclaimer_style
    ))

    # Build PDF
    doc.build(elements)
    return pdf_path
