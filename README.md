# CardioAI — Automated ECG Arrhythmia Detection System

A web-based cardiac arrhythmia detection system using a hybrid CNN-LSTM deep learning
architecture with Explainable AI (Grad-CAM) and simulated real-time streaming.

## Features

1. **Hybrid CNN-LSTM Model** — 1D-CNN extracts local ECG morphology; LSTM captures temporal cardiac rhythms
2. **Grad-CAM XAI** — Visual explanations highlighting which ECG regions drive predictions
3. **Noise Injection Slider** — Test model robustness by adding Gaussian noise
4. **Normal vs. Abnormal Comparison** — Side-by-side ECG comparison for clinical insight
5. **PDF Report Generator** — Downloadable clinical-style reports with heatmaps
6. **Confidence Threshold Alerts** — Uncertain predictions flagged for human review
7. **Simulated Real-Time Streaming** — Sliding window inference across continuous ECG signals
8. **Professional Doctor's View UI** — Dark medical-grade dashboard

## Project Structure

```
ecg_project/
├── data/                     # MIT-BIH Arrhythmia Database files
├── app/
│   ├── config.py             # Configuration & hyperparameters
│   ├── main.py               # Flask application factory
│   ├── model/
│   │   ├── preprocessing.py  # Signal filtering, normalization, segmentation
│   │   ├── architecture.py   # CNN-LSTM model definition
│   │   ├── gradcam.py        # Grad-CAM implementation
│   │   └── inference.py      # Prediction engine
│   ├── routes/
│   │   ├── api.py            # REST API endpoints
│   │   └── report.py         # PDF report generation
│   ├── static/
│   │   ├── css/style.css     # Dashboard styles
│   │   └── js/dashboard.js   # Frontend logic
│   └── templates/
│       └── dashboard.html    # Main dashboard
├── scripts/
│   └── train_model.py        # Model training script
├── saved_models/             # Trained model files
├── reports/                  # Generated PDF reports
├── requirements.txt
├── run.py                    # Application entry point
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python scripts/train_model.py
```
This loads all 48 MIT-BIH records, preprocesses signals, trains the CNN-LSTM model,
and saves it to `saved_models/cnn_lstm_model.keras`.

### 3. Run the Web Application
```bash
python run.py
```
Open **http://localhost:5000** in your browser.

## AAMI Classification (5 Classes)

| Class | Name               | MIT-BIH Symbols     |
|-------|--------------------|----------------------|
| N     | Normal             | N, L, R, e, j       |
| S     | Supraventricular   | A, a, J, S           |
| V     | Ventricular        | V, E                 |
| F     | Fusion             | F                    |
| Q     | Unknown/Paced      | /, f, Q              |

## API Endpoints

| Method | Endpoint              | Description                                |
|--------|-----------------------|--------------------------------------------|
| GET    | `/api/status`         | System status & model info                 |
| GET    | `/api/records`        | List available ECG records                 |
| GET    | `/api/record/<id>`    | Load a full ECG record                     |
| POST   | `/api/predict`        | Single window arrhythmia prediction        |
| POST   | `/api/stream`         | Simulated streaming (multiple windows)     |
| GET    | `/api/normal-template`| Normal beat template for comparison        |
| POST   | `/api/report`         | Generate & download PDF report             |
| POST   | `/api/threshold`      | Update confidence threshold                |

## Dataset

MIT-BIH Arrhythmia Database from PhysioNet (48 half-hour ECG recordings, 360 Hz).

## Disclaimer

This is an academic prototype for research and educational purposes.
Not intended for clinical use. All predictions should be reviewed by a qualified healthcare professional.
