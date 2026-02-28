/* ═══════════════════════════════════════════════════
   CardioAI — Dashboard Controller
   ═══════════════════════════════════════════════════ */

// ── State ──
const state = {
    recordId: null,
    startSample: 0,
    noiseLevel: 0,
    confidenceThreshold: 60,
    mode: "single",          // "single" | "stream"
    prediction: null,
    streamResults: [],
    normalTemplate: null,
    isLoading: false,
    cameFromStream: false,
    modelLoaded: false,
    ecgChart: null,
    heatmapChart: null,
    compNormalChart: null,
    compAbnormalChart: null,
};

// ── API Helper ──
async function apiCall(url, options = {}) {
    try {
        const resp = await fetch(url, options);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || `HTTP ${resp.status}`);
        }
        return resp;
    } catch (e) {
        showError(e.message);
        throw e;
    }
}

async function apiJson(url, options = {}) {
    const resp = await apiCall(url, options);
    return resp.json();
}

// ── Initialization ──
document.addEventListener("DOMContentLoaded", async () => {
    await checkStatus();
    await loadRecords();
    await loadNormalTemplate();
    setupEventListeners();
});

async function checkStatus() {
    try {
        const data = await apiJson("/api/status");
        state.modelLoaded = data.model_loaded;
        updateStatusIndicator(data.model_loaded);
    } catch {
        updateStatusIndicator(false);
    }
}

function updateStatusIndicator(loaded) {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    if (loaded) {
        dot.className = "status-dot online";
        text.textContent = "Model Ready";
    } else {
        dot.className = "status-dot offline";
        text.textContent = "Model Not Loaded";
    }
}

async function loadRecords() {
    try {
        const data = await apiJson("/api/records");
        const select = document.getElementById("recordSelect");
        select.innerHTML = '<option value="">— Select Record —</option>';
        data.records.forEach(r => {
            const opt = document.createElement("option");
            opt.value = r;
            opt.textContent = `Record ${r}`;
            select.appendChild(opt);
        });
    } catch { /* handled by apiCall */ }
}

async function loadNormalTemplate() {
    try {
        const data = await apiJson("/api/normal-template");
        state.normalTemplate = data;
    } catch { /* will load on demand */ }
}

// ── Event Listeners ──
function setupEventListeners() {
    document.getElementById("recordSelect").addEventListener("change", onRecordChange);
    document.getElementById("noiseSlider").addEventListener("input", onNoiseChange);
    document.getElementById("thresholdSlider").addEventListener("input", onThresholdChange);
    document.getElementById("btnAnalyze").addEventListener("click", runPrediction);
    document.getElementById("btnStream").addEventListener("click", runStream);
    document.getElementById("btnReport").addEventListener("click", downloadReport);
    document.getElementById("modeSingle").addEventListener("click", () => setMode("single"));
    document.getElementById("modeStream").addEventListener("click", () => setMode("stream"));
    document.getElementById("sampleOffset").addEventListener("change", (e) => {
        state.startSample = parseInt(e.target.value) || 0;
    });
    document.getElementById("btnBackToStream").addEventListener("click", backToStream);
}

function onRecordChange(e) {
    state.recordId = e.target.value;
    state.startSample = 0;
    document.getElementById("sampleOffset").value = 0;
    // Reset display
    hideResults();
}

function onNoiseChange(e) {
    state.noiseLevel = parseInt(e.target.value) / 100;
    document.getElementById("noiseValue").textContent = `${e.target.value}%`;
}

function onThresholdChange(e) {
    state.confidenceThreshold = parseInt(e.target.value);
    document.getElementById("thresholdValue").textContent = `${e.target.value}%`;
    // Update threshold on server
    fetch("/api/threshold", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold: state.confidenceThreshold / 100 }),
    });
}

function setMode(mode) {
    state.mode = mode;
    document.getElementById("modeSingle").classList.toggle("active", mode === "single");
    document.getElementById("modeStream").classList.toggle("active", mode === "stream");
    document.getElementById("singleControls").style.display = mode === "single" ? "block" : "none";
    document.getElementById("streamControls").style.display = mode === "stream" ? "block" : "none";
}

// ── Prediction (Single Window) ──
async function runPrediction() {
    if (!state.recordId) return showError("Please select a record first.");
    if (!state.modelLoaded) return showError("Model not loaded. Train the model first.");

    setLoading(true);
    try {
        const data = await apiJson("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                record_id: state.recordId,
                start_sample: state.startSample,
                noise_level: state.noiseLevel,
            }),
        });

        state.prediction = data;
        showResults(data);
    } catch { /* handled */ }
    setLoading(false);
}

// ── Stream Prediction ──
async function runStream() {
    if (!state.recordId) return showError("Please select a record first.");
    if (!state.modelLoaded) return showError("Model not loaded. Train the model first.");

    const numWindows = parseInt(document.getElementById("numWindows").value) || 20;
    setLoading(true);

    try {
        const data = await apiJson("/api/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                record_id: state.recordId,
                noise_level: state.noiseLevel,
                num_windows: numWindows,
                start_offset: state.startSample,
            }),
        });

        state.streamResults = data.predictions;
        showStreamResults(data);
    } catch { /* handled */ }
    setLoading(false);
}

// ── Display Results ──
function showResults(data) {
    document.getElementById("welcomeScreen").style.display = "none";
    document.getElementById("resultsContainer").style.display = "block";
    document.getElementById("streamContainer").style.display = "none";

    // Alert banner
    showAlert(data.alert_level, data.alert_message);

    // Stats
    document.getElementById("statClass").textContent = data.class_short;
    document.getElementById("statConfidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById("statNoise").textContent = `${(data.noise_level * 100).toFixed(0)}%`;
    document.getElementById("statWindow").textContent =
        `${data.time_start?.toFixed(1) || "0.0"}–${data.time_end?.toFixed(1) || "2.0"}s`;

    // Update stat card colors
    const confCard = document.getElementById("statConfidence").closest(".stat-card");
    confCard.className = "stat-card " + (data.confidence >= 0.8 ? "success" : data.confidence >= 0.6 ? "warning" : "danger");

    // ECG Chart with Grad-CAM overlay
    renderECGChart(data.signal, data.heatmap, data.class_name, data.confidence);

    // Probability bars
    renderProbBars(data.probabilities);

    // XAI explanation
    document.getElementById("xaiExplanation").innerHTML =
        `<strong>Grad-CAM Interpretation:</strong> ${data.explanation}`;

    // Comparison view
    renderComparisonCharts(data.signal, data.class_name);

    // Enable report button
    document.getElementById("btnReport").disabled = false;

    // Show "Back to Stream" if we came from a timeline click
    const backBtn = document.getElementById("backToStream");
    if (state.cameFromStream && state.streamResults.length > 0) {
        backBtn.style.display = "block";
    } else {
        backBtn.style.display = "none";
    }
}

function showStreamResults(data) {
    document.getElementById("welcomeScreen").style.display = "none";
    document.getElementById("resultsContainer").style.display = "none";
    document.getElementById("streamContainer").style.display = "block";

    const preds = data.predictions;
    const totalNormal = preds.filter(p => p.class_short === "N").length;
    const totalAbnormal = preds.filter(p => p.class_short !== "N").length;
    const totalUncertain = preds.filter(p => p.alert_level === "uncertain").length;

    document.getElementById("streamSummary").innerHTML = `
        <span style="color:var(--success-light)">● ${totalNormal} Normal</span> &nbsp;
        <span style="color:var(--accent-light)">● ${totalAbnormal} Abnormal</span> &nbsp;
        <span style="color:var(--warning-light)">● ${totalUncertain} Uncertain</span> &nbsp;
        <span style="color:var(--text-muted)">| ${preds.length} windows analyzed</span>
    `;

    // Timeline dots
    const timeline = document.getElementById("streamTimeline");
    timeline.innerHTML = "";
    preds.forEach((p, i) => {
        const dot = document.createElement("div");
        const cls = p.alert_level === "uncertain" ? "uncertain" : `class-${p.class_short}`;
        dot.className = `stream-dot ${cls}`;
        dot.textContent = p.class_short;
        dot.title = `Window ${i + 1}: ${p.class_name} (${(p.confidence * 100).toFixed(1)}%)  |  ${p.time_start?.toFixed(1)}–${p.time_end?.toFixed(1)}s`;
        dot.addEventListener("click", () => {
            // Load this specific window's full prediction
            state.startSample = p.start_sample;
            state.cameFromStream = true;
            document.getElementById("sampleOffset").value = p.start_sample;
            runPrediction();
        });
        timeline.appendChild(dot);
    });

    // Stream chart — confidence over time
    renderStreamChart(preds);
}

function showAlert(level, message) {
    const banner = document.getElementById("alertBanner");
    banner.className = `alert-banner alert-${level}`;

    const icons = { normal: "✓", arrhythmia: "⚡", uncertain: "⚠" };
    banner.innerHTML = `<span class="alert-icon">${icons[level] || "●"}</span><span>${message}</span>`;
    banner.style.display = "flex";
}

function hideResults() {
    document.getElementById("welcomeScreen").style.display = "flex";
    document.getElementById("resultsContainer").style.display = "none";
    document.getElementById("streamContainer").style.display = "none";
    document.getElementById("alertBanner").style.display = "none";
    document.getElementById("backToStream").style.display = "none";
    state.cameFromStream = false;
}

function backToStream() {
    // Return to the stream view using cached results
    state.cameFromStream = false;
    document.getElementById("resultsContainer").style.display = "none";
    document.getElementById("streamContainer").style.display = "block";
    document.getElementById("alertBanner").style.display = "none";
    document.getElementById("backToStream").style.display = "none";
}

// ── Charts ──
function renderECGChart(signal, heatmap, className, confidence) {
    const ctx = document.getElementById("ecgCanvas").getContext("2d");
    if (state.ecgChart) state.ecgChart.destroy();

    const timeLabels = signal.map((_, i) => (i / 360).toFixed(3));

    // Determine heatmap colors for each point
    const pointColors = heatmap.map(h => {
        if (h > 0.7) return "rgba(239, 83, 80, 0.9)";
        if (h > 0.5) return "rgba(255, 152, 0, 0.8)";
        if (h > 0.3) return "rgba(255, 202, 40, 0.6)";
        return "rgba(57, 73, 171, 0.6)";
    });

    state.ecgChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: "ECG Signal",
                    data: signal,
                    borderColor: "#3949ab",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false,
                },
                {
                    label: "Grad-CAM Attention",
                    data: heatmap.map((h, i) => h > 0.3 ? signal[i] : null),
                    borderColor: "#ef5350",
                    borderWidth: 2,
                    pointRadius: heatmap.map(h => h > 0.3 ? 2 : 0),
                    pointBackgroundColor: pointColors,
                    showLine: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `ECG with Grad-CAM Overlay — ${className} (${(confidence * 100).toFixed(1)}%)`,
                    color: "#1e293b",
                    font: { size: 13, weight: "bold" },
                },
                legend: {
                    labels: {
                        color: "#475569",
                        font: { size: 11 },
                        usePointStyle: true,
                        pointStyle: "rectRounded",
                        generateLabels: (chart) => {
                            return [
                                {
                                    text: "ECG Signal",
                                    fillStyle: "#3949ab",
                                    strokeStyle: "#3949ab",
                                    lineWidth: 2,
                                    pointStyle: "line",
                                    datasetIndex: 0,
                                },
                                {
                                    text: "Grad-CAM Attention",
                                    fillStyle: "#ef5350",
                                    strokeStyle: "#ef5350",
                                    lineWidth: 2,
                                    pointStyle: "circle",
                                    datasetIndex: 1,
                                },
                            ];
                        },
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (items) => `Time: ${items[0].label}s`,
                        label: (item) => {
                            if (item.datasetIndex === 0) return `Amplitude: ${item.parsed.y.toFixed(4)}`;
                            const h = heatmap[item.dataIndex];
                            return `Attention: ${(h * 100).toFixed(1)}%`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: "Time (seconds)", color: "#475569" },
                    ticks: { color: "#64748b", maxTicksLimit: 10 },
                    grid: { color: "rgba(0,0,0,0.06)" },
                },
                y: {
                    title: { display: true, text: "Amplitude (normalized)", color: "#475569" },
                    ticks: { color: "#64748b" },
                    grid: { color: "rgba(0,0,0,0.06)" },
                },
            },
        },
    });
}

function renderProbBars(probabilities) {
    const container = document.getElementById("probBars");
    const classColors = { N: "class-N", S: "class-S", V: "class-V", F: "class-F", Q: "class-Q" };
    const classLabels = {
        N: "Normal", S: "Supraventricular", V: "Ventricular", F: "Fusion", Q: "Unknown"
    };

    let html = "";
    for (const [cls, prob] of Object.entries(probabilities)) {
        const pct = (prob * 100).toFixed(1);
        html += `
            <div class="prob-bar-row">
                <span class="prob-label">${cls}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill ${classColors[cls]}" style="width: ${Math.max(pct, 2)}%">
                        ${pct}%
                    </div>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

function renderComparisonCharts(abnormalSignal, className) {
    // Abnormal (current)
    const ctxA = document.getElementById("compAbnormalCanvas").getContext("2d");
    if (state.compAbnormalChart) state.compAbnormalChart.destroy();

    const timeLabels = abnormalSignal.map((_, i) => (i / 360).toFixed(3));

    state.compAbnormalChart = new Chart(ctxA, {
        type: "line",
        data: {
            labels: timeLabels,
            datasets: [{
                label: `Current: ${className}`,
                data: abnormalSignal,
                borderColor: "#ef5350",
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.1,
                fill: { target: "origin", above: "rgba(239,83,80,0.05)" },
            }],
        },
        options: getComparisonChartOptions(`Patient ECG — ${className}`),
    });

    // Normal template
    if (state.normalTemplate) {
        const ctxN = document.getElementById("compNormalCanvas").getContext("2d");
        if (state.compNormalChart) state.compNormalChart.destroy();

        const normalLabels = state.normalTemplate.signal.map((_, i) => (i / 360).toFixed(3));

        state.compNormalChart = new Chart(ctxN, {
            type: "line",
            data: {
                labels: normalLabels,
                datasets: [{
                    label: "Normal Sinus Rhythm",
                    data: state.normalTemplate.signal,
                    borderColor: "#4caf50",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: { target: "origin", above: "rgba(76,175,80,0.05)" },
                }],
            },
            options: getComparisonChartOptions("Reference — Normal Sinus Rhythm"),
        });
    }
}

function getComparisonChartOptions(title) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: { display: true, text: title, color: "#1e293b", font: { size: 11, weight: "bold" } },
            legend: { display: false },
        },
        scales: {
            x: {
                display: true,
                ticks: { color: "#64748b", maxTicksLimit: 6, font: { size: 9 } },
                grid: { color: "rgba(0,0,0,0.06)" },
            },
            y: {
                ticks: { color: "#64748b", font: { size: 9 } },
                grid: { color: "rgba(0,0,0,0.06)" },
            },
        },
    };
}

function renderStreamChart(predictions) {
    const ctx = document.getElementById("streamCanvas").getContext("2d");
    if (state.streamChart) state.streamChart.destroy();

    const labels = predictions.map((p, i) => `W${i + 1}`);
    const confidences = predictions.map(p => (p.confidence * 100).toFixed(1));
    const bgColors = predictions.map(p => {
        if (p.alert_level === "uncertain") return "rgba(255, 202, 40, 0.7)";
        if (p.class_short === "N") return "rgba(76, 175, 80, 0.7)";
        return "rgba(239, 83, 80, 0.7)";
    });

    state.streamChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Confidence (%)",
                data: confidences,
                backgroundColor: bgColors,
                borderWidth: 0,
                borderRadius: 3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "Real-Time Streaming — Confidence per Window",
                    color: "#1e293b",
                    font: { size: 13, weight: "bold" },
                },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => {
                            const p = predictions[item.dataIndex];
                            return `${p.class_name}: ${item.parsed.y}%`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#64748b", font: { size: 9 } },
                    grid: { display: false },
                },
                y: {
                    min: 0, max: 100,
                    title: { display: true, text: "Confidence %", color: "#475569" },
                    ticks: { color: "#64748b" },
                    grid: { color: "rgba(0,0,0,0.06)" },
                },
            },
        },
    });
}

// ── PDF Report Download ──
async function downloadReport() {
    if (!state.prediction) return;
    setLoading(true, "Generating Report...");

    try {
        const resp = await apiCall("/api/report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                record_id: state.recordId,
                start_sample: state.startSample,
                noise_level: state.noiseLevel,
            }),
        });

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `ECG_Report_${state.recordId}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
    } catch { /* handled */ }

    setLoading(false);
}

// ── UI Helpers ──
function setLoading(loading, message) {
    state.isLoading = loading;
    const overlay = document.getElementById("loadingOverlay");
    overlay.style.display = loading ? "flex" : "none";
    if (message) {
        document.getElementById("loadingText").textContent = message;
    } else {
        document.getElementById("loadingText").textContent = "Analyzing ECG Signal...";
    }
    document.querySelectorAll(".btn").forEach(b => {
        if (!b.classList.contains("mode-toggle")) b.disabled = loading;
    });
}

function showError(msg) {
    const banner = document.getElementById("alertBanner");
    banner.className = "alert-banner alert-arrhythmia";
    banner.innerHTML = `<span class="alert-icon">✕</span><span>${msg}</span>`;
    banner.style.display = "flex";
    setTimeout(() => { banner.style.display = "none"; }, 5000);
}
