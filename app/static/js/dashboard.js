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
    trendChart: null,
    shortcutsVisible: false,
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
    document.getElementById("btnExportJSON").addEventListener("click", exportSessionJSON);
    document.getElementById("btnShortcuts").addEventListener("click", toggleShortcuts);
    document.getElementById("shortcutsClose").addEventListener("click", toggleShortcuts);

    // Legend card accordion
    document.querySelectorAll(".legend-card-header").forEach(header => {
        header.addEventListener("click", () => {
            header.closest(".legend-card").classList.toggle("open");
        });
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", handleKeyboardShortcut);
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
    renderXAIExplanation(data.explanation);

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
            state.startSample = p.start_sample;
            state.cameFromStream = true;
            document.getElementById("sampleOffset").value = p.start_sample;
            runPrediction();
        });
        timeline.appendChild(dot);
    });

    // Stream chart — confidence over time
    renderStreamChart(preds);

    // ── NEW FEATURES ──
    renderTrendChart(preds);
    computeAndShowRiskScore(preds);
    checkSmartAlerts(preds);
    renderSessionSummary(preds);
}

// ═══════════════════════════════════════════
// FEATURE 1: Beat-to-Beat Confidence Trend
// ═══════════════════════════════════════════
function renderTrendChart(predictions) {
    const ctx = document.getElementById("trendCanvas").getContext("2d");
    if (state.trendChart) state.trendChart.destroy();

    const labels = predictions.map((p, i) => p.time_start ? `${p.time_start.toFixed(1)}s` : `W${i + 1}`);
    const confidences = predictions.map(p => (p.confidence * 100));

    // Compute 5-window moving average
    const movingAvg = [];
    const windowSize = Math.min(5, predictions.length);
    for (let i = 0; i < confidences.length; i++) {
        const start = Math.max(0, i - Math.floor(windowSize / 2));
        const end = Math.min(confidences.length, start + windowSize);
        const slice = confidences.slice(start, end);
        movingAvg.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }

    // Color dots based on class
    const dotColors = predictions.map(p => {
        if (p.class_short === "N") return "#43a047";
        if (p.class_short === "V") return "#e53935";
        if (p.class_short === "S") return "#1e88e5";
        if (p.class_short === "F") return "#ef6c00";
        return "#8e24aa";
    });

    state.trendChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Confidence %",
                    data: confidences,
                    borderColor: "#1565c0",
                    borderWidth: 2,
                    pointRadius: 5,
                    pointBackgroundColor: dotColors,
                    pointBorderColor: dotColors,
                    pointBorderWidth: 2,
                    tension: 0.3,
                    fill: false,
                },
                {
                    label: "Trend (Moving Avg)",
                    data: movingAvg,
                    borderColor: "#ef5350",
                    borderWidth: 2,
                    borderDash: [6, 3],
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "Confidence Trend Over Time — Colored by Classification",
                    color: "#1e293b",
                    font: { size: 13, weight: "bold" },
                },
                legend: {
                    labels: { color: "#475569", font: { size: 10 } },
                },
                tooltip: {
                    callbacks: {
                        afterLabel: (item) => {
                            if (item.datasetIndex === 0) {
                                const p = predictions[item.dataIndex];
                                return `Class: ${p.class_name}`;
                            }
                            return "";
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 15 },
                    grid: { color: "rgba(0,0,0,0.04)" },
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

    // Trend insight
    const insightEl = document.getElementById("trendInsight");
    const firstHalf = confidences.slice(0, Math.floor(confidences.length / 2));
    const secondHalf = confidences.slice(Math.floor(confidences.length / 2));
    const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    const diff = avgSecond - avgFirst;

    // Detect longest abnormal streak
    let maxStreak = 0, curStreak = 0;
    predictions.forEach(p => {
        if (p.class_short !== "N") { curStreak++; maxStreak = Math.max(maxStreak, curStreak); }
        else { curStreak = 0; }
    });

    let insight = "";
    if (Math.abs(diff) < 2) {
        insight = "📊 <strong>Stable:</strong> Confidence remains steady throughout the recording.";
    } else if (diff > 0) {
        insight = `📈 <strong>Improving:</strong> Confidence increased by ${diff.toFixed(1)}% on average in the second half — rhythm may be stabilizing.`;
    } else {
        insight = `📉 <strong>Declining:</strong> Confidence dropped by ${Math.abs(diff).toFixed(1)}% on average — possible worsening trend.`;
    }
    if (maxStreak >= 3) {
        insight += ` ⚠ Longest consecutive abnormal streak: <strong>${maxStreak} beats</strong>.`;
    }
    insightEl.innerHTML = insight;
}

// ═══════════════════════════════════════════
// FEATURE 2: Risk Score Calculator (0–100)
// ═══════════════════════════════════════════
function computeAndShowRiskScore(predictions) {
    const severityWeight = { V: 1.0, F: 0.7, S: 0.4, Q: 0.3, N: 0.0 };

    let score = 0;
    const total = predictions.length;
    if (total === 0) return;

    // Factor 1: Frequency of abnormal beats (0–40 pts)
    const abnormalCount = predictions.filter(p => p.class_short !== "N").length;
    const abnormalRatio = abnormalCount / total;
    score += abnormalRatio * 40;

    // Factor 2: Severity-weighted average (0–30 pts)
    const severitySum = predictions.reduce((acc, p) => acc + (severityWeight[p.class_short] || 0), 0);
    score += (severitySum / total) * 30;

    // Factor 3: Consecutive abnormal beats (0–20 pts)
    let maxStreak = 0, curStreak = 0;
    predictions.forEach(p => {
        if (p.class_short !== "N") { curStreak++; maxStreak = Math.max(maxStreak, curStreak); }
        else { curStreak = 0; }
    });
    score += Math.min(maxStreak / 5, 1.0) * 20;

    // Factor 4: High-confidence abnormal detections (0–10 pts)
    const highConfAbnormal = predictions.filter(p => p.class_short !== "N" && p.confidence > 0.8).length;
    score += Math.min(highConfAbnormal / total * 5, 1.0) * 10;

    score = Math.min(Math.round(score), 100);

    // Update gauge
    const circumference = 2 * Math.PI * 52;
    const offset = circumference - (score / 100) * circumference;
    const ring = document.getElementById("riskRingFill");
    ring.style.strokeDasharray = circumference;
    ring.style.strokeDashoffset = offset;

    const card = document.getElementById("riskScoreCard");
    const valueEl = document.getElementById("riskScoreValue");
    const descEl = document.getElementById("riskScoreDesc");
    valueEl.textContent = score;

    if (score <= 30) {
        ring.style.stroke = "#43a047";
        valueEl.style.color = "#2e7d32";
        card.className = "risk-score-card risk-low";
        descEl.textContent = "Low Risk — Normal Rhythm";
    } else if (score <= 70) {
        ring.style.stroke = "#f57c00";
        valueEl.style.color = "#e65100";
        card.className = "risk-score-card risk-medium";
        descEl.textContent = "Moderate Risk — Monitor Closely";
    } else {
        ring.style.stroke = "#e53935";
        valueEl.style.color = "#c62828";
        card.className = "risk-score-card risk-high";
        descEl.textContent = "High Risk — Immediate Review";
    }
}

// ═══════════════════════════════════════════
// FEATURE 3: Session Summary Dashboard
// ═══════════════════════════════════════════
function renderSessionSummary(predictions) {
    const total = predictions.length;
    const normalCount = predictions.filter(p => p.class_short === "N").length;
    const abnormalCount = total - normalCount;
    const avgConf = predictions.reduce((a, p) => a + p.confidence, 0) / total * 100;

    // Quick stats
    document.getElementById("streamStatTotal").textContent = total;
    document.getElementById("streamStatNormal").textContent = `${((normalCount / total) * 100).toFixed(0)}%`;
    document.getElementById("streamStatAbnormal").textContent = `${((abnormalCount / total) * 100).toFixed(0)}%`;
    document.getElementById("streamStatAvgConf").textContent = `${avgConf.toFixed(1)}%`;

    // Class breakdown
    const classCounts = { N: 0, S: 0, V: 0, F: 0, Q: 0 };
    const classTimeSpent = { N: 0, S: 0, V: 0, F: 0, Q: 0 };
    predictions.forEach(p => {
        classCounts[p.class_short] = (classCounts[p.class_short] || 0) + 1;
        const duration = (p.time_end || 0) - (p.time_start || 0);
        classTimeSpent[p.class_short] = (classTimeSpent[p.class_short] || 0) + duration;
    });

    // Most common arrhythmia (non-N)
    const arrhythmias = Object.entries(classCounts).filter(([k]) => k !== "N").sort((a, b) => b[1] - a[1]);
    const mostCommon = arrhythmias[0] && arrhythmias[0][1] > 0 ? arrhythmias[0] : null;

    const classNames = { N: "Normal", S: "Supraventricular", V: "Ventricular", F: "Fusion", Q: "Unknown/Paced" };

    const grid = document.getElementById("sessionSummaryGrid");
    grid.innerHTML = `
        <div class="summary-item">
            <div class="summary-item-value">${total}</div>
            <div class="summary-item-label">Total Windows</div>
        </div>
        <div class="summary-item">
            <div class="summary-item-value" style="color:var(--success)">${normalCount}</div>
            <div class="summary-item-label">Normal Beats</div>
        </div>
        <div class="summary-item">
            <div class="summary-item-value" style="color:var(--accent)">${abnormalCount}</div>
            <div class="summary-item-label">Abnormal Beats</div>
        </div>
        <div class="summary-item">
            <div class="summary-item-value">${avgConf.toFixed(1)}%</div>
            <div class="summary-item-label">Avg Confidence</div>
        </div>
        <div class="summary-item">
            <div class="summary-item-value">${mostCommon ? mostCommon[0] : "—"}</div>
            <div class="summary-item-label">Most Common Arrhythmia</div>
        </div>
        <div class="summary-item">
            <div class="summary-item-value">${mostCommon ? classNames[mostCommon[0]] : "None"}</div>
            <div class="summary-item-label">${mostCommon ? `${mostCommon[1]} occurrences` : ""}</div>
        </div>
    `;

    // Class distribution stacked bar
    const distEl = document.getElementById("classDistribution");
    const colors = { N: "#43a047", S: "#1e88e5", V: "#e53935", F: "#ef6c00", Q: "#8e24aa" };
    let barHtml = '<div class="dist-bar">';
    for (const [cls, count] of Object.entries(classCounts)) {
        if (count === 0) continue;
        const pct = (count / total * 100).toFixed(1);
        barHtml += `<div class="dist-segment" style="width:${pct}%;background:${colors[cls]}" title="${cls}: ${count} (${pct}%)">${count > 0 && parseFloat(pct) > 5 ? cls : ""}</div>`;
    }
    barHtml += "</div><div class='dist-legend'>";
    for (const [cls, count] of Object.entries(classCounts)) {
        const time = classTimeSpent[cls].toFixed(1);
        barHtml += `<span class="dist-legend-item"><span class="dist-legend-dot" style="background:${colors[cls]}"></span>${classNames[cls]}: ${count} (${(count / total * 100).toFixed(0)}%) · ${time}s</span>`;
    }
    barHtml += "</div>";
    distEl.innerHTML = barHtml;
}

function exportSessionJSON() {
    if (state.streamResults.length === 0) return;

    const preds = state.streamResults;
    const total = preds.length;
    const classCounts = { N: 0, S: 0, V: 0, F: 0, Q: 0 };
    preds.forEach(p => classCounts[p.class_short]++);

    const summary = {
        export_date: new Date().toISOString(),
        record_id: state.recordId,
        total_windows: total,
        class_counts: classCounts,
        normal_percent: ((classCounts.N / total) * 100).toFixed(1),
        abnormal_percent: (((total - classCounts.N) / total) * 100).toFixed(1),
        avg_confidence: (preds.reduce((a, p) => a + p.confidence, 0) / total * 100).toFixed(1),
        risk_score: parseInt(document.getElementById("riskScoreValue").textContent),
        predictions: preds.map(p => ({
            window: p.window_index,
            class: p.class_short,
            class_name: p.class_name,
            confidence: p.confidence,
            alert_level: p.alert_level,
            time_start: p.time_start,
            time_end: p.time_end,
        })),
    };

    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `CardioAI_Session_${state.recordId}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ═══════════════════════════════════════════
// FEATURE 4: Smart Alert System
// ═══════════════════════════════════════════
function checkSmartAlerts(predictions) {
    const banner = document.getElementById("smartAlertBanner");
    const alerts = [];

    const riskNames = { V: "Ventricular Ectopy", F: "Fusion Beats", S: "Supraventricular Ectopy", Q: "Paced/Unknown" };

    // Check for 3+ consecutive abnormal beats
    let maxStreak = 0, curStreak = 0, streakClass = null;
    predictions.forEach((p) => {
        if (p.class_short !== "N") {
            curStreak++;
            if (curStreak > maxStreak) { maxStreak = curStreak; streakClass = p.class_short; }
        } else {
            curStreak = 0;
        }
    });

    if (maxStreak >= 3) {
        alerts.push({
            severity: "critical",
            icon: "🚨",
            text: `${maxStreak} consecutive abnormal beats detected (${riskNames[streakClass] || streakClass}). Immediate human review recommended.`,
        });
    }

    // Check for high-confidence abnormal (>80%)
    const highConfV = predictions.filter(p => p.class_short === "V" && p.confidence > 0.8);
    const highConfF = predictions.filter(p => p.class_short === "F" && p.confidence > 0.8);
    const highConfS = predictions.filter(p => p.class_short === "S" && p.confidence > 0.8);

    if (highConfV.length >= 2) {
        alerts.push({
            severity: "high",
            icon: "⚡",
            text: `${highConfV.length} high-confidence Ventricular ectopic beats (>80% confidence) — highest clinical priority.`,
        });
    }
    if (highConfF.length >= 2) {
        alerts.push({
            severity: "medium",
            icon: "⚠",
            text: `${highConfF.length} high-confidence Fusion beats detected — may indicate ventricular ectopy.`,
        });
    }
    if (highConfS.length >= 3) {
        alerts.push({
            severity: "low",
            icon: "ℹ",
            text: `${highConfS.length} Supraventricular ectopic beats detected — benign but worth monitoring if symptomatic.`,
        });
    }

    if (alerts.length === 0) {
        banner.style.display = "none";
        return;
    }

    // Show most severe alert
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    alerts.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

    const severityClass = alerts[0].severity === "critical" || alerts[0].severity === "high" ? "smart-alert-critical" : "smart-alert-warning";
    let html = `<div class="smart-alert ${severityClass}">`;
    alerts.forEach(a => {
        html += `<div class="smart-alert-item"><span class="smart-alert-icon">${a.icon}</span><span>${a.text}</span></div>`;
    });
    html += "</div>";
    banner.innerHTML = html;
    banner.style.display = "block";
}

function showAlert(level, message) {
    const banner = document.getElementById("alertBanner");
    banner.className = `alert-banner alert-${level}`;

    const icons = { normal: "✓", arrhythmia: "⚡", uncertain: "⚠" };
    banner.innerHTML = `<span class="alert-icon">${icons[level] || "●"}</span><span>${message}</span>`;
    banner.style.display = "flex";
}

function renderXAIExplanation(explanation) {
    // explanation is now an object: { summary, regions[], clinical_note }
    if (typeof explanation === "string") {
        // Fallback for legacy string format
        document.getElementById("xaiExplanation").innerHTML =
            `<strong>Grad-CAM Interpretation:</strong> ${explanation}`;
        document.getElementById("xaiRegions").style.display = "none";
        document.getElementById("xaiClinical").style.display = "none";
        return;
    }

    // Summary
    document.getElementById("xaiExplanation").innerHTML =
        `<strong>🔍 What the AI Sees:</strong> ${explanation.summary}`;

    // Region breakdown chips
    const regionsEl = document.getElementById("xaiRegions");
    if (explanation.regions && explanation.regions.length > 0) {
        let html = '<div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;">Attention Regions</div>';
        explanation.regions.forEach(r => {
            html += `
                <div class="xai-region-chip">
                    <span class="region-badge">${r.peak_attention}%</span>
                    <div>
                        <span class="region-component">${r.component}</span>
                        <span style="color:var(--text-muted); font-size:11px; font-family:'JetBrains Mono',monospace;">(${r.time})</span>
                        <br>
                        <span class="region-desc">${r.component_description}</span>
                    </div>
                </div>
            `;
        });
        regionsEl.innerHTML = html;
        regionsEl.style.display = "flex";
    } else {
        regionsEl.style.display = "none";
    }

    // Clinical context
    const clinicalEl = document.getElementById("xaiClinical");
    const ctx = explanation.clinical_note;
    if (ctx) {
        const riskColor = ctx.risk_level?.toLowerCase().includes("none")
            ? "background:#e8f5e9; color:#2e7d32"
            : ctx.risk_level?.toLowerCase().includes("benign")
            ? "background:#e3f2fd; color:#1565c0"
            : "background:#fff3e0; color:#e65100";

        clinicalEl.innerHTML = `
            <div class="xai-clinical-title">🩺 Clinical Context</div>
            <dl>
                <dt>What does this mean?</dt>
                <dd>${ctx.what_it_means}</dd>
                <dt>What does the model look for?</dt>
                <dd>${ctx.what_model_looks_for}</dd>
                <dt>Risk level</dt>
                <dd><span class="xai-risk-tag" style="${riskColor}">${ctx.risk_level}</span></dd>
            </dl>
        `;
        clinicalEl.style.display = "block";
    } else {
        clinicalEl.style.display = "none";
    }
}

function hideResults() {
    document.getElementById("welcomeScreen").style.display = "flex";
    document.getElementById("resultsContainer").style.display = "none";
    document.getElementById("streamContainer").style.display = "none";
    document.getElementById("alertBanner").style.display = "none";
    document.getElementById("backToStream").style.display = "none";
    document.getElementById("smartAlertBanner").style.display = "none";
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

// ═══════════════════════════════════════════
// FEATURE 6: Keyboard Shortcuts
// ═══════════════════════════════════════════
function toggleShortcuts() {
    state.shortcutsVisible = !state.shortcutsVisible;
    document.getElementById("shortcutsModal").style.display = state.shortcutsVisible ? "flex" : "none";
}

function handleKeyboardShortcut(e) {
    // Don't trigger when typing in inputs
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
    if (state.isLoading) return;

    switch (e.key.toLowerCase()) {
        case " ": // Space — Analyze
            e.preventDefault();
            if (state.recordId) runPrediction();
            break;
        case "r": // R — Stream
            if (state.recordId) runStream();
            break;
        case "d": // D — Download report
            downloadReport();
            break;
        case "n": // N — Noise +10%
            const noiseSlider = document.getElementById("noiseSlider");
            const newVal = Math.min(100, parseInt(noiseSlider.value) + 10);
            noiseSlider.value = newVal;
            noiseSlider.dispatchEvent(new Event("input"));
            break;
        case "b": // B — Back to stream
            if (state.cameFromStream) backToStream();
            break;
        case "escape": // Esc — Reset / close modal
            if (state.shortcutsVisible) {
                toggleShortcuts();
            } else {
                hideResults();
            }
            break;
        case "?": // ? — Shortcuts help
            toggleShortcuts();
            break;
        case "1": // 1 — Single mode
            setMode("single");
            break;
        case "2": // 2 — Stream mode
            setMode("stream");
            break;
        case "j": // J — Export JSON (when in stream view)
            if (state.streamResults.length > 0) exportSessionJSON();
            break;
    }
}
