/**
 * DeepScan — script.js
 * Handles: tab switching, file upload, API calls, result rendering, Chart.js
 */

"use strict";

// ── Config ────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:5000";

// ── State ─────────────────────────────────────────────────────────────────
let currentTab  = "image";   // "image" | "video"
let selectedFile = null;
let videoChart   = null;

// ── DOM refs ──────────────────────────────────────────────────────────────
const tabs          = document.querySelectorAll(".tab");
const uploadZone    = document.getElementById("uploadZone");
const fileInput     = document.getElementById("fileInput");
const fileChosen    = document.getElementById("fileChosen");
const fileNameEl    = document.getElementById("fileName");
const fileClear     = document.getElementById("fileClear");
const uploadHint    = document.getElementById("uploadHint");
const uploadIconImg = document.getElementById("uploadIconImg");
const uploadIconVid = document.getElementById("uploadIconVid");
const analyseBtn    = document.getElementById("analyseBtn");
const loadingWrap   = document.getElementById("loadingWrap");
const loadingText   = document.getElementById("loadingText");
const resultsEl     = document.getElementById("results");
const errorPanel    = document.getElementById("errorPanel");
const errorText     = document.getElementById("errorText");

// result sub-elements
const verdictBanner = document.getElementById("verdictBanner");
const verdictIcon   = document.getElementById("verdictIcon");
const verdictLabel  = document.getElementById("verdictLabel");
const verdictConf   = document.getElementById("verdictConf");
const imageResults  = document.getElementById("imageResults");
const videoResults  = document.getElementById("videoResults");
const faceGrid      = document.getElementById("faceGrid");
const fakeCountEl   = document.getElementById("fakeCount");
const realCountEl   = document.getElementById("realCount");
const sampledEl     = document.getElementById("sampledCount");
const messageEl     = document.getElementById("resultMessage");

// ── Tab switching ─────────────────────────────────────────────────────────
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    currentTab = tab.dataset.tab;
    updateTabUI();
    clearFile();
    hideAll();
  });
});

function updateTabUI() {
  if (currentTab === "image") {
    fileInput.accept = "image/png,image/jpeg,image/jpg,image/bmp,image/webp";
    uploadHint.textContent = "PNG, JPG, JPEG, BMP, WEBP";
    uploadIconImg.style.display = "";
    uploadIconVid.style.display = "none";
  } else {
    fileInput.accept = "video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm";
    uploadHint.textContent = "MP4, AVI, MOV, MKV, WEBM";
    uploadIconImg.style.display = "none";
    uploadIconVid.style.display = "";
  }
}

// ── File selection ────────────────────────────────────────────────────────
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) setFile(file);
});

uploadZone.addEventListener("click", e => {
  // Don't trigger if clicking the clear button or label
  if (e.target.closest(".file-clear") || e.target.classList.contains("file-link")) return;
  if (!selectedFile) fileInput.click();
});

fileClear.addEventListener("click", e => {
  e.stopPropagation();
  clearFile();
  hideAll();
});

// Drag & drop
uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

function setFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileChosen.style.display = "flex";
  analyseBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  fileChosen.style.display = "none";
  analyseBtn.disabled = true;
}

// ── Analyse ───────────────────────────────────────────────────────────────
analyseBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  hideAll();
  showLoading("Uploading file…");
  analyseBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  const endpoint = currentTab === "image"
    ? `${API_BASE}/detect/image`
    : `${API_BASE}/detect/video`;

  // Cycle loading messages to hint at progress
  const messages = currentTab === "image"
    ? ["Uploading file…", "Detecting faces…", "Running inference…", "Almost done…"]
    : ["Uploading video…", "Decoding frames…", "Analysing faces…", "Computing verdict…"];

  let msgIdx = 0;
  const msgInterval = setInterval(() => {
    msgIdx = (msgIdx + 1) % messages.length;
    loadingText.textContent = messages[msgIdx];
  }, 1800);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    clearInterval(msgInterval);

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      showError(err.error || "Server error");
      return;
    }

    const data = await response.json();
    hideAll();

    if (currentTab === "image") {
      renderImageResults(data);
    } else {
      renderVideoResults(data);
    }

    resultsEl.style.display = "";
  } catch (err) {
    clearInterval(msgInterval);
    showError(`Could not reach the backend. Make sure the Flask server is running on ${API_BASE}. (${err.message})`);
  } finally {
    analyseBtn.disabled = false;
  }
}

// ── Image results renderer ────────────────────────────────────────────────
function renderImageResults(data) {
  // Overall verdict
  const label = data.overall_label || (data.results?.[0]?.label) || "Unknown";
  const conf  = data.overall_confidence ?? data.results?.[0]?.confidence ?? 0;
  setVerdict(label, conf);

  // Face cards
  faceGrid.innerHTML = "";
  const faces = data.results || [];

  if (faces.length === 0) {
    faceGrid.innerHTML = `<p style="color:var(--text-muted);font-size:.88rem;">No face data returned.</p>`;
  } else {
    faces.forEach((f, i) => {
      const pct  = Math.round(f.confidence * 100);
      const cls  = f.label === "Fake" ? "fake" : "real";
      const bbox = f.bounding_box || [];
      const bboxStr = bbox.length === 4
        ? `x:${bbox[0]} y:${bbox[1]} w:${bbox[2]} h:${bbox[3]}`
        : "";

      const card = document.createElement("div");
      card.className = `face-card ${cls}`;
      card.innerHTML = `
        <div class="face-number">FACE ${i + 1}</div>
        <div class="face-label">${f.label}</div>
        <div class="face-conf">${pct}% confidence</div>
        <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
        ${bboxStr ? `<div class="face-bbox">${bboxStr}</div>` : ""}
      `;
      faceGrid.appendChild(card);
    });
  }

  imageResults.style.display  = "";
  videoResults.style.display  = "none";
  messageEl.textContent        = data.message || "";
  messageEl.style.display      = data.message ? "" : "none";
}

// ── Video results renderer ────────────────────────────────────────────────
function renderVideoResults(data) {
  const label = data.label || "Unknown";
  const conf  = data.confidence ?? 0;
  setVerdict(label, conf);

  fakeCountEl.textContent    = data.fake_count   ?? 0;
  realCountEl.textContent    = data.real_count    ?? 0;
  sampledEl.textContent      = data.frames_sampled ?? 0;
  messageEl.textContent      = data.message || "";
  messageEl.style.display    = data.message ? "" : "none";

  imageResults.style.display = "none";
  videoResults.style.display = "";

  renderVideoChart(data.fake_count ?? 0, data.real_count ?? 0);
}

// ── Chart.js donut ────────────────────────────────────────────────────────
function renderVideoChart(fake, real) {
  const ctx = document.getElementById("videoChart").getContext("2d");

  if (videoChart) {
    videoChart.destroy();
    videoChart = null;
  }

  if (fake === 0 && real === 0) return;

  videoChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Fake Frames", "Real Frames"],
      datasets: [{
        data: [fake, real],
        backgroundColor: ["rgba(255,82,82,.85)", "rgba(87,232,158,.85)"],
        borderColor:     ["rgba(255,82,82,1)",   "rgba(87,232,158,1)"],
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#9da3b4",
            font: { family: "'Space Mono', monospace", size: 11 },
            padding: 20,
            usePointStyle: true,
            pointStyleWidth: 10,
          },
        },
        tooltip: {
          backgroundColor: "#13151a",
          borderColor: "#252830",
          borderWidth: 1,
          titleColor: "#e2e4ec",
          bodyColor: "#9da3b4",
          padding: 12,
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct = total ? Math.round(ctx.parsed / total * 100) : 0;
              return `  ${ctx.parsed} frames (${pct}%)`;
            },
          },
        },
      },
      animation: { animateRotate: true, duration: 800, easing: "easeInOutQuart" },
    },
  });
}

// ── Verdict helper ────────────────────────────────────────────────────────
function setVerdict(label, confidence) {
  const isFake = label === "Fake";
  verdictBanner.className = `verdict-banner ${isFake ? "fake" : "real"}`;
  verdictIcon.textContent  = isFake ? "🔴" : "🟢";
  verdictLabel.textContent = label;
  verdictConf.textContent  = `Confidence: ${Math.round(confidence * 100)}%`;
}

// ── UI helpers ────────────────────────────────────────────────────────────
function showLoading(msg) {
  loadingText.textContent  = msg;
  loadingWrap.style.display = "";
}

function showError(msg) {
  hideAll();
  errorText.textContent   = msg;
  errorPanel.style.display = "";
}

function hideAll() {
  loadingWrap.style.display  = "none";
  resultsEl.style.display     = "none";
  errorPanel.style.display    = "none";
}