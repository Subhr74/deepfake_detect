"use strict";

const API_BASE = "http://localhost:5000";

let currentTab   = "image";
let selectedFile = null;
let videoChart   = null;

// DOM
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

const verdictBanner = document.getElementById("verdictBanner");
const verdictIcon   = document.getElementById("verdictIcon");
const verdictLabel  = document.getElementById("verdictLabel");
const verdictConf   = document.getElementById("verdictConf");
const meterFill     = document.getElementById("meterFill");
const imageResults  = document.getElementById("imageResults");
const videoResults  = document.getElementById("videoResults");
const faceGrid      = document.getElementById("faceGrid");
const signalsPanel  = document.getElementById("signalsPanel");
const fakeCountEl   = document.getElementById("fakeCount");
const realCountEl   = document.getElementById("realCount");
const sampledEl     = document.getElementById("sampledCount");
const messageEl     = document.getElementById("resultMessage");

// Signal display metadata
const SIGNAL_META = {
  frequency:    { label: "Frequency",    desc: "FFT / DCT spectral fingerprint" },
  prnu:         { label: "PRNU / Noise", desc: "Camera sensor noise pattern" },
  chromatic_ab: { label: "Chromatic AB", desc: "RGB lens aberration proxy" },
  skin_texture: { label: "Skin Texture", desc: "Micro-texture & LBP uniformity" },
  neural:       { label: "Neural",       desc: "MobileNetV2 feature anomaly" },
  colour:       { label: "Colour",       desc: "Saturation smoothness" },
};

// ── Tabs ──────────────────────────────────────────────────────────────────
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => { t.classList.remove("active"); t.setAttribute("aria-selected","false"); });
    tab.classList.add("active");
    tab.setAttribute("aria-selected","true");
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

// ── File handling ──────────────────────────────────────────────────────────
fileInput.addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });

uploadZone.addEventListener("click", e => {
  if (e.target.closest(".file-clear") || e.target.classList.contains("file-link")) return;
  if (!selectedFile) fileInput.click();
});

fileClear.addEventListener("click", e => { e.stopPropagation(); clearFile(); hideAll(); });

uploadZone.addEventListener("dragover",  e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

function setFile(f) {
  selectedFile = f;
  fileNameEl.textContent = f.name;
  fileChosen.style.display = "flex";
  analyseBtn.disabled = false;
}
function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  fileChosen.style.display = "none";
  analyseBtn.disabled = true;
}

// ── Analyse ────────────────────────────────────────────────────────────────
analyseBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  hideAll();
  showLoading("Uploading file…");
  analyseBtn.disabled = true;

  const msgs = currentTab === "image"
    ? ["Uploading file…","Detecting faces…","Running forensics…","Computing verdict…"]
    : ["Uploading video…","Decoding frames…","Running forensics…","Computing verdict…"];

  let mi = 0;
  const iv = setInterval(() => { loadingText.textContent = msgs[++mi % msgs.length]; }, 2000);

  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/detect/${currentTab}`, { method:"POST", body: fd });
    clearInterval(iv);

    if (!res.ok) {
      const e = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      showError(e.error || "Server error");
      return;
    }

    const data = await res.json();
    hideAll();
    currentTab === "image" ? renderImageResults(data) : renderVideoResults(data);
    resultsEl.style.display = "";
  } catch (err) {
    clearInterval(iv);
    showError(`Cannot reach backend at ${API_BASE}. Make sure Flask is running.\n(${err.message})`);
  } finally {
    analyseBtn.disabled = false;
  }
}

// ── Image results ──────────────────────────────────────────────────────────
function renderImageResults(data) {
  const label = data.overall_label || data.results?.[0]?.label || "Unknown";
  const conf  = data.overall_confidence ?? data.results?.[0]?.confidence ?? 0;
  setVerdict(label, conf);

  // Face cards
  faceGrid.innerHTML = "";
  (data.results || []).forEach((f, i) => {
    const pct = Math.round(f.confidence * 100);
    const cls = f.label === "Fake" ? "fake" : "real";
    const bb  = f.bounding_box || [];
    const card = document.createElement("div");
    card.className = `face-card ${cls}`;
    card.innerHTML = `
      <div class="face-number">FACE ${i + 1}</div>
      <div class="face-label">${f.label}</div>
      <div class="face-conf">${pct}% confidence</div>
      <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
      ${bb.length === 4 ? `<div class="face-bbox">x:${bb[0]} y:${bb[1]} w:${bb[2]} h:${bb[3]}</div>` : ""}
    `;
    faceGrid.appendChild(card);
  });

  // Signal breakdown — use signals from first face result
  const signals = data.results?.[0]?.signals || {};
  renderSignals(signals);

  imageResults.style.display = "";
  videoResults.style.display = "none";
  messageEl.textContent = data.message || "";
  messageEl.style.display = data.message ? "" : "none";
}

// ── Signal breakdown ───────────────────────────────────────────────────────
function renderSignals(signals) {
  signalsPanel.innerHTML = "";
  const keys = Object.keys(SIGNAL_META);

  keys.forEach(key => {
    if (!(key in signals)) return;
    const raw   = signals[key];           // 0..1, higher = more fake
    const pct   = Math.round(raw * 100);
    const isFake = raw >= 0.38;
    const meta  = SIGNAL_META[key];

    // Colour: green (real) → yellow → red (fake)
    const r = Math.round(82  + (255 - 82)  * raw);
    const g = Math.round(232 + (82  - 232) * raw);
    const b = Math.round(158 + (82  - 158) * raw);
    const barColor = `rgb(${r},${g},${b})`;

    const row = document.createElement("div");
    row.className = "signal-row";
    row.innerHTML = `
      <span class="signal-name" title="${meta.desc}">${meta.label}</span>
      <div class="signal-bar-wrap">
        <div class="signal-bar" style="width:${pct}%;background:${barColor}"></div>
      </div>
      <span class="signal-pct" style="color:${barColor}">${pct}%</span>
      <span class="signal-label-tag ${isFake ? 'fake' : 'real'}">${isFake ? 'FAKE' : 'REAL'}</span>
    `;
    signalsPanel.appendChild(row);
  });

  if (signalsPanel.children.length === 0) {
    signalsPanel.innerHTML = `<p style="color:var(--muted);font-size:.84rem;">No signal data available.</p>`;
  }
}

// ── Video results ──────────────────────────────────────────────────────────
function renderVideoResults(data) {
  setVerdict(data.label || "Unknown", data.confidence ?? 0);
  fakeCountEl.textContent = data.fake_count   ?? 0;
  realCountEl.textContent = data.real_count   ?? 0;
  sampledEl.textContent   = data.frames_sampled ?? 0;
  messageEl.textContent   = data.message || "";
  messageEl.style.display = data.message ? "" : "none";
  imageResults.style.display = "none";
  videoResults.style.display = "";
  renderVideoChart(data.fake_count ?? 0, data.real_count ?? 0);
}

// ── Chart ──────────────────────────────────────────────────────────────────
function renderVideoChart(fake, real) {
  if (videoChart) { videoChart.destroy(); videoChart = null; }
  if (fake === 0 && real === 0) return;
  const ctx = document.getElementById("videoChart").getContext("2d");
  videoChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Fake Frames","Real Frames"],
      datasets: [{
        data: [fake, real],
        backgroundColor: ["rgba(255,82,82,.85)","rgba(82,232,158,.85)"],
        borderColor:     ["rgba(255,82,82,1)",  "rgba(82,232,158,1)"],
        borderWidth: 2, hoverOffset: 6,
      }],
    },
    options: {
      responsive: true, cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { color:"#636878", font:{ family:"'Space Mono',monospace", size:11 }, padding:20, usePointStyle:true },
        },
        tooltip: {
          backgroundColor:"#111318", borderColor:"#22252f", borderWidth:1,
          titleColor:"#dde1ee", bodyColor:"#636878", padding:12,
          callbacks: { label: c => {
            const tot = c.dataset.data.reduce((a,b) => a+b, 0);
            return `  ${c.parsed} frames (${tot ? Math.round(c.parsed/tot*100) : 0}%)`;
          }},
        },
      },
      animation: { animateRotate:true, duration:800, easing:"easeInOutQuart" },
    },
  });
}

// ── Verdict ────────────────────────────────────────────────────────────────
function setVerdict(label, confidence) {
  const isFake = label === "Fake";
  verdictBanner.className = `verdict-banner ${isFake ? "fake" : "real"}`;
  verdictIcon.textContent  = isFake ? "🔴" : "🟢";
  verdictLabel.textContent = label;
  verdictConf.textContent  = `Confidence: ${Math.round(confidence * 100)}%`;
  // Meter: shows position across Real–Fake spectrum
  // fake_prob = confidence if Fake, else 1-confidence
  const fakeProbForMeter = isFake ? confidence : 1 - confidence;
  meterFill.style.width = `${Math.round(fakeProbForMeter * 100)}%`;
}

// ── Helpers ────────────────────────────────────────────────────────────────
function showLoading(msg) { loadingText.textContent = msg; loadingWrap.style.display = ""; }
function showError(msg)   { hideAll(); errorText.textContent = msg; errorPanel.style.display = ""; }
function hideAll()        { loadingWrap.style.display = resultsEl.style.display = errorPanel.style.display = "none"; }