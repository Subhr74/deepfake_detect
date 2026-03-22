"use strict";

const API = "http://localhost:5000";
let tab = "image", file = null, vChart = null, rChart = null;

const $ = id => document.getElementById(id);

// Signal metadata
const SIGS = {
  wavelet_noise:  { label: "Wavelet Noise",    cat: "Noise"     },
  skin_noise:     { label: "Skin Noise",        cat: "Noise"     },
  jpeg_ghost:     { label: "JPEG Ghost",        cat: "Noise"     },
  fft_smoothness: { label: "FFT Smoothness",    cat: "Frequency" },
  checkerboard:   { label: "GAN Checkerboard",  cat: "Frequency" },
  local_grad_var: { label: "Local Gradient",    cat: "Texture"   },
  local_contrast: { label: "Local Contrast",    cat: "Texture"   },
  edge_coherence: { label: "Edge Coherence",    cat: "Texture"   },
  neural_outlier: { label: "Neural Outlier",    cat: "Neural"    },
  channel_indep:  { label: "Channel Corr.",     cat: "Neural"    },
};

const CAT_COLORS = {
  Noise:     "#4db8ff",
  Frequency: "#b48fff",
  Texture:   "#ffb84d",
  Neural:    "#3dffaa",
};

// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => { x.classList.remove("active"); x.removeAttribute("aria-selected"); });
    t.classList.add("active");
    tab = t.dataset.tab;
    resetUI();
    if (tab === "image") {
      $("fileInput").accept = "image/png,image/jpeg,image/jpg,image/bmp,image/webp";
      $("dropHint").textContent = "PNG · JPG · JPEG · BMP · WEBP";
      $("iconImg").style.display = ""; $("iconVid").style.display = "none";
    } else {
      $("fileInput").accept = "video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm";
      $("dropHint").textContent = "MP4 · AVI · MOV · MKV · WEBM";
      $("iconImg").style.display = "none"; $("iconVid").style.display = "";
    }
  });
});

// ── File handling ─────────────────────────────────────────────────────────
$("fileInput").addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });
$("dropZone").addEventListener("click", e => {
  if (e.target.closest(".clear-btn") || e.target.classList.contains("browse-link")) return;
  if (!file) $("fileInput").click();
});
$("clearBtn").addEventListener("click", e => { e.stopPropagation(); clearFile(); });
$("dropZone").addEventListener("dragover",  e => { e.preventDefault(); $("dropZone").classList.add("over"); });
$("dropZone").addEventListener("dragleave", () => $("dropZone").classList.remove("over"));
$("dropZone").addEventListener("drop", e => {
  e.preventDefault(); $("dropZone").classList.remove("over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

function setFile(f) {
  file = f; $("chosenName").textContent = f.name;
  $("chosen").style.display = "flex"; $("runBtn").disabled = false;
}
function clearFile() {
  file = null; $("fileInput").value = "";
  $("chosen").style.display = "none"; $("runBtn").disabled = true;
  resetUI();
}

// ── Analysis ──────────────────────────────────────────────────────────────
$("runBtn").addEventListener("click", runAnalysis);

const MSGS = {
  image: ["Uploading image…", "Detecting faces…", "Running noise analysis…", "FFT + ELA forensics…", "Computing verdict…"],
  video: ["Uploading video…", "Extracting frames…", "Per-frame signal analysis…", "Optical flow + temporal…", "Computing verdict…"],
};

async function runAnalysis() {
  if (!file) return;
  resetUI();
  $("loading").style.display = "flex";
  $("runBtn").disabled = true;

  let step = 0;
  setProgress(5);
  const msgs = MSGS[tab];
  $("loadMsg").textContent = msgs[0];
  const iv = setInterval(() => {
    step = Math.min(step + 1, msgs.length - 1);
    $("loadMsg").textContent = msgs[step];
    setProgress(10 + step * 18);
  }, 2000);

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch(`${API}/detect/${tab}`, { method: "POST", body: fd });
    clearInterval(iv);
    setProgress(100);
    await sleep(200);

    if (!res.ok) {
      const e = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      showErr(e.error || "Server error."); return;
    }
    const data = await res.json();
    $("loading").style.display = "none";
    tab === "image" ? showImageResults(data) : showVideoResults(data);
    $("results").style.display = "";
  } catch (err) {
    clearInterval(iv);
    showErr(`Cannot reach Flask server at ${API}.\nRun: python app.py\n\n${err.message}`);
  } finally {
    $("runBtn").disabled = false;
  }
}

// ── Image results ─────────────────────────────────────────────────────────
function showImageResults(d) {
  const lbl  = d.overall_label || "Unknown";
  const conf = d.overall_confidence ?? 0;
  const fp   = lbl === "Fake" ? conf : 1 - conf;
  setVerdict(lbl, conf, fp);

  // ELA
  const elaScore = d.ela_fake_score ?? 0;
  $("elaScore").textContent = `${Math.round(elaScore * 100)}%`;
  $("elaScore").style.color = score2color(elaScore);
  $("elaMean").textContent  = (d.ela_mean ?? 0).toFixed(1);
  $("elaStd").textContent   = (d.ela_std  ?? 0).toFixed(1);
  $("elaBarFill").style.width = `${Math.round(elaScore * 100)}%`;

  // Radar
  drawRadar(d.pillar_scores || {});

  // Faces
  const faces = d.results || [];
  $("faceBadge").textContent = `${faces.length} face(s)`;
  const grid = $("faceRow");
  grid.innerHTML = "";
  faces.forEach((f, i) => {
    const pct = Math.round(f.confidence * 100);
    const cls = f.label === "Fake" ? "fake" : "real";
    const bb  = f.bounding_box || [];
    const ps  = f.pillar_scores || {};
    const ela = f.ela_face_score != null ? `ELA: ${Math.round(f.ela_face_score*100)}%` : "";
    const pillarsHtml = Object.entries(ps).map(([k,v]) =>
      `<span class="fcp" title="${k}: ${Math.round(v*100)}%">${k.slice(0,4).toUpperCase()} ${Math.round(v*100)}%</span>`
    ).join("");
    const card = document.createElement("div");
    card.className = `fcard ${cls}`;
    card.innerHTML = `
      <div class="fc-idx">FACE ${i+1}</div>
      <div class="fc-lbl">${f.label}</div>
      <div class="fc-conf">${pct}% confidence</div>
      <div class="fc-bar"><div class="fc-bar-fill" style="width:${pct}%"></div></div>
      ${ela ? `<div class="fc-ela">${ela}</div>` : ""}
      ${bb.length===4 ? `<div class="fc-bbox">x${bb[0]} y${bb[1]} ${bb[2]}×${bb[3]}</div>` : ""}
      <div class="fc-pillars">${pillarsHtml}</div>
    `;
    grid.appendChild(card);
  });

  // Signals
  const sigs = faces[0]?.signals || {};
  drawSignals(sigs);

  $("imgSection").style.display = "";
  $("vidSection").style.display = "none";
  const msg = d.message || "";
  $("msgBar").textContent = msg;
  $("msgBar").style.display = msg ? "" : "none";
}

// ── Video results ─────────────────────────────────────────────────────────
function showVideoResults(d) {
  const lbl  = d.label || "Unknown";
  const conf = d.confidence ?? 0;
  const fp   = lbl === "Fake" ? conf : 1 - conf;
  setVerdict(lbl, conf, fp);

  $("fakeCount").textContent  = d.fake_count   ?? 0;
  $("realCount").textContent  = d.real_count   ?? 0;
  $("sampleCount").textContent = d.frames_sampled ?? 0;
  $("tempScore").textContent  = `${Math.round((d.temporal_fake_score ?? 0)*100)}%`;
  $("tempScore").style.color  = score2color(d.temporal_fake_score ?? 0);

  const msg = d.message || "";
  $("msgBar").textContent = msg;
  $("msgBar").style.display = msg ? "" : "none";

  drawVideoChart(d.fake_count ?? 0, d.real_count ?? 0);
  $("imgSection").style.display = "none";
  $("vidSection").style.display = "";
}

// ── Verdict ───────────────────────────────────────────────────────────────
function setVerdict(label, confidence, fakeProbability) {
  const isFake = label === "Fake";
  const v = $("verdict");
  v.className = `verdict ${isFake ? "fake" : "real"}`;
  $("vDot").textContent  = isFake ? "🔴" : "🟢";
  $("vLabel").textContent = label;
  $("vSub").textContent   = `${Math.round(confidence*100)}% confidence · ${isFake ? "AI/Deepfake detected" : "Appears authentic"}`;

  const totalArc = 148;
  const offset   = totalArc - fakeProbability * totalArc;
  const fill     = $("gFill");
  fill.style.strokeDashoffset = offset;
  fill.style.stroke = isFake ? "var(--fake)" : "var(--real)";
  $("gPct").textContent = `${Math.round(fakeProbability*100)}%`;
  $("gPct").style.color = isFake ? "var(--fake)" : "var(--real)";
}

// ── Radar chart ───────────────────────────────────────────────────────────
function drawRadar(pillars) {
  if (rChart) { rChart.destroy(); rChart = null; }
  const labels = Object.keys(pillars);
  const vals   = Object.values(pillars).map(v => Math.round(v * 100));
  if (!labels.length) return;

  const colors = labels.map(l => CAT_COLORS[l] || "#888");
  rChart = new Chart($("radarChart").getContext("2d"), {
    type: "radar",
    data: {
      labels,
      datasets: [{
        data: vals,
        backgroundColor: "rgba(255,77,77,.15)",
        borderColor: "rgba(255,77,77,.75)",
        borderWidth: 2,
        pointBackgroundColor: colors,
        pointRadius: 5,
      }],
    },
    options: {
      responsive: true,
      scales: { r: {
        min:0, max:100, beginAtZero:true,
        ticks:{ stepSize:25, display:false, backdropColor:"transparent" },
        pointLabels:{ color: colors, font:{ family:"'Space Mono',monospace", size:10 } },
        grid:{ color:"rgba(255,255,255,.07)" },
        angleLines:{ color:"rgba(255,255,255,.07)" },
      }},
      plugins:{
        legend:{display:false},
        tooltip:{
          backgroundColor:"#0d0f14", borderColor:"#1e2230", borderWidth:1,
          titleColor:"#d0d4e4", bodyColor:"#555c72", padding:10,
          callbacks:{ label: c => `  ${c.raw}% fake probability` },
        },
      },
      animation:{ duration:900, easing:"easeInOutQuart" },
    },
  });
}

// ── Signal breakdown ──────────────────────────────────────────────────────
function drawSignals(signals) {
  const grid = $("sigGrid");
  grid.innerHTML = "";

  const groups = {};
  Object.entries(SIGS).forEach(([key, meta]) => {
    if (!(key in signals)) return;
    (groups[meta.cat] = groups[meta.cat] || []).push({ key, label: meta.label, val: signals[key] });
  });

  Object.entries(groups).forEach(([cat, items]) => {
    const col = document.createElement("div");
    const catColor = CAT_COLORS[cat] || "#888";
    col.innerHTML = `<div class="sig-group-title ${cat}" style="color:${catColor}">${cat}</div>`;
    items.forEach(({ label, val }) => {
      const pct    = Math.round(val * 100);
      const isFake = val >= 0.45;
      const color  = score2color(val);
      const row = document.createElement("div");
      row.className = "sig-row";
      row.innerHTML = `
        <span class="sig-name">${label}</span>
        <div class="sig-bar-wrap"><div class="sig-bar" style="width:${pct}%;background:${color}"></div></div>
        <span class="sig-pct" style="color:${color}">${pct}%</span>
        <span class="sig-tag ${isFake ? 'fake' : 'real'}">${isFake ? 'FAKE' : 'REAL'}</span>
      `;
      col.appendChild(row);
    });
    grid.appendChild(col);
  });

  if (!grid.children.length)
    grid.innerHTML = `<p style="color:var(--muted);font-size:.84rem">No signal data available.</p>`;
}

// ── Video chart ───────────────────────────────────────────────────────────
function drawVideoChart(fake, real) {
  if (vChart) { vChart.destroy(); vChart = null; }
  if (!fake && !real) return;
  vChart = new Chart($("videoChart").getContext("2d"), {
    type: "doughnut",
    data: {
      labels: ["Fake Frames", "Real Frames"],
      datasets: [{
        data: [fake, real],
        backgroundColor: ["rgba(255,77,77,.8)", "rgba(61,255,170,.8)"],
        borderColor:     ["rgba(255,77,77,1)",  "rgba(61,255,170,1)"],
        borderWidth: 2, hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, cutout: "68%",
      plugins: {
        legend: { position:"bottom", labels:{ color:"#555c72", font:{family:"'Space Mono',monospace",size:10}, padding:18, usePointStyle:true } },
        tooltip: {
          backgroundColor:"#0d0f14", borderColor:"#1e2230", borderWidth:1,
          titleColor:"#d0d4e4", bodyColor:"#555c72", padding:12,
          callbacks:{ label: c => { const t=c.dataset.data.reduce((a,b)=>a+b,0); return `  ${c.parsed} frames (${t?Math.round(c.parsed/t*100):0}%)`; } },
        },
      },
      animation:{ animateRotate:true, duration:900 },
    },
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────
function score2color(v) {
  // 0 = real green, 0.5 = yellow, 1 = fake red
  const r = Math.round(61  + (255 - 61)  * v);
  const g = Math.round(255 + (77  - 255) * v);
  const b = Math.round(170 + (77  - 170) * v);
  return `rgb(${r},${g},${b})`;
}

function setProgress(pct) {
  $("progressBar").style.width = `${pct}%`;
}

function showErr(msg) {
  $("loading").style.display = "none";
  $("errText").textContent = msg;
  $("errPanel").style.display = "flex";
}

function resetUI() {
  $("loading").style.display = "none";
  $("results").style.display = "none";
  $("errPanel").style.display = "none";
  setProgress(0);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }