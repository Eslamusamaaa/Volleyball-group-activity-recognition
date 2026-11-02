// VolleyAI Studio Frontend Platform Controller
const API_BASE = "";

// DOM Elements
const engineStatusEl = document.getElementById("engine-status");
const form = document.getElementById("upload-form");
const dropzone = document.getElementById("dropzone");
const dropLabel = document.getElementById("drop-label");
const fileInput = document.getElementById("file-input");
const submitBtn = document.getElementById("submit-btn");

const tabModeFile = document.getElementById("tab-mode-file");
const tabModeUrl = document.getElementById("tab-mode-url");
const urlZone = document.getElementById("url-zone");
const directUrlInput = document.getElementById("direct-url-input");
let studioMode = "file"; // "file" or "url"

const progressCard = document.getElementById("progress-card");
const progressPhaseEl = document.getElementById("progress-phase");
const progressPercentEl = document.getElementById("progress-percent");
const progressBarEl = document.getElementById("progress-bar");
const progressFilenameEl = document.getElementById("progress-filename");
const progressStatusMsgEl = document.getElementById("progress-status-msg");

const errorCard = document.getElementById("error-card");
const errorMsgEl = document.getElementById("error-message");

const resultsContainer = document.getElementById("results");
const videoWorkspace = document.getElementById("video-workspace");
const sourceVideo = document.getElementById("source-video");
const annotatedVideo = document.getElementById("annotated-video");
const downloadBtn = document.getElementById("btn-download-video");

// View Toggle Buttons
const btnViewAnnotated = document.getElementById("btn-view-annotated");
const btnViewSplit = document.getElementById("btn-view-split");
const btnViewSource = document.getElementById("btn-view-source");

// Analytics Stats
const statTotalPlays = document.getElementById("stat-total-plays");
const statDuration = document.getElementById("stat-duration");
const statSpikes = document.getElementById("stat-spikes");
const statSets = document.getElementById("stat-sets");
const statPasses = document.getElementById("stat-passes");
const statPoints = document.getElementById("stat-points");
const eventsListEl = document.getElementById("events-list");

// Library Elements
const libraryGrid = document.getElementById("library-grid");
const tabInternet = document.getElementById("tab-internet");
const tabDataset = document.getElementById("tab-dataset");
const countInternet = document.getElementById("count-internet");
const countDataset = document.getElementById("count-dataset");

const ytForm = document.getElementById("yt-form");
const ytUrlInput = document.getElementById("yt-url-input");
const ytSubmitBtn = document.getElementById("yt-submit-btn");
const ytStatus = document.getElementById("yt-status");

let selectedFile = null;
let currentJobId = null;
let pollTimer = null;
let libraryData = { internet_matches: [], dataset_plays: [] };
let activeLibraryTab = "internet"; // "internet" or "dataset"

// Init
document.addEventListener("DOMContentLoaded", () => {
  hideError();
  checkHealth();
  loadLibrary();
  setupViewToggles();

  const errorCloseBtn = document.getElementById("error-close");
  if (errorCloseBtn) {
    errorCloseBtn.addEventListener("click", hideError);
  }
});

// Check System Health
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      const dev = data.device ? data.device.toUpperCase() : "CPU";
      engineStatusEl.textContent = `Engine Online (${dev})`;
    } else {
      engineStatusEl.textContent = "Engine Initializing";
    }
  } catch (e) {
    engineStatusEl.textContent = "Engine Offline";
  }
}

// File Selection
function setFile(file) {
  if (!file) return;
  selectedFile = file;
  dropLabel.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  dropzone.classList.add("has-file");
  submitBtn.disabled = false;

  // Preview source video immediately
  const objectUrl = URL.createObjectURL(file);
  sourceVideo.src = objectUrl;
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
dropzone.addEventListener("click", () => fileInput.click());

// Studio Mode Toggles
if (tabModeFile && tabModeUrl) {
  tabModeFile.addEventListener("click", () => {
    studioMode = "file";
    tabModeFile.classList.add("active");
    tabModeUrl.classList.remove("active");
    dropzone.hidden = false;
    urlZone.hidden = true;
    submitBtn.disabled = !selectedFile;
  });

  tabModeUrl.addEventListener("click", () => {
    studioMode = "url";
    tabModeUrl.classList.add("active");
    tabModeFile.classList.remove("active");
    dropzone.hidden = true;
    urlZone.hidden = false;
    submitBtn.disabled = !directUrlInput.value.trim();
  });
}

if (directUrlInput) {
  directUrlInput.addEventListener("input", () => {
    if (studioMode === "url") {
      submitBtn.disabled = !directUrlInput.value.trim();
    }
  });
}

// Form Submit: Upload Video OR Analyze URL
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  resultsContainer.hidden = true;

  if (studioMode === "file") {
    if (!selectedFile) return;
    submitBtn.disabled = true;
    showProgress("Uploading match footage...", 5, selectedFile.name);

    const formData = new FormData();
    formData.append("video", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed with status ${res.status}`);
      }
      const data = await res.json();
      currentJobId = data.job_id;
      startPolling(currentJobId, selectedFile.name);
    } catch (err) {
      showError(err.message);
      hideProgress();
      submitBtn.disabled = false;
    }
  } else {
    const url = directUrlInput.value.trim();
    if (!url) return;
    submitBtn.disabled = true;
    showProgress("Fetching match footage from URL...", 5, url);

    try {
      const res = await fetch(`${API_BASE}/api/analyze/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with status ${res.status}`);
      }
      const data = await res.json();
      currentJobId = data.job_id;
      startPolling(currentJobId, data.title || url);
    } catch (err) {
      showError(err.message);
      hideProgress();
      submitBtn.disabled = false;
    }
  }
});

// Polling Job Status with graceful retry
function startPolling(jobId, filename) {
  if (pollTimer) clearInterval(pollTimer);
  hideError();

  let retryCount = 0;
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
      if (!res.ok) {
        if (res.status === 404) {
          // Stale job or server restart — cleanly stop polling without throwing error banner
          clearInterval(pollTimer);
          pollTimer = null;
          hideProgress();
          hideError();
          submitBtn.disabled = false;
          return;
        }
        retryCount++;
        if (retryCount > 8) {
          clearInterval(pollTimer);
          pollTimer = null;
          hideProgress();
          submitBtn.disabled = false;
          return;
        }
        return;
      }
      retryCount = 0; // reset counter on successful fetch
      const job = await res.json();

      updateProgress(job.progress || 0, job.phase || "Processing...", filename);

      if (job.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        onJobComplete(job);
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        showError(`Analysis failed: ${job.error || "Unknown error"}`);
        hideProgress();
        submitBtn.disabled = false;
      }
    } catch (err) {
      console.warn("Polling retry...", err);
      retryCount++;
      if (retryCount > 8) {
        clearInterval(pollTimer);
        pollTimer = null;
        hideProgress();
        submitBtn.disabled = false;
      }
    }
  }, 1200);
}

function showProgress(phase, percent, filename) {
  progressCard.hidden = false;
  updateProgress(percent, phase, filename);
}

function updateProgress(percent, phase, filename) {
  progressPhaseEl.textContent = phase;
  progressPercentEl.textContent = `${percent}%`;
  progressBarEl.style.width = `${percent}%`;
  progressFilenameEl.textContent = filename || "Video";
  progressStatusMsgEl.textContent = `${phase} (${percent}%)`;
}

function hideProgress() {
  progressCard.hidden = true;
}

function showError(msg) {
  errorCard.hidden = false;
  errorMsgEl.textContent = msg;
}

function hideError() {
  errorCard.hidden = true;
}

// Job Completion: Populate UI
function onJobComplete(job) {
  hideProgress();
  submitBtn.disabled = false;
  resultsContainer.hidden = false;

  // Set up video players
  if (job.annotated_video_url) {
    annotatedVideo.src = job.annotated_video_url;
    downloadBtn.href = job.annotated_video_url;
    downloadBtn.download = `annotated_${job.filename || "match"}.mp4`;
    downloadBtn.hidden = false;
  }

  // Populate Analytics Metrics
  const events = job.events || [];
  statTotalPlays.textContent = events.length;
  statDuration.textContent = job.duration_sec ? `${job.duration_sec}s` : "0.0s";

  let spikes = 0, sets = 0, passes = 0, points = 0;
  events.forEach((e) => {
    const act = (e.event || "").toLowerCase();
    if (act.includes("spike")) spikes++;
    else if (act.includes("set")) sets++;
    else if (act.includes("pass")) passes++;
    else if (act.includes("winpoint") || act.includes("point")) points++;
  });

  statSpikes.textContent = spikes;
  statSets.textContent = sets;
  statPasses.textContent = passes;
  statPoints.textContent = points;

  // Render Event Timeline
  renderTimeline(events);

  // Scroll to results
  resultsContainer.scrollIntoView({ behavior: "smooth" });
}

// Render Interactive Timeline
function renderTimeline(events) {
  eventsListEl.innerHTML = "";
  if (!events || events.length === 0) {
    eventsListEl.innerHTML = `<p style="color:var(--text-muted); padding:16px;">No decisive group activities detected. Try a match clip with a clearer court-wide perspective.</p>`;
    return;
  }

  events.forEach((ev, idx) => {
    const row = document.createElement("div");
    row.className = "event-row";

    const isPoint = ev.event.toLowerCase().includes("winpoint");
    const actClass = isPoint ? "winpoint" : ev.event.toLowerCase();
    const actText = isPoint ? "★ Point Won ★" : ev.event.toUpperCase();
    const teamClass = (ev.team || "left").toLowerCase();
    const teamText = `${ev.team.toUpperCase()} TEAM`;

    row.innerHTML = `
      <div class="event-left">
        <button type="button" class="event-time-btn">
          ▶ ${ev.start_sec.toFixed(1)}s - ${ev.end_sec.toFixed(1)}s
        </button>
        <span class="event-team-pill ${teamClass}">${teamText}</span>
        <span class="event-action ${actClass}">${actText}</span>
      </div>
      <div class="event-right">
        <span class="event-conf">${Math.round(ev.confidence * 100)}% Conf</span>
      </div>
    `;

    // Click to seek video players
    row.addEventListener("click", () => {
      seekBothVideos(ev.start_sec);
    });

    eventsListEl.appendChild(row);
  });
}

function seekBothVideos(timeSec) {
  if (annotatedVideo) {
    annotatedVideo.currentTime = timeSec;
    annotatedVideo.play().catch(() => {});
  }
  if (sourceVideo) {
    sourceVideo.currentTime = timeSec;
    sourceVideo.play().catch(() => {});
  }
}

// View Toggles (Annotated / Split / Source)
function setupViewToggles() {
  btnViewAnnotated.addEventListener("click", () => {
    setViewMode("annotated");
  });
  btnViewSplit.addEventListener("click", () => {
    setViewMode("split");
  });
  btnViewSource.addEventListener("click", () => {
    setViewMode("source");
  });
}

function setViewMode(mode) {
  [btnViewAnnotated, btnViewSplit, btnViewSource].forEach((b) => b.classList.remove("active"));
  videoWorkspace.className = `video-workspace mode-${mode}`;

  if (mode === "annotated") btnViewAnnotated.classList.add("active");
  else if (mode === "split") btnViewSplit.classList.add("active");
  else if (mode === "source") btnViewSource.classList.add("active");
}

// Match Library Handling
async function loadLibrary() {
  try {
    const res = await fetch(`${API_BASE}/api/library`);
    if (!res.ok) throw new Error("Could not load library");
    libraryData = await res.json();

    countInternet.textContent = libraryData.internet_matches.length;
    countDataset.textContent = libraryData.dataset_plays.length;

    renderLibraryGrid();
  } catch (e) {
    libraryGrid.innerHTML = `<p style="color:var(--text-muted); padding:20px;">Library unavailable.</p>`;
  }
}

tabInternet.addEventListener("click", () => {
  activeLibraryTab = "internet";
  tabInternet.classList.add("active");
  tabDataset.classList.remove("active");
  renderLibraryGrid();
});

tabDataset.addEventListener("click", () => {
  activeLibraryTab = "dataset";
  tabDataset.classList.add("active");
  tabInternet.classList.remove("active");
  renderLibraryGrid();
});

function renderLibraryGrid() {
  const items = (activeLibraryTab === "internet")
    ? libraryData.internet_matches
    : libraryData.dataset_plays;

  libraryGrid.innerHTML = "";
  if (!items || items.length === 0) {
    libraryGrid.innerHTML = `<p style="color:var(--text-muted); padding:20px;">No match footage found in this category.</p>`;
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "lib-card";

    const durMins = Math.floor(item.duration_sec / 60);
    const durSecs = Math.floor(item.duration_sec % 60);
    const durStr = durMins > 0 ? `${durMins}m ${durSecs}s` : `${item.duration_sec.toFixed(1)}s`;

    // Highlight items downloaded recently (within last 30 minutes)
    const isNew = item.mtime && (Date.now() / 1000 - item.mtime) < 1800;
    const newBadge = isNew ? `<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 700; margin-left: 6px;">✨ NEW</span>` : "";

    card.innerHTML = `
      <div class="lib-card-top">
        <h5>${item.title} ${newBadge}</h5>
        <div class="lib-meta">
          <span>⏱ ${durStr}</span>
          <span>💾 ${item.size_mb} MB</span>
          <span>⚡ ${item.fps} FPS</span>
        </div>
      </div>
      <div class="lib-actions">
        <button type="button" class="lib-btn primary btn-analyze-lib">
          ⚡ Analyze
        </button>
        <button type="button" class="lib-btn secondary btn-preview-lib">
          ▶ Preview
        </button>
      </div>
    `;

    // Analyze click
    card.querySelector(".btn-analyze-lib").addEventListener("click", () => {
      analyzeLibraryItem(item);
    });

    // Preview click
    card.querySelector(".btn-preview-lib").addEventListener("click", () => {
      sourceVideo.src = item.stream_url;
      resultsContainer.hidden = false;
      setViewMode("source");
      sourceVideo.play().catch(() => {});
      sourceVideo.scrollIntoView({ behavior: "smooth" });
    });

    libraryGrid.appendChild(card);
  });
}

// Analyze Library Item directly
async function analyzeLibraryItem(item) {
  hideError();
  submitBtn.disabled = true;
  resultsContainer.hidden = true;
  showProgress("Initializing library video analysis...", 5, item.filename);

  sourceVideo.src = item.stream_url;

  try {
    const res = await fetch(`${API_BASE}/api/library/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        category: item.category,
        filename: item.filename,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to start library analysis");
    }
    const data = await res.json();
    currentJobId = data.job_id;
    startPolling(currentJobId, item.filename);
  } catch (err) {
    showError(err.message);
    hideProgress();
    submitBtn.disabled = false;
  }
}

const ytProgressContainer = document.getElementById("yt-progress-container");
const ytProgressPhase = document.getElementById("yt-progress-phase");
const ytProgressPercent = document.getElementById("yt-progress-percent");
const ytProgressBar = document.getElementById("yt-progress-bar");
const ytProgressSpeed = document.getElementById("yt-progress-speed");
const ytProgressEta = document.getElementById("yt-progress-eta");

// YouTube Downloader Form with Real-Time Progress Bar
ytForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = ytUrlInput.value.trim();
  if (!input) return;

  ytSubmitBtn.disabled = true;
  ytStatus.hidden = true;
  if (ytProgressContainer) ytProgressContainer.hidden = false;
  if (ytProgressBar) ytProgressBar.style.width = "0%";
  if (ytProgressPercent) ytProgressPercent.textContent = "0%";
  if (ytProgressPhase) ytProgressPhase.textContent = "Connecting to YouTube stream...";
  if (ytProgressSpeed) ytProgressSpeed.textContent = "⚡ Initializing...";
  if (ytProgressEta) ytProgressEta.textContent = "⏱ Estimating...";

  const payload = input.startsWith("http") ? { url: input } : { query: input };

  try {
    const res = await fetch(`${API_BASE}/api/youtube/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("YouTube download request failed");
    const data = await res.json();
    const downloadId = data.download_id;

    // Poll Real-Time Progress from Backend
    const pollInterval = setInterval(async () => {
      try {
        const pRes = await fetch(`${API_BASE}/api/youtube/progress/${downloadId}`);
        if (!pRes.ok) return;
        const pData = await pRes.json();

        const pct = Math.min(100, Math.max(0, pData.percent || 0));
        if (ytProgressBar) ytProgressBar.style.width = `${pct}%`;
        if (ytProgressPercent) ytProgressPercent.textContent = `${pct.toFixed(0)}%`;
        if (ytProgressSpeed) ytProgressSpeed.textContent = `⚡ Speed: ${pData.speed || "--"}`;
        if (ytProgressEta) ytProgressEta.textContent = `⏱ ETA: ${pData.eta || "--"}`;

        if (pData.status === "downloading") {
          if (ytProgressPhase) ytProgressPhase.textContent = `Downloading: ${pct.toFixed(1)}%`;
        } else if (pData.status === "processing") {
          if (ytProgressPhase) ytProgressPhase.textContent = "Merging H.264 video & audio stream...";
        } else if (pData.status === "finished") {
          clearInterval(pollInterval);
          if (ytProgressBar) ytProgressBar.style.width = "100%";
          if (ytProgressPercent) ytProgressPercent.textContent = "100%";
          if (ytProgressPhase) {
            ytProgressPhase.innerHTML = `
              <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                <span>✔ <strong>${pData.title || pData.filename}</strong> added to library!</span>
                <button type="button" class="btn primary-btn btn-analyze-now-yt" style="padding: 6px 14px; font-size: 13px;">⚡ Analyze This Video Now</button>
              </div>
            `;
            const analyzeBtn = ytProgressPhase.querySelector(".btn-analyze-now-yt");
            if (analyzeBtn) {
              analyzeBtn.addEventListener("click", () => {
                analyzeLibraryItem({
                  category: "from internet",
                  filename: pData.filename,
                  stream_url: `/api/library/stream/from_internet/${pData.filename}`
                });
              });
            }
          }
          ytUrlInput.value = "";
          ytSubmitBtn.disabled = false;
          loadLibrary();
        } else if (pData.status === "error") {
          clearInterval(pollInterval);
          if (ytProgressPhase) ytProgressPhase.textContent = `❌ Error: ${pData.error || "Download failed"}`;
          ytSubmitBtn.disabled = false;
        }
      } catch (err) {
        console.warn("Download poll retry...", err);
      }
    }, 700);

  } catch (err) {
    if (ytProgressContainer) ytProgressContainer.hidden = true;
    ytStatus.hidden = false;
    ytStatus.textContent = `Error: ${err.message}`;
    ytSubmitBtn.disabled = false;
  }
});
