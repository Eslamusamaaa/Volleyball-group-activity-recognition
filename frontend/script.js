const API_BASE = ""; // same origin as this page when served by the backend

const form = document.getElementById("upload-form");
const dropzone = document.getElementById("dropzone");
const dropLabel = document.getElementById("drop-label");
const fileInput = document.getElementById("file-input");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const eventsListEl = document.getElementById("events-list");

let selectedFile = null;

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  dropLabel.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  dropzone.classList.add("has-file");
  submitBtn.disabled = false;
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

function showStatus(message, isError = false) {
  statusEl.hidden = false;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function renderEvents(events) {
  eventsListEl.innerHTML = "";
  if (events.length === 0) {
    eventsListEl.innerHTML = '<p style="color:var(--sub)">No events detected -- try a clip with a clearer view of the court.</p>';
    return;
  }
  for (const ev of events) {
    const row = document.createElement("div");
    row.className = "event";
    row.innerHTML = `
      <span class="time">${ev.start_sec}s - ${ev.end_sec}s</span>
      <span class="badge ${ev.team}">${ev.team}</span>
      <span class="label">${ev.event}</span>
      <span class="conf">${Math.round(ev.confidence * 100)}%</span>
    `;
    eventsListEl.appendChild(row);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  submitBtn.disabled = true;
  resultsEl.hidden = true;
  showStatus("Uploading and analyzing — this can take a little while for longer clips...");

  const body = new FormData();
  body.append("video", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed (${res.status})`);
    }
    const data = await res.json();
    showStatus(`Done — ${data.n_windows} windows analyzed at ${data.fps} fps.`);
    resultsEl.hidden = false;
    renderEvents(data.events);
  } catch (err) {
    showStatus(`Error: ${err.message}`, true);
  } finally {
    submitBtn.disabled = false;
  }
});
