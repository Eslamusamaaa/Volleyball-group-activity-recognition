// Analysis studio: upload / URL / library jobs, live pipeline tracker, results,
// timeline and charts. Talks to the existing FastAPI endpoints unchanged.
import { $, $$, esc, getJSON, apiUrl, authHeader, toast, fmtTime, fmtDuration, activityOf, ACTIVITY, tabGroup, icons } from "./util.js";
import * as charts from "./charts.js";

const els = {
  form: $("#upload-form"), dropzone: $("#dropzone"), dropLabel: $("#drop-label"), dropSub: $("#drop-sub"),
  fileInput: $("#file-input"), submit: $("#submit-btn"), previewBox: $("#preview-box"), previewVideo: $("#preview-video"),
  urlInput: $("#url-input"), panelFile: $("#panel-file"), panelUrl: $("#panel-url"),
  runName: $("#run-name"), runPct: $("#run-pct"), runBar: $("#run-bar"), runSub: $("#run-sub"), stages: $$(".stage"),
  runError: $("#run-error"), runErrorMsg: $("#run-error-msg"),
  results: $("#results-panel"), resultSub: $("#result-sub"), download: $("#download-btn"),
  vidWrap: $("#vid-wrap"), vidSource: $("#vid-source"), vidAi: $("#vid-ai"), src: $("#source-video"), ai: $("#annotated-video"),
  kpis: $("#kpis"), timeline: $("#timeline"), legend: $("#timeline-legend"), events: $("#events"),
  dist: $("#chart-dist"), conf: $("#chart-conf"),
};

let mode = "file";
let file = null;
let objectUrl = null;
let pollTimer = null;
let clock = null;
let busy = false;
let current = { events: [], duration: 0, view: "annotated" };

/* ---------- input handling ---------- */
function setMode(m) {
  mode = m;
  els.panelFile.hidden = m !== "file";
  els.panelUrl.hidden = m !== "url";
  refreshSubmit();
}
const refreshSubmit = () => { els.submit.disabled = busy || (mode === "file" ? !file : !els.urlInput.value.trim()); };

function setFile(f) {
  if (!f) return;
  if (!/\.(mp4|avi|mov|mkv|webm)$/i.test(f.name) && !f.type.startsWith("video/")) { toast("Please choose a video file (MP4, AVI, MOV, MKV or WEBM).", "error"); return; }
  if (f.size > 500 * 1024 * 1024) { toast("That file is larger than the 500 MB limit.", "error"); return; }
  file = f;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(f);
  els.dropLabel.textContent = f.name;
  els.dropSub.textContent = `${(f.size / 1048576).toFixed(1)} MB · click to replace`;
  els.dropzone.classList.add("has-file");
  els.previewVideo.src = objectUrl;
  els.previewBox.hidden = false;
  refreshSubmit();
}

/* ---------- live tracker ---------- */
import { signedIn, sessionExpired } from "./auth.js";

const STAGE_BY_PHASE = [[/render/i, 3], [/track|recogni|detect|extract|process/i, 1], [/upload|download|fetch|initiali|queue|prob/i, 0]];
function stageIndex(phase, status) {
  if (status === "done") return 4;
  for (const [re, i] of STAGE_BY_PHASE) if (re.test(phase || "")) return i;
  return 0;
}

function setRun({ name, pct, phase, status, sub, pctLabel }) {
  if (name != null) els.runName.textContent = name;
  els.runPct.textContent = pctLabel ?? `${Math.round(pct)}%`;
  $("i", els.runBar).style.width = `${pct}%`;
  els.runBar.classList.toggle("done", status === "done");
  if (sub != null) els.runSub.textContent = sub;
  const idx = stageIndex(phase, status);
  els.stages.forEach((s, i) => {
    s.classList.toggle("done", i < idx || (status === "done"));
    s.classList.toggle("active", i === idx && status !== "done");
  });
  const active = els.stages[idx];
  els.stages.forEach((s) => ($("small", s).textContent = ""));
  if (active && status !== "done") $("small", active).textContent = phase || "";
}

function resetRun() {
  clearInterval(pollTimer); clearInterval(clock); pollTimer = clock = null;
  busy = false; refreshSubmit();
  els.runError.hidden = true;
  setRun({ name: "No job running", pct: 0, phase: "", status: "idle", sub: "Idle — waiting for a video" });
  els.stages.forEach((s) => s.classList.remove("done", "active"));
}

function startClock() {
  const t0 = Date.now();
  clearInterval(clock);
  clock = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    els.runSub.textContent = `Elapsed ${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  }, 1000);
}

function fail(msg) {
  clearInterval(pollTimer); clearInterval(clock);
  busy = false; refreshSubmit();
  els.runError.hidden = false;
  els.runErrorMsg.textContent = msg || "Unknown error";
  els.runSub.textContent = "Stopped";
  els.stages.forEach((s) => s.classList.remove("active"));
}

/* ---------- job lifecycle ---------- */
function begin(name) {
  if (!signedIn()) throw new Error("Sign in with Google (button above) to run an analysis.");
  busy = true; refreshSubmit();
  els.runError.hidden = true;
  els.results.hidden = true;
  setRun({ name, pct: 0, phase: "Initializing", status: "queued", sub: "Starting…" });
  startClock();
  $(".studio").scrollIntoView({ behavior: "smooth", block: "start" });
}

function poll(jobId, name, sourceUrl) {
  clearInterval(pollTimer);
  let misses = 0;
  pollTimer = setInterval(async () => {
    try {
      const job = await getJSON(`/api/jobs/${jobId}`);
      misses = 0;
      setRun({ name, pct: job.progress || 0, phase: job.phase, status: job.status, sub: null });
      if (job.status === "done") { clearInterval(pollTimer); clearInterval(clock); busy = false; refreshSubmit(); showResults(job, sourceUrl); }
      else if (job.status === "error") fail(job.error);
    } catch (e) {
      if (e.status === 404) { resetRun(); return; }      // stale job after a server restart: reset quietly
      if (++misses > 8) fail("Lost connection to the analysis engine.");
    }
  }, 1200);
}

async function submit() {
  try {
    if (mode === "file") {
      begin(file.name);
      const data = await upload(file);
      poll(data.job_id, file.name, objectUrl);
    } else {
      const url = els.urlInput.value.trim();
      begin(url);
      const data = await getJSON("/api/analyze/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
      poll(data.job_id, data.title || url, null);
    }
  } catch (e) { if (e.status === 401) sessionExpired(); fail(e.message); }
}

function upload(f) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/analyze"));
    Object.entries(authHeader()).forEach(([k, v]) => xhr.setRequestHeader(k, v));
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = (e.loaded / e.total) * 100;
      setRun({ name: f.name, pct: p, phase: "Uploading", status: "queued", sub: null, pctLabel: `${Math.round(p)}%` });
      $("small", els.stages[0]).textContent = `Uploading ${Math.round(p)}%`;
    };
    xhr.onload = () => {
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error body */ }
      if (xhr.status === 401) sessionExpired();
      xhr.status >= 200 && xhr.status < 300 ? resolve(body) : reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error while uploading."));
    const fd = new FormData(); fd.append("video", f);
    xhr.send(fd);
  });
}

export async function analyzeLibrary(item) {
  if (busy) { toast("An analysis is already running.", ""); return; }
  try {
    begin(item.filename);
    const data = await getJSON("/api/library/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ category: item.category, filename: item.filename }) });
    poll(data.job_id, item.filename, item.stream_url);
  } catch (e) { if (e.status === 401) sessionExpired(); fail(e.message); }
}

/* ---------- results ---------- */
function showResults(job, sourceUrl) {
  const events = (job.events || []).slice().sort((a, b) => a.start_sec - b.start_sec);
  const duration = job.duration_sec || Math.max(0, ...events.map((e) => e.end_sec));
  current = { events, duration, view: "annotated" };
  els.results.hidden = false;

  const src = sourceUrl || (job.source_type === "url" && job.filename ? `/api/library/stream/from_internet/${encodeURIComponent(job.filename)}` : null);
  els.src.src = apiUrl(src) || "";
  els.ai.src = apiUrl(job.annotated_video_url) || "";
  els.download.hidden = !job.annotated_video_url;
  if (job.annotated_video_url) { els.download.href = apiUrl(job.annotated_video_url); els.download.download = `annotated_${job.filename || "match"}.mp4`; }
  setView("annotated");
  $$("[data-view]").forEach((b) => (b.hidden = b.dataset.view !== "annotated" && !src));

  const took = job.finished_at && job.started_at ? job.finished_at - job.started_at : null;
  els.resultSub.textContent = `${job.filename || "Video"} · ${fmtDuration(duration)} · ${Math.round(job.fps || 0)} fps`;
  renderKpis(events, duration, took);
  renderTimeline(events, duration);
  renderCharts(events);
  renderEventList(events);
  els.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderKpis(events, duration, took) {
  const n = (k) => events.filter((e) => e.event.toLowerCase() === k).length;
  const mean = events.length ? events.reduce((a, e) => a + e.confidence, 0) / events.length : 0;
  const k = [
    [events.length, "Events detected"],
    [fmtDuration(duration), "Video length"],
    [took ? fmtDuration(took) : "—", took && duration ? `Processing time (${(duration / took).toFixed(2)}× real time)` : "Processing time"],
    [events.length ? `${Math.round(mean * 100)}%` : "—", "Mean confidence"],
    [n("spike"), "Spikes"],
    [n("winpoint"), "Points won"],
  ];
  els.kpis.innerHTML = k.map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
}

function renderTimeline(events, duration) {
  if (!events.length) {
    els.timeline.innerHTML = `<div class="state">${icons.empty}<strong>No group activity detected</strong><span>Try footage with a clearer court-wide view. Predictions below 40% confidence are discarded.</span></div>`;
    els.legend.innerHTML = "";
    return;
  }
  const lane = (team) => `<div class="lane"><span class="lane-label" style="color:var(--team-${team})">${team}</span><div class="lane-track" data-lane="${team}">${
    events.map((e, i) => e.team.toLowerCase() === team ? seg(e, i, duration) : "").join("")}</div></div>`;
  const ticks = Array.from({ length: 6 }, (_, i) => `<span>${fmtTime((duration * i) / 5)}</span>`).join("");
  els.timeline.innerHTML = `<div style="position:relative">${lane("left")}${lane("right")}<div class="playhead" id="playhead" style="left:64px"></div></div>
    <div class="axis"><span></span><div class="axis-ticks">${ticks}</div></div>`;
  els.legend.innerHTML = Object.entries(ACTIVITY).map(([k, v]) => `<span><i style="--c:${v.color}"></i>${v.label}</span>`).join("");

  $$(".lane-track", els.timeline).forEach((track) => {
    track.addEventListener("click", (e) => {
      if (e.target.closest(".seg")) return;
      const r = track.getBoundingClientRect();
      seek(((e.clientX - r.left) / r.width) * duration);
    });
  });
  $$(".seg", els.timeline).forEach((s) => s.addEventListener("click", () => selectEvent(+s.dataset.i, true)));
}

function seg(e, i, duration) {
  const a = activityOf(e.event);
  const left = (e.start_sec / duration) * 100, w = Math.max(0.4, ((e.end_sec - e.start_sec) / duration) * 100);
  const tip = `${a.label} · ${e.team} team · ${fmtTime(e.start_sec)}–${fmtTime(e.end_sec)} · ${Math.round(e.confidence * 100)}% confidence`;
  return `<button class="seg" data-i="${i}" style="--c:${a.color}; left:${left}%; width:${w}%" data-tip="${esc(tip)}" aria-label="${esc(tip)}"></button>`;
}

function renderCharts(events) {
  if (!events.length) { els.dist.innerHTML = els.conf.innerHTML = `<div class="state">${icons.empty}<span>No events to chart.</span></div>`; return; }
  const items = Object.entries(ACTIVITY).map(([k, v]) => {
    const ev = events.filter((e) => e.event.toLowerCase() === k);
    const l = ev.filter((e) => e.team.toLowerCase() === "left").length, r = ev.length - l;
    return { label: v.label, value: ev.length, segments: [{ value: l, color: "var(--team-left)", tip: `${v.label} · left team: ${l}` }, { value: r, color: "var(--team-right)", tip: `${v.label} · right team: ${r}` }], tip: `${v.label}: ${ev.length} events (${l} left · ${r} right)` };
  });
  els.dist.innerHTML = charts.hbars(items, { max: Math.max(...items.map((i) => i.value), 1), labelW: 90 }) +
    `<div class="chart-legend"><span><i style="--c:var(--team-left)"></i>Left team</span><span><i style="--c:var(--team-right)"></i>Right team</span></div>`;
  els.conf.innerHTML = charts.vbars(events.map((e) => ({ value: e.confidence, color: activityOf(e.event).color, tip: `#${events.indexOf(e) + 1} ${activityOf(e.event).label} · ${e.team} · ${Math.round(e.confidence * 100)}%` })), { ref: 0.4, refLabel: "merge threshold 40%" });
  $$(".vbar", els.conf).forEach((b) => b.addEventListener("click", () => selectEvent(+b.dataset.idx, true)));
}

function renderEventList(events) {
  if (!events.length) { els.events.innerHTML = ""; return; }
  els.events.innerHTML = events.map((e, i) => {
    const a = activityOf(e.event), team = e.team.toLowerCase();
    return `<button class="ev" data-i="${i}" style="--c:${a.color}"><span class="t">${fmtTime(e.start_sec)} – ${fmtTime(e.end_sec)}</span><span class="team ${team}">${team}</span>
      <span class="nm"><i></i>${esc(a.label)}</span><span class="conf"><span class="meter" style="--c:${a.color}"><i style="width:${e.confidence * 100}%"></i></span>${Math.round(e.confidence * 100)}%</span></button>`;
  }).join("");
  $$(".ev", els.events).forEach((b) => b.addEventListener("click", () => selectEvent(+b.dataset.i, true)));
}

function selectEvent(i, play) {
  const e = current.events[i];
  if (!e) return;
  $$(".seg.sel, .ev.sel").forEach((x) => x.classList.remove("sel"));
  $$(`[data-i="${i}"]`, els.timeline).forEach((x) => x.classList.add("sel"));
  const row = $(`.ev[data-i="${i}"]`, els.events);
  row?.classList.add("sel");
  row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  seek(e.start_sec, play);
}

function visibleVideos() {
  const v = current.view;
  return v === "split" ? [els.ai, els.src] : v === "source" ? [els.src] : [els.ai];
}

function seek(t, play = false) {
  const vids = visibleVideos();
  vids.forEach((v) => { if (v.readyState || v.src) { try { v.currentTime = t; } catch { /* metadata not ready */ } } });
  if (play) vids.forEach((v) => v.play().catch(() => {}));
  updatePlayhead();
}

function setView(view) {
  current.view = view;
  els.vidSource.hidden = view === "annotated";
  els.vidAi.hidden = view === "source";
  els.vidWrap.classList.toggle("split-view", view === "split");
  $$("[data-view]").forEach((b) => { const on = b.dataset.view === view; b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on)); });
  els.src.pause(); els.ai.pause();
}

function updatePlayhead() {
  const ph = $("#playhead");
  if (!ph || !current.duration) return;
  const v = visibleVideos()[0];
  const track = $(".lane-track", els.timeline);
  if (!track) return;
  const frac = Math.min(1, (v.currentTime || 0) / current.duration);
  ph.style.left = `${track.offsetLeft + frac * track.offsetWidth}px`;
  // Highlight the active event while playing.
  const t = v.currentTime || 0;
  const idx = current.events.findIndex((e) => t >= e.start_sec && t <= e.end_sec);
  $$(".ev.active-now").forEach((x) => x.classList.remove("active-now"));
  if (idx >= 0) $(`.ev[data-i="${idx}"]`, els.events)?.classList.add("active-now");
}

/* ---------- wiring ---------- */
export function initStudio() {
  tabGroup($(".studio .tabs"), (t) => setMode(t.id === "tab-url" ? "url" : "file"));
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => setFile(els.fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag-over"); }));
  ["dragleave", "drop"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag-over"); }));
  els.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  els.urlInput.addEventListener("input", refreshSubmit);
  els.form.addEventListener("submit", (e) => { e.preventDefault(); if (!els.submit.disabled) submit(); });
  $$("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  ["timeupdate", "seeked", "play"].forEach((ev) => els.ai.addEventListener(ev, updatePlayhead));
  els.src.addEventListener("timeupdate", updatePlayhead);
  let raf;
  els.ai.addEventListener("play", () => { const loop = () => { updatePlayhead(); if (!els.ai.paused) raf = requestAnimationFrame(loop); }; loop(); });
  els.ai.addEventListener("pause", () => cancelAnimationFrame(raf));
  addEventListener("resize", updatePlayhead);
}
