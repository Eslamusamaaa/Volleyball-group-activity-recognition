// Match library + YouTube importer.
import { $, $$, esc, getJSON, toast, fmtDuration, tabGroup, icons } from "./util.js";
import { analyzeLibrary } from "./studio.js";

let data = { internet_matches: [], dataset_plays: [] };
let tab = "internet";
const grid = $("#lib-grid");
const dlg = $("#preview-dialog");

export async function loadLibrary() {
  grid.innerHTML = Array.from({ length: 3 }, () => `<div class="skeleton" style="min-height:150px"></div>`).join("");
  try {
    data = await getJSON("/api/library");
    $("#count-internet").textContent = data.internet_matches.length;
    $("#count-dataset").textContent = data.dataset_plays.length;
    render();
  } catch {
    grid.innerHTML = `<div class="state error" style="grid-column:1/-1">${icons.error}<strong>Library unavailable</strong><span>The backend did not respond. Start it with <code>uvicorn backend.app:app</code>.</span></div>`;
  }
}

function render() {
  const items = tab === "internet" ? data.internet_matches : data.dataset_plays;
  if (!items.length) {
    grid.innerHTML = `<div class="state" style="grid-column:1/-1">${icons.empty}<strong>No footage in this category yet</strong><span>${tab === "internet" ? "Import a match from YouTube above." : "Dataset rallies appear here once placed in production Data/from our dataset."}</span></div>`;
    return;
  }
  grid.innerHTML = items.map((it, i) => {
    const isNew = it.mtime && Date.now() / 1000 - it.mtime < 1800;
    return `<article class="lib">
      <h4>${esc(it.title)}${isNew ? ' <span class="new">NEW</span>' : ""}</h4>
      <div class="meta"><span class="chip">${fmtDuration(it.duration_sec)}</span><span class="chip">${it.size_mb} MB</span><span class="chip">${it.fps} fps</span></div>
      <div class="actions"><button class="btn btn-primary btn-sm" data-act="analyze" data-i="${i}">Analyse</button><button class="btn btn-sm" data-act="preview" data-i="${i}">Preview</button></div>
    </article>`;
  }).join("");
  $$("button[data-act]", grid).forEach((b) => b.addEventListener("click", () => {
    const it = items[+b.dataset.i];
    if (b.dataset.act === "analyze") analyzeLibrary(it); else preview(it);
  }));
}

function preview(it) {
  $("#preview-title").textContent = it.title;
  const v = $("#preview-dialog-video");
  v.src = it.stream_url;
  dlg.showModal();
  v.play().catch(() => {});
}

/* ---------- YouTube importer ---------- */
function initImporter() {
  const form = $("#yt-form"), btn = $("#yt-btn"), box = $("#yt-progress");
  const set = (id, v) => ($(id).textContent = v);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#yt-input").value.trim();
    if (!input) return;
    btn.disabled = true; box.hidden = false;
    $("#yt-bar").style.width = "0%"; set("#yt-pct", "0%"); set("#yt-phase", "Connecting to source…"); set("#yt-speed", ""); set("#yt-eta", "");
    const dur = parseInt($("#yt-duration").value, 10);
    const body = { [/^https?:/i.test(input) ? "url" : "query"]: input, max_duration_sec: dur > 0 ? dur : null };
    try {
      const { download_id } = await getJSON("/api/youtube/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const timer = setInterval(async () => {
        try {
          const p = await getJSON(`/api/youtube/progress/${download_id}`);
          const pct = Math.min(100, Math.max(0, p.percent || 0));
          $("#yt-bar").style.width = `${pct}%`; set("#yt-pct", `${pct.toFixed(0)}%`);
          set("#yt-speed", p.speed ? `Speed ${p.speed}` : ""); set("#yt-eta", p.eta ? `ETA ${p.eta}` : "");
          if (p.status === "downloading") set("#yt-phase", "Downloading…");
          else if (p.status === "processing") set("#yt-phase", "Merging video and audio…");
          else if (p.status === "finished") {
            clearInterval(timer); btn.disabled = false; $("#yt-input").value = "";
            set("#yt-phase", `Added: ${p.title || p.filename}`);
            toast("Match added to the library.", "ok");
            await loadLibrary();
            tab = "internet"; render();
          } else if (p.status === "error") { clearInterval(timer); btn.disabled = false; set("#yt-phase", `Download failed: ${p.error || "unknown error"}`); toast("Download failed.", "error"); }
        } catch { /* transient poll error */ }
      }, 700);
    } catch (err) { btn.disabled = false; box.hidden = true; toast(err.message, "error"); }
  });
}

export function initLibrary() {
  tabGroup($("#lib-tab-internet").parentElement, (t) => { tab = t.id === "lib-tab-dataset" ? "dataset" : "internet"; render(); });
  $("#preview-close").addEventListener("click", () => dlg.close());
  dlg.addEventListener("close", () => { const v = $("#preview-dialog-video"); v.pause(); v.removeAttribute("src"); v.load(); });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  initImporter();
  loadLibrary();
}
