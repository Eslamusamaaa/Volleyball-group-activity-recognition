// Verified sample viewer: a test-split frame, the pipeline's group-activity decision,
// and a card per tracked player. Data: data/samples.json (scripts/build_sample_data.py),
// where every player was matched to a dataset ground-truth box by IoU >= 0.5.
import { $, esc, getJSON, tabGroup, pct } from "./util.js";

const nice = (s) => s.replace(/^([lr])[-_]/, (_, t) => (t === "l" ? "left " : "right "));

function render(box, s, kind) {
  const checked = s.players.filter((p) => p.correct !== null);
  const right = checked.filter((p) => p.correct).length;
  const hasClip = !!s.clip_video;
  const title = kind === "group" ? `Group activity: ${nice(s.gt_scene)}` : `Player action: ${s.label}`;
  box.innerHTML = `
    <div class="sv">
      <div class="sv-media">
        ${hasClip ? `<div class="tabs sv-switch" role="group" aria-label="Frame or clip"><button class="tab active" data-m="img" aria-pressed="true">Frame</button><button class="tab" data-m="vid" aria-pressed="false">Clip</button></div>` : ""}
        <img src="${s.image}" alt="${esc(title)} — test frame with tracked players and predicted actions" width="${s.width}" height="${s.height}" loading="lazy">
        ${hasClip ? `<video src="${s.clip_video}" controls muted loop playsinline preload="none" hidden aria-label="Annotated clip for ${esc(title)}"></video>` : ""}
      </div>
      <div class="sv-meta">
        <span class="verdict">✓ group activity matches ground truth</span>
        <div class="kv">
          <div><span>Ground truth</span><span>${esc(s.gt_scene)}</span></div>
          <div><span>Predicted</span><span>${esc(s.pred_scene)}</span></div>
          <div><span>Confidence</span><span>${pct(s.scene_conf, 1)}</span></div>
          <div><span>Source</span><span>test · video ${s.video} · clip ${esc(s.clip)}</span></div>
          <div><span>Players checked</span><span>${right} / ${checked.length} correct</span></div>
        </div>
        <p class="note" style="margin-top:0">${hasClip ? "The clip is the pipeline's full annotated output for this 41-frame sequence, slowed to 8 fps. " : ""}Tracked IDs (#) are assigned by ByteTrack.</p>
      </div>
    </div>
  `;

  if (hasClip) {
    const img = $("img", box), vid = $("video", box);
    box.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => {
      const v = b.dataset.m === "vid";
      img.hidden = v; vid.hidden = !v;
      box.querySelectorAll("[data-m]").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-pressed", String(x === b)); });
      v ? vid.play().catch(() => {}) : vid.pause();
    }));
  }
}

function mount(tabsEl, viewEl, items, kind, tabLabel) {
  if (!items.length) { viewEl.innerHTML = `<div class="state">No verified samples available.</div>`; return; }
  tabsEl.innerHTML = items.map((s, i) => `<button class="tab" role="tab" data-i="${i}" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}">${esc(tabLabel(s))}</button>`).join("");
  tabGroup(tabsEl, (t) => render(viewEl, items[+t.dataset.i], kind));
  render(viewEl, items[0], kind);
}

export async function initSamples() {
  let data;
  try { data = await getJSON("data/samples.json"); }
  catch { $("#grp-view").innerHTML = $("#ply-view").innerHTML = `<div class="state error">Sample data could not be loaded.</div>`; return; }
  mount($("#grp-tabs"), $("#grp-view"), data.group, "group", (s) => s.gt_scene);
  mount($("#ply-tabs"), $("#ply-view"), data.player, "player", (s) => s.label);
}
