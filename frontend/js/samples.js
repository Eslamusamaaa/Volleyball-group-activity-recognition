// Verified sample viewer: a test-split frame, the pipeline's group-activity decision,
// and a card per tracked player. Data: data/samples.json (scripts/build_sample_data.py),
// where every player was matched to a dataset ground-truth box by IoU >= 0.5.
import { $, esc, getJSON, tabGroup, pct } from "./util.js";

const nice = (s) => s.replace(/^([lr])[-_]/, (_, t) => (t === "l" ? "left " : "right "));

function render(box, s, kind) {
  const checked = s.players.filter((p) => p.correct !== null);
  const right = checked.filter((p) => p.correct).length;
  const title = kind === "group" ? `Group activity: ${nice(s.gt_scene)}` : `Player action: ${s.label}`;
  box.innerHTML = `
    <div class="sv">
      <div class="sv-media">
        <img src="${s.image}" alt="${esc(title)} — test frame with tracked players and predicted actions" width="${s.width}" height="${s.height}" loading="lazy">
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
        <p class="note" style="margin-top:0">Tracked IDs (#) are assigned by ByteTrack.</p>
      </div>
    </div>
  `;
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
