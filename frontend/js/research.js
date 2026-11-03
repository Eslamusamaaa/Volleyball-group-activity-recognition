// Research sections: classes, dataset, baseline ladder, results. All numbers come
// from data/research.json (built by scripts/build_frontend_data.py from outputs/results).
import { $, $$, esc, getJSON, tabGroup, pct } from "./util.js";
import * as charts from "./charts.js";

const SCENE_CLASSES = ["l-pass", "l-spike", "l_set", "l_winpoint", "r-pass", "r_set", "r_spike", "r_winpoint"];
const ACTION_CLASSES = ["blocking", "digging", "falling", "jumping", "moving", "setting", "spiking", "standing", "waiting"];

const GLOSSARY = {
  "l-pass": "Left team receives / passes", "l-spike": "Left team attacks", "l_set": "Left team sets",
  "l_winpoint": "Left team wins the point", "r-pass": "Right team receives / passes", "r_set": "Right team sets",
  "r_spike": "Right team attacks", "r_winpoint": "Right team wins the point",
  blocking: "Jumping at the net to block", digging: "Defensive floor save", falling: "Falling or diving to the floor",
  jumping: "Airborne without attacking or blocking", moving: "Running / repositioning", setting: "Overhead set",
  spiking: "Attack hit", standing: "Standing in place", waiting: "Ready stance awaiting play",
};

// What each rung isolates (README ablation table) and what the experiment found (RESULTS.md).
const NOTES = {
  "B1": { asks: "Baseline: how far does a fine-tuned whole-image CNN go, with no players and no time?", found: "Most errors are pass↔set for the same team — adjacent phases look alike in a single frame." },
  "B2": { asks: "B1 vs B2/B3: do individual players help? B2 vs B3: does fine-tuning matter?", found: "Frozen ImageNet features: the small classifier head had not converged in 15 epochs. Kept as an ablation control, 22.4 pts under the paper's target." },
  "B3": { asks: "Does fine-tuning the backbone on player actions help scene recognition?", found: "Player-level information in a single frame is worse than B1 (65.7 vs 72.2): max-pooling loses which player is where. r_winpoint precision is only 0.33." },
  "B4": { asks: "B1 vs B4: does adding time help, with no player information?", found: "Slightly beats B1, reversing the paper's finding. ResNet50 features appear robust enough to camera motion that the LSTM still gains signal." },
  "B5": { asks: "B3 vs B5: does time help once players are in the picture?", found: "Real per-frame tracking barely moves aggregate accuracy (+0.15 pts vs static boxes) but redistributes errors across classes." },
  "B6": { asks: "B5 vs B6: does the order of pooling vs temporal modelling matter?", found: "Needed 65 epochs to reach 73.2%. Pooling before temporal smoothing lets the max flip between players frame to frame." },
  "B7": { asks: "B5/B6 vs B7: does a second, scene-level LSTM help?", found: "Slightly below our own B5. The two winpoint classes stay weakest (0.46–0.63 recall) because mixed pooling loses court side." },
  "B8": { asks: "B7 vs B8: does pooling each team separately reduce confusion?", found: "Yes — the best result in the ladder. l_winpoint recall 0.63 → 0.88, r_winpoint 0.46 → 0.86; every class now above 0.81 recall." },
  "B8-mt": { asks: "Adds a per-player action head trained jointly with the scene loss (λ = 1.0).", found: "Costs 1.4 pts of scene accuracy vs B8 but yields every player's action in one pass, and improves recall on jumping, digging and waiting over standalone B3-A." },
};

let D;
const state = { model: "B8-mt", task: "scene", rung: "B8-mt" };

export async function initResearch() {
  try {
    D = await getJSON("data/research.json");
  } catch (e) {
    $$("[data-research]").forEach((el) => (el.innerHTML = `<div class="state error">Research data could not be loaded.</div>`));
    return;
  }
  D.ladder.push({ id: "B3-A", label: "Player-action backbone", test_acc: D.b3a.test_acc, val_acc: null, person: D.b3a.person, person_acc: D.b3a.test_acc, curve: D.b3a.curve, personOnly: true });
  renderHero();
  renderClasses();
  renderSupport();
  renderLadder();
  renderGroup();
  renderResultsControls();
  renderResults();
  renderDetector();
  renderFollowups();
  renderRecall();
}

const byId = (id) => D.ladder.find((m) => m.id === id);

function renderHero() {
  const m = byId("B8-mt");
  const set = (k, v) => $$(`[data-hero="${k}"]`).forEach((el) => (el.innerHTML = v));
  set("scene", `${m.test_acc.toFixed(1)}<small>%</small>`);
  set("person", `${m.person_acc.toFixed(1)}<small>%</small>`);
  set("map", `${D.detector.map50.toFixed(1)}<small>%</small>`);
  set("track", `${D.tracker.match_rate.toFixed(1)}<small>%</small>`);
  set("b8", `${byId("B8").test_acc.toFixed(1)}<small>%</small>`);
}

function renderClasses() {
  const chip = (n, c) => `<span class="chip" data-tip="${esc(GLOSSARY[n])}" tabindex="0" style="--c:${c}"><i></i>${esc(n)}</span>`;
  const col = (n) => (n.includes("spike") ? "var(--c-spike)" : n.includes("set") ? "var(--c-set)" : n.includes("win") ? "var(--c-winpoint)" : "var(--c-pass)");
  $("#scene-classes").innerHTML = SCENE_CLASSES.map((n) => chip(n, col(n))).join("");
  $("#action-classes").innerHTML = ACTION_CLASSES.map((n) => chip(n, "var(--accent-2)")).join("");
}

function renderSupport() {
  const m = byId("B8-mt");
  $("#chart-scene-support").innerHTML = charts.hbars(
    m.scene.classes.map((c) => ({ label: c.name, value: c.support, color: c.name.startsWith("l") ? "var(--team-left)" : "var(--team-right)", tip: `${c.name}: ${c.support} test clips` })),
    { fmt: (v) => v.toLocaleString() });
  $("#chart-action-support").innerHTML = charts.hbars(
    [...m.person.classes].sort((a, b) => b.support - a.support).map((c) => ({ label: c.name, value: c.support, color: "var(--accent-2)", tip: `${c.name}: ${c.support.toLocaleString()} player instances` })),
    { log: true, fmt: (v) => v.toLocaleString() });
}

/* ---------- Baseline ladder ---------- */
function renderLadder() {
  const rungs = D.ladder.filter((m) => !m.personOnly);
  $("#ladder").innerHTML = rungs.map((m) => `
    <button class="rung ${m.id === "B8-mt" ? "final" : ""}" data-id="${m.id}" aria-pressed="${m.id === state.rung}">
      <span class="id">${m.id}</span>
      <span class="track"><span class="fill" style="width:0" data-w="${m.test_acc}"></span>${m.paper_target ? `<span class="paper" style="left:${m.paper_target}%" data-tip="Paper target ${m.paper_target}%"></span>` : ""}</span>
      <span class="val">${m.test_acc.toFixed(1)}%</span>
    </button>`).join("");
  requestAnimationFrame(() => $$("#ladder .fill").forEach((f) => (f.style.width = f.dataset.w + "%")));
  $$("#ladder .rung").forEach((b) => b.addEventListener("click", () => { state.rung = b.dataset.id; $$("#ladder .rung").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); renderRungDetail(); }));
  renderRungDetail();
}

function renderRungDetail() {
  const rungs = D.ladder.filter((m) => !m.personOnly);
  const i = rungs.findIndex((m) => m.id === state.rung);
  const m = rungs[i], prev = rungs[i - 1], note = NOTES[m.id];
  const delta = (a, b) => { const d = a - b; return `<span class="${d >= 0 ? "delta-up" : "delta-down"}">${d >= 0 ? "+" : ""}${d.toFixed(1)} pts</span>`; };
  const rows = [
    ["Test accuracy", `${m.test_acc.toFixed(2)}%`],
    ["Validation (best)", `${m.val_acc.toFixed(1)}%`],
    ["Epochs", m.epochs],
    m.paper_target ? ["vs paper target", `${m.paper_target}% · ${delta(m.test_acc, m.paper_target)}`] : null,
    prev ? [`vs ${prev.id}`, delta(m.test_acc, prev.test_acc)] : null,
    m.person_acc ? ["Player-action accuracy", `${m.person_acc.toFixed(2)}%`] : null,
  ].filter(Boolean);
  $("#model-detail").innerHTML = `
    <div>
      <h3 style="font-size:22px">${m.id} · ${esc(m.label)}</h3>
      <p class="muted" style="margin:8px 0 14px">${esc(m.description)}</p>
      <div class="kv">${rows.map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join("")}</div>
      <details class="disclose" style="margin-top:14px" open><summary>What this rung isolates</summary><div class="body"><p>${esc(note.asks)}</p></div></details>
      <details class="disclose" open><summary>What the experiment found</summary><div class="body"><p>${esc(note.found)}</p></div></details>
    </div>
    <figure class="fig"><img src="assets/research/${m.id.toLowerCase().replace("-", "")}_pipeline.webp" alt="${m.id} pipeline drawn on a real clip" loading="lazy"><figcaption>${m.id} pipeline on a real test clip (video 4, clip 94540, <code>r_spike</code>).</figcaption></figure>`;
}

/* ---------- Results explorer ---------- */
function renderResultsControls() {
  const tabs = $("#results-model-tabs");
  tabs.innerHTML = D.ladder.map((m) => `<button class="tab" role="tab" data-id="${m.id}" aria-selected="${m.id === state.model}" tabindex="${m.id === state.model ? 0 : -1}">${m.id}</button>`).join("");
  tabGroup(tabs, (t) => { state.model = t.dataset.id; const m = byId(state.model); state.task = m.personOnly ? "person" : "scene"; renderResults(); });
}

function renderResults() {
  const m = byId(state.model);
  // task toggle
  const hasBoth = !!m.person && !m.personOnly;
  const taskBox = $("#cm-task");
  taskBox.hidden = !hasBoth;
  if (hasBoth) {
    taskBox.innerHTML = ["scene", "person"].map((t) => `<button class="tab ${state.task === t ? "active" : ""}" data-t="${t}" aria-pressed="${state.task === t}">${t === "scene" ? "Group activity" : "Player action"}</button>`).join("");
    $$("button", taskBox).forEach((b) => b.addEventListener("click", () => { state.task = b.dataset.t; renderResults(); }));
  } else if (!m.person) state.task = "scene";
  const sec = state.task === "person" ? m.person : m.scene;

  $("#cm-sub").textContent = `${m.id} · ${state.task === "person" ? "9-way player action" : "8-way group activity"} · accuracy ${pct(sec.accuracy, 2)}`;
  $("#chart-cm").innerHTML = charts.confusion(sec.labels, sec.matrix);

  $("#class-table").innerHTML = `<thead><tr><th>Class</th><th class="num">Prec</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th><th>F1</th></tr></thead><tbody>${sec.classes.map((c) => `
    <tr><td>${esc(c.name)}</td><td class="num">${c.precision.toFixed(3)}</td><td class="num">${c.recall.toFixed(3)}</td><td class="num">${c.f1.toFixed(3)}</td><td class="num">${c.support.toLocaleString()}</td>
    <td style="width:120px"><div class="meter" style="--c:${c.f1 < 0.5 ? "var(--bad)" : c.f1 < 0.75 ? "var(--warn)" : "var(--ok)"}"><i style="width:${c.f1 * 100}%"></i></div></td></tr>`).join("")}</tbody>`;

  const curve = m.curve;
  if (curve && curve.length) {
    $("#curve-sub").textContent = `${m.id} · ${curve.length} epochs · best validation ${m.val_acc ? m.val_acc.toFixed(1) + "%" : "—"}`;
    const all = curve.flatMap((r) => [r.train_acc, r.val_acc]);
    const yMin = Math.max(0, Math.floor((Math.min(...all) - 0.03) * 10) / 10);
    $("#chart-curve").innerHTML = charts.lines([
      { name: "train", color: "var(--accent-2)", values: curve.map((r) => r.train_acc) },
      { name: "val", color: "var(--accent)", values: curve.map((r) => r.val_acc) },
    ], curve.map((r) => r.epoch), { yMin, yMax: 1 });
  } else {
    $("#curve-sub").textContent = "";
    $("#chart-curve").innerHTML = `<div class="state">No training log recorded for this model.</div>`;
  }
}

function renderRecall() {
  const a = D.b3a.person.classes, b = byId("B8-mt").person.classes;
  const order = [...b].sort((x, y) => x.recall - y.recall).map((c) => c.name);
  const get = (arr, n) => arr.find((c) => c.name === n).recall;
  $("#chart-recall").innerHTML = charts.grouped(order, order.map((n) => get(a, n)), order.map((n) => get(b, n)), { nameA: "B3-A", nameB: "B8-mt" });
}

function renderDetector() {
  const d = D.detector, t = D.tracker;
  $("#det-stats").innerHTML = [
    [d.map50.toFixed(1) + "%", "mAP50"], [d.precision.toFixed(1) + "%", "Precision"], [d.recall.toFixed(1) + "%", "Recall"], [t.match_rate.toFixed(1) + "%", "Tracker match"],
  ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  $("#chart-detector").innerHTML = charts.lines([{ name: "mAP50", color: "var(--accent-2)", values: d.curve.map((r) => r.map50) }], d.curve.map((r) => r.epoch), { yMin: Math.floor(Math.min(...d.curve.map((r) => r.map50)) * 10) / 10, yMax: 1 });
}

function renderFollowups() {
  const classes = ["jumping", "digging", "waiting", "standing"];
  $("#followup-table").innerHTML = `<thead><tr><th>Variant</th><th class="num">Scene</th><th class="num">Person</th>${classes.map((c) => `<th class="num">${c} recall</th>`).join("")}</tr></thead><tbody>${D.followups.map((f) => `
    <tr><td>${esc(f.name)}</td><td class="num">${f.scene_acc.toFixed(2)}%</td><td class="num">${f.person_acc.toFixed(2)}%</td>${classes.map((c) => `<td class="num">${f.recall[c].toFixed(3)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

/* ---------- Group activity explorer ---------- */
const SCENE_TEXT = {
  pass: "The team is receiving serve or passing the ball up.",
  set: "The team's setter is delivering the ball to an attacker.",
  spike: "A player of this team is attacking the ball over the net.",
  winpoint: "This team has just won the point (end of the rally).",
};
const sceneColor = (n) => (n.includes("spike") ? "var(--c-spike)" : n.includes("set") ? "var(--c-set)" : n.includes("win") ? "var(--c-winpoint)" : "var(--c-pass)");

function renderGroup() {
  const m = byId("B8-mt").scene, b7 = byId("B7").scene, b8 = byId("B8").scene;
  const wp = (s, n) => s.classes.find((c) => c.name === n).recall;
  $("#gd-delta").textContent = `Measured: l_winpoint recall ${wp(b7, "l_winpoint").toFixed(2)} to ${wp(b8, "l_winpoint").toFixed(2)}, r_winpoint ${wp(b7, "r_winpoint").toFixed(2)} to ${wp(b8, "r_winpoint").toFixed(2)}.`;
  const chips = $("#group-chips");
  chips.innerHTML = m.labels.map((n) => `<button class="chip btn-chip" style="--c:${sceneColor(n)}" data-n="${n}" aria-pressed="false"><i></i>${n}</button>`).join("");
  const show = (n) => {
    $$(".btn-chip", chips).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.n === n)));
    const i = m.labels.indexOf(n), c = m.classes[i], row = m.matrix[i], tot = row.reduce((a, b) => a + b, 0);
    const side = n.startsWith("l") ? "left" : "right";
    const phase = n.replace(/^[lr][-_]/, "");
    const errs = row.map((v, j) => ({ j, v })).filter((x) => x.j !== i && x.v > 0).sort((a, b) => b.v - a.v).slice(0, 3);
    $("#group-detail").innerHTML = `
      <p><span class="team ${side}">${side} team</span> <b style="margin-left:8px;text-transform:capitalize">${phase}</b></p>
      <p class="muted" style="margin:8px 0 14px">${SCENE_TEXT[phase]}</p>
      <div class="grid g3">
        <div class="stat"><b>${c.precision.toFixed(2)}</b><span>Precision</span></div>
        <div class="stat"><b>${c.recall.toFixed(2)}</b><span>Recall</span></div>
        <div class="stat"><b>${c.f1.toFixed(2)}</b><span>F1 · ${c.support} clips</span></div>
      </div>
      <p style="margin-top:18px;font-weight:600">Most often mistaken for</p>
      <div class="confuse">${errs.length ? errs.map((e) => `<div class="row"><span>${m.labels[e.j]}</span><div class="meter" style="--c:var(--bad)"><i style="width:${Math.min(100, (e.v / tot) * 300)}%"></i></div><span>${((e.v / tot) * 100).toFixed(1)}%</span></div>`).join("") : '<span class="muted">No errors on the test split.</span>'}</div>
      <p class="note">Shares are of the ${tot} true ${n} test clips.</p>`;
  };
  $$(".btn-chip", chips).forEach((b) => b.addEventListener("click", () => show(b.dataset.n)));
  show("r_spike");
}
