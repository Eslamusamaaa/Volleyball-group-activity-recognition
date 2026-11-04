// Sample videos: the pipeline's annotated output for dataset test clips, looping on
// their own with no controls (nothing to pause or scrub). Data: data/videos.json
// (scripts/render_sample_videos.py). Player clips ring the verified player.
import { $, esc, getJSON, tabGroup, pct } from "./util.js";

const nice = (s) => s.replace(/^([lr])[-_]/, (_, t) => (t === "l" ? "left " : "right "));

export function loopVideo(vid) {
  vid.muted = true; vid.loop = true; vid.autoplay = true; vid.playsInline = true;
  vid.controls = false; vid.disablePictureInPicture = true;
  vid.setAttribute("controlslist", "nodownload noplaybackrate noremoteplayback");
  vid.addEventListener("contextmenu", (e) => e.preventDefault());
  vid.addEventListener("pause", () => { if (!vid.ended && document.visibilityState === "visible") vid.play().catch(() => {}); });
  return vid;
}

function render(box, c, kind) {
  const title = kind === "group" ? `Group activity: ${nice(c.label)}` : `Player action: ${c.label}`;
  const facts = kind === "group"
    ? `<div><span>Group activity</span><span>${esc(nice(c.label))}</span></div><div><span>Confidence</span><span>${pct(c.confidence, 1)}</span></div>`
    : `<div><span>Player action</span><span>${esc(c.label)}</span></div><div><span>Highlighted</span><span>yellow box</span></div>`;
  box.innerHTML = `
    <div class="sv">
      <div class="sv-media"><video class="loop-video" muted loop autoplay playsinline preload="auto" poster="${c.poster}" src="${c.video_file}" aria-label="${esc(title)}, annotated test clip"></video></div>
      <div class="sv-meta">
        <span class="verdict">✓ matches the dataset annotation</span>
        <div class="kv">${facts}<div><span>Source</span><span>test · video ${c.source_video} · clip ${esc(c.source_clip)}</span></div></div>
        <p class="note" style="margin-top:0">${kind === "group" ? "The pipeline's annotated output for a test sequence, trimmed to the stretch where the banner shows this class, slowed to about 12 seconds and looping." : "The pipeline's annotated output for a 41-frame test sequence, slowed to about 12 seconds and looping."}</p>
      </div>
    </div>`;
  const v = loopVideo($("video", box));
  v.play().catch(() => {});
}

function mount(tabsEl, viewEl, items, kind) {
  if (!items.length) { viewEl.innerHTML = `<div class="state">No verified videos available.</div>`; return; }
  tabsEl.innerHTML = items.map((s, i) => `<button class="tab" role="tab" data-i="${i}" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}">${esc(kind === "group" ? nice(s.label) : s.label)}</button>`).join("");
  tabGroup(tabsEl, (t) => render(viewEl, items[+t.dataset.i], kind));
  render(viewEl, items[0], kind);
}

export async function initVideos() {
  let data;
  try { data = await getJSON("data/videos.json"); }
  catch { $("#vgrp-view").innerHTML = $("#vply-view").innerHTML = `<div class="state error">Sample videos could not be loaded.</div>`; return; }
  mount($("#vgrp-tabs"), $("#vgrp-view"), data.group, "group");
  mount($("#vply-tabs"), $("#vply-view"), data.player, "player");
}
