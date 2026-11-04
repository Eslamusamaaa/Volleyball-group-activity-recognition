// Hero showcase: the first thing a visitor sees is the model's annotated output
// playing on its own (muted, looping, no controls, slowed to about 8 s), with a switch to
// move between one verified clip per group activity.
import { $, esc, getJSON, pct } from "./util.js";
import { loopVideo } from "./videos.js";

const nice = (c) => `${c.team} ${c.event}`;

export async function initShowcase() {
  const card = $("#hero-card"), bar = $("#hero-switch"), cap = $("#hero-cap");
  if (!card || !bar) return;
  let clips;
  try { clips = await getJSON("data/showcase.json"); } catch { return; }
  if (!clips.length) return;

  const img = $("img", card);
  const vid = document.createElement("video");
  loopVideo(vid);
  vid.setAttribute("aria-label", "Annotated model output on a held-out test clip");
  vid.preload = "auto";
  img.after(vid);

  const show = (i) => {
    const c = clips[i];
    vid.poster = c.poster;
    vid.src = c.video_file;
    cap.textContent = `group activity: ${nice(c)} · ${pct(c.confidence, 0)}`;
    bar.querySelectorAll("button").forEach((b, j) => {
      b.classList.toggle("active", j === i);
      b.setAttribute("aria-pressed", String(j === i));
    });
    vid.play().catch(() => {});
  };

  bar.innerHTML = clips.map((c, i) => `<button type="button" class="chip-btn" aria-pressed="false" data-i="${i}">${esc(nice(c))}</button>`).join("");
  bar.addEventListener("click", (e) => { const b = e.target.closest("button[data-i]"); if (b) show(+b.dataset.i); });
  vid.addEventListener("loadeddata", () => { img.hidden = true; vid.play().catch(() => {}); });
  show(Math.max(0, clips.findIndex((c) => c.label === "r_spike")));
}
