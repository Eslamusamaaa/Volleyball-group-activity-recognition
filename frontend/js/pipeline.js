// Interactive visual of the actual inference pipeline (src/inference/*, backend/app.py).
import { $, $$, icons } from "./util.js";

const STAGES = [
  { id: "video", icon: icons.video, label: "Video input", sub: "OpenCV decode",
    text: "The video is decoded frame by frame with OpenCV and its native FPS is read, so every window can be converted into real seconds later.",
    points: ["Upload, library clip or downloaded link", "FPS from the file itself — no assumed 25 fps", "Sliding windows of 9 frames, stride 9 by default"],
    kv: [["Module", "video_io.py"], ["Window", "9 frames"], ["Stride", "9 frames"]] },
  { id: "detect", icon: icons.detect, label: "Player detection", sub: "YOLOv8n fine-tuned",
    text: "A COCO-pretrained detector fires on referees, coaches and ball kids too, matching only 77.8% of ground-truth players. It was fine-tuned on this dataset's own player boxes as a single class.",
    points: ["Single class: player", "Fine-tuned for 30 epochs on ~2,150 clips' centre frames", "Held-out val: 95.2% precision, 94.2% recall, 98.0% mAP50"],
    kv: [["Module", "detector.py"], ["Model", "YOLOv8n"], ["mAP50", "98.0%"]] },
  { id: "track", icon: icons.track, label: "Tracking", sub: "ByteTrack",
    text: "ByteTrack gives each player a consistent ID across frames. A window only uses players tracked in every one of its 9 frames, because LSTM1 needs a full sequence per player.",
    points: ["Validated against ground truth before use", "97.7% of ground-truth players matched on 18 clips (72.1% with stock detector)", "Top-12 confidence capping tested and left off — it made matching worse"],
    kv: [["Module", "tracker.py"], ["Match rate", "97.7%"], ["Clips", "18"]] },
  { id: "features", icon: icons.features, label: "Crops & features", sub: "ResNet50 (B3-A)",
    text: "Each tracked box is cropped in every frame and encoded by a ResNet50 fine-tuned on the nine player actions, producing a 2048-d feature per player per frame.",
    points: ["Team side assigned per window by player x-position", "Feature sequence: players × 9 frames × 2048", "Same loader as training, so checkpoints load unchanged"],
    kv: [["Module", "model_io.py"], ["Backbone", "ResNet50"], ["Feature", "2048-d"]] },
  { id: "model", icon: icons.model, label: "Temporal model", sub: "B8-mt",
    text: "LSTM1 runs per player, outputs are max-pooled per team, and LSTM2 runs over the pooled sequence. One forward pass gives a group-activity prediction and a 9-way action for each player.",
    points: ["Scene head: 8 classes · test accuracy 87.4%", "Player head: 9 classes · test accuracy 75.7%", "Hidden size 512 for both LSTMs"],
    kv: [["Module", "sliding_window.py"], ["Scene acc.", "87.4%"], ["Player acc.", "75.7%"]] },
  { id: "events", icon: icons.events, label: "Event timeline", sub: "Merge windows",
    text: "Consecutive windows predicting the same class are merged into one event. Weak predictions are dropped so idle moments are not labelled as plays.",
    points: ["Windows below 0.40 confidence are discarded", "Win-point events are capped at ~100 frames", "Output: team, event, confidence, start / end seconds"],
    kv: [["Module", "timeline.py"], ["Min. confidence", "0.40"], ["Winpoint cap", "100 frames"]] },
  { id: "render", icon: icons.render, label: "Annotated video", sub: "H.264 render",
    text: "A second pass burns player boxes, per-player action labels and a broadcast-style HUD into the video, which the studio streams back with the event list.",
    points: ["Boxes labelled with track ID and action", "Event HUD synchronised with the timeline", "Served by the FastAPI backend as MP4"],
    kv: [["Module", "annotator.py"], ["Output", "MP4 (H.264)"], ["API", "/api/jobs/{id}/video"]] },
];

export function initPipeline() {
  const flow = $("#flow");
  flow.innerHTML = STAGES.map((s, i) => `
    <button class="flow-node" role="tab" data-i="${i}" aria-selected="false" aria-controls="flow-detail">
      <span class="orb">${s.icon}</span><span class="lbl">${s.label}</span><span class="sub">${s.sub}</span>
    </button>`).join("");
  const nodes = $$(".flow-node", flow);
  const show = (i) => {
    const s = STAGES[i];
    nodes.forEach((n, k) => { n.classList.toggle("active", k === i); n.setAttribute("aria-selected", String(k === i)); });
    $("#flow-detail").innerHTML = `
      <div><h3>${i + 1}. ${s.label}</h3><p class="muted">${s.text}</p><ul>${s.points.map((p) => `<li>${p}</li>`).join("")}</ul></div>
      <div class="kv">${s.kv.map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join("")}</div>`;
  };
  nodes.forEach((n, i) => {
    n.addEventListener("click", () => show(i));
    n.addEventListener("mouseenter", () => show(i));
    n.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { e.preventDefault(); const k = (i + d + nodes.length) % nodes.length; nodes[k].focus(); show(k); }
    });
  });
  show(4);
}
