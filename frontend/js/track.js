// Visitor tracking: one "visit" event when the page loads and a "leave" summary (time on page, scroll depth,
// sections seen, clicks) whenever the tab is hidden, closed or every 30 s while it stays open. Events carry a
// random browser id kept in localStorage, so a later Google sign-in can be linked to the same browser.
// What is collected is listed on /privacy.html. Events go to POST /api/track as text/plain so sendBeacon
// needs no CORS preflight.
import { visitorId } from "./util.js";

const API = (window.API_BASE || "").replace(/\/$/, "");
const sid = Math.random().toString(36).slice(2, 10);
const seen = new Set();
let clicks = 0, maxScroll = 0, shown = 0, since = performance.now();

function send(payload) {
  const body = JSON.stringify({ ...payload, vid: visitorId(), sid });
  try {
    if (navigator.sendBeacon && navigator.sendBeacon(`${API}/api/track`, new Blob([body], { type: "text/plain" }))) return;
  } catch { /* fall through */ }
  fetch(`${API}/api/track`, { method: "POST", body, headers: { "content-type": "text/plain" }, keepalive: true }).catch(() => {});
}

const seconds = () => Math.round(shown + (document.visibilityState === "visible" ? (performance.now() - since) / 1000 : 0));

function leave() {
  send({ type: "leave", dur: seconds(), scroll: maxScroll, clicks, sections: [...seen] });
}

export async function initTrack() {
  let geo = {};
  try {
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 2500);
    geo = await fetch("/__geo", { signal: ctl.signal }).then((r) => r.json()); clearTimeout(t);
  } catch { /* geo is best effort */ }

  let returning = 0;
  try { returning = +(localStorage.getItem("volleyai.visits") || 0); localStorage.setItem("volleyai.visits", String(returning + 1)); } catch { /* storage blocked */ }
  const q = new URLSearchParams(location.search);
  send({
    type: "visit", country: geo.country || "", region: geo.region || "", city: geo.city || "",
    tz: Intl.DateTimeFormat().resolvedOptions().timeZone || geo.tz || "", lang: navigator.language || "",
    screen: `${innerWidth}x${innerHeight}`, touch: navigator.maxTouchPoints > 0, ref: document.referrer || "", returning,
    utm: [q.get("utm_source"), q.get("utm_medium"), q.get("utm_campaign"), q.get("ref"), q.get("source")].filter(Boolean).join("/"),
  });

  const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) seen.add(e.target.id); }), { threshold: 0.35 });
  document.querySelectorAll("section[id]").forEach((s) => io.observe(s));
  addEventListener("scroll", () => {
    const h = document.documentElement;
    maxScroll = Math.max(maxScroll, Math.round(((scrollY + innerHeight) / h.scrollHeight) * 100));
  }, { passive: true });
  addEventListener("click", (e) => { if (e.target.closest("a, button, summary")) clicks++; }, { passive: true });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") { shown += (performance.now() - since) / 1000; leave(); } else since = performance.now();
  });
  addEventListener("pagehide", leave);
  setInterval(() => { if (document.visibilityState === "visible") leave(); }, 30000);
}
