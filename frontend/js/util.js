// Shared helpers: DOM, formatting, toast, tooltips, reveal, nav.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

// Backend origin when the UI is hosted apart from the API (see config.js).
// Empty string = same origin, which is how the FastAPI app serves this UI.
const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
export const apiUrl = (u) => (u && u.startsWith("/api") ? API_BASE + u : u);

export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

export function fmtTime(sec) {
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

export function fmtDuration(sec) {
  if (!sec) return "0s";
  const m = Math.floor(sec / 60);
  return m > 0 ? `${m}m ${Math.round(sec % 60)}s` : `${sec.toFixed(1)}s`;
}

// Activity classes used everywhere (timeline, charts, chips).
export const ACTIVITY = {
  pass: { label: "Pass", color: "var(--c-pass)" },
  set: { label: "Set", color: "var(--c-set)" },
  spike: { label: "Spike", color: "var(--c-spike)" },
  winpoint: { label: "Point won", color: "var(--c-winpoint)" },
};
export const activityOf = (name) => ACTIVITY[(name || "").toLowerCase()] || { label: name, color: "var(--text-3)" };

let toastTimer;
export function toast(msg, kind = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4200);
}

// Google ID token for the costly endpoints; set by auth.js after sign-in.
let authToken = "";
export const setAuthToken = (t) => { authToken = t || ""; };
// Random id of this browser (kept in localStorage); the API uses it to link a visit to a later sign-in.
export function visitorId() {
  try {
    let v = localStorage.getItem("volleyai.vid") || "";
    if (!v) { v = crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now(); localStorage.setItem("volleyai.vid", v); }
    return v;
  } catch { return ""; }
}
export const authHeader = () => ({ "X-Visitor-Id": visitorId(), ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}) });

export async function getJSON(url, opts = {}) {
  const res = await fetch(apiUrl(url), { ...opts, headers: { ...authHeader(), ...opts.headers } });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const e = new Error(err.detail || `Request failed (${res.status})`);
    e.status = res.status;
    throw e;
  }
  return res.json();
}

// Delegated tooltip: any element (HTML or SVG) with [data-tip].
export function initTooltips() {
  const box = $("#tipbox");
  let current = null;
  const place = (target, x, y) => {
    const r = box.getBoundingClientRect();
    let left = x + 14, top = y + 16;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 12;
    box.style.left = `${Math.max(8, left)}px`;
    box.style.top = `${Math.max(8, top)}px`;
  };
  const show = (el, x, y) => {
    current = el;
    box.textContent = el.getAttribute("data-tip");
    box.classList.add("show");
    place(el, x, y);
  };
  const hide = () => { current = null; box.classList.remove("show"); };
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest?.("[data-tip]");
    if (el) show(el, e.clientX, e.clientY); else if (current) hide();
  });
  document.addEventListener("mousemove", (e) => { if (current) place(current, e.clientX, e.clientY); });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest?.("[data-tip]");
    if (el) { const r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.bottom); }
  });
  document.addEventListener("focusout", hide);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
  document.addEventListener("scroll", hide, { passive: true });
  // Make glossary terms keyboard-reachable.
  $$(".term[data-tip]").forEach((t) => t.setAttribute("tabindex", "0"));
}

export function initReveal() {
  const els = $$(".reveal");
  if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { rootMargin: "0px 0px -8% 0px" });
  els.forEach((e) => io.observe(e));
}

export function initNav() {
  const links = $$(".nav-links a");
  const toggle = $("#nav-toggle");
  const menu = $("#nav-links");
  toggle.addEventListener("click", () => {
    const open = menu.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  links.forEach((a) => a.addEventListener("click", () => { menu.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }));

  const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      links.forEach((l) => l.classList.remove("active"));
      map.get(en.target.id)?.classList.add("active");
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  map.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });

  // Sub-navigation inside the Technical section.
  const tlinks = $$("#tech-nav a");
  const tmap = new Map(tlinks.map((a) => [a.getAttribute("href").slice(1), a]));
  const tio = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      tlinks.forEach((l) => l.classList.remove("active"));
      const a = tmap.get(en.target.id);
      a?.classList.add("active");
      a?.scrollIntoView({ block: "nearest", inline: "center" });
    });
  }, { rootMargin: "-30% 0px -60% 0px" });
  tmap.forEach((_, id) => { const s = document.getElementById(id); if (s) tio.observe(s); });
}

// Accessible tab group: arrow keys move + select.
export function tabGroup(container, onSelect) {
  const tabs = $$('[role="tab"]', container);
  const select = (tab, focus = false) => {
    tabs.forEach((t) => { t.setAttribute("aria-selected", String(t === tab)); t.tabIndex = t === tab ? 0 : -1; });
    if (focus) tab.focus();
    onSelect(tab);
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length], true); }
    });
  });
  return select;
}

export const icons = {
  video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m10 9 5 3-5 3z"/></svg>',
  detect: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="10" r="2.2"/><path d="M8.5 17c.5-2.3 1.8-3.5 3.5-3.5s3 1.2 3.5 3.5"/></svg>',
  track: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="18" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="6" r="1.6"/><path d="M6.4 16.8 10.6 13.2M13.4 10.8 17.6 7.2" stroke-dasharray="2 2.5"/></svg>',
  features: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><path d="M14 17.5h7M17.5 14v7"/></svg>',
  model: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="6" r="1.8"/><circle cx="5" cy="18" r="1.8"/><circle cx="12" cy="12" r="2.2"/><circle cx="19" cy="12" r="1.8"/><path d="M6.6 7l3.8 4M6.6 17l3.8-4M14.2 12h3"/></svg>',
  events: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>',
  render: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4M7 9h4v4H7z"/></svg>',
  empty: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 15l5-4 4 3 3-2 6 4"/></svg>',
  error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5m0 3h.01"/></svg>',
};
