// Admin section: totals plus two tables from GET /api/visitors: the people who signed in, and every browser
// that opened the site (device, place, visits, time, what they looked at, analyses). It stays hidden unless
// the API says the signed-in account is an admin; it reacts to the "volleyai:auth" event from auth.js.
import { $, getJSON } from "./util.js";

const when = (t) => new Date(t * 1000).toLocaleString();
const dur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`);
const host = (u) => { if (!u) return "direct"; try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return u; } };
const place = (b) => [b.city, b.region, b.country].filter(Boolean).join(", ") || "unknown";
let data = null;

function row(cells, cls) {
  const tr = document.createElement("tr"); if (cls) tr.className = cls;
  cells.forEach((x) => { const td = document.createElement("td"); td.textContent = x; tr.appendChild(td); });
  return tr;
}

function renderTotals(t) {
  const items = [
    ["Browsers", t.browsers], ["Visits", t.visits], ["Never signed in", t.never_signed_in], ["Signed-in people", t.signed_in_people],
    ["People who analysed", t.people_who_analysed], ["Analyses", t.analyses], ["Active last 24 h", t.browsers_24h], ["Active last 7 days", t.browsers_7d],
    ["Avg time on site", dur(t.avg_time_sec)],
  ];
  $("#admin-totals").replaceChildren(...items.map(([k, v]) => {
    const d = document.createElement("div"); d.className = "stat-card";
    const b = document.createElement("b"); b.textContent = v; const s = document.createElement("span"); s.textContent = k;
    d.append(b, s); return d;
  }));
}

function renderBrowsers() {
  const f = $("#admin-filter").value;
  const list = data.browsers.filter((b) => f === "all" || (f === "anon" ? !b.signed_in_as.length : b.signed_in_as.length));
  $("#admin-br-rows").replaceChildren(...list.map((b) => row([
    when(b.last_seen), b.signed_in_as.length ? b.signed_in_as.join(", ") : "anonymous", b.device || "unknown", place(b), b.visits, dur(b.time_sec),
    `${b.scroll}%`, b.sections.join(", ") || "-", b.clicks, b.analyses, host(b.ref) + (b.utm ? ` (${b.utm})` : ""),
  ], b.signed_in_as.length ? "" : "anon-never")));
  $("#admin-br-note").textContent = `${list.length} browser${list.length === 1 ? "" : "s"} shown.`;
}

function csv() {
  const head = ["last_seen", "first_seen", "who", "device", "country", "region", "city", "language", "timezone", "screen", "touch", "visits", "returning_count", "time_sec", "scroll_pct", "clicks", "sections", "analyses", "came_from", "campaign", "ip_shortened"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = data.browsers.map((b) => [new Date(b.last_seen * 1000).toISOString(), new Date(b.first_seen * 1000).toISOString(), b.signed_in_as.join(" "), b.device, b.country, b.region, b.city,
    b.lang, b.tz, b.screen, b.touch, b.visits, b.returning, b.time_sec, b.scroll, b.clicks, b.sections.join(" "), b.analyses, b.ref, b.utm, b.ip].map(q).join(","));
  const url = URL.createObjectURL(new Blob([[head.join(","), ...lines].join("\n")], { type: "text/csv" }));
  const a = document.createElement("a"); a.href = url; a.download = `volleyai-visitors-${new Date().toISOString().slice(0, 10)}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function load() {
  const note = $("#admin-note");
  note.textContent = "Loading…";
  try {
    data = await getJSON("/api/visitors");
    renderTotals(data.totals);
    $("#admin-rows").replaceChildren(...data.visitors.map((v) => row([v.email, when(v.first_seen), when(v.last_seen), v.visits, v.analyses])));
    renderBrowsers();
    note.textContent = `${data.visitors.length} signed-in ${data.visitors.length === 1 ? "person" : "people"}. Updated ${new Date().toLocaleTimeString()}.`;
  } catch (e) { note.textContent = e.message; }
}

export function initAdmin() {
  const sec = $("#admin"), link = $("#nav-admin");
  if (!sec) return;
  $("#admin-refresh").addEventListener("click", load);
  $("#admin-filter").addEventListener("change", () => data && renderBrowsers());
  $("#admin-csv").addEventListener("click", () => data && csv());
  let shownBefore = false;
  addEventListener("volleyai:auth", (e) => {
    const on = !!e.detail.isAdmin;
    sec.hidden = !on; if (link) link.hidden = !on;
    if (on && !shownBefore) { shownBefore = true; load(); }
  });
}
