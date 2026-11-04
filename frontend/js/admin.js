// Admin section: the list of everyone who signed in with Google, from GET /api/visitors.
// The section and its nav link stay hidden unless the API says the signed-in account is an admin.
import { $, getJSON } from "./util.js";

const when = (t) => new Date(t * 1000).toLocaleString();

async function load() {
  const note = $("#admin-note"), body = $("#admin-rows");
  try {
    const { visitors, anonymous = [] } = await getJSON("/api/visitors");
    $("#admin-anon-rows").replaceChildren(...anonymous.map((b) => {
      const tr = document.createElement("tr");
      const place = [b.city, b.country].filter(Boolean).join(", ") || "unknown";
      const who = b.signed_in_as && b.signed_in_as.length ? b.signed_in_as.join(", ") : "never signed in";
      [when(b.last_seen), b.device || "unknown", place, b.ip || "", b.ref || "direct", b.hits, who].forEach((x) => { const td = document.createElement("td"); td.textContent = x; tr.appendChild(td); });
      if (!(b.signed_in_as && b.signed_in_as.length)) tr.className = "anon-never";
      return tr;
    }));
    const never = anonymous.filter((b) => !(b.signed_in_as && b.signed_in_as.length)).length;
    $("#admin-anon-note").textContent = `${anonymous.length} browser${anonymous.length === 1 ? "" : "s"} reached the login page, ${never} never signed in.`;
    body.replaceChildren(...visitors.map((v) => {
      const tr = document.createElement("tr");
      [v.email, when(v.first_seen), when(v.last_seen), v.visits, v.analyses].forEach((x) => { const td = document.createElement("td"); td.textContent = x; tr.appendChild(td); });
      return tr;
    }));
    note.textContent = `${visitors.length} visitor${visitors.length === 1 ? "" : "s"}, newest first.`;
  } catch (e) { note.textContent = e.message; }
}

export function initAdmin(isAdmin) {
  const sec = $("#admin"), link = $("#nav-admin");
  if (!sec || !isAdmin) return;
  sec.hidden = false; if (link) link.hidden = false;
  $("#admin-refresh").addEventListener("click", load);
  load();
}
