// Admin section: the list of everyone who signed in with Google, from GET /api/visitors.
// The section and its nav link stay hidden unless the API says the signed-in account is an admin.
import { $, getJSON } from "./util.js";

const when = (t) => new Date(t * 1000).toLocaleString();

async function load() {
  const note = $("#admin-note"), body = $("#admin-rows");
  try {
    const { visitors } = await getJSON("/api/visitors");
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
