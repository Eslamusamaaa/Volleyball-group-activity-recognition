// Login wall: with a Google client ID configured, nothing on the site loads until the
// visitor signs in with Google. The ID token is also sent as a bearer token to the API,
// which verifies it and records the email. Without a client ID the wall is off.
import { $, authHeader, getJSON, setAuthToken } from "./util.js";

const CLIENT_ID = (window.GOOGLE_CLIENT_ID || "").trim();
const KEY = "volleyai.idtoken";
let email = "";

export const authEnabled = !!CLIENT_ID;
export const signedIn = () => !authEnabled || !!email;

const root = document.documentElement;
const msg = (t) => { const m = $("#gate-msg"); if (m) m.textContent = t || ""; };

function claims(t) {
  try { return JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))); } catch { return null; }
}

function setToken(token) {
  const c = token && claims(token);
  if (!c || c.exp * 1000 < Date.now() + 30000) {
    email = ""; setAuthToken("");
    try { localStorage.removeItem(KEY); } catch { /* storage blocked */ }
    return false;
  }
  email = c.email || ""; setAuthToken(token);
  try { localStorage.setItem(KEY, token); } catch { /* storage blocked */ }
  return true;
}

export function signOut() { setToken(""); window.google?.accounts.id.disableAutoSelect(); location.reload(); }

// Called when the API answers 401: the token expired, so send the visitor back to the wall.
export function sessionExpired() { setToken(""); location.reload(); }

function showIdentity() {
  const bar = $("#auth-bar");
  if (!bar) return;
  bar.hidden = false;
  bar.innerHTML = `<span>Signed in as <b></b></span><button type="button" class="btn btn-sm">Sign out</button>`;
  $("b", bar).textContent = email;
  $("button", bar).addEventListener("click", signOut);
}

// Resolves with true/false once the API says whether the account is an admin.
let resolveAdmin;
export const adminReady = new Promise((r) => { resolveAdmin = r; });

// The visitor already holds a valid token, so the site opens at once; the API is told in the
// background (it records the email and reports admin status) and is retried if it is still waking up.
async function registerVisit(tries = 4) {
  for (let i = 0; i < tries; i++) {
    try {
      resolveAdmin(!!(await getJSON("/api/session", { method: "POST" })).is_admin);
      return;
    } catch (e) {
      if (e.status === 401 || e.status === 403) { if (e.status === 401) sessionExpired(); else { msg(e.message); setToken(""); location.reload(); } return; }
      await new Promise((r) => setTimeout(r, 6000));
    }
  }
  resolveAdmin(false);
}

const cookieToken = () => (document.cookie.split(/;\s*/).find((c) => c.startsWith("volleyai_session=")) || "").split("=")[1] || "";

export function initAuth() {
  if (!authEnabled) { root.classList.remove("locked"); resolveAdmin(false); return Promise.resolve(); }
  return new Promise((resolve) => {
    const enter = () => { root.classList.remove("locked"); showIdentity(); resolve(); registerVisit(); };
    let saved = ""; try { saved = localStorage.getItem(KEY) || ""; } catch { /* storage blocked */ }
    // The login page set a cookie too, so a new tab or browser session needs no second sign-in.
    if (setToken(saved) || setToken(cookieToken())) return enter();
    const s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client"; s.async = true; s.defer = true;
    s.onload = () => {
      window.google.accounts.id.initialize({ client_id: CLIENT_ID, auto_select: true, callback: (r) => { if (setToken(r.credential)) enter(); } });
      window.google.accounts.id.renderButton($("#gate-btn"), { theme: "filled_black", size: "large", shape: "pill" });
      window.google.accounts.id.prompt();
    };
    s.onerror = () => msg("Google sign-in could not be loaded. Check your connection and reload.");
    document.head.appendChild(s);
  });
}
