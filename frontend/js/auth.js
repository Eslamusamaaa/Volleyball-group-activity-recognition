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
    try { sessionStorage.removeItem(KEY); } catch { /* storage blocked */ }
    return false;
  }
  email = c.email || ""; setAuthToken(token);
  try { sessionStorage.setItem(KEY, token); } catch { /* storage blocked */ }
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

export function initAuth() {
  if (!authEnabled) { root.classList.remove("locked"); return Promise.resolve(); }
  return new Promise((resolve) => {
    // Register the visit with the API (records the email); resolves only when it accepts the token.
    const enter = async () => {
      msg("Signing you in…");
      try {
        await getJSON("/api/session", { method: "POST" });
        root.classList.remove("locked"); showIdentity(); resolve();
      } catch (e) {
        setToken("");
        msg(e.status ? e.message : "The server is starting up or unreachable. Wait a moment and try again.");
      }
    };
    let saved = ""; try { saved = sessionStorage.getItem(KEY) || ""; } catch { /* storage blocked */ }
    const restored = setToken(saved);
    const s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client"; s.async = true; s.defer = true;
    s.onload = () => {
      window.google.accounts.id.initialize({ client_id: CLIENT_ID, callback: (r) => { if (setToken(r.credential)) enter(); } });
      if (!restored) window.google.accounts.id.renderButton($("#gate-btn"), { theme: "filled_black", size: "large", shape: "pill" });
    };
    s.onerror = () => msg("Google sign-in could not be loaded. Check your connection and reload.");
    document.head.appendChild(s);
    if (restored) enter();
  });
}
