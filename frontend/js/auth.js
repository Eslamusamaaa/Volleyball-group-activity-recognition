// Optional Google sign-in, from inside the site. Nobody is asked to sign in on arrival: the whole site is open.
// Signing in identifies the visitor (their email is recorded) and is required to run an analysis, which costs
// compute. The ID token is sent as a bearer token to the API, which verifies it. Without a client ID in
// config.js the sign-in controls stay hidden.
import { $, getJSON, setAuthToken, toast } from "./util.js";

const CLIENT_ID = (window.GOOGLE_CLIENT_ID || "").trim();
const KEY = "volleyai.idtoken";
let email = "", admin = false;

export const authEnabled = !!CLIENT_ID;
export const signedIn = () => !authEnabled || !!email;

// Other modules (the admin section) listen for this instead of importing each other's state.
const emit = () => window.dispatchEvent(new CustomEvent("volleyai:auth", { detail: { email, isAdmin: admin } }));

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

/* ---------- Google Identity Services, loaded only when someone wants to sign in ---------- */
let gis;
function loadGIS() {
  if (gis) return gis;
  gis = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client"; s.async = true; s.defer = true;
    s.onload = () => {
      window.google.accounts.id.initialize({ client_id: CLIENT_ID, auto_select: false, callback: onCredential });
      resolve();
    };
    s.onerror = () => { gis = null; reject(new Error("Google sign-in could not be loaded.")); };
    document.head.appendChild(s);
  });
  return gis;
}

async function onCredential(r) {
  if (!setToken(r.credential)) { toast("That sign-in has already expired. Try again.", "error"); return; }
  closeDialog(); renderBars();
  toast(`Signed in as ${email}`, "");
  await registerVisit();
}

// Tells the API who signed in (it records the email and reports whether the account is an admin).
async function registerVisit(tries = 4) {
  for (let i = 0; i < tries; i++) {
    try {
      admin = !!(await getJSON("/api/session", { method: "POST" })).is_admin;
      emit();
      return;
    } catch (e) {
      if (e.status === 401 || e.status === 403) { signOut(); toast(e.message, "error"); return; }
      await new Promise((r) => setTimeout(r, 6000));   // the server may still be waking up
    }
  }
  emit();
}

/* ---------- UI ---------- */
const dlg = () => $("#signin-dialog");
const closeDialog = () => { const d = dlg(); if (d && d.open) d.close(); };

function renderBars() {
  const nav = $("#nav-signin"), bar = $("#auth-bar");
  if (nav) {
    nav.hidden = false;
    nav.textContent = email ? email.split("@")[0].slice(0, 14) : "Sign in";
    nav.title = email ? `Signed in as ${email}` : "Sign in with Google";
    nav.classList.toggle("signed", !!email);
  }
  if (bar) {
    bar.hidden = false;
    bar.innerHTML = email
      ? `<span>Signed in as <b></b></span><button type="button" class="btn btn-sm" data-act="out">Sign out</button>`
      : `<span>Everything on the site is open. <b>Sign in with Google</b> to run an analysis.</span><button type="button" class="btn btn-sm btn-primary" data-act="in">Sign in</button>`;
    if (email) $("b", bar).textContent = email;
  }
}

export async function openSignIn() {
  const d = dlg(); if (!d) return;
  const body = $("#signin-body");
  if (email) {
    body.innerHTML = `<h3>Your account</h3><p>Signed in as <b></b>.</p><button type="button" class="btn" data-act="out">Sign out</button>`;
    $("b", body).textContent = email;
  } else {
    body.innerHTML = `<h3>Sign in with Google</h3>
      <p>Browsing the site needs no account. Signing in lets you run an analysis on your own video. We record the email address of people who sign in; see the <a href="/privacy.html">privacy policy</a>.</p>
      <div id="g-btn" class="g-btn"></div><p class="gate-msg" id="g-msg" role="alert"></p>`;
  }
  if (!d.open) d.showModal();
  if (!email) {
    try { await loadGIS(); window.google.accounts.id.renderButton($("#g-btn"), { theme: "filled_black", size: "large", shape: "pill" }); }
    catch (e) { $("#g-msg").textContent = e.message; }
  }
}

// Used before an analysis: true when allowed, otherwise opens the sign-in dialog.
export function requireSignIn() {
  if (signedIn()) return true;
  openSignIn();
  return false;
}

// The API answered 401: the token expired.
export function sessionExpired() { setToken(""); admin = false; renderBars(); emit(); openSignIn(); }

export function signOut() {
  setToken(""); admin = false;
  window.google?.accounts.id.disableAutoSelect();
  closeDialog(); renderBars(); emit();
}

export function initAuth() {
  const nav = $("#nav-signin"), bar = $("#auth-bar");
  if (!authEnabled) { if (nav) nav.hidden = true; if (bar) bar.hidden = true; return; }
  let saved = ""; try { saved = localStorage.getItem(KEY) || ""; } catch { /* storage blocked */ }
  const restored = setToken(saved);
  renderBars();
  const act = (e) => { const b = e.target.closest("[data-act]"); if (b) (b.dataset.act === "out" ? signOut : openSignIn)(); };
  nav?.addEventListener("click", openSignIn);
  bar?.addEventListener("click", act);
  dlg()?.addEventListener("click", (e) => { if (e.target === dlg() || e.target.closest(".dlg-x")) closeDialog(); else act(e); });
  if (restored) registerVisit(); else emit();
}
