// Google sign-in (Google Identity Services). The ID token is sent as a bearer token
// to the costly endpoints (upload and library analysis); the backend verifies it.
import { $, setAuthToken, toast } from "./util.js";

const CLIENT_ID = (window.GOOGLE_CLIENT_ID || "").trim();
const KEY = "volleyai.idtoken";
let email = "";

export const authEnabled = !!CLIENT_ID;
export const signedIn = () => !authEnabled || !!email;

function claims(t) {
  try { return JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))); } catch { return null; }
}

function apply(token) {
  const c = token && claims(token);
  if (!c || c.exp * 1000 < Date.now() + 30000) { email = ""; setAuthToken(""); try { sessionStorage.removeItem(KEY); } catch { /* storage blocked */ } return false; }
  email = c.email || "";
  setAuthToken(token);
  try { sessionStorage.setItem(KEY, token); } catch { /* storage blocked */ }
  return true;
}

export function signOut() {
  apply("");
  window.google?.accounts.id.disableAutoSelect();
  draw();
}

// Called when the backend answers 401: the token is gone or expired.
export function sessionExpired() { apply(""); draw(); }

function draw() {
  const bar = $("#auth-bar");
  if (!bar) return;
  if (email) {
    bar.innerHTML = `<span>Signed in as <b></b></span><button type="button" class="btn btn-sm">Sign out</button>`;
    $("b", bar).textContent = email;
    $("button", bar).addEventListener("click", signOut);
    return;
  }
  bar.innerHTML = `<span>Sign in with Google to run an analysis.</span><div id="g-btn"></div>`;
  window.google?.accounts.id.renderButton($("#g-btn", bar), { theme: "filled_black", size: "medium", shape: "pill" });
}

export function initAuth() {
  const bar = $("#auth-bar");
  if (!bar) return;
  bar.hidden = !authEnabled;
  if (!authEnabled) return;
  let saved = ""; try { saved = sessionStorage.getItem(KEY) || ""; } catch { /* storage blocked */ }
  apply(saved);
  const s = document.createElement("script");
  s.src = "https://accounts.google.com/gsi/client"; s.async = true; s.defer = true;
  s.onload = () => {
    window.google.accounts.id.initialize({
      client_id: CLIENT_ID,
      callback: (r) => { if (apply(r.credential)) { draw(); toast(`Signed in as ${email}`, ""); } },
    });
    draw();
  };
  s.onerror = () => { bar.textContent = "Google sign-in could not be loaded."; };
  document.head.appendChild(s);
  draw();
}
