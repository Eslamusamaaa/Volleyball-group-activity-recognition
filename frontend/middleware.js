// Vercel Routing Middleware: nothing on this site is served without a valid Google sign-in.
// The visitor signs in on /login.html; POST /__login verifies the Google ID token and sets it
// as a cookie (readable by the page, so a new tab needs no second sign-in). Every other request must carry a cookie with a token that still
// verifies (signature against Google's keys, issuer, audience, expiry, verified email).
// Without it, pages get the login page and other files get 401. The pages needed to sign in,
// and the privacy policy that Google requires to be public, stay open.

const CLIENT_ID = (typeof process !== "undefined" && process.env && process.env.GOOGLE_CLIENT_ID) ||
  "143704568226-u1sp1ahv23j5rft2sfpn0ntfbf3bfu8m.apps.googleusercontent.com";
const COOKIE = "volleyai_session";
const JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs";
const PUBLIC = new Set(["/login.html", "/privacy.html", "/config.js", "/__login"]);
const PUBLIC_PREFIX = ["/css/"];

export const config = { matcher: "/:path*" };

let jwks = { keys: null, at: 0 };

const b64 = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(s.length / 4) * 4, "=")), (c) => c.charCodeAt(0));
const text = (u8) => new TextDecoder().decode(u8);

async function keys() {
  if (jwks.keys && Date.now() - jwks.at < 3600e3) return jwks.keys;
  const r = await fetch(JWKS_URL);
  if (!r.ok) throw new Error("jwks");
  jwks = { keys: (await r.json()).keys, at: Date.now() };
  return jwks.keys;
}

// Returns the claims of a valid Google ID token, or null.
export async function verify(token, now = Date.now(), getKeys = keys) {
  try {
    const [h, p, s] = token.split(".");
    const head = JSON.parse(text(b64(h)));
    if (head.alg !== "RS256") return null;
    const jwk = (await getKeys()).find((k) => k.kid === head.kid);
    if (!jwk) return null;
    const key = await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
    const ok = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, b64(s), new TextEncoder().encode(`${h}.${p}`));
    if (!ok) return null;
    const c = JSON.parse(text(b64(p)));
    const issOk = c.iss === "accounts.google.com" || c.iss === "https://accounts.google.com";
    if (!issOk || c.aud !== CLIENT_ID || !(c.exp * 1000 > now) || !c.email_verified || !c.email) return null;
    return c;
  } catch {
    return null;
  }
}

const cookieOf = (req) => (req.headers.get("cookie") || "").split(/;\s*/).map((x) => x.split("=")).find(([k]) => k === COOKIE)?.[1] || "";
const next = () => new Response(null, { headers: { "x-middleware-next": "1" } });
const rewrite = (req, path) => new Response(null, { headers: { "x-middleware-rewrite": new URL(path, req.url).toString() } });

export default async function middleware(req, ctx, getKeys) {
  const { pathname } = new URL(req.url);

  if (pathname === "/__login" && req.method === "POST") {
    let credential = "";
    try { credential = (await req.json()).credential || ""; } catch { /* bad body */ }
    const c = await verify(credential, Date.now(), getKeys);
    if (!c) return new Response(JSON.stringify({ detail: "Your Google sign-in is invalid or expired." }), { status: 401, headers: { "content-type": "application/json" } });
    const maxAge = Math.max(60, Math.floor(c.exp - Date.now() / 1000));
    return new Response(JSON.stringify({ email: c.email }), {
      headers: { "content-type": "application/json", "set-cookie": `${COOKIE}=${credential}; Max-Age=${maxAge}; Path=/; Secure; SameSite=Lax` },
    });
  }

  if (PUBLIC.has(pathname) || PUBLIC_PREFIX.some((p) => pathname.startsWith(p))) return next();

  if (await verify(cookieOf(req), Date.now(), getKeys)) return next();

  const isPage = pathname === "/" || pathname.endsWith(".html") || !pathname.split("/").pop().includes(".");
  return isPage ? rewrite(req, "/login.html") : new Response("Sign in with Google to view this file.", { status: 401 });
}
