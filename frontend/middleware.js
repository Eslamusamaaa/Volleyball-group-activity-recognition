// Vercel Routing Middleware. It only answers /__geo: the visitor's approximate place from Vercel's edge headers,
// which the site's tracking script sends along with its visit event. Every other request is served as normal;
// the site is open and Google sign-in is optional (see js/auth.js).
export const config = { matcher: "/__geo" };

export default function middleware(req) {
  const h = (k) => req.headers.get(k) || "";
  const dec = (v) => { try { return decodeURIComponent(v); } catch { return v; } };
  return new Response(JSON.stringify({
    country: h("x-vercel-ip-country"), region: dec(h("x-vercel-ip-country-region")), city: dec(h("x-vercel-ip-city")), tz: h("x-vercel-ip-timezone"),
  }), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
