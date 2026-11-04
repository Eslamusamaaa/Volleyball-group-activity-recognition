"""Google sign-in for the costly endpoints.

The browser signs in with Google Identity Services and sends the ID token as
`Authorization: Bearer <token>`. The token is verified here against GOOGLE_CLIENT_ID.
Optional ALLOWED_EMAILS (comma separated) limits access to those accounts.
With no GOOGLE_CLIENT_ID set, auth is off (local development).
"""
import json
import os
import threading
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
ALLOWED_EMAILS = {e.strip().lower() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()}
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}
VISITS_FILE = os.environ.get("VISITS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "visits.jsonl"))
_LOCK = threading.Lock()


def _verify(token: str) -> dict:
    from google.auth.transport import requests
    from google.oauth2 import id_token
    return id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)


def require_user(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """FastAPI dependency: returns the signed-in email, or None when auth is off."""
    if not CLIENT_ID:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in with Google to run an analysis.")
    try:
        claims = _verify(authorization.split(" ", 1)[1].strip())
    except ValueError:
        raise HTTPException(401, "Your Google sign-in expired or is invalid. Sign in again.")
    email = (claims.get("email") or "").lower()
    if not claims.get("email_verified") or not email:
        raise HTTPException(403, "Your Google account has no verified email.")
    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        raise HTTPException(403, f"{email} is not allowed to use this service.")
    return email


def record(email: Optional[str], kind: str, detail: str = "", vid: str = "") -> None:
    """Append one visit/analysis event; never lets logging break a request."""
    if not email:
        return
    row = {"t": int(time.time()), "email": email, "kind": kind, "detail": detail[:200], "vid": vid[:64]}
    print("VISIT", json.dumps(row), flush=True)
    try:
        with _LOCK:
            os.makedirs(os.path.dirname(VISITS_FILE), exist_ok=True)
            with open(VISITS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def _rows():
    try:
        with open(VISITS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def visitors() -> list:
    """Per-email summary: first/last seen, sign-ins, analyses."""
    people = {}
    for r in _rows():
        if r.get("kind") not in ("visit", "analyze") or not r.get("email"):
            continue
        p = people.setdefault(r["email"], {"email": r["email"], "first_seen": r["t"], "last_seen": r["t"], "visits": 0, "analyses": 0})
        p["first_seen"], p["last_seen"] = min(p["first_seen"], r["t"]), max(p["last_seen"], r["t"])
        p["visits" if r["kind"] == "visit" else "analyses"] += 1
    return sorted(people.values(), key=lambda p: -p["last_seen"])


def browsers() -> list:
    """One entry per browser (visitor id): who they are if they signed in, how often they came, for how long,
    what they looked at, where from, and how many analyses they ran. Newest first."""
    rows = list(_rows())
    signed, analyses = {}, {}
    for r in rows:
        if r.get("vid") and r.get("kind") == "visit":
            signed.setdefault(r["vid"], set()).add(r["email"])
        if r.get("vid") and r.get("kind") == "analyze":
            analyses[r["vid"]] = analyses.get(r["vid"], 0) + 1
    out = {}
    for r in rows:
        if r.get("kind") not in ("hit", "leave"):
            continue
        key = r.get("vid") or f'{r.get("ip")}|{r.get("ua")}'
        b = out.setdefault(key, {"vid": r.get("vid", ""), "first_seen": r["t"], "last_seen": r["t"], "sids": set(), "time": {}, "clicks": {},
                                 "sections": set(), "scroll": 0, "returning": 0})
        b["first_seen"], b["last_seen"] = min(b["first_seen"], r["t"]), max(b["last_seen"], r["t"])
        sid = r.get("sid") or str(r["t"])
        b["sids"].add(sid)
        if r["kind"] == "hit":
            for k in ("device", "country", "region", "city", "tz", "lang", "screen", "ip", "ref", "utm"):
                if r.get(k):
                    b[k] = r[k]
            b["returning"] = max(b["returning"], int(r.get("returning") or 0))
            b["touch"] = bool(r.get("touch"))
        else:
            b["time"][sid] = max(b["time"].get(sid, 0), int(r.get("dur") or 0))
            b["clicks"][sid] = max(b["clicks"].get(sid, 0), int(r.get("clicks") or 0))
            b["scroll"] = max(b["scroll"], int(r.get("scroll") or 0))
            b["sections"].update(r.get("sections") or [])
    result = []
    for b in out.values():
        who = sorted(signed.get(b["vid"], [])) if b["vid"] else []
        result.append({**{k: v for k, v in b.items() if k not in ("sids", "time", "clicks", "sections")},
                       "visits": len(b["sids"]), "time_sec": sum(b["time"].values()), "clicks": sum(b["clicks"].values()),
                       "sections": sorted(b["sections"]), "signed_in_as": who, "analyses": analyses.get(b["vid"], 0)})
    return sorted(result, key=lambda x: -x["last_seen"])


def totals(people: list, brs: list) -> dict:
    now = time.time()
    timed = [b for b in brs if b["time_sec"]]
    return {
        "browsers": len(brs),
        "visits": sum(b["visits"] for b in brs),
        "never_signed_in": sum(1 for b in brs if not b["signed_in_as"]),
        "signed_in_people": len(people),
        "people_who_analysed": sum(1 for p in people if p["analyses"]),
        "analyses": sum(p["analyses"] for p in people),
        "browsers_24h": sum(1 for b in brs if now - b["last_seen"] < 86400),
        "browsers_7d": sum(1 for b in brs if now - b["last_seen"] < 7 * 86400),
        "avg_time_sec": int(sum(b["time_sec"] for b in timed) / max(1, len(timed))),
    }


# --- tracking events from the page: a visit when it loads, a summary when the visitor leaves --------
_HITS: dict = {}
EVENTS_PER_HOUR = 240


def _short_ip(ip: str) -> str:
    """Keep the network, drop the host part (a.b.c.0 or the first three IPv6 groups)."""
    if ":" in ip:
        return ":".join(ip.split(":")[:3]) + "::"
    parts = ip.split(".")
    return ".".join(parts[:3] + ["0"]) if len(parts) == 4 else ""


def _device(ua: str) -> str:
    ua = ua or ""
    browser = next((n for k, n in (("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"), ("Chrome/", "Chrome"), ("Safari/", "Safari")) if k in ua), "Other browser")
    system = next((n for k, n in (("Android", "Android"), ("iPhone", "iPhone"), ("iPad", "iPad"), ("Windows", "Windows"), ("Mac OS", "macOS"), ("Linux", "Linux")) if k in ua), "unknown system")
    bot = any(w in ua.lower() for w in ("bot", "crawler", "spider", "curl", "python-requests", "headless"))
    return f"{browser} on {system}" + (" (bot)" if bot else "")


def _s(v, n):
    return str(v or "")[:n]


def _i(v, lo, hi):
    try:
        return max(lo, min(int(v), hi))
    except (TypeError, ValueError):
        return lo


def record_event(ip: str, ua: str, p: dict) -> bool:
    """Log a page event ({"type": "visit"|"leave", ...}). Rate limited per IP; returns False when dropped."""
    kind = p.get("type")
    if kind not in ("visit", "leave"):
        return False
    now = time.time()
    with _LOCK:
        n, start = _HITS.get(ip, (0, now))
        if now - start > 3600:
            n, start = 0, now
        if n >= EVENTS_PER_HOUR:
            return False
        _HITS[ip] = (n + 1, start)
    row = {"t": int(now), "kind": "hit" if kind == "visit" else "leave", "email": "", "vid": _s(p.get("vid"), 64), "sid": _s(p.get("sid"), 32),
           "ip": _short_ip(ip), "ua": ua[:200], "device": _device(ua)}
    if kind == "visit":
        row.update({"country": _s(p.get("country"), 2).upper(), "region": _s(p.get("region"), 40), "city": _s(p.get("city"), 60), "tz": _s(p.get("tz"), 40),
                    "lang": _s(p.get("lang"), 12), "screen": _s(p.get("screen"), 16), "touch": bool(p.get("touch")), "ref": _s(p.get("ref"), 200),
                    "utm": _s(p.get("utm"), 80), "returning": _i(p.get("returning"), 0, 9999)})
    else:
        secs = p.get("sections") if isinstance(p.get("sections"), list) else []
        row.update({"dur": _i(p.get("dur"), 0, 6 * 3600), "scroll": _i(p.get("scroll"), 0, 100), "clicks": _i(p.get("clicks"), 0, 9999),
                    "sections": [_s(x, 30) for x in secs[:20]]})
    print("VISIT", json.dumps(row), flush=True)
    try:
        with _LOCK:
            os.makedirs(os.path.dirname(VISITS_FILE), exist_ok=True)
            with open(VISITS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    except OSError:
        pass
    return True


def require_admin(email: Optional[str] = Depends(require_user)) -> str:
    if not email or email not in ADMIN_EMAILS:
        raise HTTPException(403, "Only an admin can see the visitor list.")
    return email
