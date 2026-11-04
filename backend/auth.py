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
        if r.get("kind") == "hit":
            continue
        p = people.setdefault(r["email"], {"email": r["email"], "first_seen": r["t"], "last_seen": r["t"], "visits": 0, "analyses": 0})
        p["first_seen"], p["last_seen"] = min(p["first_seen"], r["t"]), max(p["last_seen"], r["t"])
        p["visits" if r["kind"] == "visit" else "analyses"] += 1
    return sorted(people.values(), key=lambda p: -p["last_seen"])


def anonymous() -> list:
    """Browsers that reached the login page, newest first, with the email they signed in as, if any."""
    signed = {}
    for r in _rows():
        if r.get("kind") == "visit" and r.get("vid"):
            signed.setdefault(r["vid"], set()).add(r["email"])
    browsers = {}
    for r in _rows():
        if r.get("kind") != "hit":
            continue
        key = r.get("vid") or f'{r.get("ip")}|{r.get("ua")}'
        b = browsers.setdefault(key, {"first_seen": r["t"], "last_seen": r["t"], "hits": 0, "signed_in_as": sorted(signed.get(r.get("vid"), []))})
        b["first_seen"], b["last_seen"] = min(b["first_seen"], r["t"]), max(b["last_seen"], r["t"])
        b["hits"] += 1
        for k in ("device", "country", "city", "ip", "ref"):
            if r.get(k):
                b[k] = r[k]
    return sorted(browsers.values(), key=lambda b: -b["last_seen"])


# --- anonymous visits: someone opened the site and got the login page -------------------------
_HITS: dict = {}
HITS_PER_HOUR = 40


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
    kind = "bot" if any(w in ua.lower() for w in ("bot", "crawler", "spider", "curl", "python-requests")) else ""
    return f"{browser} on {system}" + (" (bot)" if kind else "")


def record_hit(ip: str, ua: str, vid: str, country: str = "", city: str = "", ref: str = "") -> bool:
    """Log that a browser reached the login page. Rate limited per IP; returns False when dropped."""
    now = time.time()
    with _LOCK:
        n, start = _HITS.get(ip, (0, now))
        if now - start > 3600:
            n, start = 0, now
        if n >= HITS_PER_HOUR:
            return False
        _HITS[ip] = (n + 1, start)
    row = {"t": int(now), "kind": "hit", "email": "", "vid": vid[:64], "ip": _short_ip(ip), "ua": ua[:200], "device": _device(ua),
           "country": country[:2].upper(), "city": city[:60], "ref": ref[:200]}
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
