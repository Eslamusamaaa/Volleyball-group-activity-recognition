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


def record(email: Optional[str], kind: str, detail: str = "") -> None:
    """Append one visit/analysis event; never lets logging break a request."""
    if not email:
        return
    row = {"t": int(time.time()), "email": email, "kind": kind, "detail": detail[:200]}
    print("VISIT", json.dumps(row), flush=True)
    try:
        with _LOCK:
            os.makedirs(os.path.dirname(VISITS_FILE), exist_ok=True)
            with open(VISITS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def visitors() -> list:
    """Per-email summary: first/last seen, sign-ins, analyses."""
    people = {}
    try:
        with open(VISITS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                p = people.setdefault(r["email"], {"email": r["email"], "first_seen": r["t"], "last_seen": r["t"], "visits": 0, "analyses": 0})
                p["first_seen"], p["last_seen"] = min(p["first_seen"], r["t"]), max(p["last_seen"], r["t"])
                p["visits" if r["kind"] == "visit" else "analyses"] += 1
    except OSError:
        pass
    return sorted(people.values(), key=lambda p: -p["last_seen"])


def require_admin(email: Optional[str] = Depends(require_user)) -> str:
    if not email or email not in ADMIN_EMAILS:
        raise HTTPException(403, "Only an admin can see the visitor list.")
    return email
