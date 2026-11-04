"""Google sign-in for the costly endpoints.

The browser signs in with Google Identity Services and sends the ID token as
`Authorization: Bearer <token>`. The token is verified here against GOOGLE_CLIENT_ID.
Optional ALLOWED_EMAILS (comma separated) limits access to those accounts.
With no GOOGLE_CLIENT_ID set, auth is off (local development).
"""
import os
from typing import Optional

from fastapi import Header, HTTPException

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
ALLOWED_EMAILS = {e.strip().lower() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()}


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
