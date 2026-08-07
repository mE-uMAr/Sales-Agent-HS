"""Session tokens, rate limiting and admin auth.

The threat here is not a determined attacker stealing secrets — there are none
in the chat surface. It is an open endpoint that spends money: every unmetered
POST is an LLM call on someone else's card. So the design is about binding a
conversation to the browser that started it and capping how fast anyone can
talk.

A session token is an HMAC over ``session_id`` and an expiry. It is not a
bearer credential for anything valuable; it just proves the caller is the one
who opened this conversation.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.observability import get_logger

logger = get_logger(__name__)


# ── session tokens ───────────────────────────────────────────────────
def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_session_token(session_id: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    expires_at = int(time.time()) + settings.session_token_ttl_minutes * 60
    payload = f"{session_id}.{expires_at}"
    secret = settings.session_token_secret.get_secret_value()
    return f"{payload}.{_sign(payload, secret)}"


def verify_session_token(
    token: str, session_id: str, settings: Settings | None = None
) -> bool:
    settings = settings or get_settings()
    try:
        token_session_id, expires_raw, signature = token.rsplit(".", 2)
        expires_at = int(expires_raw)
    except (ValueError, AttributeError):
        return False

    if token_session_id != session_id:
        return False
    if expires_at < time.time():
        return False

    expected = _sign(f"{token_session_id}.{expires_at}", settings.session_token_secret.get_secret_value())
    return hmac.compare_digest(expected, signature)


# ── rate limiting ────────────────────────────────────────────────────
@dataclass
class _Window:
    limit: int
    period_seconds: float


class SlidingWindowLimiter:
    """In-process sliding window.

    Adequate for a single instance, which is what this service is sized for.
    Running more than one replica means moving these counters to Redis —
    otherwise each replica enforces the limit independently.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, window: _Window) -> bool:
        now = time.monotonic()
        cutoff = now - window.period_seconds
        hits = self._hits[key]

        while hits and hits[0] < cutoff:
            hits.popleft()

        if len(hits) >= window.limit:
            return False

        hits.append(now)
        return True

    def prune(self, older_than_seconds: float = 3600.0) -> int:
        """Drop idle keys so the dict does not grow without bound."""
        cutoff = time.monotonic() - older_than_seconds
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] < cutoff]
        for key in stale:
            del self._hits[key]
        return len(stale)


limiter = SlidingWindowLimiter()


def client_fingerprint(request: Request) -> str:
    """A stable, non-identifying key for rate limiting.

    ``X-Forwarded-For`` is trusted only because this service is expected to sit
    behind a reverse proxy that sets it. If yours does not, strip the header at
    the edge or this becomes spoofable.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


# ── dependencies ─────────────────────────────────────────────────────
def require_widget_key(
    request: Request, x_widget_key: str | None = Header(default=None)
) -> str:
    """Gate session creation on the site's public key, then rate limit."""
    settings = get_settings()

    if not hmac.compare_digest(x_widget_key or "", settings.widget_public_key):
        logger.warning("rejected session start with a bad widget key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid widget key"
        )

    fingerprint = client_fingerprint(request)
    if not limiter.check(
        f"start:{fingerprint}",
        _Window(limit=settings.rate_limit_sessions_per_hour, period_seconds=3600),
    ):
        logger.warning("session start rate limited", extra={"client": fingerprint})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many conversations started; try again later",
        )

    return fingerprint


def require_session_token(
    session_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    """Prove the caller opened this conversation, then rate limit the turn."""
    settings = get_settings()
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if not verify_session_token(token, session_id, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired session token",
        )

    if not limiter.check(
        f"msg:{session_id}",
        _Window(limit=settings.rate_limit_messages_per_minute, period_seconds=60),
    ):
        logger.warning("message rate limited", extra={"session_id": session_id})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="you're sending messages too quickly; give it a moment",
        )

    return session_id


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = settings.admin_api_key.get_secret_value()

    if settings.is_production and expected == "change_me_in_production":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin API key is not configured",
        )

    if not hmac.compare_digest(x_admin_key or "", expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin key"
        )
