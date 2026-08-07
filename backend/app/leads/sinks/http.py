"""Forward leads to an external CRM over HTTP.

Signs each request so the receiver can verify it came from us, and sends a
stable ``Idempotency-Key`` so a retried delivery does not create a duplicate
record downstream.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime

import httpx

from app.leads.models import LeadRecord
from app.leads.sink import LeadDeliveryError
from app.observability import get_logger

logger = get_logger(__name__)

#: Statuses worth trying again. Everything else in 4xx is our bug, not a blip.
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpLeadSink:
    """POST the lead as JSON to a webhook endpoint."""

    name = "http"

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._secret = secret
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def _sign(self, body: bytes, timestamp: str) -> str:
        digest = hmac.new(
            self._secret.encode(),  # type: ignore[union-attr]
            f"{timestamp}.".encode() + body,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"

    async def deliver(self, record: LeadRecord) -> str:
        body = json.dumps(
            record.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        ).encode()

        timestamp = datetime.now(UTC).isoformat()
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": record.id,
            "X-Hashed-Timestamp": timestamp,
        }
        if self._secret:
            headers["X-Hashed-Signature"] = self._sign(body, timestamp)

        try:
            response = await self._client.post(self._url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LeadDeliveryError(f"transport error: {exc}", retryable=True) from exc

        if response.is_success:
            reference = record.id
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    reference = str(payload.get("id") or payload.get("reference") or reference)
            except ValueError:
                pass  # A bare 200 with no body is a perfectly good acknowledgement.
            logger.info(
                "lead delivered",
                extra={"lead_id": record.id, "reference": reference},
            )
            return reference

        raise LeadDeliveryError(
            f"CRM returned {response.status_code}: {response.text[:300]}",
            retryable=response.status_code in RETRYABLE_STATUSES,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
