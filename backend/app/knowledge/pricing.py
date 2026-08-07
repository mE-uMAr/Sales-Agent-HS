"""Deterministic pricing lookup.

Prices deliberately do **not** go through the vector store. A model that
paraphrases a retrieved chunk will eventually transpose a digit or blend two
tiers together, and a wrong number in a sales conversation is a real-world
problem, not a quality metric.

So: prices live in ``content/public/pricing.yaml`` as typed records, and the
``lookup_pricing`` tool is an ordinary dictionary lookup. The model can only
relay what this module returns.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config import get_settings
from app.observability import get_logger

logger = get_logger(__name__)

#: Words too generic to identify a service on their own.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "app", "build", "building", "cost", "costs", "do",
        "for", "how", "i", "is", "it", "much", "my", "need", "of", "price",
        "pricing", "the", "to", "want", "we", "what", "with", "you", "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")


def _normalise(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


class PriceQuote(BaseModel):
    """One tier of one service, exactly as written in the catalog."""

    service_id: str
    service_name: str
    tier_id: str
    tier_name: str
    scope: str
    currency: str
    market_price: float
    our_price: float
    typical_timeline: str
    disclaimer: str

    @property
    def discount_pct(self) -> float:
        """Derived, never authored — so it cannot contradict the prices."""
        if self.market_price <= 0:
            return 0.0
        return round((1 - self.our_price / self.market_price) * 100, 1)

    @property
    def savings(self) -> float:
        return round(self.market_price - self.our_price, 2)

    def to_display(self) -> dict[str, Any]:
        """Flat, unambiguous shape handed to the model as a tool result."""
        return {
            "service_id": self.service_id,
            "service": self.service_name,
            "tier": self.tier_name,
            "scope": self.scope,
            "currency": self.currency,
            "typical_market_price": self.market_price,
            "our_price": self.our_price,
            "you_save": self.savings,
            "discount_pct": self.discount_pct,
            "typical_timeline": self.typical_timeline,
        }


class PricingTier(BaseModel):
    id: str
    name: str
    scope: str
    market_price: float
    our_price: float
    typical_timeline: str = ""


class PricingService(BaseModel):
    id: str
    name: str
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    tiers: list[PricingTier] = Field(default_factory=list)


class PricingCatalog:
    """In-memory view of the pricing file, with deterministic name matching."""

    def __init__(
        self,
        services: list[PricingService],
        currency: str = "USD",
        disclaimer: str = "",
    ) -> None:
        self.currency = currency
        self.disclaimer = disclaimer
        self._services = {service.id: service for service in services}

    # ── loading ──────────────────────────────────────────────────────
    @classmethod
    def from_file(cls, path: Path) -> PricingCatalog:
        if not path.exists():
            raise FileNotFoundError(f"pricing catalog not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        services = [PricingService.model_validate(item) for item in raw.get("services", [])]

        seen: set[str] = set()
        for service in services:
            if service.id in seen:
                raise ValueError(f"duplicate service id in pricing catalog: {service.id}")
            seen.add(service.id)
            if not service.tiers:
                raise ValueError(f"service '{service.id}' has no pricing tiers")

        logger.info(
            "pricing catalog loaded",
            extra={"services": len(services), "path": str(path)},
        )
        return cls(
            services=services,
            currency=raw.get("currency", "USD"),
            disclaimer=(raw.get("disclaimer") or "").strip(),
        )

    # ── access ───────────────────────────────────────────────────────
    @property
    def service_ids(self) -> list[str]:
        return list(self._services)

    def get(self, service_id: str) -> PricingService | None:
        return self._services.get(service_id.strip().lower())

    def match(self, query: str) -> PricingService | None:
        """Best service for a free-text phrase, or ``None``.

        Pure string matching in a fixed precedence order, so the same phrase
        always resolves to the same service and the result is testable without
        a model in the loop.
        """
        if not query or not query.strip():
            return None

        normalised = _normalise(query)

        # 1. The query is (or contains) a service id.
        for service_id in self._services:
            if normalised == service_id or service_id.replace("_", " ") == normalised:
                return self._services[service_id]

        # 2. Longest alias appearing in the query wins — "mobile app" beats "app".
        best: tuple[int, PricingService] | None = None
        for service in self._services.values():
            for alias in [service.name, *service.aliases]:
                alias_norm = _normalise(alias)
                matches = alias_norm and alias_norm in normalised
                if matches and (best is None or len(alias_norm) > best[0]):
                    best = (len(alias_norm), service)
        if best is not None:
            return best[1]

        # 3. Fall back to token overlap against name + aliases.
        query_tokens = _tokens(query)
        if not query_tokens:
            return None

        scored: list[tuple[int, PricingService]] = []
        for service in self._services.values():
            vocabulary = _tokens(" ".join([service.id.replace("_", " "), service.name, *service.aliases]))
            overlap = len(query_tokens & vocabulary)
            if overlap:
                scored.append((overlap, service))

        if not scored:
            return None
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored[0][1]

    def quote(self, service_id: str, tier_id: str | None = None) -> list[PriceQuote]:
        """Quotes for a service. All tiers unless ``tier_id`` narrows it."""
        service = self.get(service_id)
        if service is None:
            return []

        tiers = service.tiers
        if tier_id:
            wanted = tier_id.strip().lower()
            tiers = [t for t in service.tiers if t.id.lower() == wanted]
            if not tiers:
                return []

        return [
            PriceQuote(
                service_id=service.id,
                service_name=service.name,
                tier_id=tier.id,
                tier_name=tier.name,
                scope=tier.scope,
                currency=self.currency,
                market_price=tier.market_price,
                our_price=tier.our_price,
                typical_timeline=tier.typical_timeline,
                disclaimer=self.disclaimer,
            )
            for tier in tiers
        ]

    def catalog_overview(self) -> str:
        """Compact service list injected into the system prompt.

        Gives the model the exact ``service_id`` values it may pass to the
        pricing tool, without putting a single number in the prompt.
        """
        lines = [
            f"- {service.id}: {service.name} — {' '.join(service.summary.split())}"
            for service in self._services.values()
        ]
        return "\n".join(lines)


_catalog: PricingCatalog | None = None
_lock = threading.Lock()


def get_catalog(*, reload: bool = False) -> PricingCatalog:
    global _catalog
    with _lock:
        if _catalog is None or reload:
            settings = get_settings()
            _catalog = PricingCatalog.from_file(settings.public_content_dir / "pricing.yaml")
        return _catalog
