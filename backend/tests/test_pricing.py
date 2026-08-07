"""Pricing must be exact. These tests are the reason it is not a RAG lookup."""

from __future__ import annotations

import pytest
import yaml

from app.config import get_settings
from app.knowledge.pricing import PricingCatalog


def test_every_quote_byte_matches_the_yaml(catalog: PricingCatalog) -> None:
    """A quoted number must equal the file, not merely resemble it."""
    raw = yaml.safe_load(
        (get_settings().public_content_dir / "pricing.yaml").read_text()
    )

    for service in raw["services"]:
        for tier in service["tiers"]:
            quote = next(
                q
                for q in catalog.quote(service["id"])
                if q.tier_id == tier["id"]
            )
            assert quote.market_price == tier["market_price"]
            assert quote.our_price == tier["our_price"]
            assert quote.service_name == service["name"]
            assert quote.currency == raw["currency"]


def test_discount_is_derived_not_authored(catalog: PricingCatalog) -> None:
    """The discount cannot contradict the prices because it is computed."""
    quote = catalog.quote("web_app", "starter")[0]
    expected = round((1 - quote.our_price / quote.market_price) * 100, 1)
    assert quote.discount_pct == expected
    assert quote.savings == quote.market_price - quote.our_price


def test_our_price_is_always_below_market(catalog: PricingCatalog) -> None:
    for service_id in catalog.service_ids:
        for quote in catalog.quote(service_id):
            assert quote.our_price < quote.market_price, (
                f"{service_id}/{quote.tier_id} is not actually a discount"
            )


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("I need a mobile app", "mobile_app"),
        ("we want an ios app", "mobile_app"),
        ("how much for an online store", "ecommerce"),
        ("shopify storefront", "ecommerce"),
        ("a chatbot over our docs", "ai_integration"),
        ("we need a customer portal dashboard", "web_app"),
        ("help with our aws infrastructure", "cloud_devops"),
        ("a monthly support retainer", "support"),
        ("redesign our ux", "ui_ux"),
    ],
)
def test_free_text_resolves_deterministically(
    catalog: PricingCatalog, phrase: str, expected: str
) -> None:
    assert catalog.match(phrase) is not None
    assert catalog.match(phrase).id == expected


def test_longest_alias_wins(catalog: PricingCatalog) -> None:
    """'mobile app' must beat the shorter 'app' alias on web_app."""
    assert catalog.match("we need a mobile app built").id == "mobile_app"


def test_unmatchable_phrase_returns_none(catalog: PricingCatalog) -> None:
    assert catalog.match("xyzzy plugh nothing relevant") is None
    assert catalog.match("") is None


def test_unknown_service_and_tier_yield_no_quotes(catalog: PricingCatalog) -> None:
    assert catalog.quote("not_a_service") == []
    assert catalog.quote("web_app", "not_a_tier") == []


def test_catalog_overview_names_ids_but_no_prices(catalog: PricingCatalog) -> None:
    """The prompt gets service ids; numbers stay behind the tool."""
    overview = catalog.catalog_overview()
    for service_id in catalog.service_ids:
        assert service_id in overview
    for service_id in catalog.service_ids:
        for quote in catalog.quote(service_id):
            assert str(int(quote.our_price)) not in overview
