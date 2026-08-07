"""Lead capture — a deliberately self-contained module.

The rest of the application depends on exactly two names from here:
:class:`LeadRecord` (the payload) and :class:`LeadService` (the entry point).
Nothing in this package imports from ``app.chat``. That keeps the seam clean if
lead storage is later lifted into its own service or pointed at a CRM.
"""

from app.leads.models import (
    Contact,
    DeliveryStatus,
    HandoffReason,
    LeadRecord,
    QuotedPrice,
)
from app.leads.service import LeadService, get_lead_service
from app.leads.sink import LeadSink

__all__ = [
    "Contact",
    "DeliveryStatus",
    "HandoffReason",
    "LeadRecord",
    "LeadService",
    "LeadSink",
    "QuotedPrice",
    "get_lead_service",
]
