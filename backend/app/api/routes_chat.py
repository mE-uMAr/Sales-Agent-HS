"""The three endpoints the website calls."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    CloseSessionOut,
    MessageIn,
    MessageOut,
    StartSessionIn,
    StartSessionOut,
)
from app.api.security import (
    issue_session_token,
    require_session_token,
    require_widget_key,
)
from app.chat.copy import HANDOFF_LINE
from app.chat.service import (
    SessionClosedError,
    SessionNotFoundError,
    StartSessionRequest,
    get_chat_service,
)
from app.config import get_settings
from app.leads import HandoffReason
from app.observability import get_logger, session_id_var

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/sessions",
    response_model=StartSessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Start a conversation from the contact-form payload",
)
async def start_session(
    payload: StartSessionIn,
    fingerprint: str = Depends(require_widget_key),
) -> StartSessionOut:
    settings = get_settings()

    session_id, opening = await get_chat_service().start_session(
        StartSessionRequest(
            name=payload.name,
            email=payload.email,
            phone=payload.phone,
            company=payload.company,
            page_url=payload.page_url,
            utm=payload.utm,
            client_hash=fingerprint,
        )
    )

    return StartSessionOut(
        session_id=session_id,
        token=issue_session_token(session_id, settings),
        expires_in=settings.session_token_ttl_minutes * 60,
        message=opening,
        stage="greeting",
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=MessageOut,
    summary="Send a visitor message and get the assistant's reply",
)
async def send_message(
    payload: MessageIn,
    session_id: str = Depends(require_session_token),
) -> MessageOut:
    session_id_var.set(session_id)

    try:
        result = await get_chat_service().send_message(session_id, payload.message)
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="session not found"
        ) from None
    except SessionClosedError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this conversation has already ended",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from None

    return MessageOut(
        session_id=result.session_id,
        reply=result.reply,
        stage=result.stage,
        closed=result.closed,
        handoff_reason=result.handoff_reason,
    )


@router.post(
    "/sessions/{session_id}/close",
    response_model=CloseSessionOut,
    summary="End the conversation and capture the lead",
)
async def close_session(
    session_id: str = Depends(require_session_token),
) -> CloseSessionOut:
    session_id_var.set(session_id)

    try:
        lead = await get_chat_service().close_session(
            session_id, HandoffReason.COMPLETED
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="session not found"
        ) from None

    # `lead is None` means it was already closed — still a success, and still
    # exactly one lead. Closing twice must not look like a failure to the client.
    return CloseSessionOut(
        session_id=session_id,
        closed=True,
        lead_captured=lead is not None,
        message=HANDOFF_LINE,
    )
