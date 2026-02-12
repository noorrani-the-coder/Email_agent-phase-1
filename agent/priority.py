from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any

from ai.llm import call_llm
from ai.prompts import PRIORITY_PROMPT
from db.models import EmailMemory
from db.session import get_session, init_db

logger = logging.getLogger(__name__)


@dataclass
class PriorityResult:
    label: str
    score: int
    confidence: float
    reasons: list[str]
    llm_hook: dict[str, Any]
    tier: str


def _parse_email_address(value: str) -> tuple[str, str]:
    _, addr = parseaddr(value or "")
    domain = addr.split("@")[-1].lower() if "@" in addr else ""
    return addr, domain


def _normalize_timestamp(value: Any) -> tuple[str, datetime | None]:
    if value is None or value == "":
        return "", None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10**12:
            timestamp = timestamp / 1000.0
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat(), dt
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.isoformat(), dt
        except Exception:
            try:
                timestamp = float(value)
                if timestamp > 10**12:
                    timestamp = timestamp / 1000.0
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return dt.isoformat(), dt
            except Exception:
                return value, None
    return str(value), None


def _history_stats(session, sender: str) -> dict[str, Any]:
    if not sender:
        return {"count": 0, "last_timestamp": None, "recent_decisions": []}

    rows = session.query(EmailMemory).filter_by(sender=sender).all()
    if not rows:
        return {"count": 0, "last_timestamp": None, "recent_decisions": []}

    timestamps = [r.timestamp for r in rows if r.timestamp]
    last_ts = max(timestamps) if timestamps else None
    recent = [
        {
            "label": r.priority_label,
            "confidence": r.priority_confidence,
            "reasons": r.priority_reasons.split(" | ") if r.priority_reasons else [],
            "timestamp": r.decision_timestamp,
        }
        for r in rows
        if r.priority_label
    ]
    recent = sorted(recent, key=lambda x: x.get("timestamp") or "", reverse=True)[:5]
    return {
        "count": len(rows),
        "last_timestamp": last_ts,
        "recent_decisions": recent,
    }


def _build_context(email: dict[str, Any], record: EmailMemory | None, session) -> dict[str, Any]:
    sender = (record.sender if record else email.get("from") or email.get("sender") or "")
    subject = (record.subject if record else email.get("subject") or "")
    body = (record.body if record else email.get("content") or email.get("body") or "")
    sender_addr, sender_domain = _parse_email_address(sender)
    timestamp_value = record.timestamp if record else email.get("timestamp") or ""
    timestamp_iso, _ = _normalize_timestamp(timestamp_value)
    history = _history_stats(session, sender_addr or sender)

    return {
        "email_id": email.get("email_id") or email.get("id") or "",
        "thread_id": email.get("thread_id") or "",
        "sender": sender,
        "sender_address": sender_addr,
        "sender_domain": sender_domain,
        "subject": subject,
        "body": body[:4000],
        "timestamp": timestamp_iso,
        "history": {
            "count": history["count"],
            "last_timestamp": history["last_timestamp"],
            "recent_decisions": history["recent_decisions"],
        },
    }


def _coerce_llm_output(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _validate_llm_output(payload: dict[str, Any]) -> tuple[str, float, list[str]]:
    label = str(payload.get("label", "")).strip().lower()
    if label not in {"high", "medium", "low"}:
        raise ValueError("invalid label")

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError("invalid confidence")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence out of range")

    reasons = payload.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        raise ValueError("invalid reasons")
    reasons = [r.strip() for r in reasons if r.strip()][:4]
    if not reasons:
        reasons = ["llm decision"]

    return label, confidence, reasons


def compute_priority(email: dict[str, Any]) -> PriorityResult:
    init_db()
    session = get_session()
    try:
        email_id = email.get("email_id") or email.get("id")
        record = None
        if email_id:
            record = session.query(EmailMemory).filter_by(email_id=email_id).first()

        context = _build_context(email, record, session)
        user_payload = json.dumps(context, ensure_ascii=True)
        raw = call_llm(PRIORITY_PROMPT, user_payload, temperature=0)

        try:
            payload = _coerce_llm_output(raw)
            if payload is None:
                raise ValueError("non-json response")
            label, confidence, reasons = _validate_llm_output(payload)
            error = None
        except Exception as exc:
            label = "medium"
            confidence = 0.2
            reasons = ["invalid llm response"]
            error = exc

        threshold = 0.6
        tier = "tier-2" if confidence < threshold else "tier-1"
        llm_hook = {
            "eligible": confidence < threshold,
            "reason": "low confidence" if confidence < threshold else "not needed",
            "prompt_seed": {
                "email_id": context.get("email_id"),
                "sender": context.get("sender"),
                "subject": context.get("subject"),
                "timestamp": context.get("timestamp"),
                "history_count": context.get("history", {}).get("count", 0),
            },
        }

        if error:
            logger.warning(
                "priority llm parse failed",
                extra={
                    "email_id": context.get("email_id"),
                    "error": str(error),
                    "raw": raw[:500],
                },
            )
        else:
            logger.info(
                "priority llm decision",
                extra={
                    "email_id": context.get("email_id"),
                    "label": label,
                    "confidence": confidence,
                    "tier": tier,
                },
            )

        score = int(round(confidence * 100))

        decision_ts = datetime.now(tz=timezone.utc).isoformat()
        if record:
            record.priority_label = label
            record.priority_confidence = confidence
            record.priority_score = score
            record.priority_reasons = " | ".join(reasons)
            record.priority_tier = tier
            record.decision_timestamp = decision_ts
            session.add(record)
            session.commit()
        else:
            sender = context.get("sender") or ""
            session.add(
                EmailMemory(
                    email_id=context.get("email_id") or "",
                    sender=sender,
                    sender_type="unknown",
                    promo=False,
                    urgency="",
                    subject=context.get("subject") or "",
                    body=context.get("body") or "",
                    timestamp=context.get("timestamp") or "",
                    priority_label=label,
                    priority_confidence=confidence,
                    priority_score=score,
                    priority_reasons=" | ".join(reasons),
                    priority_tier=tier,
                    decision_timestamp=decision_ts,
                )
            )
            session.commit()

        return PriorityResult(
            label=label,
            score=score,
            confidence=confidence,
            reasons=reasons,
            llm_hook=llm_hook,
            tier=tier,
        )
    finally:
        session.close()
