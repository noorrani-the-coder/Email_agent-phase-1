from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any

from db.models import EmailMemory
from db.session import get_session, init_db


def _normalize_timestamp(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10**12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.isoformat()
        except Exception:
            try:
                ts = float(value)
                if ts > 10**12:
                    ts = ts / 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            except Exception:
                return value
    return str(value)


def persist_observation(observed: dict[str, Any]) -> EmailMemory:
    init_db()
    session = get_session()
    try:
        email_id = observed.get("email_id") or observed.get("id") or ""
        record = session.query(EmailMemory).filter_by(email_id=email_id).first()

        sender = observed.get("from") or observed.get("sender") or ""
        _, addr = parseaddr(sender)

        subject = observed.get("subject") or ""
        body = observed.get("content") or observed.get("body") or ""
        timestamp = _normalize_timestamp(observed.get("timestamp"))

        if record:
            record.sender = sender
            record.subject = subject
            record.body = body
            record.timestamp = timestamp
            session.add(record)
        else:
            session.add(
                EmailMemory(
                    email_id=email_id,
                    sender=sender,
                    sender_type="unknown",
                    promo=False,
                    urgency="",
                    subject=subject,
                    body=body,
                    timestamp=timestamp,
                )
            )
        session.commit()
        return record or session.query(EmailMemory).filter_by(email_id=email_id).first()
    finally:
        session.close()


def store_reply_draft(observed: dict[str, Any], reply: str) -> None:
    init_db()
    session = get_session()
    try:
        email_id = observed.get("email_id") or observed.get("id") or ""
        record = session.query(EmailMemory).filter_by(email_id=email_id).first()
        if not record:
            return
        record.reply_draft = reply or ""
        record.reply_timestamp = datetime.now(tz=timezone.utc).isoformat()
        session.add(record)
        session.commit()
    finally:
        session.close()


def store_action_state(
    observed: dict[str, Any],
    next_action: str,
    action_reason: str = "",
    *,
    task_status: str | None = None,
    urgent_flag: bool | None = None,
    needs_human_review: bool | None = None,
    reply_json: str | None = None,
) -> None:
    init_db()
    session = get_session()
    try:
        email_id = observed.get("email_id") or observed.get("id") or ""
        record = session.query(EmailMemory).filter_by(email_id=email_id).first()
        if not record:
            return
        record.next_action = (next_action or "").strip()
        record.action_reason = (action_reason or "").strip()
        record.action_timestamp = datetime.now(tz=timezone.utc).isoformat()
        if task_status is not None:
            record.task_status = task_status
        if urgent_flag is not None:
            record.urgent_flag = bool(urgent_flag)
        if needs_human_review is not None:
            record.needs_human_review = bool(needs_human_review)
        if reply_json is not None:
            record.reply_draft = reply_json
            record.reply_timestamp = datetime.now(tz=timezone.utc).isoformat()
        session.add(record)
        session.commit()
    finally:
        session.close()
