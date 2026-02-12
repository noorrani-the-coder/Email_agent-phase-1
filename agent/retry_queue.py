from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.actions import execute_next_action
from agent.decision import analyze_email_with_status
from db.models import RetryQueue
from db.session import get_session, init_db

_RETRY_MAX_ATTEMPTS = max(1, int(os.getenv("RETRY_MAX_ATTEMPTS", "8")))
_RETRY_BASE_DELAY_SECONDS = max(1.0, float(os.getenv("RETRY_BASE_DELAY_SECONDS", "15")))
_RETRY_MAX_DELAY_SECONDS = max(1.0, float(os.getenv("RETRY_MAX_DELAY_SECONDS", "3600")))
_RETRY_BATCH_SIZE = max(1, int(os.getenv("RETRY_BATCH_SIZE", "10")))


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _next_retry_timestamp(attempts: int) -> str:
    delay = min(_RETRY_MAX_DELAY_SECONDS, _RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempts - 1)))
    return _iso(_now() + timedelta(seconds=delay))


def enqueue_retry(observed: dict[str, Any], operation: str, error: str = "") -> None:
    init_db()
    session = get_session()
    now = _iso(_now())
    try:
        email_id = observed.get("email_id") or observed.get("id") or ""
        if not email_id:
            return

        pending = (
            session.query(RetryQueue)
            .filter_by(email_id=email_id, operation=operation, status="pending")
            .order_by(RetryQueue.id.desc())
            .first()
        )
        if pending:
            pending.payload = json.dumps({"observed": observed}, ensure_ascii=True)
            pending.last_error = error or pending.last_error
            pending.updated_at = now
            session.add(pending)
            session.commit()
            return

        session.add(
            RetryQueue(
                email_id=email_id,
                operation=operation or "analyze_and_execute",
                payload=json.dumps({"observed": observed}, ensure_ascii=True),
                status="pending",
                attempts=0,
                next_retry_at=now,
                last_error=error or "",
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    finally:
        session.close()


def _mark_done(session, row: RetryQueue) -> None:
    row.status = "done"
    row.updated_at = _iso(_now())
    session.add(row)
    session.commit()


def _schedule_retry(session, row: RetryQueue, error: str) -> None:
    row.attempts = int(row.attempts or 0) + 1
    row.last_error = error
    row.updated_at = _iso(_now())
    if row.attempts >= _RETRY_MAX_ATTEMPTS:
        row.status = "failed"
    else:
        row.status = "pending"
        row.next_retry_at = _next_retry_timestamp(row.attempts)
    session.add(row)
    session.commit()


def _run_analyze_and_execute(observed: dict[str, Any]) -> tuple[bool, str]:
    analysis, analysis_ok = analyze_email_with_status(observed)
    if not analysis_ok:
        return False, str(analysis.get("Reasoning", "analysis failed"))
    _, action_ok, action_error = execute_next_action(observed, analysis)
    return action_ok, action_error


def process_retry_queue(limit: int | None = None) -> int:
    init_db()
    session = get_session()
    processed = 0
    now = _now()
    batch_limit = max(1, int(limit or _RETRY_BATCH_SIZE))
    try:
        rows = (
            session.query(RetryQueue)
            .filter_by(status="pending")
            .order_by(RetryQueue.id.asc())
            .all()
        )
        for row in rows:
            if processed >= batch_limit:
                break
            due = _parse_iso(row.next_retry_at)
            if due and due > now:
                continue

            payload = {}
            try:
                payload = json.loads(row.payload or "{}")
            except Exception:
                payload = {}
            observed = payload.get("observed") if isinstance(payload, dict) else None
            if not isinstance(observed, dict):
                _schedule_retry(session, row, "retry payload invalid")
                processed += 1
                continue

            try:
                if row.operation in {"analyze_and_execute", "analyze_and_draft"}:
                    ok, error = _run_analyze_and_execute(observed)
                else:
                    ok, error = False, f"unsupported retry operation: {row.operation}"
            except Exception as exc:
                ok = False
                error = f"retry execution failed: {exc.__class__.__name__}"

            if ok:
                _mark_done(session, row)
            else:
                _schedule_retry(session, row, error or "retry failed")
            processed += 1
        return processed
    finally:
        session.close()
