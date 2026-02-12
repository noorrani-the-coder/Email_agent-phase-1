from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any

from db.models import BehaviorLog
from db.session import get_session, init_db

FINAL_ACTIONS = {"sent_reply", "ignored", "edited_draft", "deleted"}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def sender_domain_from_observed(observed: dict[str, Any]) -> str:
    sender = str(observed.get("from") or observed.get("sender") or "")
    _, addr = parseaddr(sender)
    if "@" not in addr:
        return ""
    return addr.split("@", 1)[-1].strip().lower()


def log_behavior_event(
    *,
    email_id: str,
    intent: str,
    sender_domain: str,
    requires_reply: bool | None,
    proposed_action: str,
    agent_action: str,
    llm_confidence: float,
    behavior_match_score: float,
    final_decision_score: float,
    user_final_action: str = "",
    user_opened: bool | None = None,
) -> None:
    if not email_id:
        return
    init_db()
    session = get_session()
    now = _now_iso()
    try:
        row = session.query(BehaviorLog).filter_by(email_id=email_id).first()
        clean_final = user_final_action.strip().lower()
        if clean_final not in FINAL_ACTIONS:
            clean_final = ""

        if row:
            row.intent = (intent or "").strip()
            row.sender_domain = (sender_domain or "").strip().lower()
            row.requires_reply = bool(requires_reply) if isinstance(requires_reply, bool) else False
            row.proposed_action = (proposed_action or "").strip().lower()
            row.agent_action = (agent_action or "").strip().lower()
            if user_opened is not None:
                row.user_opened = bool(user_opened)
            row.llm_confidence = max(0.0, min(1.0, float(llm_confidence or 0.0)))
            row.behavior_match_score = max(0.0, min(1.0, float(behavior_match_score or 0.0)))
            row.final_decision_score = max(0.0, min(1.0, float(final_decision_score or 0.0)))
            if clean_final:
                row.user_final_action = clean_final
            row.updated_at = now
            session.add(row)
            session.commit()
            return

        session.add(
            BehaviorLog(
                email_id=email_id,
                intent=(intent or "").strip(),
                sender_domain=(sender_domain or "").strip().lower(),
                requires_reply=bool(requires_reply) if isinstance(requires_reply, bool) else False,
                user_final_action=clean_final,
                user_opened=bool(user_opened) if user_opened is not None else False,
                proposed_action=(proposed_action or "").strip().lower(),
                agent_action=(agent_action or "").strip().lower(),
                llm_confidence=max(0.0, min(1.0, float(llm_confidence or 0.0))),
                behavior_match_score=max(0.0, min(1.0, float(behavior_match_score or 0.0))),
                final_decision_score=max(0.0, min(1.0, float(final_decision_score or 0.0))),
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    finally:
        session.close()


def record_user_final_action(email_id: str, user_final_action: str) -> bool:
    clean = (user_final_action or "").strip().lower()
    if clean not in FINAL_ACTIONS:
        return False
    init_db()
    session = get_session()
    try:
        row = session.query(BehaviorLog).filter_by(email_id=email_id).first()
        if not row:
            return False
        row.user_final_action = clean
        row.user_opened = True
        row.updated_at = _now_iso()
        session.add(row)
        session.commit()
        return True
    finally:
        session.close()


def record_user_opened(email_id: str) -> bool:
    if not email_id:
        return False
    init_db()
    session = get_session()
    try:
        row = session.query(BehaviorLog).filter_by(email_id=email_id).first()
        if not row:
            return False
        row.user_opened = True
        row.updated_at = _now_iso()
        session.add(row)
        session.commit()
        return True
    finally:
        session.close()


def _is_reply_action(user_final_action: str) -> bool:
    clean = (user_final_action or "").strip().lower()
    return clean in {"sent_reply", "edited_draft"}


def _is_manual_override(agent_action: str, user_final_action: str) -> bool:
    action = (agent_action or "").strip().lower()
    final = (user_final_action or "").strip().lower()
    expected = {
        "draft_reply": {"sent_reply", "edited_draft"},
        "ignore": {"ignored", "deleted"},
        "create_task": {"edited_draft", "sent_reply"},
        "flag_high_urgency": {"sent_reply", "edited_draft", "ignored"},
        "escalate_human_review": {"edited_draft", "sent_reply", "ignored"},
    }
    allowed = expected.get(action)
    if not allowed:
        return False
    return final not in allowed


def _safe_rate(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return max(0.0, min(1.0, numerator / denominator))


def compute_behavior_profile(intent: str, sender_domain: str) -> dict[str, float | int]:
    init_db()
    session = get_session()
    try:
        rows = session.query(BehaviorLog).all()
        clean_intent = (intent or "").strip().lower()
        clean_domain = (sender_domain or "").strip().lower()

        sender_total = 0
        sender_replies = 0
        intent_total = 0
        intent_replies = 0
        open_total = 0
        open_opened = 0
        auto_total = 0
        overrides = 0

        for row in rows:
            row_intent = (row.intent or "").strip().lower()
            row_domain = (row.sender_domain or "").strip().lower()
            row_final = (row.user_final_action or "").strip().lower()

            if row_domain == clean_domain and clean_domain:
                sender_total += 1
                if row_final in FINAL_ACTIONS and _is_reply_action(row_final):
                    sender_replies += 1

            if row_intent == clean_intent and clean_intent:
                intent_total += 1
                if row_final in FINAL_ACTIONS and _is_reply_action(row_final):
                    intent_replies += 1

            if row_domain == clean_domain and clean_domain:
                open_total += 1
                if bool(getattr(row, "user_opened", False)):
                    open_opened += 1

            if row_final not in FINAL_ACTIONS:
                continue
            auto_total += 1
            if _is_manual_override(row.agent_action, row_final):
                overrides += 1

        sample_size = max(sender_total, intent_total, auto_total)
        if sample_size <= 0:
            return {
                "reply_rate_by_sender": 0.0,
                "reply_rate_by_intent": 0.0,
                "open_rate": 0.0,
                "manual_override_rate": 0.0,
                "importance_score": 0.0,
                "sample_size": 0,
            }

        reply_rate_by_sender = _safe_rate(sender_replies, sender_total, default=0.0)
        reply_rate_by_intent = _safe_rate(intent_replies, intent_total, default=0.0)
        open_rate = _safe_rate(open_opened, open_total, default=0.0)
        manual_override_rate = _safe_rate(overrides, auto_total, default=0.0)

        # Unified behavior importance: reply-history dominant, open-rate minor.
        importance_score = (
            (0.60 * reply_rate_by_sender)
            + (0.30 * reply_rate_by_intent)
            + (0.05 * open_rate)
            + (0.05 * (1.0 - manual_override_rate))
        )

        return {
            "reply_rate_by_sender": max(0.0, min(1.0, reply_rate_by_sender)),
            "reply_rate_by_intent": max(0.0, min(1.0, reply_rate_by_intent)),
            "open_rate": max(0.0, min(1.0, open_rate)),
            "manual_override_rate": max(0.0, min(1.0, manual_override_rate)),
            "importance_score": max(0.0, min(1.0, importance_score)),
            "sample_size": int(sample_size),
        }
    finally:
        session.close()
