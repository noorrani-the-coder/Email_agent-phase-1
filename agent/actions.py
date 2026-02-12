from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agent.behavior import compute_behavior_profile, sender_domain_from_observed
from agent.decision import generate_reply_with_status
from agent.persist import store_action_state
from db.models import TaskQueue
from db.session import get_session, init_db

ALLOWED_ACTIONS = {
    "ignore",
    "draft_reply",
    "create_task",
    "flag_high_urgency",
    "escalate_human_review",
}
_DRAFT_MIN_FINAL_SCORE = 0.55
_DRAFT_AUTO_THRESHOLD = 0.65
_REVIEW_TASK_THRESHOLD = 0.60
_REVIEW_THRESHOLD = 0.45
_MAX_BEHAVIOR_INFLUENCE = 0.40
_FULL_BEHAVIOR_AT_SAMPLES = 25
_LOW_SAMPLE_INVARIANT_LIMIT = 8
_CLEAR_IGNORE_CONFIDENCE = 0.90


def _safe_action(value: Any, requires_reply: Any) -> str:
    if isinstance(value, str):
        action = value.strip().lower()
        if action in ALLOWED_ACTIONS:
            return action
    if requires_reply is True:
        return "draft_reply"
    if requires_reply is False:
        return "ignore"
    return "escalate_human_review"


def _safe_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _enqueue_task(observed: dict[str, Any], reason: str) -> None:
    init_db()
    session = get_session()
    now = datetime.now(tz=timezone.utc).isoformat()
    try:
        email_id = observed.get("email_id") or observed.get("id") or ""
        if not email_id:
            return
        title = str(observed.get("subject", "")).strip() or "Email follow-up task"
        body = str(observed.get("content", "")).strip()
        description = reason.strip() or "Follow up required based on email analysis."
        if body:
            description = f"{description}\n\nEmail excerpt:\n{body[:1500]}"

        row = session.query(TaskQueue).filter_by(email_id=email_id).first()
        if row:
            row.title = title
            row.description = description
            if not row.status:
                row.status = "open"
            row.updated_at = now
            session.add(row)
        else:
            session.add(
                TaskQueue(
                    email_id=email_id,
                    title=title,
                    description=description,
                    status="open",
                    created_at=now,
                    updated_at=now,
                )
            )
        session.commit()
    finally:
        session.close()


def execute_next_action(observed: dict[str, Any], analysis: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    proposed_action = _safe_action(analysis.get("NextAction"), analysis.get("RequiresReply"))
    llm_confidence = _safe_confidence(analysis.get("Confidence", 0.0))
    behavior = compute_behavior_profile(
        str(analysis.get("Intent") or ""),
        sender_domain_from_observed(observed),
    )
    sample_size = int(behavior.get("sample_size", 0) or 0)
    importance_score = _safe_confidence(behavior.get("importance_score", 0.0))
    behavior_weight = min(_MAX_BEHAVIOR_INFLUENCE, _MAX_BEHAVIOR_INFLUENCE * (sample_size / _FULL_BEHAVIOR_AT_SAMPLES))
    final_score = ((1.0 - behavior_weight) * llm_confidence) + (behavior_weight * importance_score)

    next_action = proposed_action
    requires_action = analysis.get("RequiresAction") is True

    # Stability invariant: clear ignore with high LLM confidence should survive cold start.
    if (
        proposed_action == "ignore"
        and llm_confidence >= _CLEAR_IGNORE_CONFIDENCE
        and sample_size < _LOW_SAMPLE_INVARIANT_LIMIT
    ):
        next_action = "ignore"
    
    if proposed_action == "draft_reply":
        if final_score >= _DRAFT_AUTO_THRESHOLD:
            next_action = "draft_reply"
        elif final_score >= _REVIEW_TASK_THRESHOLD and requires_action:
            next_action = "create_task"
        elif final_score >= _REVIEW_THRESHOLD:
            next_action = "escalate_human_review"
        elif final_score < _DRAFT_MIN_FINAL_SCORE:
            next_action = "escalate_human_review"
    elif proposed_action == "ignore":
        if (
            llm_confidence >= _CLEAR_IGNORE_CONFIDENCE
            and sample_size < _LOW_SAMPLE_INVARIANT_LIMIT
        ):
            next_action = "ignore"
        elif final_score >= _REVIEW_TASK_THRESHOLD and requires_action:
            next_action = "create_task"
        elif final_score >= _REVIEW_THRESHOLD:
            next_action = "escalate_human_review"

    action_reason = str(analysis.get("ActionReason") or analysis.get("Reasoning") or "").strip()
    result = {
        "ProposedAction": proposed_action,
        "Action": next_action,
        "ActionReason": action_reason,
        "LLMConfidence": llm_confidence,
        "ReplyRateBySender": behavior.get("reply_rate_by_sender"),
        "ReplyRateByIntent": behavior.get("reply_rate_by_intent"),
        "OpenRate": behavior.get("open_rate"),
        "ManualOverrideRate": behavior.get("manual_override_rate"),
        "ImportanceScore": importance_score,
        "BehaviorInfluenceWeight": behavior_weight,
        "BehaviorSampleSize": sample_size,
        "FinalDecisionScore": final_score,
        "Draft": {"DraftReply": "", "Reasoning": "No draft generated for this action.", "Confidence": 1.0},
    }
    if proposed_action != next_action:
        result["ActionReason"] = (
            f"{action_reason} Adaptive routing changed action from {proposed_action} "
            f"to {next_action} because unified final score was {final_score:.2f} "
            f"(behavior weight {behavior_weight:.2f}, samples {sample_size})."
        ).strip()
    persisted_reason = result["ActionReason"]

    if next_action == "ignore":
        store_action_state(observed, next_action, persisted_reason, task_status="", urgent_flag=False, needs_human_review=False)
        return result, True, ""

    if next_action == "draft_reply":
        draft, draft_ok = generate_reply_with_status(observed, analysis)
        if not draft_ok:
            return result, False, str(draft.get("Reasoning", "draft unavailable"))
        draft_json = json.dumps(draft, ensure_ascii=True)
        store_action_state(
            observed,
            next_action,
            persisted_reason,
            task_status="",
            urgent_flag=False,
            needs_human_review=False,
            reply_json=draft_json,
        )
        result["Draft"] = draft
        return result, True, ""

    if next_action == "create_task":
        _enqueue_task(observed, persisted_reason)
        store_action_state(observed, next_action, persisted_reason, task_status="open")
        return result, True, ""

    if next_action == "flag_high_urgency":
        store_action_state(observed, next_action, persisted_reason, urgent_flag=True)
        return result, True, ""

    if next_action == "escalate_human_review":
        store_action_state(observed, next_action, persisted_reason, needs_human_review=True)
        return result, True, ""

    return result, False, "unsupported action"
