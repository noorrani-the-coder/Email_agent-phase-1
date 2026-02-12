import json
from typing import Any

from ai.llm import call_llm


EXEC_EMAIL_ANALYSIS_PROMPT = """
You are an intelligent executive email assistant.

Analyze the email carefully and answer the following:

1. Intent: What is the main purpose of the email?
2. RequiresReply: Does the sender expect a response from the recipient? (true/false)
3. RequiresAction: Does the email require the recipient to take any action? (true/false)
4. NextAction: choose exactly one of:
   - ignore
   - draft_reply
   - create_task
   - flag_high_urgency
   - escalate_human_review
5. ActionReason: Brief reason for NextAction.
6. Urgency: low / medium / high
7. Reasoning: Brief explanation.
8. Confidence: 0.0 to 1.0

Important:
- Do not use keyword rules.
- Infer expectations socially and professionally.
- Newsletters, automated notifications, and marketing emails usually do not require replies.
- Direct questions, requests, proposals, confirmations usually require replies.

Return output strictly in JSON.
Use exactly these keys: Intent, RequiresReply, RequiresAction, NextAction, ActionReason, Urgency, Reasoning, Confidence.
Do not include markdown or any extra keys.
"""

EXEC_EMAIL_REPLY_PROMPT = """
You are an executive email assistant that writes concise, professional drafts.

You will receive:
- The original email content.
- A structured analysis of that email.

Write a suitable reply draft that matches the analysis.
If analysis indicates no reply is required, set DraftReply to an empty string and explain briefly.
Do not invent facts or commitments not present in the email.

Return output strictly in JSON.
Use exactly these keys: DraftReply, Reasoning, Confidence.
Confidence must be a number from 0.0 to 1.0.
Do not include markdown or any extra keys.
"""


def _email_content(email: Any) -> str:
    if isinstance(email, dict):
        return str(email.get("content", ""))
    return ""


def _fallback_analysis(reasoning: str = "Model response could not be parsed reliably.") -> dict[str, Any]:
    return {
        "Intent": "Unknown",
        "RequiresReply": None,
        "RequiresAction": None,
        "NextAction": "escalate_human_review",
        "ActionReason": "Analysis is uncertain; routing to human review.",
        "Urgency": "low",
        "Reasoning": reasoning,
        "Confidence": 0.2,
    }


def _coerce_bool_or_none(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _normalize_next_action(value: Any, requires_reply: Any) -> str:
    allowed = {
        "ignore",
        "draft_reply",
        "create_task",
        "flag_high_urgency",
        "escalate_human_review",
    }
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "draft": "draft_reply",
            "reply": "draft_reply",
            "create task": "create_task",
            "task": "create_task",
            "flag high urgency": "flag_high_urgency",
            "high_urgency": "flag_high_urgency",
            "escalate": "escalate_human_review",
            "human_review": "escalate_human_review",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in allowed:
            return normalized
    if requires_reply is True:
        return "draft_reply"
    if requires_reply is False:
        return "ignore"
    return "escalate_human_review"


def _coerce_analysis_payload(raw: str) -> tuple[dict[str, Any], bool]:
    parse_ok = True
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("response is not an object")
    except Exception:
        parsed = _fallback_analysis()
        parse_ok = False

    payload = {
        "Intent": str(parsed.get("Intent", "Unknown")).strip() or "Unknown",
        "RequiresReply": _coerce_bool_or_none(parsed.get("RequiresReply")),
        "RequiresAction": _coerce_bool_or_none(parsed.get("RequiresAction")),
        "NextAction": "",
        "ActionReason": str(parsed.get("ActionReason", "")).strip(),
        "Urgency": str(parsed.get("Urgency", "low")).strip().lower(),
        "Reasoning": str(parsed.get("Reasoning", "")).strip() or "No reasoning provided.",
        "Confidence": parsed.get("Confidence", 0.2),
    }
    payload["NextAction"] = _normalize_next_action(parsed.get("NextAction"), payload["RequiresReply"])
    if not payload["ActionReason"]:
        payload["ActionReason"] = payload["Reasoning"]

    if payload["Urgency"] not in {"low", "medium", "high"}:
        payload["Urgency"] = "low"

    try:
        payload["Confidence"] = float(payload["Confidence"])
    except Exception:
        payload["Confidence"] = 0.2
    payload["Confidence"] = max(0.0, min(1.0, payload["Confidence"]))

    return payload, parse_ok


def _fallback_reply(reasoning: str = "Reply draft could not be generated reliably.") -> dict[str, Any]:
    return {
        "DraftReply": "",
        "Reasoning": reasoning,
        "Confidence": 0.2,
    }


def _coerce_reply_payload(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("response is not an object")
    except Exception:
        parsed = _fallback_reply()

    payload = {
        "DraftReply": str(parsed.get("DraftReply", "")).strip(),
        "Reasoning": str(parsed.get("Reasoning", "")).strip() or "No reasoning provided.",
        "Confidence": parsed.get("Confidence", 0.2),
    }

    try:
        payload["Confidence"] = float(payload["Confidence"])
    except Exception:
        payload["Confidence"] = 0.2
    payload["Confidence"] = max(0.0, min(1.0, payload["Confidence"]))

    return payload


def analyze_email(email: Any) -> dict[str, Any]:
    payload, _ = analyze_email_with_status(email)
    return payload


def analyze_email_with_status(email: Any) -> tuple[dict[str, Any], bool]:
    try:
        raw = call_llm(
            EXEC_EMAIL_ANALYSIS_PROMPT,
            _email_content(email),
            temperature=0.1,
        )
    except Exception as exc:
        return _fallback_analysis(reasoning=f"analysis unavailable: {exc.__class__.__name__}"), False
    return _coerce_analysis_payload(raw)


def generate_reply(email: Any, analysis: Any) -> dict[str, Any]:
    payload, _ = generate_reply_with_status(email, analysis)
    return payload


def generate_reply_with_status(email: Any, analysis: Any) -> tuple[dict[str, Any], bool]:
    user_payload = {
        "email_content": _email_content(email),
        "analysis": analysis if isinstance(analysis, dict) else _fallback_analysis(),
    }

    try:
        raw = call_llm(
            EXEC_EMAIL_REPLY_PROMPT,
            json.dumps(user_payload, ensure_ascii=True),
            temperature=0.2,
        )
    except Exception as exc:
        return _fallback_reply(reasoning=f"draft unavailable: {exc.__class__.__name__}"), False
    return _coerce_reply_payload(raw), True


def summarize(email: Any) -> str:
    return json.dumps(analyze_email(email), ensure_ascii=True)


def draft_reply(email: Any) -> str:
    analysis = analyze_email(email)
    return json.dumps(generate_reply(email, analysis), ensure_ascii=True)
