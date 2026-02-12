from __future__ import annotations

from agent.behavior import record_user_final_action


def record_feedback(email_id: str, replied: bool, edited: bool) -> bool:
    if replied and edited:
        action = "edited_draft"
    elif replied:
        action = "sent_reply"
    elif edited:
        action = "edited_draft"
    else:
        action = "ignored"
    return record_user_final_action(email_id, action)
