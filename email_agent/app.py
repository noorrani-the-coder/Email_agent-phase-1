from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.ingestion import ingest_emails
from agent.observation import observe_email
from agent.decision import analyze_email_with_status
from agent.persist import persist_observation
from agent.memory import store_email
from agent.actions import execute_next_action
from agent.behavior import log_behavior_event, sender_domain_from_observed
from agent.retry_queue import enqueue_retry, process_retry_queue
from db.models import EmailMemory
from db.session import get_session, init_db
from gmail.auth import get_credentials
from googleapiclient.discovery import build


def run_agent() -> None:
    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    emails = ingest_emails()
    init_db()
    session = get_session()

    try:
        process_retry_queue()
        for e in emails:
            email_id = e.get("id")
            if email_id and session.query(EmailMemory).filter_by(email_id=email_id).first():
                continue

            observed = observe_email(service, e["id"])
            persist_observation(observed)
            try:
                store_email(observed.get("content", ""))
            except Exception:
                pass

            analysis, analysis_ok = analyze_email_with_status(observed)
            action_result = {
                "Action": "escalate_human_review",
                "ActionReason": "Analysis failed and was queued for retry.",
                "Draft": {"DraftReply": "", "Reasoning": "No draft generated.", "Confidence": 0.0},
            }

            if not analysis_ok:
                action_result, _, _ = execute_next_action(observed, analysis)
                enqueue_retry(observed, operation="analyze_and_execute", error=str(analysis.get("Reasoning", "")))
            else:
                action_result, action_ok, action_error = execute_next_action(observed, analysis)
                if not action_ok:
                    enqueue_retry(observed, operation="analyze_and_execute", error=action_error)

            log_behavior_event(
                email_id=observed.get("email_id") or observed.get("id") or email_id or "",
                intent=str(analysis.get("Intent") or ""),
                sender_domain=sender_domain_from_observed(observed),
                requires_reply=analysis.get("RequiresReply"),
                proposed_action=str(action_result.get("ProposedAction") or action_result.get("Action") or ""),
                agent_action=str(action_result.get("Action") or ""),
                llm_confidence=float(action_result.get("LLMConfidence", analysis.get("Confidence", 0.0)) or 0.0),
                behavior_match_score=float(action_result.get("ImportanceScore", 0.0) or 0.0),
                final_decision_score=float(action_result.get("FinalDecisionScore", 0.0) or 0.0),
                user_final_action="",
            )

            process_retry_queue(limit=1)

            output = {
                "EmailId": observed.get("email_id") or observed.get("id") or email_id or "",
                "Analysis": analysis,
                "ProposedAction": action_result.get("ProposedAction"),
                "Action": action_result.get("Action"),
                "ActionReason": action_result.get("ActionReason"),
                "LLMConfidence": action_result.get("LLMConfidence"),
                "ReplyRateBySender": action_result.get("ReplyRateBySender"),
                "ReplyRateByIntent": action_result.get("ReplyRateByIntent"),
                "OpenRate": action_result.get("OpenRate"),
                "ManualOverrideRate": action_result.get("ManualOverrideRate"),
                "ImportanceScore": action_result.get("ImportanceScore"),
                "BehaviorInfluenceWeight": action_result.get("BehaviorInfluenceWeight"),
                "BehaviorSampleSize": action_result.get("BehaviorSampleSize"),
                "FinalDecisionScore": action_result.get("FinalDecisionScore"),
                "Draft": action_result.get("Draft"),
            }
            print(json.dumps(output, ensure_ascii=True))
    finally:
        session.close()


if __name__ == "__main__":
    run_agent()
