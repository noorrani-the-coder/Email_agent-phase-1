from __future__ import annotations

from pathlib import Path
import base64
import html as html_lib
import re
import sys
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Iterable

from googleapiclient.discovery import build

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gmail.auth import get_credentials
from db.models import EmailMemory
from db.session import get_session, init_db

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify"
]

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "aol.com", "proton.me", "protonmail.com", "pm.me", "gmx.com", "zoho.com",
}

PROMO_KEYWORDS = {
    "sale", "deal", "discount", "offer", "promo", "promotion", "save",
    "limited time", "limited-time", "coupon", "free shipping", "last chance",
}

URGENCY_KEYWORDS = {
    "urgent", "asap", "action required", "time sensitive", "deadline",
    "respond today", "immediate", "final notice", "due today",
}

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
WHITESPACE_RE = re.compile(r"\s+")


def _header_dict(headers: Iterable[dict[str, str]]) -> dict[str, str]:
    return {h.get("name", ""): h.get("value", "") for h in headers}


def _decode_body(data: str) -> str:
    if not data:
        return ""
    try:
        decoded = base64.urlsafe_b64decode(data + "===")
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parts = []
    stack = [payload]
    while stack:
        part = stack.pop()
        parts.append(part)
        stack.extend(part.get("parts", []) or [])
    return parts


def _get_message_body(message: dict[str, Any]) -> tuple[str, str]:
    payload = message.get("payload", {}) or {}
    parts = _extract_parts(payload)

    html_body = ""
    text_body = ""

    for part in parts:
        mime = (part.get("mimeType") or "").lower()
        body = part.get("body", {}) or {}
        data = body.get("data")
        if not data:
            continue
        content = _decode_body(data)
        if mime == "text/html" and not html_body:
            html_body = content
        elif mime == "text/plain" and not text_body:
            text_body = content

    return html_body, text_body


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    cleaned = SCRIPT_STYLE_RE.sub(" ", html)
    cleaned = TAG_RE.sub(" ", cleaned)
    cleaned = html_lib.unescape(cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _sender_type(from_value: str, headers: dict[str, str]) -> str:
    _, addr = parseaddr(from_value)
    domain = addr.split("@")[-1].lower() if "@" in addr else ""

    if any(h in headers for h in ("List-Unsubscribe", "List-Id", "List-Post")):
        return "automated"
    if headers.get("Precedence", "").lower() in {"bulk", "junk", "list"}:
        return "automated"
    if domain in FREE_EMAIL_DOMAINS:
        return "personal"
    if domain:
        return "organizational"
    return "unknown"


def _promotional(headers: dict[str, str], subject: str, body_text: str) -> bool:
    if any(h in headers for h in ("List-Unsubscribe", "List-Id")):
        return True
    haystack = f"{subject}\n{body_text}".lower()
    return any(k in haystack for k in PROMO_KEYWORDS)


def _urgency_clues(subject: str, body_text: str) -> list[str]:
    haystack = f"{subject}\n{body_text}".lower()
    return [k for k in URGENCY_KEYWORDS if k in haystack]


def _parse_timestamp(date_value: str) -> str | None:
    if not date_value:
        return None
    try:
        dt = parsedate_to_datetime(date_value)
        if dt:
            return dt.isoformat()
    except Exception:
        return None
    return None


def main():
    creds = get_credentials()

    service = build("gmail", "v1", credentials=creds)
    init_db()

    messages = []
    page_token = None
    while True:
        results = service.users().messages().list(
            userId="me",
            maxResults=500,
            pageToken=page_token,
        ).execute()
        messages.extend(results.get("messages", []) or [])
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(messages)} emails\n")

    session = get_session()
    stored = 0
    skipped = 0

    for msg in messages:
        message = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="full"
        ).execute()

        headers = _header_dict(message.get("payload", {}).get("headers", []) or [])
        subject = headers.get("Subject", "")
        from_value = headers.get("From", "")
        date_value = headers.get("Date", "")

        html_body, text_body = _get_message_body(message)
        body_text = _html_to_text(html_body) if html_body else text_body

        sender_type = _sender_type(from_value, headers)
        is_promotional = _promotional(headers, subject, body_text)
        urgency = _urgency_clues(subject, body_text)
        timestamp = _parse_timestamp(date_value)

        existing = session.query(EmailMemory).filter_by(email_id=msg["id"]).first()
        if existing:
            skipped += 1
        else:
            urgency_value = ", ".join(urgency)
            session.add(
                EmailMemory(
                    email_id=msg["id"],
                    sender=from_value,
                    sender_type=sender_type,
                    promo=is_promotional,
                    urgency=urgency_value,
                    subject=subject,
                    body=body_text,
                    timestamp=timestamp or "",
                )
            )
            stored += 1

        print("From      :", from_value)
        print("Subject   :", subject)
        print("Sender    :", sender_type)
        print("Promo     :", "yes" if is_promotional else "no")
        print("Urgency   :", ", ".join(urgency) if urgency else "none")
        print("Timestamp :", timestamp or "unknown")
        print("Body      :", body_text[:400] + ("..." if len(body_text) > 400 else ""))
        print("-" * 40)

    session.commit()
    session.close()
    print(f"Stored {stored} emails, skipped {skipped} existing.")

if __name__ == "__main__":
    main()
