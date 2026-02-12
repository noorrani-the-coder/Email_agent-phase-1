import base64
from bs4 import BeautifulSoup


def _decode_body(data: str) -> str:
    if not data:
        return ""
    try:
        decoded = base64.urlsafe_b64decode(data + "===")
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_parts(payload):
    parts = []
    stack = [payload]
    while stack:
        part = stack.pop()
        parts.append(part)
        stack.extend(part.get("parts", []) or [])
    return parts

def extract_body(payload):
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

    return html_body or text_body or ""

def observe_email(service, message_id):
    msg = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

    body = extract_body(msg["payload"])
    text = BeautifulSoup(body, "html.parser").get_text()

    return {
        "email_id": msg["id"],
        "thread_id": msg["threadId"],
        "from": headers.get("From"),
        "subject": headers.get("Subject"),
        "timestamp": int(msg["internalDate"]),
        "content": text.strip()
    }
