from googleapiclient.discovery import build


def fetch_emails(creds, max_results=None, page_size=500):
    service = build("gmail", "v1", credentials=creds)

    messages = []
    page_token = None
    remaining = max_results

    while True:
        batch_size = page_size
        if remaining is not None:
            batch_size = min(page_size, max(0, remaining))
            if batch_size == 0:
                break

        results = service.users().messages().list(
            userId="me",
            maxResults=batch_size,
            pageToken=page_token,
        ).execute()

        messages.extend(results.get("messages", []) or [])
        page_token = results.get("nextPageToken")

        if remaining is not None:
            remaining -= batch_size
            if remaining <= 0:
                break
        if not page_token:
            break

    return messages
