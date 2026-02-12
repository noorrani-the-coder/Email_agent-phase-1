from gmail.fetch import fetch_emails
from gmail.auth import get_credentials


def ingest_emails(max_results=None):
    creds = get_credentials()
    messages = fetch_emails(creds, max_results=max_results)

    return messages
