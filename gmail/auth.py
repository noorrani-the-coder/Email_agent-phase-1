from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify"
]

def get_credentials():
    root_dir = Path(__file__).resolve().parents[1]
    token_path = root_dir / "token.json"
    creds_path = root_dir / "credentials.json"
    alt_creds_path = root_dir / "email_agent" / "credentials.json"

    if not creds_path.exists() and alt_creds_path.exists():
        creds_path = alt_creds_path

    if token_path.exists():
        return Credentials.from_authorized_user_file(str(token_path), SCOPES)

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0)

    token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds
