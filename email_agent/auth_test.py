from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify"
]

def main():
    flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json",
        SCOPES
    )

    creds = flow.run_local_server(port=0)

    service = build("gmail", "v1", credentials=creds)

    profile = service.users().getProfile(userId="me").execute()
    print("✅ Agent connected to Gmail:")
    print("📧 Email:", profile["emailAddress"])

if __name__ == "__main__":
    main()
