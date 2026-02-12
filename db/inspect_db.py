from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db.session import get_session, init_db
from db.models import EmailMemory


def main() -> None:
    init_db()
    session = get_session()
    try:
        count = session.query(EmailMemory).count()
        print(f"count: {count}")

        sample = session.query(EmailMemory).first()
        if not sample:
            print("sample: None")
            return

        print("sample:")
        print(f"  email_id: {sample.email_id}")
        print(f"  sender: {sample.sender}")
        print(f"  sender_type: {sample.sender_type}")
        print(f"  promo: {sample.promo}")
        print(f"  urgency: {sample.urgency}")
        print(f"  subject: {sample.subject}")
        print(f"  body: {sample.body[:200] + ('...' if len(sample.body) > 200 else '')}")
        print(f"  timestamp: {sample.timestamp}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
