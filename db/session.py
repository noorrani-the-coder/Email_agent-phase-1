from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.models import Base

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "db" / "email_memory.sqlite"

engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def get_session():
    return SessionLocal()


def _ensure_columns() -> None:
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(email_memory)")).fetchall()
        existing = {row[1] for row in rows}
        desired = {
            "priority_label": "TEXT NOT NULL DEFAULT ''",
            "priority_confidence": "REAL NOT NULL DEFAULT 0.0",
            "priority_score": "INTEGER NOT NULL DEFAULT 0",
            "priority_reasons": "TEXT NOT NULL DEFAULT ''",
            "priority_tier": "TEXT NOT NULL DEFAULT ''",
            "decision_timestamp": "TEXT NOT NULL DEFAULT ''",
            "reply_draft": "TEXT NOT NULL DEFAULT ''",
            "reply_timestamp": "TEXT NOT NULL DEFAULT ''",
            "next_action": "TEXT NOT NULL DEFAULT ''",
            "action_reason": "TEXT NOT NULL DEFAULT ''",
            "task_status": "TEXT NOT NULL DEFAULT ''",
            "urgent_flag": "INTEGER NOT NULL DEFAULT 0",
            "needs_human_review": "INTEGER NOT NULL DEFAULT 0",
            "action_timestamp": "TEXT NOT NULL DEFAULT ''",
        }
        for name, ddl in desired.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE email_memory ADD COLUMN {name} {ddl}"))

        behavior_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='behavior_log'")
        ).fetchone()
        if behavior_exists:
            rows = conn.execute(text("PRAGMA table_info(behavior_log)")).fetchall()
            behavior_existing = {row[1] for row in rows}
            behavior_desired = {
                "user_opened": "INTEGER NOT NULL DEFAULT 0",
                "proposed_action": "TEXT NOT NULL DEFAULT ''",
            }
            for name, ddl in behavior_desired.items():
                if name not in behavior_existing:
                    conn.execute(text(f"ALTER TABLE behavior_log ADD COLUMN {name} {ddl}"))
