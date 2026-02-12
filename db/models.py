from sqlalchemy import Column, String, Integer, Boolean, Text, UniqueConstraint, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EmailMemory(Base):
    __tablename__ = "email_memory"
    __table_args__ = (
        UniqueConstraint("email_id", name="uq_email_memory_email_id"),
    )

    id = Column(Integer, primary_key=True)
    email_id = Column(String, nullable=False)
    sender = Column(String, nullable=False)
    sender_type = Column(String, nullable=False)
    promo = Column(Boolean, nullable=False, default=False)
    urgency = Column(String, nullable=False, default="")
    subject = Column(String, nullable=False, default="")
    body = Column(Text, nullable=False, default="")
    timestamp = Column(String, nullable=False, default="")
    priority_label = Column(String, nullable=False, default="")
    priority_confidence = Column(Float, nullable=False, default=0.0)
    priority_score = Column(Integer, nullable=False, default=0)
    priority_reasons = Column(Text, nullable=False, default="")
    priority_tier = Column(String, nullable=False, default="")
    decision_timestamp = Column(String, nullable=False, default="")
    reply_draft = Column(Text, nullable=False, default="")
    reply_timestamp = Column(String, nullable=False, default="")
    next_action = Column(String, nullable=False, default="")
    action_reason = Column(Text, nullable=False, default="")
    task_status = Column(String, nullable=False, default="")
    urgent_flag = Column(Boolean, nullable=False, default=False)
    needs_human_review = Column(Boolean, nullable=False, default=False)
    action_timestamp = Column(String, nullable=False, default="")


class RetryQueue(Base):
    __tablename__ = "retry_queue"

    id = Column(Integer, primary_key=True)
    email_id = Column(String, nullable=False)
    operation = Column(String, nullable=False, default="analyze_and_execute")
    payload = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(String, nullable=False, default="")
    last_error = Column(Text, nullable=False, default="")
    created_at = Column(String, nullable=False, default="")
    updated_at = Column(String, nullable=False, default="")


class TaskQueue(Base):
    __tablename__ = "task_queue"

    id = Column(Integer, primary_key=True)
    email_id = Column(String, nullable=False)
    title = Column(String, nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="open")
    created_at = Column(String, nullable=False, default="")
    updated_at = Column(String, nullable=False, default="")


class BehaviorLog(Base):
    __tablename__ = "behavior_log"
    __table_args__ = (
        UniqueConstraint("email_id", name="uq_behavior_log_email_id"),
    )

    id = Column(Integer, primary_key=True)
    email_id = Column(String, nullable=False)
    intent = Column(String, nullable=False, default="")
    sender_domain = Column(String, nullable=False, default="")
    requires_reply = Column(Boolean, nullable=False, default=False)
    user_final_action = Column(String, nullable=False, default="")
    user_opened = Column(Boolean, nullable=False, default=False)
    proposed_action = Column(String, nullable=False, default="")
    agent_action = Column(String, nullable=False, default="")
    llm_confidence = Column(Float, nullable=False, default=0.0)
    behavior_match_score = Column(Float, nullable=False, default=0.0)
    final_decision_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(String, nullable=False, default="")
    updated_at = Column(String, nullable=False, default="")
