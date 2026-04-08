"""
Email Triage Environment — Typed Pydantic Models
Defines Action, Observation, and State for the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Base classes (mirrors openenv-core interface)
# ──────────────────────────────────────────────

class Action(BaseModel):
    """Base Action class."""
    pass


class Observation(BaseModel):
    """Base Observation class."""
    pass


class State(BaseModel):
    """Base State class."""
    episode_id: str = ""
    step_count: int = 0


# ──────────────────────────────────────────────
# Email domain models
# ──────────────────────────────────────────────

class Email(BaseModel):
    """A single email in the inbox."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    is_read: bool = False
    labels: List[str] = Field(default_factory=list)
    priority: Optional[str] = None  # "urgent", "normal", "low"
    category: Optional[str] = None  # "support", "sales", "internal", "spam", etc.


# ──────────────────────────────────────────────
# Action
# ──────────────────────────────────────────────

class EmailTriageAction(Action):
    """
    Action the agent takes on a single email.

    action_type options:
      - "label"    : assign category + priority (required fields: email_id, category, priority)
      - "archive"  : mark as handled / archive (required: email_id)
      - "flag"     : flag for human review (required: email_id, reason)
      - "respond"  : draft a reply (required: email_id, draft_reply)
      - "skip"     : do nothing this step (no required fields)
    """
    action_type: str = Field(..., description="One of: label | archive | flag | respond | skip")
    email_id: Optional[str] = Field(None, description="Target email ID")
    category: Optional[str] = Field(
        None,
        description="Email category: support | sales | internal | spam | newsletter | urgent_escalation"
    )
    priority: Optional[str] = Field(None, description="Priority: urgent | normal | low")
    reason: Optional[str] = Field(None, description="Reason for flagging (required for 'flag')")
    draft_reply: Optional[str] = Field(None, description="Reply draft text (required for 'respond')")


# ──────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────

class EmailTriageObservation(Observation):
    """What the agent sees after each step."""
    inbox: List[Email] = Field(default_factory=list, description="Current inbox emails")
    last_action_result: str = Field("", description="Result/feedback from last action")
    last_action_error: Optional[str] = Field(None, description="Error message if action failed")
    emails_processed: int = Field(0, description="Number of emails triaged so far")
    emails_remaining: int = Field(0, description="Emails not yet processed")
    task_description: str = Field("", description="Natural language description of the current task")
    reward: float = Field(0.0, description="Reward from the last step")
    done: bool = Field(False, description="Whether the episode is complete")


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class EmailTriageState(State):
    """Full internal state of the environment."""
    task_name: str = ""
    inbox: List[Email] = Field(default_factory=list)
    ground_truth_labels: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="email_id -> {category, priority} ground truth"
    )
    agent_labels: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="email_id -> agent's decisions"
    )
    flagged_emails: List[str] = Field(default_factory=list)
    archived_emails: List[str] = Field(default_factory=list)
    draft_replies: Dict[str, str] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    max_steps: int = 20
