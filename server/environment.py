"""
Email Triage Environment — Core Logic
Implements reset(), step(), state() with 3 graded tasks.

Tasks:
  1. basic_triage   (EASY)   — label 5 obvious emails by category + priority
  2. mixed_inbox    (MEDIUM) — 10 emails including ambiguous cases, spam, escalations
  3. crisis_inbox   (HARD)   — 15 emails under simulated SLA pressure with
                               partial-credit reply drafting
"""

import uuid
import copy
from typing import Any, Dict, List, Optional, Tuple

try:
    from models import (
        Email, EmailTriageAction, EmailTriageObservation, EmailTriageState
    )
except ImportError:
    from server.models import (  # type: ignore
        Email, EmailTriageAction, EmailTriageObservation, EmailTriageState
    )


# ──────────────────────────────────────────────────────────────
# Static email datasets
# ──────────────────────────────────────────────────────────────

TASK_EASY_EMAILS: List[Dict] = [
    {
        "id": "e001",
        "sender": "billing@acmecorp.com",
        "subject": "Invoice #1042 - Payment Overdue",
        "body": "Your invoice #1042 for $2,400 is now 30 days overdue. Please remit payment immediately or contact our billing department.",
        "timestamp": "2024-03-10T09:15:00Z",
        "ground_truth": {"category": "urgent_escalation", "priority": "urgent"},
    },
    {
        "id": "e002",
        "sender": "newsletter@techdigest.io",
        "subject": "This week in AI — March 2024 Digest",
        "body": "Welcome to this week's AI digest. Top stories: GPT-5 rumours, new open-source models, robotics breakthroughs...",
        "timestamp": "2024-03-10T08:00:00Z",
        "ground_truth": {"category": "newsletter", "priority": "low"},
    },
    {
        "id": "e003",
        "sender": "john.smith@company.internal",
        "subject": "Team lunch tomorrow - vote for restaurant",
        "body": "Hey team, voting for Friday lunch. Options: Thai Palace, Pizza Hub, Sushi Go. Reply with your choice!",
        "timestamp": "2024-03-10T10:30:00Z",
        "ground_truth": {"category": "internal", "priority": "low"},
    },
    {
        "id": "e004",
        "sender": "customer.jenny@gmail.com",
        "subject": "My order hasn't arrived - order #78312",
        "body": "Hello, I placed order #78312 on Feb 28 and it's now been 10 days with no delivery. The tracking number doesn't work. Please help ASAP.",
        "timestamp": "2024-03-10T11:00:00Z",
        "ground_truth": {"category": "support", "priority": "urgent"},
    },
    {
        "id": "e005",
        "sender": "sales.lead@b2bprospects.biz",
        "subject": "Partnership opportunity for your business",
        "body": "Hi, I'm reaching out about a potential partnership between our companies. We help SaaS businesses grow 10x. Can we schedule a call?",
        "timestamp": "2024-03-10T12:45:00Z",
        "ground_truth": {"category": "sales", "priority": "normal"},
    },
]

TASK_MEDIUM_EMAILS: List[Dict] = [
    *TASK_EASY_EMAILS,
    {
        "id": "e006",
        "sender": "no-reply@phishing-spoof.xyz",
        "subject": "URGENT: Your account will be suspended! Verify NOW",
        "body": "Dear user, your account will be suspended in 24 hours. Click here immediately: http://totally-not-phishing.xyz/verify",
        "timestamp": "2024-03-10T07:00:00Z",
        "ground_truth": {"category": "spam", "priority": "low"},
    },
    {
        "id": "e007",
        "sender": "sarah.dev@company.internal",
        "subject": "Production server down - P0 incident",
        "body": "CRITICAL: Production API server is returning 500 errors. Users are impacted. Incident #INC-2024-0310 opened. All hands needed.",
        "timestamp": "2024-03-10T13:00:00Z",
        "ground_truth": {"category": "urgent_escalation", "priority": "urgent"},
    },
    {
        "id": "e008",
        "sender": "mike.prospect@newclient.com",
        "subject": "Following up on our demo last week",
        "body": "Hi, loved the demo! I've discussed with my team and we're ready to move forward. Can we get a pricing proposal by EOW?",
        "timestamp": "2024-03-10T14:00:00Z",
        "ground_truth": {"category": "sales", "priority": "urgent"},
    },
    {
        "id": "e009",
        "sender": "hr@company.internal",
        "subject": "Mandatory: Updated Remote Work Policy - action required",
        "body": "Please review and sign the updated remote work policy by March 15. Non-compliance may affect payroll processing.",
        "timestamp": "2024-03-10T09:00:00Z",
        "ground_truth": {"category": "internal", "priority": "urgent"},
    },
    {
        "id": "e010",
        "sender": "amazon-orders@amazon.com",
        "subject": "Your Amazon order has shipped",
        "body": "Good news! Your order #113-2948-00183 has shipped and will arrive Thursday. Track your package with code: 1Z999AA10123456784.",
        "timestamp": "2024-03-10T08:30:00Z",
        "ground_truth": {"category": "newsletter", "priority": "low"},
    },
]

TASK_HARD_EMAILS: List[Dict] = [
    *TASK_MEDIUM_EMAILS,
    {
        "id": "e011",
        "sender": "angry.customer@bigclient.com",
        "subject": "Absolutely unacceptable service - escalating to CEO",
        "body": "I have been a customer for 5 years and your support team has failed me 3 times this month. I am writing to your CEO and posting on social media unless this is resolved TODAY.",
        "timestamp": "2024-03-10T15:00:00Z",
        "ground_truth": {
            "category": "urgent_escalation", "priority": "urgent",
            "expected_reply_keywords": ["apologize", "resolve", "priority", "escalate", "contact"]
        },
    },
    {
        "id": "e012",
        "sender": "legal@externalfirm.com",
        "subject": "Notice of Potential Litigation - Confidential",
        "body": "Our client intends to pursue legal action regarding contract breach on project Alpha. Please engage your legal counsel immediately. Response required within 5 business days.",
        "timestamp": "2024-03-10T15:30:00Z",
        "ground_truth": {
            "category": "urgent_escalation", "priority": "urgent",
            "expected_reply_keywords": ["legal", "review", "counsel", "acknowledge"]
        },
    },
    {
        "id": "e013",
        "sender": "vendor@cloudinfra.io",
        "subject": "Your subscription renews in 3 days — $15,000 annual plan",
        "body": "This is a reminder that your Enterprise Cloud subscription ($15,000/year) will auto-renew on March 13. To cancel or modify, log in to your portal.",
        "timestamp": "2024-03-10T08:00:00Z",
        "ground_truth": {"category": "urgent_escalation", "priority": "urgent"},
    },
    {
        "id": "e014",
        "sender": "ceo@company.internal",
        "subject": "Board meeting prep - slides needed by 5pm TODAY",
        "body": "Team, I need the Q1 performance slides ready for the board meeting at 6pm. Please send to me by 5pm. This is highest priority.",
        "timestamp": "2024-03-10T10:00:00Z",
        "ground_truth": {"category": "internal", "priority": "urgent"},
    },
    {
        "id": "e015",
        "sender": "curious.student@university.edu",
        "subject": "Question about your product for my thesis",
        "body": "Hi, I'm a grad student researching AI tools for my thesis. Could you share some stats or a brief interview? Happy to share the final paper!",
        "timestamp": "2024-03-10T16:00:00Z",
        "ground_truth": {"category": "sales", "priority": "low"},
    },
]

TASK_CONFIGS = {
    "basic_triage": {
        "description": (
            "EASY — Triage 5 emails: assign the correct category "
            "(support|sales|internal|spam|newsletter|urgent_escalation) "
            "and priority (urgent|normal|low) to each email. "
            "Use the 'label' action for each email."
        ),
        "emails": TASK_EASY_EMAILS,
        "max_steps": 10,
        "requires_reply": False,
    },
    "mixed_inbox": {
        "description": (
            "MEDIUM — Triage 10 emails including phishing, escalations, and ambiguous cases. "
            "Correctly label category + priority. Spam should be archived. "
            "Urgent escalations must be flagged for human review."
        ),
        "emails": TASK_MEDIUM_EMAILS,
        "max_steps": 20,
        "requires_reply": False,
    },
    "crisis_inbox": {
        "description": (
            "HARD — Triage 15 emails under SLA pressure. Correctly label all emails. "
            "For emails e011 and e012 (angry customer + legal notice), you MUST use "
            "the 'respond' action with a professional, empathetic reply draft containing "
            "the right keywords. All P0 and legal emails must be flagged."
        ),
        "emails": TASK_HARD_EMAILS,
        "max_steps": 30,
        "requires_reply": True,
    },
}


# ──────────────────────────────────────────────────────────────
# Grader helpers
# ──────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"support", "sales", "internal", "spam", "newsletter", "urgent_escalation"}
VALID_PRIORITIES = {"urgent", "normal", "low"}


def _grade_label(
    email_id: str,
    agent_label: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> float:
    """Return 0.0–1.0 for a single email label decision."""
    score = 0.0
    gt_cat = ground_truth.get("category", "")
    gt_pri = ground_truth.get("priority", "")
    ag_cat = agent_label.get("category", "")
    ag_pri = agent_label.get("priority", "")

    if ag_cat == gt_cat:
        score += 0.5
    if ag_pri == gt_pri:
        score += 0.5
    return score


def _grade_reply(draft: str, expected_keywords: List[str]) -> float:
    """Return 0.0–1.0 based on how many expected keywords appear in the draft."""
    if not draft:
        return 0.0
    draft_lower = draft.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in draft_lower)
    return round(hits / len(expected_keywords), 2) if expected_keywords else 0.0


# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    OpenEnv-compatible Email Triage Environment.

    Implements reset(), step(), state() per OpenEnv spec.
    """

    def __init__(self, task_name: str = "basic_triage"):
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASK_CONFIGS)}")
        self._task_name = task_name
        self._state: EmailTriageState = EmailTriageState()
        self._config = TASK_CONFIGS[task_name]

    # ── Core API ──────────────────────────────────────────────

    def reset(self) -> EmailTriageObservation:
        """Reset environment, load fresh inbox."""
        config = self._config
        raw_emails = config["emails"]

        inbox = [
            Email(
                id=e["id"],
                sender=e["sender"],
                subject=e["subject"],
                body=e["body"],
                timestamp=e["timestamp"],
            )
            for e in raw_emails
        ]

        ground_truth = {
            e["id"]: e["ground_truth"]
            for e in raw_emails
        }

        self._state = EmailTriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task_name,
            inbox=inbox,
            ground_truth_labels=ground_truth,
            agent_labels={},
            flagged_emails=[],
            archived_emails=[],
            draft_replies={},
            cumulative_reward=0.0,
            max_steps=config["max_steps"],
        )

        return self._build_observation(
            last_result="Inbox loaded. Begin triage.",
            reward=0.0,
            done=False,
        )

    def step(self, action: EmailTriageAction) -> Tuple[EmailTriageObservation, float, bool, Dict]:
        """
        Process one action. Returns (observation, reward, done, info).
        Reward is in [0.0, 1.0] per step (normalised by inbox size).
        """
        self._state.step_count += 1
        reward = 0.0
        error: Optional[str] = None
        result_msg = ""

        atype = action.action_type.lower()

        if atype == "label":
            reward, result_msg, error = self._handle_label(action)
        elif atype == "archive":
            reward, result_msg, error = self._handle_archive(action)
        elif atype == "flag":
            reward, result_msg, error = self._handle_flag(action)
        elif atype == "respond":
            reward, result_msg, error = self._handle_respond(action)
        elif atype == "skip":
            reward, result_msg, error = 0.0, "Skipped.", None
        else:
            error = f"Unknown action_type '{atype}'"
            result_msg = "Action failed."

        self._state.cumulative_reward += reward

        done = self._check_done()

        obs = self._build_observation(
            last_result=result_msg,
            last_error=error,
            reward=reward,
            done=done,
        )

        info = {
            "step": self._state.step_count,
            "cumulative_reward": self._state.cumulative_reward,
            "emails_processed": len(self._state.agent_labels),
        }

        return obs, reward, done, info

    def state(self) -> EmailTriageState:
        """Return current state (serialisable)."""
        return copy.deepcopy(self._state)

    # ── Action handlers ───────────────────────────────────────

    def _handle_label(self, action: EmailTriageAction) -> Tuple[float, str, Optional[str]]:
        if not action.email_id:
            return 0.0, "", "email_id is required for 'label' action"
        if action.email_id not in {e.id for e in self._state.inbox}:
            return 0.0, "", f"Email '{action.email_id}' not found in inbox"
        if action.category not in VALID_CATEGORIES:
            return 0.0, "", f"Invalid category '{action.category}'. Must be one of {VALID_CATEGORIES}"
        if action.priority not in VALID_PRIORITIES:
            return 0.0, "", f"Invalid priority '{action.priority}'. Must be one of {VALID_PRIORITIES}"

        self._state.agent_labels[action.email_id] = {
            "category": action.category,
            "priority": action.priority,
        }

        gt = self._state.ground_truth_labels.get(action.email_id, {})
        score = _grade_label(action.email_id, self._state.agent_labels[action.email_id], gt)
        # Normalise reward by total inbox size
        n = len(self._state.inbox)
        reward = round(score / n, 4) if n else 0.0

        feedback = f"Labelled {action.email_id} as {action.category}/{action.priority}."
        if score == 1.0:
            feedback += " ✓ Correct!"
        elif score == 0.5:
            feedback += " ~ Partially correct (one dimension wrong)."
        else:
            feedback += " ✗ Incorrect."

        return reward, feedback, None

    def _handle_archive(self, action: EmailTriageAction) -> Tuple[float, str, Optional[str]]:
        if not action.email_id:
            return 0.0, "", "email_id is required for 'archive' action"
        if action.email_id not in {e.id for e in self._state.inbox}:
            return 0.0, "", f"Email '{action.email_id}' not found"

        gt = self._state.ground_truth_labels.get(action.email_id, {})
        # Archiving spam/newsletters correctly earns reward
        correct_archive = gt.get("category") in ("spam", "newsletter")
        reward = 0.05 if correct_archive else -0.02
        reward = max(0.0, reward)

        if action.email_id not in self._state.archived_emails:
            self._state.archived_emails.append(action.email_id)

        msg = f"Archived {action.email_id}."
        if correct_archive:
            msg += " ✓ Correct — spam/newsletter emails should be archived."
        else:
            msg += " ⚠ This email may need attention before archiving."

        return reward, msg, None

    def _handle_flag(self, action: EmailTriageAction) -> Tuple[float, str, Optional[str]]:
        if not action.email_id:
            return 0.0, "", "email_id is required for 'flag' action"
        if not action.reason:
            return 0.0, "", "reason is required for 'flag' action"

        gt = self._state.ground_truth_labels.get(action.email_id, {})
        # Correct to flag urgent_escalations
        correct_flag = gt.get("category") == "urgent_escalation"
        n = len(self._state.inbox)
        reward = round(0.3 / n, 4) if correct_flag else 0.0

        if action.email_id not in self._state.flagged_emails:
            self._state.flagged_emails.append(action.email_id)

        msg = f"Flagged {action.email_id} for human review. Reason: {action.reason}"
        if correct_flag:
            msg += " ✓ Good — this is an urgent escalation."
        else:
            msg += " (Note: not classified as urgent escalation in ground truth.)"

        return reward, msg, None

    def _handle_respond(self, action: EmailTriageAction) -> Tuple[float, str, Optional[str]]:
        if not action.email_id:
            return 0.0, "", "email_id is required for 'respond' action"
        if not action.draft_reply:
            return 0.0, "", "draft_reply is required for 'respond' action"

        gt = self._state.ground_truth_labels.get(action.email_id, {})
        expected_kws = gt.get("expected_reply_keywords", [])

        reply_score = _grade_reply(action.draft_reply, expected_kws)
        n = len(self._state.inbox)
        reward = round((reply_score * 0.5) / max(n, 1), 4)  # reply contributes up to 0.5

        self._state.draft_replies[action.email_id] = action.draft_reply

        msg = f"Draft reply saved for {action.email_id}. Reply quality score: {reply_score:.2f}"
        if reply_score >= 0.8:
            msg += " ✓ Excellent reply."
        elif reply_score >= 0.4:
            msg += " ~ Acceptable reply, missing some key elements."
        elif expected_kws:
            msg += f" ✗ Reply missing key elements: {expected_kws}"

        return reward, msg, None

    # ── Done / scoring helpers ─────────────────────────────────

    def _check_done(self) -> bool:
        total = len(self._state.inbox)
        processed = len(self._state.agent_labels) + len(self._state.archived_emails)
        all_processed = processed >= total
        over_steps = self._state.step_count >= self._state.max_steps
        return all_processed or over_steps

    def _build_observation(
        self,
        last_result: str = "",
        last_error: Optional[str] = None,
        reward: float = 0.0,
        done: bool = False,
    ) -> EmailTriageObservation:
        processed_ids = (
            set(self._state.agent_labels.keys())
            | set(self._state.archived_emails)
        )
        remaining = [e for e in self._state.inbox if e.id not in processed_ids]
        return EmailTriageObservation(
            inbox=remaining,
            last_action_result=last_result,
            last_action_error=last_error,
            emails_processed=len(processed_ids),
            emails_remaining=len(remaining),
            task_description=self._config["description"],
            reward=reward,
            done=done,
        )

    # ── Final grader (called at episode end) ──────────────────

    def compute_final_score(self) -> float:
        """
        Compute 0.0–1.0 episode score.
        Called by the grader after the episode ends.
        """
        config = self._config
        raw_emails = config["emails"]
        n = len(raw_emails)
        if n == 0:
            return 0.0

        label_scores = []
        for e in raw_emails:
            eid = e["id"]
            gt = e["ground_truth"]
            if eid in self._state.agent_labels:
                label_scores.append(
                    _grade_label(eid, self._state.agent_labels[eid], gt)
                )
            elif eid in self._state.archived_emails:
                correct_archive = gt.get("category") in ("spam", "newsletter")
                label_scores.append(1.0 if correct_archive else 0.0)
            else:
                label_scores.append(0.0)

        label_score = sum(label_scores) / n  # 0–1

        # Reply quality (hard task only)
        reply_score = 0.0
        if config["requires_reply"]:
            reply_emails = [
                e for e in raw_emails
                if "expected_reply_keywords" in e.get("ground_truth", {})
            ]
            if reply_emails:
                rscores = []
                for e in reply_emails:
                    draft = self._state.draft_replies.get(e["id"], "")
                    kws = e["ground_truth"].get("expected_reply_keywords", [])
                    rscores.append(_grade_reply(draft, kws))
                reply_score = sum(rscores) / len(rscores)

        # Efficiency bonus: fewer wasted steps = small bonus
        steps_used = self._state.step_count
        max_steps = self._state.max_steps
        efficiency = max(0.0, 1.0 - (steps_used / max_steps))
        efficiency_bonus = efficiency * 0.05  # up to +5%

        if config["requires_reply"]:
            final = 0.7 * label_score + 0.25 * reply_score + efficiency_bonus
        else:
            final = 0.95 * label_score + efficiency_bonus

        return round(min(final, 1.0), 4)
