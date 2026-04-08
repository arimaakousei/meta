"""
Tests for Email Triage Environment
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from server.environment import EmailTriageEnvironment, _grade_label, _grade_reply


# ── Unit tests: grader helpers ────────────────────────────────

class TestGraders:
    def test_label_perfect(self):
        score = _grade_label(
            "e001",
            {"category": "support", "priority": "urgent"},
            {"category": "support", "priority": "urgent"},
        )
        assert score == 1.0

    def test_label_wrong_category(self):
        score = _grade_label(
            "e001",
            {"category": "spam", "priority": "urgent"},
            {"category": "support", "priority": "urgent"},
        )
        assert score == 0.5  # priority correct, category wrong

    def test_label_both_wrong(self):
        score = _grade_label(
            "e001",
            {"category": "spam", "priority": "low"},
            {"category": "support", "priority": "urgent"},
        )
        assert score == 0.0

    def test_reply_all_keywords(self):
        score = _grade_reply(
            "We apologize and will resolve this as our highest priority. We are escalating your issue and will contact you shortly.",
            ["apologize", "resolve", "priority", "escalate", "contact"],
        )
        assert score == 1.0

    def test_reply_partial_keywords(self):
        score = _grade_reply(
            "We apologize for the inconvenience.",
            ["apologize", "resolve", "priority", "escalate", "contact"],
        )
        assert score == 0.2  # only 1/5 keywords

    def test_reply_empty(self):
        assert _grade_reply("", ["key1", "key2"]) == 0.0

    def test_reply_no_keywords_required(self):
        assert _grade_reply("some reply", []) == 0.0


# ── Integration: basic_triage task ───────────────────────────

class TestBasicTriage:
    def setup_method(self):
        self.env = EmailTriageEnvironment(task_name="basic_triage")

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, EmailTriageObservation)
        assert len(obs.inbox) == 5
        assert obs.emails_remaining == 5
        assert obs.emails_processed == 0
        assert obs.done is False

    def test_reset_is_deterministic(self):
        obs1 = self.env.reset()
        obs2 = self.env.reset()
        assert [e.id for e in obs1.inbox] == [e.id for e in obs2.inbox]

    def test_state_returns_state_model(self):
        self.env.reset()
        state = self.env.state()
        assert isinstance(state, EmailTriageState)
        assert state.task_name == "basic_triage"
        assert state.step_count == 0

    def test_step_label_correct(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type="label",
            email_id="e001",
            category="urgent_escalation",
            priority="urgent",
        )
        obs, reward, done, info = self.env.step(action)
        assert reward > 0
        assert not done
        assert obs.emails_processed == 1
        assert obs.emails_remaining == 4
        assert info["step"] == 1

    def test_step_label_wrong(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type="label",
            email_id="e001",
            category="spam",
            priority="low",
        )
        obs, reward, done, info = self.env.step(action)
        assert reward == 0.0  # completely wrong

    def test_step_archive_spam(self):
        self.env.reset()
        # e002 is newsletter → archive is correct
        action = EmailTriageAction(action_type="archive", email_id="e002")
        obs, reward, done, info = self.env.step(action)
        assert reward > 0  # rewarded for archiving newsletter

    def test_step_invalid_action_type(self):
        self.env.reset()
        action = EmailTriageAction(action_type="delete", email_id="e001")
        obs, reward, done, info = self.env.step(action)
        assert reward == 0.0
        assert obs.last_action_error is not None

    def test_step_invalid_category(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type="label",
            email_id="e001",
            category="nonsense",
            priority="urgent",
        )
        obs, reward, done, info = self.env.step(action)
        assert obs.last_action_error is not None

    def test_episode_completes_when_all_processed(self):
        self.env.reset()
        email_ids = ["e001", "e002", "e003", "e004", "e005"]
        label_map = {
            "e001": ("urgent_escalation", "urgent"),
            "e002": ("newsletter", "low"),
            "e003": ("internal", "low"),
            "e004": ("support", "urgent"),
            "e005": ("sales", "normal"),
        }
        done = False
        for eid in email_ids:
            cat, pri = label_map[eid]
            action = EmailTriageAction(
                action_type="label", email_id=eid, category=cat, priority=pri
            )
            obs, reward, done, info = self.env.step(action)
        assert done is True

    def test_perfect_score_basic_triage(self):
        self.env.reset()
        label_map = {
            "e001": ("urgent_escalation", "urgent"),
            "e002": ("newsletter", "low"),
            "e003": ("internal", "low"),
            "e004": ("support", "urgent"),
            "e005": ("sales", "normal"),
        }
        for eid, (cat, pri) in label_map.items():
            action = EmailTriageAction(
                action_type="label", email_id=eid, category=cat, priority=pri
            )
            self.env.step(action)
        score = self.env.compute_final_score()
        assert score >= 0.90, f"Expected >=0.90, got {score}"

    def test_zero_score_all_wrong(self):
        self.env.reset()
        for eid in ["e001", "e002", "e003", "e004", "e005"]:
            action = EmailTriageAction(
                action_type="label", email_id=eid, category="spam", priority="low"
            )
            self.env.step(action)
        score = self.env.compute_final_score()
        assert score < 0.25


# ── Integration: mixed_inbox task ────────────────────────────

class TestMixedInbox:
    def setup_method(self):
        self.env = EmailTriageEnvironment(task_name="mixed_inbox")

    def test_reset_10_emails(self):
        obs = self.env.reset()
        assert len(obs.inbox) == 10

    def test_step_flag_escalation(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type="flag",
            email_id="e007",  # production down → urgent escalation
            reason="P0 production incident requires immediate human escalation",
        )
        obs, reward, done, info = self.env.step(action)
        assert reward > 0

    def test_state_step_count_increments(self):
        self.env.reset()
        for _ in range(3):
            self.env.step(EmailTriageAction(action_type="skip"))
        state = self.env.state()
        assert state.step_count == 3


# ── Integration: crisis_inbox task ───────────────────────────

class TestCrisisInbox:
    def setup_method(self):
        self.env = EmailTriageEnvironment(task_name="crisis_inbox")

    def test_reset_15_emails(self):
        obs = self.env.reset()
        assert len(obs.inbox) == 15

    def test_respond_action_scored(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type="respond",
            email_id="e011",
            draft_reply=(
                "Dear customer, we sincerely apologize for the repeated issues. "
                "We will resolve this immediately and treat it as our highest priority. "
                "Our team will contact you within the hour and escalate to senior management."
            ),
        )
        obs, reward, done, info = self.env.step(action)
        assert reward > 0

    def test_respond_empty_draft_returns_error(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type="respond",
            email_id="e011",
            draft_reply="",
        )
        obs, reward, done, info = self.env.step(action)
        assert obs.last_action_error is not None
        assert reward == 0.0

    def test_max_steps_triggers_done(self):
        self.env.reset()
        done = False
        for _ in range(35):  # exceeds max_steps=30
            _, _, done, _ = self.env.step(EmailTriageAction(action_type="skip"))
            if done:
                break
        assert done is True

    def test_score_range_is_valid(self):
        self.env.reset()
        self.env.step(EmailTriageAction(action_type="skip"))
        score = self.env.compute_final_score()
        assert 0.0 <= score <= 1.0


# ── Reward signal tests (partial progress) ───────────────────

class TestRewardSignal:
    def test_reward_non_zero_on_correct_label(self):
        env = EmailTriageEnvironment(task_name="basic_triage")
        env.reset()
        _, r, _, _ = env.step(EmailTriageAction(
            action_type="label", email_id="e004",
            category="support", priority="urgent"
        ))
        assert r > 0

    def test_reward_zero_on_wrong_label(self):
        env = EmailTriageEnvironment(task_name="basic_triage")
        env.reset()
        _, r, _, _ = env.step(EmailTriageAction(
            action_type="label", email_id="e004",
            category="spam", priority="low"
        ))
        assert r == 0.0

    def test_cumulative_reward_tracked(self):
        env = EmailTriageEnvironment(task_name="basic_triage")
        env.reset()
        env.step(EmailTriageAction(
            action_type="label", email_id="e001",
            category="urgent_escalation", priority="urgent"
        ))
        state = env.state()
        assert state.cumulative_reward > 0
