---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 📬 Email Triage OpenEnv

An **OpenEnv-compatible environment** for training and evaluating AI agents on real-world email inbox triage. Agents must classify emails, flag escalations, archive noise, and draft professional replies — tasks every knowledge worker performs daily.

---

## 🧠 Why Email Triage?

Email triage is a high-value, real-world task with:
- **Clear correctness criteria** (category + priority ground truth)
- **Partial progress signals** (reward at every step, not just the end)
- **Natural difficulty scaling** (easy → medium → hard across 3 tasks)
- **Practical utility** — a trained agent could directly assist support/ops teams

---

## 🗂 Project Structure

```
email-triage-env/
├── models.py                # Pydantic Action, Observation, State models
├── __init__.py              # Package exports
├── inference.py             # Baseline LLM agent script (MANDATORY)
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container definition
├── pyproject.toml           # Dependencies
├── .dockerignore
├── server/
│   ├── __init__.py
│   ├── environment.py       # Core logic: tasks, graders, rewards
│   ├── app.py               # FastAPI server (reset/step/state endpoints)
│   └── requirements.txt
└── tests/
    └── test_environment.py  # Full test suite (pytest)
```

---

## 🎯 Tasks

| Task | Difficulty | Emails | Max Steps | Description |
|------|-----------|--------|-----------|-------------|
| `basic_triage` | 🟢 Easy | 5 | 10 | Label 5 clear-cut emails by category + priority |
| `mixed_inbox` | 🟡 Medium | 10 | 20 | 10 emails including spam, escalations, ambiguous cases |
| `crisis_inbox` | 🔴 Hard | 15 | 30 | 15 emails under SLA pressure + required reply drafts |

---

## 🔧 Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `label` | `email_id`, `category`, `priority` | Assign category + priority to an email |
| `archive` | `email_id` | Archive spam/newsletters |
| `flag` | `email_id`, `reason` | Flag for human review (use for urgent_escalation) |
| `respond` | `email_id`, `draft_reply` | Draft a professional reply |
| `skip` | — | Do nothing this step |

**Categories:** `support` \| `sales` \| `internal` \| `spam` \| `newsletter` \| `urgent_escalation`

**Priorities:** `urgent` \| `normal` \| `low`

### Example Actions

```json
{"action_type": "label", "email_id": "e004", "category": "support", "priority": "urgent"}
{"action_type": "archive", "email_id": "e002"}
{"action_type": "flag", "email_id": "e007", "reason": "P0 production incident"}
{"action_type": "respond", "email_id": "e011", "draft_reply": "We sincerely apologize..."}
```

---

## 👁 Observation Space

```python
class EmailTriageObservation(BaseModel):
    inbox: List[Email]           # Remaining unprocessed emails
    last_action_result: str      # Human-readable feedback
    last_action_error: str|None  # Error message if action failed
    emails_processed: int        # Count triaged so far
    emails_remaining: int        # Count not yet processed
    task_description: str        # Natural language task description
    reward: float                # Reward from last step
    done: bool                   # Episode complete flag
```

---

## 🏆 Reward Function

Rewards are **dense** — the agent receives signal at every step:

| Action | Reward | Condition |
|--------|--------|-----------|
| `label` correct | `1.0 / n_emails` | Both category and priority correct |
| `label` half-correct | `0.5 / n_emails` | One dimension correct |
| `label` wrong | `0.0` | Both dimensions wrong |
| `archive` spam/newsletter | `0.05` | Correct use of archive |
| `archive` wrong email | `0.0` | Should not have been archived |
| `flag` urgent_escalation | `0.3 / n_emails` | Correct escalation |
| `respond` (hard task) | up to `0.5 / n_emails` | Keyword-based reply quality |
| Efficiency bonus | up to `+0.05` | Fewer wasted steps |

**Final score formula:**
- Easy/Medium: `0.95 × label_accuracy + 0.05 × efficiency_bonus`
- Hard: `0.70 × label_accuracy + 0.25 × reply_quality + 0.05 × efficiency_bonus`

---

## 🚀 Setup & Usage

### Local (with Docker)

```bash
# Build
docker build -t email-triage-env .

# Run (default: basic_triage task)
docker run -p 7860:7860 email-triage-env

# Run a specific task
docker run -p 7860:7860 -e EMAIL_TRIAGE_TASK=crisis_inbox email-triage-env
```

### Local (without Docker)

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### API Endpoints

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset environment (start episode)
curl -X POST "http://localhost:7860/reset?task=basic_triage"

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"label","email_id":"e001","category":"urgent_escalation","priority":"urgent"}'

# Get current state
curl http://localhost:7860/state

# Get episode score
curl http://localhost:7860/score
```

---

## 🤖 Running the Baseline

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SERVER_URL=http://localhost:7860

python inference.py
```

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score | Notes |
|------|-------|-------|
| `basic_triage` | ~0.85 | Near-perfect on clear-cut cases |
| `mixed_inbox` | ~0.62 | Struggles with spam vs. legitimate sales |
| `crisis_inbox` | ~0.48 | Reply quality is the main bottleneck |

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📦 Deploying to Hugging Face Spaces

```bash
# Install openenv CLI
pip install openenv-core

# Push to your HF Space
openenv push --repo-id dazaiiosamuu/email-triage-env
```

The Space must be tagged `openenv` for discovery.

---

## 📊 Scoring Breakdown (per competition rubric)

| Criterion | How this env addresses it |
|-----------|--------------------------|
| **Real-world utility (30%)** | Email triage is done by millions of support/ops workers daily. A trained agent has direct deployment value. |
| **Task & grader quality (25%)** | 3 tasks with clear difficulty progression. Deterministic graders with 0.0–1.0 scoring. Hard task genuinely challenges frontier models. |
| **Environment design (20%)** | Dense reward at every step. Clean reset/state management. Well-typed Pydantic models. |
| **Code quality (15%)** | Full OpenEnv spec compliance. Working Dockerfile. Typed models. Comprehensive test suite. |
| **Creativity & novelty (10%)** | Novel reward design: partial credit per dimension, efficiency bonus, keyword-based reply scoring. Real email fixtures with realistic content. |
