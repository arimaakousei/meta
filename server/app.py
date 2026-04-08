"""
Email Triage Environment — FastAPI Server
Exposes POST /reset, POST /step, GET /state, GET /health, GET /score
"""

import os
import uvicorn
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
    from server.environment import EmailTriageEnvironment
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation, EmailTriageState  # type: ignore
    from environment import EmailTriageEnvironment  # type: ignore

# ── App init ──────────────────────────────────────────────────
app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compatible environment for training and evaluating "
        "AI agents on real-world email triage tasks. "
        "Three difficulty levels: basic_triage (easy), mixed_inbox (medium), "
        "crisis_inbox (hard)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "basic_triage")
_env: EmailTriageEnvironment = EmailTriageEnvironment(task_name=TASK_NAME)


# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "task": TASK_NAME}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "basic_triage",
                "difficulty": "easy",
                "description": "Label 5 clear-cut emails by category and priority.",
                "num_emails": 5,
                "max_steps": 10,
            },
            {
                "name": "mixed_inbox",
                "difficulty": "medium",
                "description": "Triage 10 emails including spam, escalations and ambiguous cases.",
                "num_emails": 10,
                "max_steps": 20,
            },
            {
                "name": "crisis_inbox",
                "difficulty": "hard",
                "description": "Handle 15 emails under SLA pressure; draft replies for escalations.",
                "num_emails": 15,
                "max_steps": 30,
            },
        ]
    }


@app.post("/reset", response_model=EmailTriageObservation)
def reset(task: str = None):
    global _env
    global TASK_NAME

    if task and task != TASK_NAME:
        if task not in ("basic_triage", "mixed_inbox", "crisis_inbox"):
            raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")
        TASK_NAME = task
        _env = EmailTriageEnvironment(task_name=task)

    obs = _env.reset()
    return obs


@app.post("/step")
def step(action: EmailTriageAction):
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=EmailTriageState)
def get_state():
    return _env.state()


@app.get("/score")
def get_score():
    score = _env.compute_final_score()
    state = _env.state()
    return {
        "score": score,
        "task": state.task_name,
        "steps": state.step_count,
        "emails_labelled": len(state.agent_labels),
        "emails_archived": len(state.archived_emails),
        "emails_flagged": len(state.flagged_emails),
        "draft_replies": len(state.draft_replies),
        "cumulative_reward": state.cumulative_reward,
    }


# ── Required by openenv validate ─────────────────────────────

def main():
    """Entry point for openenv / project.scripts."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
