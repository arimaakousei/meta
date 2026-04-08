"""
Inference Script — Email Triage OpenEnv
=======================================
Runs an LLM agent against all 3 tasks using the OpenAI client.

MANDATORY env vars:
  API_BASE_URL  — LLM endpoint (default: HuggingFace router)
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace / API key

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=email-triage model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import json
import os
import sys
import time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:7860")
BENCHMARK    = "email-triage"
MAX_STEPS    = 25
TEMPERATURE  = 0.2

TASKS = ["basic_triage", "mixed_inbox", "crisis_inbox"]

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Helpers ───────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    url = f"{SERVER_URL}{path}"
    r = getattr(requests, method)(url, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def build_system_prompt(task_desc: str) -> str:
    return f"""You are an expert email triage agent. Your job is to process an email inbox.

TASK: {task_desc}

Available actions (respond with ONLY valid JSON, no markdown):

1. Label an email:
   {{"action_type": "label", "email_id": "eXXX", "category": "<category>", "priority": "<priority>"}}
   Categories: support | sales | internal | spam | newsletter | urgent_escalation
   Priorities: urgent | normal | low

2. Archive an email (use for spam/newsletters):
   {{"action_type": "archive", "email_id": "eXXX"}}

3. Flag for human review (use for urgent_escalation):
   {{"action_type": "flag", "email_id": "eXXX", "reason": "<why>"}}

4. Respond with a draft reply (use for escalations requiring response):
   {{"action_type": "respond", "email_id": "eXXX", "draft_reply": "<professional reply text>"}}

5. Skip (do nothing):
   {{"action_type": "skip"}}

RULES:
- Process each email exactly once using label, archive, flag, or respond.
- Spam and newsletters → archive.
- Urgent escalations → flag AND label.
- For angry customers / legal notices → respond with a professional reply.
- Output ONLY the JSON action, nothing else.
"""


def build_user_prompt(obs: dict) -> str:
    inbox = obs.get("inbox", [])
    if not inbox:
        return '{"action_type": "skip"}'
    
    emails_text = []
    for e in inbox[:5]:  # show up to 5 emails at a time
        emails_text.append(
            f"ID: {e['id']}\n"
            f"From: {e['sender']}\n"
            f"Subject: {e['subject']}\n"
            f"Body: {e['body'][:300]}\n"
            "---"
        )

    prev_result = obs.get("last_action_result", "")
    prev_error  = obs.get("last_action_error", "")
    remaining   = obs.get("emails_remaining", 0)
    processed   = obs.get("emails_processed", 0)

    prompt = f"Emails remaining: {remaining} | Processed: {processed}\n\n"
    if prev_result:
        prompt += f"Last result: {prev_result}\n"
    if prev_error:
        prompt += f"Last error: {prev_error}\n"
    prompt += "\nINBOX (showing next emails to process):\n\n"
    prompt += "\n".join(emails_text)
    prompt += "\n\nOutput your next action as a single JSON object:"
    return prompt


def parse_action(text: str) -> dict:
    """Extract JSON from LLM output, handling markdown fences."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().strip("`").strip()
    # Find first { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


# ── Per-task runner ───────────────────────────────────────────

def run_task(task_name: str) -> dict:
    rewards = []
    step_n  = 0
    success = False
    score   = 0.0
    error_last = None

    # Reset
    try:
        obs = api("post", f"/reset?task={task_name}")
    except Exception as exc:
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        return {"success": False, "steps": 0, "score": 0.0, "rewards": []}

    system_prompt = build_system_prompt(obs.get("task_description", ""))
    conversation  = []

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    for _step in range(MAX_STEPS):
        step_n += 1
        user_msg = build_user_prompt(obs)
        conversation.append({"role": "user", "content": user_msg})

        # LLM call
        action_str = "{}"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}] + conversation,
                temperature=TEMPERATURE,
                max_tokens=512,
            )
            action_str = response.choices[0].message.content or "{}"
            conversation.append({"role": "assistant", "content": action_str})
        except Exception as exc:
            error_last = str(exc)
            print(
                f"[STEP] step={step_n} action=skip reward=0.00 done=false error={error_last}",
                flush=True,
            )
            rewards.append(0.0)
            time.sleep(2)
            continue

        # Parse + execute action
        reward    = 0.0
        done      = False
        error_msg = "null"
        try:
            action_dict = parse_action(action_str)
            step_resp = api("post", "/step", json=action_dict)
            reward    = float(step_resp.get("reward", 0.0))
            done      = bool(step_resp.get("done", False))
            obs       = step_resp.get("observation", obs)
            err       = obs.get("last_action_error")
            error_msg = err if err else "null"
        except Exception as exc:
            error_msg = str(exc)[:80]
            done = False

        rewards.append(reward)
        print(
            f"[STEP] step={step_n} action={action_dict.get('action_type','?')}({action_dict.get('email_id','')}) "
            f"reward={reward:.2f} done={str(done).lower()} error={error_msg}",
            flush=True,
        )

        if done:
            break

    # Get final score
    try:
        score_resp = api("get", "/score")
        score      = float(score_resp.get("score", 0.0))
        success    = score >= 0.5
    except Exception:
        score   = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_n} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {"success": success, "steps": step_n, "score": score, "rewards": rewards}


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow overriding tasks via CLI args
    tasks_to_run = sys.argv[1:] if len(sys.argv) > 1 else TASKS

    all_results = {}
    for task in tasks_to_run:
        result = run_task(task)
        all_results[task] = result

    # Summary to stderr so it doesn't pollute stdout format
    print("\n=== BASELINE SUMMARY ===", file=sys.stderr)
    for task, res in all_results.items():
        print(f"  {task:20s}  score={res['score']:.2f}  steps={res['steps']}", file=sys.stderr)

    avg = sum(r["score"] for r in all_results.values()) / max(len(all_results), 1)
    print(f"\n  AVERAGE SCORE: {avg:.2f}", file=sys.stderr)
