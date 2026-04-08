"""
Email Triage Environment — Python Client
For use in training loops / external code.

Usage:
    from client import EmailTriageEnvClient

    client = EmailTriageEnvClient(base_url="http://localhost:7860")
    obs = client.reset(task="basic_triage")

    action = {"action_type": "label", "email_id": "e001",
               "category": "support", "priority": "urgent"}
    result = client.step(action)
    print(result["reward"], result["done"])

    state = client.state()
    score = client.score()
"""

import requests
from typing import Any, Dict, Optional


class EmailTriageEnvClient:
    """Synchronous HTTP client for the Email Triage OpenEnv environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _get(self, path: str) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}{path}", timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: Any = None, params: Dict = None) -> Dict[str, Any]:
        r = self._session.post(
            f"{self.base_url}{path}", json=json, params=params, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        """Check environment health."""
        return self._get("/health")

    def tasks(self) -> Dict[str, Any]:
        """List available tasks."""
        return self._get("/tasks")

    def reset(self, task: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.
        Optionally switch to a different task.
        """
        params = {"task": task} if task else {}
        return self._post("/reset", params=params)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take one step. action is a dict with action_type and relevant fields.

        Returns:
            {
                "observation": {...},
                "reward": float,
                "done": bool,
                "info": {...}
            }
        """
        return self._post("/step", json=action)

    def state(self) -> Dict[str, Any]:
        """Return the full current environment state."""
        return self._get("/state")

    def score(self) -> Dict[str, Any]:
        """Compute and return the final episode score."""
        return self._get("/score")

    def run_episode(
        self,
        task: str,
        agent_fn,
        max_steps: int = 30,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method: run a full episode with a callable agent.

        Args:
            task: task name
            agent_fn: callable(observation: dict) -> action: dict
            max_steps: safety cap
            verbose: print step-by-step output

        Returns:
            {"score": float, "steps": int, "rewards": list}
        """
        obs = self.reset(task=task)
        rewards = []

        for step_n in range(1, max_steps + 1):
            action = agent_fn(obs)
            result = self.step(action)
            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            obs    = result.get("observation", obs)
            rewards.append(reward)

            if verbose:
                err = obs.get("last_action_error")
                print(
                    f"  step={step_n} action={action.get('action_type')} "
                    f"reward={reward:.3f} done={done}"
                    + (f" error={err}" if err else "")
                )

            if done:
                break

        final = self.score()
        return {
            "score":   final.get("score", 0.0),
            "steps":   step_n,
            "rewards": rewards,
        }
