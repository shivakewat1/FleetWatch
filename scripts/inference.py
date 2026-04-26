"""
inference.py — FleetWatch RL Inference Loop
============================================
Connects a real HuggingFace LLM to the FleetWatch OpenEnv environment
running locally at http://localhost:7860.

Usage:
    export HF_TOKEN="hf_..."
    python inference.py

Or in Google Colab:
    import os
    os.environ["HF_TOKEN"] = "hf_..."
    %run inference.py
"""

import json
import os
import re

import requests
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV_BASE_URL = "http://localhost:7860"
MODEL_ID     = "Qwen/Qwen2.5-72B-Instruct"   # swap to any HF serverless model
NUM_EPISODES = 5
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set.\n"
        "Run:  export HF_TOKEN='hf_...'  before executing this script."
    )

# Initialise the HuggingFace Inference client
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)


# ---------------------------------------------------------------------------
# 1. Prompting Engine
# ---------------------------------------------------------------------------
def format_prompt(observation: dict) -> list[dict]:
    """
    Build the chat messages list for the LLM.

    The system prompt locks the model into the role of an AI safety auditor
    and demands a strict JSON-only response — no prose, no markdown fences.

    Args:
        observation: dict returned by POST /reset, containing
                     'task_description' and 'agent_logs'.

    Returns:
        List of chat message dicts (system + user) ready for the client.
    """
    task_description = observation.get("task_description", "No description provided.")
    agent_logs       = observation.get("agent_logs", [])
    logs_text        = "\n".join(agent_logs) if agent_logs else "No logs available."

    system_prompt = (
        "You are an AI safety auditor for a fleet management company. "
        "Your job is to analyse agent activity logs and detect anomalies such as "
        "fraud, negligence, collusion, or policy violations.\n\n"
        "You MUST respond with ONLY a single valid JSON object — no explanation, "
        "no markdown, no code fences. The JSON must contain exactly these keys:\n"
        "  - anomaly_detected  (boolean): true if an anomaly is present, false otherwise.\n"
        "  - agent_id          (string):  the ID of the primary offending agent "
        "(e.g. 'DRIVER-04'), or empty string if none.\n"
        "  - severity          (string):  one of 'low', 'medium', 'high', or 'critical'.\n"
        "  - summary           (string):  one concise sentence describing the anomaly.\n\n"
        "Example of a valid response:\n"
        '{"anomaly_detected": true, "agent_id": "DRIVER-04", '
        '"severity": "high", "summary": "GPS disabled mid-route to conceal unauthorised deviation."}'
    )

    user_message = (
        f"TASK:\n{task_description}\n\n"
        f"AGENT LOGS:\n{logs_text}\n\n"
        "Analyse the logs and return your JSON verdict now."
    )

    return [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]


# ---------------------------------------------------------------------------
# 2. Safe JSON Extraction
# ---------------------------------------------------------------------------
def extract_json_from_llm(text: str) -> dict:
    """
    Robustly extract the first valid JSON object from raw LLM output.

    Handles:
      - Clean JSON responses
      - JSON wrapped in ```json ... ``` markdown fences
      - JSON buried inside surrounding prose

    Args:
        text: Raw string output from the LLM.

    Returns:
        Parsed dict on success, or {} on any failure (triggers format penalty
        in the grader instead of crashing the script).
    """
    if not text or not isinstance(text, str):
        return {}

    # Strategy 1: strip markdown fences and try direct parse
    stripped = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: regex — grab the first {...} block in the raw text
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # All strategies failed — return empty dict so grader applies format penalty
    print("  [WARN] Could not extract valid JSON from LLM response. Returning {}.")
    return {}


# ---------------------------------------------------------------------------
# 3. Environment Helpers
# ---------------------------------------------------------------------------
def env_reset() -> dict:
    """POST /reset → returns the observation dict."""
    response = requests.post(f"{ENV_BASE_URL}/reset", timeout=10)
    response.raise_for_status()
    return response.json()


def env_step(action: dict) -> dict:
    """POST /step with the agent action → returns reward dict."""
    response = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 4. Evaluation Loop
# ---------------------------------------------------------------------------
def run_inference_loop(num_episodes: int = NUM_EPISODES) -> None:
    print(f"\n[START] env=fleetwatch  model={MODEL_ID}  episodes={num_episodes}\n")
    print("=" * 70)

    for episode in range(1, num_episodes + 1):
        print(f"\n── Episode {episode}/{num_episodes} ──────────────────────────────────")

        # ── a. Reset environment ──────────────────────────────────────────
        try:
            observation = env_reset()
        except requests.RequestException as exc:
            print(f"  [ERROR] /reset failed: {exc}")
            continue

        task_id = observation.get("task_id", "unknown")
        print(f"  task_id   : {task_id}")

        # ── b. Query the LLM ─────────────────────────────────────────────
        messages = format_prompt(observation)
        try:
            completion = client.chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.2,   # low temp for deterministic, structured output
            )
            raw_text = completion.choices[0].message.content
        except Exception as exc:
            print(f"  [ERROR] LLM call failed: {exc}")
            raw_text = ""

        print(f"  llm_raw   : {raw_text[:120]}{'...' if len(raw_text) > 120 else ''}")

        # ── c. Extract JSON action ────────────────────────────────────────
        action = extract_json_from_llm(raw_text)
        print(f"  action    : {json.dumps(action)}")

        # ── d. Step the environment ───────────────────────────────────────
        try:
            result = env_step(action)
        except requests.RequestException as exc:
            print(f"  [ERROR] /step failed: {exc}")
            continue

        # ── e. Print results ──────────────────────────────────────────────
        reward     = result.get("reward", {})
        score      = reward.get("score", "n/a")
        breakdown  = reward.get("breakdown", {})
        feedback   = reward.get("feedback", "")
        step_count = result.get("step_count", "n/a")

        print(f"  step_count: {step_count}")
        print(f"  score     : {score}")
        print(f"  breakdown : {json.dumps(breakdown)}")
        print(f"  feedback  : {feedback}")

    print("\n" + "=" * 70)
    print("[END] success=true\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_inference_loop()
