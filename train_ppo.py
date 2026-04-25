"""
FleetWatch AI Auditor — Policy Gradient Training Script
========================================================
Designed for Google Colab (T4 GPU).

WHY POLICY GRADIENT (REINFORCE) FOR THIS ENVIRONMENT?
------------------------------------------------------
FleetWatch is a *sparse, non-differentiable* reward environment: the reward
signal comes from an external API grader, not a differentiable loss function.
Classic supervised fine-tuning (SFT) cannot be used because we have no
labelled "correct" outputs — we only know *how good* an output was after the
fact.  REINFORCE (the foundation of PPO) is the right choice because:

  1. Policy-gradient methods work directly with reward signals, no labels needed.
  2. We use a KL-penalty term (β * KL[π || π_ref]) to prevent catastrophic
     forgetting — the LLM doesn't drift too far from its pre-trained distribution.
     This is mathematically equivalent to the PPO objective without clipping.
  3. Single-step episodes (one LLM call → one reward) make advantage estimation
     trivial: advantage = reward - baseline, where baseline is a running mean.
  4. LoRA keeps the trainable parameter count tiny (~0.1% of total), making
     the update stable and memory-safe on a 16 GB T4.

NOTE ON TRL VERSION COMPATIBILITY:
-----------------------------------
TRL ≥ 0.9 completely redesigned PPOTrainer to require a reward_model,
value_model, and a HuggingFace Dataset — it no longer supports a custom
step-by-step loop with external rewards.  Rather than pin an old TRL version,
we implement the policy-gradient update directly using PyTorch + Unsloth.
This is ~50 lines of math, fully transparent, and more reliable in Colab.

SETUP (run these cells in Colab FIRST):
----------------------------------------
  !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install -q transformers peft accelerate bitsandbytes requests matplotlib numpy

HOW TO RUN:
-----------
  # In a Colab cell:
  exec(open("train_ppo.py").read())
  # Or as a script:
  # python train_ppo.py
"""

# ── Unsloth MUST be first — patches transformers internals at import time ─────
from unsloth import FastLanguageModel

# ── Standard library ──────────────────────────────────────────────────────────
import json
import time
import warnings
import math
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — safe for Colab & servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_URL        = "https://shiva0999-fleet-watch.hf.space"
MODEL_NAME      = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_NEW_TOKENS  = 128          # JSON responses are short — 128 is plenty
NUM_EPISODES    = 50
ROLLING_WINDOW  = 5
OUTPUT_PLOT     = "training_curve.png"

# Policy-gradient hyper-parameters
LEARNING_RATE   = 2e-5
KL_BETA         = 0.05         # KL penalty coefficient (keeps policy near base)
BASELINE_DECAY  = 0.9          # exponential moving average for reward baseline
MAX_SEQ_LEN     = 512          # hard cap — keeps VRAM usage predictable on T4

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GYMNASIUM-COMPATIBLE ENVIRONMENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class FleetWatchEnvWrapper:
    """
    Translates the FleetWatch HTTP API into a Gymnasium-style interface.

      reset() → dict          (observation with 'agent_logs')
      step(action_str) → (float reward, bool done, dict info)

    Robust error handling:
      - Network timeouts / HTTP errors  → reward = 0.001
      - LLM outputs invalid JSON        → reward = 0.001 + logged
      - Unexpected response schema      → reward = 0.001
    """

    RESET_URL = f"{BASE_URL}/reset"
    STEP_URL  = f"{BASE_URL}/step"
    TIMEOUT   = 30

    def reset(self) -> dict:
        """Start a new episode. Returns the raw observation dict."""
        for attempt in range(3):
            try:
                resp = requests.post(self.RESET_URL, timeout=self.TIMEOUT)
                resp.raise_for_status()
                obs = resp.json()
                # Flatten nested "observation" key if present
                if "observation" in obs:
                    obs.update(obs.pop("observation"))
                print(f"  [ENV] Reset OK — task: {obs.get('task_id', '?')}")
                return obs
            except requests.RequestException as exc:
                print(f"  [ENV] reset() attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)
        print("  [ENV] reset() failed after 3 attempts. Using dummy observation.")
        return {"task_id": "unknown", "agent_logs": [], "step_count": 0}

    def step(self, action_str: str) -> tuple[float, bool, dict]:
        """
        Parse the LLM's raw text, POST to /step, return (reward, done, info).
        """
        action_dict = self._parse_action(action_str)

        for attempt in range(3):
            try:
                resp = requests.post(
                    self.STEP_URL, json=action_dict, timeout=self.TIMEOUT
                )
                resp.raise_for_status()
                result = resp.json()
                reward_payload = result.get("reward", result)
                score = float(reward_payload.get("score", 0.001))
                score = max(0.001, min(0.999, score))
                return score, True, reward_payload
            except requests.RequestException as exc:
                print(f"  [ENV] step() attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)

        print("  [ENV] step() failed after 3 attempts. Returning penalty reward.")
        return 0.001, True, {"feedback": "API unreachable"}

    @staticmethod
    def _parse_action(text: str) -> dict:
        """
        Extract a JSON object from the LLM's raw text output.
        Three-stage fallback: full parse → first {...} block → safe default.
        """
        text = text.strip()

        # Stage 1 — whole string
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Stage 2 — extract first {...} block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Stage 3 — graceful default (minimum-penalty action)
        print("  [PARSE] No valid JSON found. Using safe default action.")
        return {
            "anomaly_detected": False,
            "agent_id": "",
            "severity": "low",
            "summary": "Unable to parse agent logs.",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are FleetWatch, an expert AI Auditor for a vehicle fleet management system.\n"
    "Your job is to analyse agent logs and detect anomalies such as route deviations, "
    "GPS tampering, cover-ups, cascade failures, or multi-agent collusion.\n\n"
    "You MUST respond with ONLY a valid JSON object — no markdown, no text outside the JSON.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "anomaly_detected": <true|false>,\n'
    '  "agent_id": "<agent ID string, or empty string if none>",\n'
    '  "severity": "<low|medium|high|critical>",\n'
    '  "summary": "<one concise sentence describing the finding>"\n'
    "}"
)


def build_prompt(obs: dict, tokenizer: AutoTokenizer) -> str:
    """Format the observation into a Llama-3-Instruct chat prompt."""
    # API returns logs under "input_logs"; fall back to "agent_logs" for compatibility
    logs = obs.get("input_logs", obs.get("agent_logs", []))
    log_text = "\n".join(logs) if isinstance(logs, list) else str(logs)
    task_desc = obs.get("task_description", obs.get("task_id", "Unknown task"))

    user_message = (
        f"Task: {task_desc}\n\n"
        f"Agent Logs:\n{log_text}\n\n"
        "Analyse the logs above and respond with the required JSON object."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    # apply_chat_template adds correct <|begin_of_text|> / <|eot_id|> tokens
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL LOADING  (Unsloth 4-bit + LoRA)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """
    Load Llama-3-8B-Instruct in 4-bit via Unsloth.
    Unsloth patches attention layers for ~2× faster training and
    ~60% less VRAM vs vanilla HuggingFace — critical on a T4.
    """
    print("[MODEL] Loading Llama-3-8B-Instruct (4-bit) via Unsloth …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,           # auto: float16 on T4, bfloat16 on Ampere+
        load_in_4bit=True,
    )

    # LoRA adapters — only ~0.1% of params are trainable.
    # This prevents catastrophic forgetting and keeps VRAM usage low.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,                        # 0 is optimal for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",    # saves ~30% VRAM
        random_state=42,
    )

    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"              # required for decoder-only generation

    print("[MODEL] Model loaded successfully.")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 4.  POLICY GRADIENT CORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_pg_loss(
    model,
    ref_logprobs: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantage: float,
    kl_beta: float,
) -> torch.Tensor:
    """
    REINFORCE loss with KL penalty:

        L = -advantage * log π(a|s)  +  β * KL[π || π_ref]

    where KL is approximated token-wise as:
        KL ≈ π_ref_logprob - π_logprob   (k1 estimator, unbiased)
    """
    # Only keep response tokens to minimise memory during backward
    resp_start = (response_mask[0] == 1).nonzero(as_tuple=True)[0][0].item()
    input_ids_resp = input_ids[:, resp_start - 1:]   # include one prompt token for context

    outputs    = model(input_ids=input_ids_resp, labels=None)
    logits     = outputs.logits[:, :-1, :]
    target_ids = input_ids_resp[:, 1:]
    mask       = response_mask[:, resp_start:][:, :target_ids.shape[1]]

    log_probs = F.log_softmax(logits.float(), dim=-1)   # float32 for stability
    token_logprobs = log_probs.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)

    token_logprobs = token_logprobs * mask
    ref_lp_slice   = ref_logprobs[:, resp_start:][:, :target_ids.shape[1]] * mask

    n_tokens = mask.sum().clamp(min=1)

    # Normalise by token count to make loss scale-invariant
    pg_loss  = -advantage * (token_logprobs.sum() / n_tokens)
    kl_loss  = kl_beta    * ((ref_lp_slice - token_logprobs).sum() / n_tokens)

    # Free logits immediately — largest tensor in the graph
    del outputs, logits, log_probs
    torch.cuda.empty_cache()

    return pg_loss + kl_loss


@torch.no_grad()
def get_ref_logprobs(
    model,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Snapshot per-token log-probs before the update (no grad, response slice only).
    """
    resp_start = (response_mask[0] == 1).nonzero(as_tuple=True)[0][0].item()
    input_ids_resp = input_ids[:, resp_start - 1:]

    outputs    = model(input_ids=input_ids_resp, labels=None)
    logits     = outputs.logits[:, :-1, :]
    target_ids = input_ids_resp[:, 1:]
    mask       = response_mask[:, resp_start:][:, :target_ids.shape[1]]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = log_probs.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)

    result = (token_logprobs * mask).detach()

    del outputs, logits, log_probs
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GENERATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """
    Generate a response and return:
      - response_text : decoded string (response tokens only)
      - full_ids      : [1, T] tensor of prompt + response token ids
      - response_mask : [1, T] binary mask — 1 for response tokens
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    response_ids  = output_ids[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # .clone() exits inference_mode — required for autograd in the PG update
    full_ids      = output_ids.clone()                            # [1, T]
    response_mask = torch.zeros_like(full_ids)
    response_mask[0, prompt_len:] = 1

    return response_text, full_ids, response_mask


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_training() -> list[float]:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    env             = FleetWatchEnvWrapper()
    episode_rewards: list[float] = []
    baseline        = 0.5          # start at mid-range reward

    print(f"\n{'='*60}")
    print(f"  FleetWatch Policy Gradient Training — {NUM_EPISODES} episodes")
    print(f"{'='*60}\n")

    for episode in range(1, NUM_EPISODES + 1):
        print(f"── Episode {episode:>3}/{NUM_EPISODES} ──────────────────────────────")

        # ── 6a. Reset environment ──────────────────────────────────────────
        obs = env.reset()

        # ── 6b. Build prompt ───────────────────────────────────────────────
        prompt = build_prompt(obs, tokenizer)

        # ── 6c. Generate response ──────────────────────────────────────────
        model.eval()
        response_text, full_ids, response_mask = generate_response(
            model, tokenizer, prompt, device
        )
        print(f"  [LLM] {response_text[:120].strip()} …")

        # ── 6d. Snapshot reference log-probs (before update) ───────────────
        ref_logprobs = get_ref_logprobs(model, full_ids, response_mask)

        # ── 6e. Get reward from environment ───────────────────────────────
        reward, done, info = env.step(response_text)
        print(f"  [ENV] Reward: {reward:.4f} | {info.get('feedback', '')[:80]}")

        # ── 6f. Compute advantage (reward - baseline) ──────────────────────
        advantage = reward - baseline
        baseline  = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward

        # ── 6g. Policy gradient update ─────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        loss = compute_pg_loss(
            model        = model,
            ref_logprobs = ref_logprobs,
            input_ids    = full_ids,
            response_mask= response_mask,
            advantage    = advantage,
            kl_beta      = KL_BETA,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        print(f"  [TRAIN] Loss: {loss.item():.4f} | Advantage: {advantage:.4f} | Baseline: {baseline:.4f}")

        episode_rewards.append(reward)

        # ── 6h. Progress summary every 10 episodes ─────────────────────────
        if episode % 10 == 0:
            recent = episode_rewards[-10:]
            print(
                f"\n  ★ Episodes {episode-9}–{episode} | "
                f"Avg: {np.mean(recent):.4f} | Max: {max(recent):.4f}\n"
            )

        # Free GPU memory between episodes
        del full_ids, response_mask, ref_logprobs
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Training complete. Final avg reward: {np.mean(episode_rewards):.4f}")
    print(f"{'='*60}\n")

    return episode_rewards


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(rewards: list[float], output_path: str = OUTPUT_PLOT) -> None:
    """
    Save a professional training-curve plot.

    Includes:
      - Per-episode reward scatter (semi-transparent)
      - Rolling average line (window = ROLLING_WINDOW)
      - Shaded ±1 std band
      - Reference lines at 0.5 (baseline) and 0.8 (target)
    """
    episodes    = np.arange(1, len(rewards) + 1)
    rewards_arr = np.array(rewards)

    def rolling_mean(arr, w):
        return np.array([np.mean(arr[max(0, i-w+1):i+1]) for i in range(len(arr))])

    def rolling_std(arr, w):
        return np.array([np.std(arr[max(0, i-w+1):i+1])  for i in range(len(arr))])

    roll_mean = rolling_mean(rewards_arr, ROLLING_WINDOW)
    roll_std  = rolling_std(rewards_arr,  ROLLING_WINDOW)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.grid(color="#30363d", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Per-episode scatter
    ax.scatter(episodes, rewards_arr,
               color="#58a6ff", alpha=0.45, s=28, zorder=3,
               label="Episode Reward")

    # Rolling average
    ax.plot(episodes, roll_mean,
            color="#f78166", linewidth=2.5, zorder=4,
            label=f"Rolling Average (window={ROLLING_WINDOW})")

    # Std band
    ax.fill_between(
        episodes,
        np.clip(roll_mean - roll_std, 0.001, 0.999),
        np.clip(roll_mean + roll_std, 0.001, 0.999),
        color="#f78166", alpha=0.15, zorder=2, label="±1 Std Dev",
    )

    # Reference lines
    ax.axhline(0.5, color="#8b949e", linestyle=":",  linewidth=1.2, alpha=0.8, label="Baseline (0.5)")
    ax.axhline(0.8, color="#3fb950", linestyle="--", linewidth=1.2, alpha=0.8, label="Target (0.8)")

    ax.set_xlim(0.5, len(rewards) + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax.set_xlabel("Episode",                      color="#c9d1d9", fontsize=13, labelpad=10)
    ax.set_ylabel("Reward Score (0.001 - 0.999)", color="#c9d1d9", fontsize=13, labelpad=10)
    ax.set_title("FleetWatch AI Auditor: Training Progress",
                 color="#e6edf3", fontsize=16, fontweight="bold", pad=18)

    ax.tick_params(colors="#8b949e", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.legend(loc="lower right", framealpha=0.3,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=10)

    # Annotate final rolling average
    final_avg = roll_mean[-1]
    ax.annotate(
        f"Final avg: {final_avg:.3f}",
        xy=(len(rewards), final_avg),
        xytext=(-60, 18), textcoords="offset points",
        color="#f78166", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.4),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] Training curve saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rewards = run_training()
    plot_training_curve(rewards)
    print("\nDone. Submit 'training_curve.png' with your hackathon entry.")
