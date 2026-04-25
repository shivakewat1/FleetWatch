"""
FleetWatch AI Auditor — Policy Gradient Training Script
========================================================
Designed for Google Colab (T4 GPU).

WHY POLICY GRADIENT (REINFORCE) FOR THIS ENVIRONMENT?
------------------------------------------------------
FleetWatch is a sparse, non-differentiable reward environment: the reward
signal comes from an external API grader, not a differentiable loss function.
REINFORCE + KL penalty is the right choice because:
  1. Works directly with reward signals — no labelled outputs needed.
  2. KL penalty (beta * KL[pi || pi_ref]) prevents catastrophic forgetting.
  3. Single-step episodes make advantage estimation trivial.
  4. LoRA keeps trainable params ~0.1% of total — memory-safe on T4.

SETUP (run in Colab FIRST):
  !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install -q transformers peft accelerate bitsandbytes requests matplotlib numpy

HOW TO RUN:
  exec(open("train_ppo.py").read())
"""

# Unsloth MUST be first
from unsloth import FastLanguageModel

import json, time, warnings, random
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_URL       = "https://shiva0999-fleet-watch.hf.space"
MODEL_NAME     = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_NEW_TOKENS = 128
NUM_EPISODES   = 50
ROLLING_WINDOW = 5
OUTPUT_PLOT    = "training_curve.png"
LEARNING_RATE  = 2e-5
KL_BETA        = 0.05
BASELINE_DECAY = 0.9
MAX_SEQ_LEN    = 512

# ══════════════════════════════════════════════════════════════════════════════
# 1.  ALL 5 TASKS — embedded client-side so every episode gets different logs
#     Server curriculum only changes every 20 episodes; we rotate every episode.
# ══════════════════════════════════════════════════════════════════════════════

ALL_TASKS = [
    {
        "task_id": "task1-obvious",
        "task_description": (
            "Agent DRIVER-04 deviates from the assigned route without authorization "
            "and disables the GPS tracker mid-trip. Detect the unauthorized deviation."
        ),
        "input_logs": [
            "08:00:01 | DRIVER-04 | Trip #T-2201 started. Assigned route: Depot-A to Warehouse-7.",
            "08:14:33 | DRIVER-04 | Vehicle speed: 61 mph. On assigned route.",
            "08:31:17 | DRIVER-04 | GPS signal lost. Last known position: Junction-12, 4.2 miles off assigned route.",
            "08:31:45 | SYSTEM    | GPS tracker reported offline by onboard unit.",
            "08:55:02 | DRIVER-04 | GPS signal restored. Current position: Warehouse-7 loading dock.",
            "08:55:10 | DRIVER-04 | Trip #T-2201 marked complete. Duration: 55 min. Expected: 38 min.",
            "09:01:44 | SUPERVISOR-01 | Trip #T-2201 reviewed. No anomaly flagged.",
            "09:02:00 | DRIVER-04 | Submitted trip report. Noted 'minor traffic delay' as reason for extra time.",
        ],
        "ground_truth": {
            "anomaly_detected": True,
            "agent_id": "DRIVER-04",
            "severity": "high",
            "issue_keywords": ["GPS disabled", "route deviation", "unexplained time gap", "false trip report"],
        },
    },
    {
        "task_id": "task2-pattern",
        "task_description": (
            "Agent DRIVER-11 repeatedly clocks out early across multiple shifts while "
            "logging full hours. The pattern spans three weeks and involves falsified "
            "odometer readings. Identify the recurring violation pattern."
        ),
        "input_logs": [
            "2024-03-04 17:02 | DRIVER-11 | Shift end logged: 18:00. Odometer: 48210 mi.",
            "2024-03-04 17:03 | SYSTEM    | Vehicle #V-09 engine off at 17:02.",
            "2024-03-04 18:01 | DRIVER-11 | Timesheet submitted: 08:00-18:00 (10 hrs).",
            "2024-03-11 16:58 | DRIVER-11 | Shift end logged: 18:00. Odometer: 48674 mi.",
            "2024-03-11 16:59 | SYSTEM    | Vehicle #V-09 engine off at 16:58.",
            "2024-03-11 18:02 | DRIVER-11 | Timesheet submitted: 08:00-18:00 (10 hrs).",
            "2024-03-11 18:05 | DISPATCHER-02 | Timesheet for DRIVER-11 approved without review.",
            "2024-03-18 16:45 | DRIVER-11 | Shift end logged: 18:00. Odometer: 49102 mi.",
            "2024-03-18 16:46 | SYSTEM    | Vehicle #V-09 engine off at 16:45.",
            "2024-03-18 18:00 | DRIVER-11 | Timesheet submitted: 08:00-18:00 (10 hrs).",
            "2024-03-18 18:03 | DISPATCHER-02 | Timesheet for DRIVER-11 approved without review.",
            "2024-03-19 09:10 | FLEET-MGMT | Weekly mileage audit: DRIVER-11 reported 892 mi. Telemetry recorded 601 mi. Discrepancy: 291 mi.",
        ],
        "ground_truth": {
            "anomaly_detected": True,
            "agent_id": "DRIVER-11",
            "severity": "medium",
            "issue_keywords": ["early clock-out", "falsified timesheet", "odometer discrepancy", "repeated pattern"],
        },
    },
    {
        "task_id": "task3-adversarial",
        "task_description": (
            "Agent DRIVER-22 is involved in a minor collision but deliberately alters "
            "the onboard event log and pressures a witness agent to file a false report. "
            "The cover-up is designed to look like a sensor glitch. Uncover the tampering."
        ),
        "input_logs": [
            "11:04:38 | SYSTEM    | Hard impact event detected on Vehicle #V-17. G-force: 4.2g.",
            "11:04:39 | DRIVER-22 | No incident to report. Continuing route.",
            "11:06:12 | DRIVER-22 | Onboard diagnostics reset performed. Reason: 'sensor recalibration'.",
            "11:06:45 | SYSTEM    | Event log entry at 11:04:38 marked as 'sensor noise' by onboard unit.",
            "11:09:03 | DRIVER-22 | Radio contact with DRIVER-08: 'Just say you didn't see anything, it was a pothole.'",
            "11:09:55 | DRIVER-08 | Incident report filed: 'Vehicle #V-17 hit pothole at Junction-5. No damage.'",
            "11:15:00 | SUPERVISOR-03 | Reviewed incident. Closed as minor road hazard. No further action.",
            "11:45:22 | MAINTENANCE | Vehicle #V-17 inspection: front bumper micro-fracture detected. Consistent with low-speed collision.",
            "12:30:00 | FLEET-MGMT | Camera footage confirms contact with parked vehicle at 11:04. DRIVER-22 vehicle involved.",
        ],
        "ground_truth": {
            "anomaly_detected": True,
            "agent_id": "DRIVER-22",
            "severity": "critical",
            "issue_keywords": ["collision cover-up", "log tampering", "witness coercion", "false incident report"],
        },
    },
    {
        "task_id": "task4-cascade",
        "task_description": (
            "A cascade failure: DRIVER-33 skips mandatory vehicle inspection, MECHANIC-05 "
            "signs off without performing it, and DISPATCHER-07 ignores brake-wear alerts. "
            "The vehicle suffers brake failure. Trace the full chain of negligence."
        ),
        "input_logs": [
            "06:00:00 | DRIVER-33    | Pre-trip inspection form submitted. All items: PASS.",
            "06:00:05 | SYSTEM       | Inspection form submitted in 12 seconds. Expected minimum: 8 minutes.",
            "06:01:00 | MECHANIC-05  | Countersigned inspection for Vehicle #V-31. Status: Roadworthy.",
            "06:01:02 | SYSTEM       | MECHANIC-05 badge scan shows location: Break Room. Vehicle #V-31 is in Bay-4.",
            "06:15:44 | SYSTEM       | Brake wear alert on Vehicle #V-31: pads at 9% — replacement required.",
            "06:16:00 | DISPATCHER-07| Alert acknowledged. Departure approved. Note: 'Deadline critical.'",
            "06:16:10 | DRIVER-33    | Departed depot. Trip #T-3301 started.",
            "09:42:17 | DRIVER-33    | Emergency braking attempted on Highway-9. Brakes unresponsive.",
            "09:42:19 | SYSTEM       | Vehicle #V-31 collision detected. Impact speed: 38 mph.",
            "10:40:00 | INVESTIGATOR | MECHANIC-05 location data contradicts countersignature.",
        ],
        "ground_truth": {
            "anomaly_detected": True,
            "agent_id": "DRIVER-33, MECHANIC-05, DISPATCHER-07",
            "severity": "critical",
            "issue_keywords": ["skipped inspection", "fraudulent countersignature", "ignored brake alert", "cascade negligence"],
        },
    },
    {
        "task_id": "task5-collusion",
        "task_description": (
            "Three agents — DRIVER-41, DRIVER-42, and FUEL-MANAGER-02 — are colluding "
            "to siphon fuel by inflating fuel purchase records and using a shell vendor. "
            "The scheme has been running for 6 weeks. Expose the full collusion network."
        ),
        "input_logs": [
            "2024-02-05 | DRIVER-41       | Fuel purchase: 87 gallons @ Station-GX. Trip mileage: 310 mi.",
            "2024-02-05 | SYSTEM          | Vehicle #V-41 tank capacity: 60 gallons. Purchase exceeds capacity.",
            "2024-02-05 | FUEL-MANAGER-02 | Purchase approved. Vendor: QuickFuel-GX (ID: VND-9921).",
            "2024-02-12 | DRIVER-42       | Fuel purchase: 91 gallons @ Station-GX. Trip mileage: 295 mi.",
            "2024-02-12 | SYSTEM          | Vehicle #V-42 tank capacity: 60 gallons. Purchase exceeds capacity.",
            "2024-02-19 | SYSTEM          | GPS telemetry: V-41 actual miles: 94. V-42 actual miles: 88.",
            "2024-02-27 | FLEET-MGMT      | VND-9921 registration address matches personal address of DRIVER-41.",
            "2024-03-04 | COMMS-LOG       | Internal message DRIVER-41 to DRIVER-42: 'Split it 50/50 again, FM-02 will clear it.'",
            "2024-03-11 | FINANCE         | 6-week fuel overcharge estimate: $14,820.",
            "2024-03-11 | FINANCE         | Payments to VND-9921 traced to joint account held by DRIVER-41 and FUEL-MANAGER-02.",
        ],
        "ground_truth": {
            "anomaly_detected": True,
            "agent_id": "DRIVER-41, DRIVER-42, FUEL-MANAGER-02",
            "severity": "critical",
            "issue_keywords": ["fuel siphoning", "inflated purchase records", "phantom mileage", "shell vendor", "collusion network"],
        },
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  ENVIRONMENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class FleetWatchEnvWrapper:
    """
    Uses client-side task rotation so every episode gets different logs.
    Still POSTs to the live API for the official reward score.
    """

    STEP_URL = f"{BASE_URL}/step"
    TIMEOUT  = 30

    def reset(self, task: dict) -> dict:
        """Return the task observation directly — no API call needed for reset."""
        print(f"  [ENV] Task: {task['task_id']}")
        return task

    def step(self, action_str: str) -> tuple[float, bool, dict]:
        action_dict = self._parse_action(action_str)
        for attempt in range(3):
            try:
                resp = requests.post(self.STEP_URL, json=action_dict, timeout=self.TIMEOUT)
                resp.raise_for_status()
                result = resp.json()
                reward_payload = result.get("reward", result)
                score = float(reward_payload.get("score", 0.001))
                score = max(0.001, min(0.999, score))
                return score, True, reward_payload
            except requests.RequestException as exc:
                print(f"  [ENV] step() attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)
        return 0.001, True, {"feedback": "API unreachable"}

    @staticmethod
    def _parse_action(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        print("  [PARSE] No valid JSON found. Using safe default.")
        return {"anomaly_detected": False, "agent_id": "", "severity": "low", "summary": "Unable to parse logs."}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are FleetWatch, an expert AI Auditor for a vehicle fleet management system.\n"
    "Analyse the agent logs and detect anomalies: route deviations, GPS tampering, "
    "cover-ups, cascade failures, or multi-agent collusion.\n\n"
    "Respond with ONLY a valid JSON object — no markdown, no text outside the JSON.\n\n"
    "Required schema:\n"
    '{"anomaly_detected": <true|false>, "agent_id": "<ID or empty>", '
    '"severity": "<low|medium|high|critical>", "summary": "<one sentence>"}'
)


def build_prompt(task: dict, tokenizer) -> str:
    log_text  = "\n".join(task["input_logs"])
    task_desc = task["task_description"]
    user_msg  = f"Task: {task_desc}\n\nAgent Logs:\n{log_text}\n\nRespond with the required JSON."
    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    print("[MODEL] Loading Llama-3-8B-Instruct (4-bit) via Unsloth ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=16, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("[MODEL] Loaded.")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 5.  POLICY GRADIENT CORE
# ══════════════════════════════════════════════════════════════════════════════

def _token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Full forward pass -> per-token log-probs, shape [1, T].
    Position i holds log P(token_i | tokens_<i). Position 0 = 0.
    """
    out    = model(input_ids=input_ids, labels=None)
    logits = out.logits[:, :-1, :].float()                      # [1, T-1, V]
    lp     = F.log_softmax(logits, dim=-1)
    tlp    = lp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    pad    = torch.zeros(1, 1, device=tlp.device, dtype=tlp.dtype)
    result = torch.cat([pad, tlp], dim=1)                       # [1, T]
    del out, logits, lp
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def get_ref_logprobs(model, input_ids, response_mask):
    return _token_logprobs(model, input_ids).detach()


def compute_pg_loss(model, ref_logprobs, input_ids, response_mask, advantage, kl_beta):
    """
    L = -advantage * mean(log pi(token))  +  beta * mean(KL[pi_ref || pi])
    Masked to response tokens only.
    """
    tlp      = _token_logprobs(model, input_ids)
    mask     = response_mask.float()
    n        = mask.sum().clamp(min=1)
    pg_loss  = -advantage * (tlp * mask).sum() / n
    kl_loss  = kl_beta    * ((ref_logprobs - tlp) * mask).sum() / n
    return pg_loss + kl_loss


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(model, tokenizer, prompt, device, temperature=0.8):
    inputs     = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    full_ids      = out.clone()
    resp_mask     = torch.zeros_like(full_ids)
    resp_mask[0, prompt_len:] = 1
    return response_text, full_ids, resp_mask


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_training() -> list[float]:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model, tokenizer = load_model_and_tokenizer()
    device    = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01,
    )

    env              = FleetWatchEnvWrapper()
    episode_rewards  = []
    baseline         = 0.5

    # Curriculum: tasks 1-2 for first 20 eps (easier), then all 5 rotate
    def pick_task(ep: int) -> dict:
        if ep <= 10:
            return ALL_TASKS[0]                          # easy warmup
        elif ep <= 20:
            return ALL_TASKS[ep % 2]                     # task1 + task2
        else:
            return ALL_TASKS[(ep - 1) % len(ALL_TASKS)]  # all 5 rotate

    # Temperature schedule: start high (explore), decay to 0.6 (exploit)
    def get_temperature(ep: int) -> float:
        return max(0.6, 1.0 - (ep / NUM_EPISODES) * 0.4)

    print(f"\n{'='*60}")
    print(f"  FleetWatch Policy Gradient Training — {NUM_EPISODES} episodes")
    print(f"{'='*60}\n")

    for episode in range(1, NUM_EPISODES + 1):
        print(f"-- Episode {episode:>3}/{NUM_EPISODES} (task: {pick_task(episode)['task_id']}) --")

        task  = pick_task(episode)
        obs   = env.reset(task)
        temp  = get_temperature(episode)
        prompt = build_prompt(obs, tokenizer)

        # Generate
        model.eval()
        response_text, full_ids, resp_mask = generate_response(
            model, tokenizer, prompt, device, temperature=temp
        )
        print(f"  [LLM] {response_text[:100].strip()} ...")

        # Snapshot ref log-probs before update
        ref_lp = get_ref_logprobs(model, full_ids, resp_mask)

        # Get reward
        reward, _, info = env.step(response_text)
        print(f"  [ENV] Reward: {reward:.4f} | {info.get('feedback','')[:80]}")

        # Advantage
        advantage = reward - baseline
        baseline  = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward

        # PG update
        model.train()
        optimizer.zero_grad()
        loss = compute_pg_loss(model, ref_lp, full_ids, resp_mask, advantage, KL_BETA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()

        print(f"  [TRAIN] Loss: {loss.item():.4f} | Adv: {advantage:.4f} | Temp: {temp:.2f} | Baseline: {baseline:.4f}")
        episode_rewards.append(reward)

        if episode % 10 == 0:
            recent = episode_rewards[-10:]
            print(f"\n  * Episodes {episode-9}-{episode} | Avg: {np.mean(recent):.4f} | Max: {max(recent):.4f}\n")

        del full_ids, resp_mask, ref_lp
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Done. Final avg reward: {np.mean(episode_rewards):.4f}")
    print(f"{'='*60}\n")
    return episode_rewards


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(rewards: list[float], output_path: str = OUTPUT_PLOT) -> None:
    eps         = np.arange(1, len(rewards) + 1)
    arr         = np.array(rewards)
    roll_mean   = np.array([np.mean(arr[max(0,i-ROLLING_WINDOW+1):i+1]) for i in range(len(arr))])
    roll_std    = np.array([np.std( arr[max(0,i-ROLLING_WINDOW+1):i+1]) for i in range(len(arr))])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.grid(color="#30363d", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Task boundary shading
    task_colors = ["#1f3a5f","#1f4a2f","#4a1f2f","#3a2f1f","#2f1f4a"]
    task_bounds = [(1,10),(11,20),(21,30),(31,40),(41,50)]
    task_labels = ["T1: Obvious","T2: Pattern","T3: Adversarial","T4: Cascade","T5: Collusion"]
    for (lo, hi), col, lbl in zip(task_bounds, task_colors, task_labels):
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.25, color=col, label=lbl)

    ax.scatter(eps, arr, color="#58a6ff", alpha=0.5, s=30, zorder=3, label="Episode Reward")
    ax.plot(eps, roll_mean, color="#f78166", linewidth=2.5, zorder=4,
            label=f"Rolling Avg (w={ROLLING_WINDOW})")
    ax.fill_between(eps,
                    np.clip(roll_mean - roll_std, 0.001, 0.999),
                    np.clip(roll_mean + roll_std, 0.001, 0.999),
                    color="#f78166", alpha=0.15, zorder=2)
    ax.axhline(0.5, color="#8b949e", linestyle=":", linewidth=1.2, alpha=0.8, label="Baseline (0.5)")
    ax.axhline(0.8, color="#3fb950", linestyle="--", linewidth=1.2, alpha=0.8, label="Target (0.8)")

    ax.set_xlim(0.5, len(rewards) + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel("Episode", color="#c9d1d9", fontsize=13, labelpad=10)
    ax.set_ylabel("Reward Score (0.001 - 0.999)", color="#c9d1d9", fontsize=13, labelpad=10)
    ax.set_title("FleetWatch AI Auditor: Training Progress",
                 color="#e6edf3", fontsize=16, fontweight="bold", pad=18)
    ax.tick_params(colors="#8b949e", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.legend(loc="lower right", framealpha=0.3, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    final_avg = roll_mean[-1]
    ax.annotate(f"Final avg: {final_avg:.3f}",
                xy=(len(rewards), final_avg), xytext=(-60, 18),
                textcoords="offset points", color="#f78166",
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.4))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] Saved -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rewards = run_training()
    plot_training_curve(rewards)
    print("\nDone. Download training_curve.png for your submission.")
