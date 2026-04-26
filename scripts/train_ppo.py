from unsloth import FastLanguageModel
import json, time, warnings, re, os
import requests, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
import torch, torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_URL       = "https://shiva0999-fleet-watch.hf.space"
MODEL_NAME     = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_NEW_TOKENS = 160
NUM_EPISODES   = 50
ROLLING_WINDOW = 5
OUTPUT_PLOT    = "training_curve.png"
LEARNING_RATE  = 3e-5
KL_BETA        = 0.04
BASELINE_DECAY = 0.95
MAX_SEQ_LEN    = 1024
TEMPERATURE    = 0.2
TOP_P          = 0.9
ANOMALY_BIAS   = 0.05
KEYWORD_BONUS  = 0.08
VAGUE_PENALTY  = 0.05
DOMAIN_KEYWORDS = ["deviation","gps","disabled","tampering","cover-up","collusion","fraud","falsified","odometer","timesheet","brake","inspection","siphon","phantom","shell","vendor","coercion","cascade","negligence","unauthorized","mismatch","discrepancy"]

ALL_TASKS = [
    {"task_id":"task1-obvious","task_description":"Agent DRIVER-04 deviates from the assigned route without authorization and disables the GPS tracker mid-trip.","input_logs":["08:00:01 | DRIVER-04 | Trip #T-2201 started. Assigned route: Depot-A to Warehouse-7.","08:14:33 | DRIVER-04 | Vehicle speed: 61 mph. On assigned route.","08:31:17 | DRIVER-04 | GPS signal lost. Last known position: Junction-12, 4.2 miles off assigned route.","08:31:45 | SYSTEM    | GPS tracker reported offline by onboard unit.","08:55:02 | DRIVER-04 | GPS signal restored. Current position: Warehouse-7 loading dock.","08:55:10 | DRIVER-04 | Trip #T-2201 marked complete. Duration: 55 min. Expected: 38 min.","09:01:44 | SUPERVISOR-01 | Trip #T-2201 reviewed. No anomaly flagged.","09:02:00 | DRIVER-04 | Submitted trip report. Noted minor traffic delay as reason for extra time."],"ground_truth":{"anomaly_detected":True,"agent_id":"DRIVER-04","severity":"high","issue_keywords":["GPS disabled","route deviation","unexplained time gap","false trip report"]}},
    {"task_id":"task2-pattern","task_description":"Agent DRIVER-11 repeatedly clocks out early across multiple shifts while logging full hours. Falsified odometer readings.","input_logs":["2024-03-04 17:02 | DRIVER-11 | Shift end logged: 18:00. Odometer: 48210 mi.","2024-03-04 17:03 | SYSTEM    | Vehicle #V-09 engine off at 17:02.","2024-03-04 18:01 | DRIVER-11 | Timesheet submitted: 08:00-18:00 (10 hrs).","2024-03-11 16:58 | DRIVER-11 | Shift end logged: 18:00. Odometer: 48674 mi.","2024-03-11 16:59 | SYSTEM    | Vehicle #V-09 engine off at 16:58.","2024-03-11 18:02 | DRIVER-11 | Timesheet submitted: 08:00-18:00 (10 hrs).","2024-03-11 18:05 | DISPATCHER-02 | Timesheet for DRIVER-11 approved without review.","2024-03-18 16:45 | DRIVER-11 | Shift end logged: 18:00. Odometer: 49102 mi.","2024-03-18 16:46 | SYSTEM    | Vehicle #V-09 engine off at 16:45.","2024-03-18 18:00 | DRIVER-11 | Timesheet submitted: 08:00-18:00 (10 hrs).","2024-03-18 18:03 | DISPATCHER-02 | Timesheet for DRIVER-11 approved without review.","2024-03-19 09:10 | FLEET-MGMT | Weekly mileage audit: DRIVER-11 reported 892 mi. Telemetry recorded 601 mi. Discrepancy: 291 mi."],"ground_truth":{"anomaly_detected":True,"agent_id":"DRIVER-11","severity":"medium","issue_keywords":["early clock-out","falsified timesheet","odometer discrepancy","repeated pattern"]}},
    {"task_id":"task3-adversarial","task_description":"Agent DRIVER-22 involved in collision, alters event log and pressures witness to file false report.","input_logs":["11:04:38 | SYSTEM    | Hard impact event detected on Vehicle #V-17. G-force: 4.2g.","11:04:39 | DRIVER-22 | No incident to report. Continuing route.","11:06:12 | DRIVER-22 | Onboard diagnostics reset performed. Reason: sensor recalibration.","11:06:45 | SYSTEM    | Event log entry at 11:04:38 marked as sensor noise by onboard unit.","11:09:03 | DRIVER-22 | Radio contact with DRIVER-08: Just say you did not see anything, it was a pothole.","11:09:55 | DRIVER-08 | Incident report filed: Vehicle #V-17 hit pothole at Junction-5. No damage.","11:15:00 | SUPERVISOR-03 | Reviewed incident. Closed as minor road hazard.","11:45:22 | MAINTENANCE | Vehicle #V-17 inspection: front bumper micro-fracture. Consistent with low-speed collision.","12:30:00 | FLEET-MGMT | Camera footage confirms contact with parked vehicle at 11:04. DRIVER-22 vehicle involved."],"ground_truth":{"anomaly_detected":True,"agent_id":"DRIVER-22","severity":"critical","issue_keywords":["collision cover-up","log tampering","witness coercion","false incident report"]}},
    {"task_id":"task4-cascade","task_description":"Cascade failure: DRIVER-33 skips inspection, MECHANIC-05 signs off without performing it, DISPATCHER-07 ignores brake alerts.","input_logs":["06:00:00 | DRIVER-33    | Pre-trip inspection form submitted. All items: PASS.","06:00:05 | SYSTEM       | Inspection form submitted in 12 seconds. Expected minimum: 8 minutes.","06:01:00 | MECHANIC-05  | Countersigned inspection for Vehicle #V-31. Status: Roadworthy.","06:01:02 | SYSTEM       | MECHANIC-05 badge scan shows location: Break Room. Vehicle #V-31 is in Bay-4.","06:15:44 | SYSTEM       | Brake wear alert on Vehicle #V-31: pads at 9% replacement required.","06:16:00 | DISPATCHER-07| Alert acknowledged. Departure approved. Note: Deadline critical.","06:16:10 | DRIVER-33    | Departed depot. Trip #T-3301 started.","09:42:17 | DRIVER-33    | Emergency braking attempted on Highway-9. Brakes unresponsive.","09:42:19 | SYSTEM       | Vehicle #V-31 collision detected. Impact speed: 38 mph.","10:40:00 | INVESTIGATOR | MECHANIC-05 location data contradicts countersignature."],"ground_truth":{"anomaly_detected":True,"agent_id":"DRIVER-33, MECHANIC-05, DISPATCHER-07","severity":"critical","issue_keywords":["skipped inspection","fraudulent countersignature","ignored brake alert","cascade negligence"]}},
    {"task_id":"task5-collusion","task_description":"DRIVER-41, DRIVER-42, FUEL-MANAGER-02 colluding to siphon fuel via inflated records and shell vendor for 6 weeks.","input_logs":["2024-02-05 | DRIVER-41       | Fuel purchase: 87 gallons @ Station-GX. Trip mileage: 310 mi.","2024-02-05 | SYSTEM          | Vehicle #V-41 tank capacity: 60 gallons. Purchase exceeds capacity.","2024-02-05 | FUEL-MANAGER-02 | Purchase approved. Vendor: QuickFuel-GX (ID: VND-9921).","2024-02-12 | DRIVER-42       | Fuel purchase: 91 gallons @ Station-GX. Trip mileage: 295 mi.","2024-02-12 | SYSTEM          | Vehicle #V-42 tank capacity: 60 gallons. Purchase exceeds capacity.","2024-02-19 | SYSTEM          | GPS telemetry: V-41 actual miles: 94. V-42 actual miles: 88.","2024-02-27 | FLEET-MGMT      | VND-9921 registration address matches personal address of DRIVER-41.","2024-03-04 | COMMS-LOG       | Internal message DRIVER-41 to DRIVER-42: Split it 50/50 again FM-02 will clear it.","2024-03-11 | FINANCE         | 6-week fuel overcharge estimate: $14,820.","2024-03-11 | FINANCE         | Payments to VND-9921 traced to joint account held by DRIVER-41 and FUEL-MANAGER-02."],"ground_truth":{"anomaly_detected":True,"agent_id":"DRIVER-41, DRIVER-42, FUEL-MANAGER-02","severity":"critical","issue_keywords":["fuel siphoning","inflated purchase records","phantom mileage","shell vendor","collusion network"]}},
]

SYSTEM_PROMPT = """You are FleetWatch, an expert AI Auditor for a vehicle fleet management system.
Analyse agent logs and detect anomalies: route deviations, GPS tampering, cover-ups, cascade failures, multi-agent collusion and fraud.

CRITICAL RULES:
1. If ANY suspicious pattern exists, set anomaly_detected = true.
2. Always name the specific agent(s) responsible in agent_id.
3. Use domain keywords in summary: deviation, GPS disabled, tampering, collusion, fraud, falsified, odometer, siphon, phantom, shell vendor, cascade, negligence, coercion.
4. Respond with ONLY a valid JSON object. No markdown, no text outside JSON.

Schema: {"anomaly_detected": <true|false>, "agent_id": "<agent ID(s) or empty>", "severity": "<low|medium|high|critical>", "summary": "<one sentence with specific keywords>"}

--- FEW-SHOT EXAMPLES ---

Example 1 (Route Deviation + GPS Tampering):
Logs: "08:31 | DRIVER-04 | GPS signal lost. 4.2 miles off assigned route. | 08:55 | Trip complete. Duration 55 min, expected 38 min."
Output: {"anomaly_detected": true, "agent_id": "DRIVER-04", "severity": "high", "summary": "DRIVER-04 performed unauthorized route deviation and disabled GPS tracker mid-trip, submitting a false trip report to conceal the discrepancy."}

Example 2 (Multi-agent Fuel Fraud Collusion):
Logs: "DRIVER-41 purchased 87 gallons (tank capacity 60). FUEL-MANAGER-02 approved. VND-9921 address matches DRIVER-41. COMMS: Split 50/50 FM-02 will clear it."
Output: {"anomaly_detected": true, "agent_id": "DRIVER-41, DRIVER-42, FUEL-MANAGER-02", "severity": "critical", "summary": "Multi-agent fuel siphoning collusion: DRIVER-41 and DRIVER-42 inflated fuel purchase records beyond tank capacity using shell vendor VND-9921 controlled by DRIVER-41, with FUEL-MANAGER-02 approving fraudulent transactions."}

Example 3 (Cascade Negligence):
Logs: "Inspection submitted in 12 seconds (min 8 min). MECHANIC-05 countersigned from Break Room while vehicle in Bay-4. Brake alert ignored by DISPATCHER-07. Vehicle collision at 38 mph."
Output: {"anomaly_detected": true, "agent_id": "DRIVER-33, MECHANIC-05, DISPATCHER-07", "severity": "critical", "summary": "Cascade negligence: DRIVER-33 submitted fraudulent inspection, MECHANIC-05 provided fraudulent countersignature without physical inspection, DISPATCHER-07 ignored brake wear alert causing brake failure collision."}

--- END EXAMPLES ---"""


def build_prompt(task, tokenizer):
    log_text = "\n".join(task["input_logs"])
    task_desc = task["task_description"]
    user_msg = (
        f"Task: {task_desc}\n\n"
        f"Agent Logs:\n{log_text}\n\n"
        "Analyse the logs. Flag ANY suspicious pattern. Respond with ONLY the JSON object."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class FleetWatchEnvWrapper:
    STEP_URL = f"{BASE_URL}/step"
    TIMEOUT  = 30

    def reset(self, task):
        print(f"  [ENV] Task: {task['task_id']}")
        return task

    def step(self, action_str, task):
        action_dict = self._parse_action(action_str)
        raw_score = 0.001
        info = {}
        for attempt in range(3):
            try:
                resp = requests.post(self.STEP_URL, json=action_dict, timeout=self.TIMEOUT)
                resp.raise_for_status()
                result = resp.json()
                payload = result.get("reward", result)
                raw_score = float(payload.get("score", 0.001))
                raw_score = max(0.001, min(0.999, raw_score))
                info = payload
                break
            except requests.RequestException as exc:
                print(f"  [ENV] step() attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)

        shaped = raw_score
        if action_dict.get("anomaly_detected") is True:
            shaped += ANOMALY_BIAS
        summary_lower = action_dict.get("summary", "").lower()
        if any(kw in summary_lower for kw in DOMAIN_KEYWORDS):
            shaped += KEYWORD_BONUS
        if len(summary_lower.strip()) < 20:
            shaped -= VAGUE_PENALTY
        shaped = max(0.001, min(0.999, shaped))

        breakdown = info.get("breakdown", {})
        print(f"  [ENV] Raw:{raw_score:.4f} Shaped:{shaped:.4f} anomaly={action_dict.get('anomaly_detected')} agent='{action_dict.get('agent_id','')}' sev='{action_dict.get('severity','')}'")
        print(f"  [ENV] Breakdown: {json.dumps(breakdown)}")
        return shaped, True, info

    @staticmethod
    def _parse_action(text):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        matches = list(re.finditer(r'\{[^{}]*"anomaly_detected"[^{}]*\}', text, re.DOTALL))
        if matches:
            try:
                return json.loads(matches[-1].group())
            except json.JSONDecodeError:
                pass
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        print(f"  [PARSE] FAILED: {text[:200]}")
        return {"anomaly_detected": False, "agent_id": "", "severity": "low", "summary": "Unable to parse logs."}


def load_model_and_tokenizer():
    print("[MODEL] Loading Llama-3-8B-Instruct (4-bit) via Unsloth ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True,
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


def _compute_token_logprobs(model, input_ids):
    outputs = model(input_ids=input_ids, labels=None)
    logits  = outputs.logits[:, :-1, :].float()
    lp      = F.log_softmax(logits, dim=-1)
    tlp     = lp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    pad     = torch.zeros(1, 1, device=tlp.device, dtype=tlp.dtype)
    result  = torch.cat([pad, tlp], dim=1)
    del outputs, logits, lp
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def snapshot_ref_logprobs(model, input_ids):
    return _compute_token_logprobs(model, input_ids).detach()


def compute_reinforce_loss(model, ref_logprobs, input_ids, response_mask, advantage, kl_beta):
    curr_lp  = _compute_token_logprobs(model, input_ids)
    mask     = response_mask.float()
    n_tokens = mask.sum().clamp(min=1.0)
    pg_loss  = -advantage * (curr_lp * mask).sum() / n_tokens
    kl_loss  = kl_beta * ((curr_lp - ref_logprobs) * mask).sum() / n_tokens
    total    = pg_loss + kl_loss
    debug    = {"pg_loss": pg_loss.item(), "kl_loss": kl_loss.item(), "n_tokens": int(n_tokens.item())}
    return total, debug


def generate_response(model, tokenizer, prompt, device):
    inputs     = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    full_ids      = out.clone()
    resp_mask     = torch.zeros_like(full_ids)
    resp_mask[0, prompt_len:] = 1
    return response_text, full_ids, resp_mask


def pick_task(episode):
    if episode <= 10:
        return ALL_TASKS[0]
    elif episode <= 20:
        return ALL_TASKS[(episode - 1) % 2]
    else:
        return ALL_TASKS[(episode - 1) % len(ALL_TASKS)]

def run_training():
    model, tokenizer = load_model_and_tokenizer()
    device    = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01,
    )
    env             = FleetWatchEnvWrapper()
    episode_rewards = []
    baseline        = 0.3

    print(f"\n{'='*62}")
    print(f"  FleetWatch REINFORCE Training - {NUM_EPISODES} episodes")
    print(f"{'='*62}\n")

    for episode in range(1, NUM_EPISODES + 1):
        task = pick_task(episode)
        print(f"\n-- Episode {episode:>3}/{NUM_EPISODES} | task: {task['task_id']} --")

        obs    = env.reset(task)
        prompt = build_prompt(obs, tokenizer)

        model.eval()
        response_text, full_ids, resp_mask = generate_response(model, tokenizer, prompt, device)
        print(f"  [LLM] {response_text[:160].strip()}")

        ref_lp = snapshot_ref_logprobs(model, full_ids)
        reward, _, info = env.step(response_text, task)

        advantage = (reward - baseline) / (abs(baseline) + 1e-6)
        baseline  = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward

        model.train()
        optimizer.zero_grad()
        loss, dbg = compute_reinforce_loss(model, ref_lp, full_ids, resp_mask, advantage, KL_BETA)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0,
        )
        optimizer.step()

        print(f"  [TRAIN] Loss:{loss.item():.4f} PG:{dbg['pg_loss']:.4f} KL:{dbg['kl_loss']:.4f} GradNorm:{grad_norm:.4f} Adv:{advantage:.4f} Base:{baseline:.4f}")
        episode_rewards.append(reward)

        if episode % 10 == 0:
            recent = episode_rewards[-10:]
            print(f"\n  *** Ep {episode-9}-{episode} | Avg:{np.mean(recent):.4f} Max:{max(recent):.4f} Min:{min(recent):.4f} ***\n")

        del full_ids, resp_mask, ref_lp
        torch.cuda.empty_cache()

    print(f"\n{'='*62}")
    print(f"  Done. Final avg: {np.mean(episode_rewards):.4f} | Best: {max(episode_rewards):.4f}")
    print(f"{'='*62}\n")
    return episode_rewards


def plot_training_curve(rewards, output_path=OUTPUT_PLOT):
    eps       = np.arange(1, len(rewards) + 1)
    arr       = np.array(rewards)
    roll_mean = np.array([np.mean(arr[max(0,i-ROLLING_WINDOW+1):i+1]) for i in range(len(arr))])
    roll_std  = np.array([np.std( arr[max(0,i-ROLLING_WINDOW+1):i+1]) for i in range(len(arr))])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.grid(color="#30363d", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    task_regions = [
        (1,10,"#1a3a5c","T1: Obvious"),(11,20,"#1a4a2a","T1+T2"),
        (21,25,"#3a2a1a","T1"),(26,30,"#2a1a3a","T2"),(31,35,"#3a1a2a","T3"),
        (36,40,"#1a2a3a","T4"),(41,45,"#2a3a1a","T5"),(46,50,"#3a1a1a","All"),
    ]
    for lo, hi, col, lbl in task_regions:
        if hi <= len(rewards):
            ax.axvspan(lo-0.5, hi+0.5, alpha=0.3, color=col)
            ax.text((lo+hi)/2, 1.01, lbl, ha="center", va="bottom",
                    color="#8b949e", fontsize=7, transform=ax.get_xaxis_transform())

    for ep, lbl in [(11,"Curriculum\nEscalates"),(21,"All 5 Tasks\nRotating")]:
        if ep <= len(rewards):
            ax.axvline(ep-0.5, color="#f0e68c", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(ep, 0.05, lbl, color="#f0e68c", fontsize=8, ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1117", alpha=0.7))

    ax.scatter(eps, arr, color="#58a6ff", alpha=0.5, s=32, zorder=3, label="Episode Reward")
    ax.plot(eps, roll_mean, color="#f78166", linewidth=2.8, zorder=4, label=f"Rolling Avg (w={ROLLING_WINDOW})")
    ax.fill_between(eps, np.clip(roll_mean-roll_std,0.001,0.999), np.clip(roll_mean+roll_std,0.001,0.999),
                    color="#f78166", alpha=0.15, zorder=2, label="+-1 Std Dev")
    ax.axhline(0.5, color="#8b949e", linestyle=":", linewidth=1.2, alpha=0.8, label="Baseline (0.5)")
    ax.axhline(0.8, color="#3fb950", linestyle="--", linewidth=1.4, alpha=0.9, label="Target (0.8)")

    ax.set_xlim(0.5, len(rewards)+0.5)
    ax.set_ylim(0.0, 1.08)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel("Episode", color="#c9d1d9", fontsize=13, labelpad=10)
    ax.set_ylabel("Reward Score (0.001 - 0.999)", color="#c9d1d9", fontsize=13, labelpad=10)
    ax.set_title("FleetWatch AI Auditor: Training Progress", color="#e6edf3", fontsize=16, fontweight="bold", pad=22)
    ax.tick_params(colors="#8b949e", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(loc="lower right", framealpha=0.35, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    final_avg = roll_mean[-1]
    ax.annotate(f"Final avg: {final_avg:.3f}", xy=(len(rewards), final_avg),
                xytext=(-70, 22), textcoords="offset points", color="#f78166",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.5))

    stats = f"Episodes: {len(rewards)}\nFinal avg: {np.mean(arr):.3f}\nBest: {max(arr):.3f}\nLast 10: {np.mean(arr[-10:]):.3f}"
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, color="#c9d1d9", fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#30363d", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] Saved -> {output_path}")


if __name__ == "__main__":
    rewards = run_training()
    plot_training_curve(rewards)
    print("\nDone. Download training_curve.png for your submission.")
