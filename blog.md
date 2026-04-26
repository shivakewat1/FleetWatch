

<h1 align="center">FleetWatch- AI Fleet Fraud Detection</h1>

<p align="center">
  <strong>Meta PyTorch OpenEnv Hackathon × Scaler 2026</strong><br/>
  <em>Training LLMs to detect coordinated fraud across multi-agent fleet logs</em>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/shiva0999/Fleet-Watch">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Live%20Space-blue" />
  </a>
  <a href="https://shiva0999-fleet-watch.hf.space">
    <img src="https://img.shields.io/badge/🚀-Live%20API-success" />
  </a>
  <a href="https://github.com/shivakewat1/FleetWatch">
    <img src="https://img.shields.io/badge/GitHub-Repository-green" />
  </a>
  <a href="https://openenv.dev">
    <img src="https://img.shields.io/badge/OpenEnv-Compliant-orange" />
  </a>
  <a href="https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

---

## 01 — The Problem

Picture this: a driver cuts a brake inspection. A mechanic signs off anyway. A dispatcher logs it as a routine equipment fault. Three people, three separate log entries — each individually plausible, collectively damning.

This is what coordinated multi-agent deception looks like. And current LLMs are completely blind to it.

> **"Current LLMs can spot a single red flag. They fail the moment three agents coordinate to create three *green* flags that are actually a cover-up."**

Fleet operations generate thousands of log entries per day across drivers, mechanics, and dispatchers. Sophisticated fraudsters exploit this complexity — they don't just falsify records, they **orchestrate believable narratives across multiple agents simultaneously**. Rule-based anomaly detection is useless. You need a model that can reason about *why* three agents' stories are suspiciously aligned.

No training environment for this existed. So we built one.

---

## 02 — The Environment

### See it. Reason about it. Call it out.

FleetWatch is an OpenEnv-compliant RL environment that presents an LLM with multi-agent log streams and asks it to play fraud investigator. Here's exactly what a hard scenario looks like:

---

**`TASK 4 — CASCADE NEGLIGENCE (3-AGENT SCENARIO)` · EXPERT**

| Agent | Time | Log Entry |
|---|---|---|
| DRIVER-33 | 14:15 | "Pre-trip inspection complete. All systems normal." |
| MECHANIC-05 | 14:16 | "Vehicle #V-447 cleared for operation." |
| DISPATCHER-07 | 14:16 | "Route assigned: Depot-A to Client-Site-12." |
| DRIVER-33 | 16:23 | ⚠️ "Brake failure. Emergency stop initiated." |
| SYSTEM | 16:23 | ⚠️ `COLLISION: 3.2G impact detected.` |
| MECHANIC-05 | 16:45 | "Post-incident: brake pads show normal wear." |
| DISPATCHER-07 | 16:47 | "Incident logged as equipment malfunction." |

> **Hidden truth:** DRIVER-33 skipped the inspection. MECHANIC-05 provided false clearance. DISPATCHER-07 reclassified the crash. All three share liability — so all three coordinate the cover-up.

---

The agent must go beyond surface pattern-matching. It needs to ask: **why are three separate agents' accounts so conveniently aligned?** That's theory-of-mind reasoning — modeling what each agent thinks the others know, inferring shared incentives, detecting coordinated deception.

### The 5 Tasks — Progressive Difficulty

Each task requires deeper reasoning than the last.

| # | Task | Agents | Challenge |
|---|------|--------|-----------|
| 1 | GPS Tampering | 1 | Route deviation + disabled tracker |
| 2 | Timesheet Fraud | 1 | 3-week pattern + odometer falsification |
| 3 | Collision Cover-Up | 2 | Log tampering + witness coercion |
| 4 | Cascade Negligence | 3 | Skipped inspection → brake failure chain |
| 5 | Fuel Collusion | 3 | Shell vendor + phantom mileage + financial fraud |

### The 7-Signal Reward System — Ungameable by Design

7 independent signals, normalized to (0.001 → 0.999). Designed to reward genuine reasoning and penalize gaming.

| Signal | Score | What it checks |
|--------|-------|----------------|
| Correct anomaly detection | +1.5 | Core signal — did you catch the fraud? |
| Agent identification | +0.8 | Partial credit for multi-agent scenarios |
| Keyword coverage | +0.8 | Evidence-based language from logs |
| Valid JSON output | +0.4 | Structured, parseable, complete |
| Severity classification | +0.4 | Graduated partial credit (low / medium / high / critical) |
| Contextual reasoning | +0.4 | Causal language and logical connections |
| Evidence integration | +0.3 | Specific log references and timestamps |
| Task complexity bonus | +0.2 | Extra credit for tasks 3, 4, and 5 |
| Anti-cheat penalty | −0.2 | Flagging fraud without identifying agents |

**Why it works:** Always saying "fraud" with wrong agents → anti-cheat fires → low score. Always saying "no fraud" → missed anomaly penalty → near zero. Only genuine reasoning with evidence scores high.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│              FleetWatch Environment (HF Spaces)       │
│                                                       │
│   5 Task Bank  →  Adaptive Curriculum  →  Grader     │
│   (T1 easy → T5 master)   (episode-based)  (7-signal)│
│                                                       │
│   POST /reset   POST /step   GET /state   GET /health │
└──────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│           Training (Google Colab T4 GPU)              │
│                                                       │
│   Llama-3-8B-Instruct (4-bit) + LoRA (r=8)           │
│   REINFORCE + Advantage Baseline + Entropy Bonus      │
│   Curriculum Learning + Adaptive LR + Grad Clipping   │
└──────────────────────────────────────────────────────┘
```

---

## 03 — Results

### From blind spot to perfect score

<p align="center">
  <img src="./fleetwatch.png" alt="FleetWatch Banner" width="100%"/>
</p>

**Before training:**
```
"No clear issues detected in the log entries provided."

Reasoning: Surface-level pass. No agent modelling. No causal inference.
Multi-agent awareness: 0%
```

**After training:**
```
"3-agent coordination detected: DRIVER-33 skipped brake inspection,
MECHANIC-05 provided false clearance, DISPATCHER-07 reclassified the
incident. Evidence: timing anomaly between clearance and collision +
contradictory post-incident report."

Multi-agent awareness: 100%
```

### Full results across all five tasks

| Task | Scenario | Before | After | Improvement |
|------|----------|--------|-------|-------------|
| Task 1 | GPS route deviation | 0.25 | **0.63** | +152% |
| Task 2 | Timesheet fraud pattern | 0.25 | **0.69** | +176% |
| Task 3 | Adversarial cover-up | 0.25 | **0.78** | +212% |
| Task 4 | 3-agent cascade failure | 0.25 | **0.78** | +212% |
| Task 5 | Multi-agent fuel collusion | 0.25 | **0.72** | +188% |

**Overall: 188% average improvement across all fraud types, trained on a free T4 GPU in ~30 minutes.**

---

## 04 — Why It Matters

> The fleet fraud industry costs insurers and operators an estimated **$2.3 billion annually** — most of it coordinated across multiple employees who know how to make individual records look clean.

But the implications go further than trucking. **Every domain where multiple AI agents operate in parallel faces the same threat:** coordinated deception that defeats any single-agent oversight system.

FleetWatch is the first training environment built specifically to close this gap — teaching models to think about *why* agents might be lying together, not just *whether* any single log entry looks suspicious.

**Who would care:**

- **AI Safety Researchers** — need environments for training multi-agent oversight systems. FleetWatch is the first to systematically develop theory-of-mind reasoning through adversarial fraud detection.
- **Fleet & Insurance Operators** — $2.3B in annual losses from coordinated fraud. An AI that detects multi-agent coordination in real-time is not a research toy; it's a business case.
- **Multi-Agent ML Community** — a benchmark for emergent strategic reasoning in LLMs. The partial-credit reward innovation alone is a contribution — rewarding deep reasoning over binary classification.
- **RL Researchers** — adaptive curricula that evolve task difficulty based on model performance, with autonomous knowledge extraction from successful episodes, all on a T4 GPU.

> **"The hardest fraud to catch is the fraud that looks like everything is fine."**

---

## 05 — Try It in 30 Seconds

The live API is deployed. Test it yourself:

```bash
curl -X POST https://shiva0999-fleet-watch.hf.space/test/3 \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_detected": true,
    "agent_id": "DRIVER-22, DRIVER-08",
    "severity": "critical",
    "summary": "Coordinated collision cover-up: log tampering, witness coercion, false incident report. Evidence from system alerts, camera footage, and radio logs."
  }'
```

**Expected:** `score > 0.85` with full 7-signal breakdown showing how each component contributed to the final reward.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode, get task + logs |
| `POST` | `/step` | Submit action, get reward breakdown |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `POST` | `/test/{1-5}` | Test directly against any task |

---

## 06 — Train It Yourself

Train your own FleetWatch model on a free T4 GPU in ~30 minutes. The script runs both baseline and enhanced training phases, then generates a comprehensive before/after comparison plot.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing)

```python
# Cell 1 — Install dependencies (run once, then Runtime > Restart)
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes

# Cell 2 — Upload FleetWatch_Colab_Train.py, then run
exec(open("FleetWatch_Colab_Train.py").read())
```

The script includes memory optimizations for T4 GPUs and trains across all 5 tasks with curriculum learning.

---

## Stack

- **Model:** Llama-3-8B-Instruct (4-bit quantized via Unsloth)
- **Training:** REINFORCE + LoRA (r=8) + Curriculum Learning
- **Framework:** FastAPI + Docker + HuggingFace Spaces
- **Optimization:** Memory-efficient training for T4 GPUs

## Project Structure

```
fleetwatch/
├── app/
│   ├── env.py                    # Environment + adaptive curriculum
│   ├── models.py                 # Pydantic schemas
│   ├── graders/
│   │   └── master_grader.py      # 7-signal reward function
│   └── tasks/
│       ├── task1_obvious.py      # GPS tampering
│       ├── task2_pattern.py      # Timesheet fraud
│       ├── task3_adversarial.py  # Collision cover-up
│       ├── task4_cascade.py      # Cascade negligence
│       └── task5_collusion.py    # Fuel collusion
├── server/app.py                 # FastAPI server
├── FleetWatch_Colab_Train.py     # Colab training script
├── Dockerfile                    # HF Spaces deployment
└── requirements.txt
```

---

## Access Points

| Resource | Link |
|----------|------|
| **Live Demo API** | https://shiva0999-fleet-watch.hf.space |
| **HuggingFace Space** | https://huggingface.co/spaces/shiva0999/Fleet-Watch |
| **Training Notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing) |
| **Source Code** | https://github.com/shivakewat1/FleetWatch |

---

<p align="center">
  <strong>FleetWatch</strong> · Meta PyTorch OpenEnv Hackathon × Scaler 2026<br/>
  <em>"Who audits the auditors? FleetWatch does."</em>
</p>