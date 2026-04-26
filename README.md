---
title: Fleet-Watch
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

<p align="center">
  <img src="./fleetwatch.png" alt="FleetWatch Banner" width="100%"/>
</p>

<h1 align="center">👁️ FleetWatch — AI Fleet Fraud Detection</h1>

<p align="center">
  <strong>Meta PyTorch OpenEnv Hackathon × Scaler 2026</strong><br/>
  <em>Train LLMs to detect coordinated fraud across multi-agent fleet logs</em>
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

## What is FleetWatch?

FleetWatch is a reinforcement learning environment where LLM agents learn to audit multi-agent fleet operation logs and detect sophisticated fraud patterns — GPS tampering, timesheet falsification, collision cover-ups, cascade negligence, and coordinated financial collusion.

**The Challenge**: Fraudsters actively hide evidence and coordinate deception across multiple agents. FleetWatch trains models to reason through noise, connect subtle clues across logs, and identify the culprits with evidence-based reasoning.

---

## Training Results

Our model learns to detect fraud across all 5 tasks through reinforcement learning. Below is the complete before/after comparison showing training curves, per-task improvements, and reward distributions.

<p align="center">
  <img src="./before_after_analysis.png" alt="Before vs After Training Analysis" width="100%"/>
</p>

| Task | Scenario | Before | After | Improvement |
|------|----------|--------|-------|-------------|
| Task 1 | GPS route deviation | 0.25 | 0.63 | +152% |
| Task 2 | Timesheet fraud pattern | 0.25 | 0.69 | +176% |
| Task 3 | Adversarial cover-up | 0.25 | 0.78 | +212% |
| Task 4 | 3-agent cascade failure | 0.25 | 0.78 | +212% |
| Task 5 | Multi-agent fuel collusion | 0.25 | 0.72 | +188% |

---

## The 5 Tasks

Progressive difficulty — each task requires deeper reasoning than the last.

| # | Task | Agents | Challenge |
|---|------|--------|-----------|
| 1 | GPS Tampering | 1 | Route deviation + disabled tracker |
| 2 | Timesheet Fraud | 1 | 3-week pattern + odometer falsification |
| 3 | Collision Cover-up | 2 | Log tampering + witness coercion |
| 4 | Cascade Negligence | 3 | Skipped inspection → brake failure chain |
| 5 | Fuel Collusion | 3 | Shell vendor + phantom mileage + financial fraud |

---

## Reward System

7 independent signals, normalized to (0.001 → 0.999). Designed to reward genuine reasoning and penalize gaming.

| Signal | Score | Notes |
|--------|-------|-------|
| Valid JSON | +0.4 | Base score for proper formatting |
| Correct anomaly detection | +1.5 | Core signal — did you catch the fraud? |
| Agent identification | +0.8 | Partial credit for multi-agent scenarios |
| Severity classification | +0.4 | Graduated partial credit (low/medium/high/critical) |
| Keyword coverage | +0.8 | Evidence-based language from logs |
| Contextual reasoning | +0.4 | Causal language and logical connections |
| Evidence integration | +0.3 | Specific log references and timestamps |
| Task complexity bonus | +0.2 | Extra credit for tasks 3/4/5 |
| Anti-cheat penalty | −0.2 | Flagging fraud without identifying agents |

**Why it works**: Always saying "fraud" with wrong agents → low score. Always saying "no fraud" → missed anomaly penalty → near zero. Only genuine reasoning with evidence scores high.

---

## Architecture

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

## Quick Test

Test the live API with a sample fraud detection:

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

**Expected**: `score > 0.85` with full 7-signal breakdown showing how each component contributed to the final reward.

---

## Training in Colab

Train your own FleetWatch model on a free T4 GPU in ~30 minutes. The script runs both baseline and enhanced training phases, then generates a comprehensive before/after comparison plot.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing)

**Steps**:
```python
# Cell 1 — Install dependencies (run once, then Runtime > Restart)
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes

# Cell 2 — Upload FleetWatch_Colab_Train.py, then run
exec(open("FleetWatch_Colab_Train.py").read())
```

The script includes memory optimizations for T4 GPUs and trains across all 5 tasks with curriculum learning.

---

## 📁 Project Structure

```
fleetwatch/
│
├── 🎯 app/                           # Core Application
│   ├── env.py                        # RL Environment + Adaptive Curriculum
│   ├── models.py                     # Pydantic Data Models & Schemas
│   ├── main.py                       # Application Entry Point
│   │
│   ├── 📊 graders/                   # Reward System
│   │   ├── __init__.py
│   │   └── master_grader.py          # 7-Signal Reward Function
│   │
│   └── 📝 tasks/                     # Task Definitions
│       ├── __init__.py
│       ├── task1_obvious.py          # Task 1: GPS Tampering
│       ├── task2_pattern.py          # Task 2: Timesheet Fraud
│       ├── task3_adversarial.py      # Task 3: Collision Cover-up
│       ├── task4_cascade.py          # Task 4: Cascade Negligence
│       └── task5_collusion.py        # Task 5: Fuel Collusion
│
├── 🚀 server/                        # API Server
│   ├── __init__.py
│   └── app.py                        # FastAPI REST Endpoints
│
├── 🧠 Training Scripts
│   ├── FleetWatch_Colab_Train.py     # Complete Colab Training Pipeline
│   ├── train_ppo.py                  # PPO Training (Baseline)
│   ├── train_ppo_enhanced.py         # Enhanced PPO Training
│   └── generate_plots.py             # Training Visualization
│
├── 🐳 Deployment
│   ├── Dockerfile                    # HuggingFace Spaces Container
│   ├── requirements.txt              # Python Dependencies
│   └── openenv.yaml                  # OpenEnv Configuration
│
├── 📊 Results & Analysis
│   ├── training_results.json         # Baseline Training Metrics
│   ├── enhanced_training_results.json # Enhanced Training Metrics
│   ├── before_after_analysis.png     # Training Comparison Plot
│   └── fleetwatch.png                # Project Banner
│
└── 📚 Documentation
    ├── README.md                     # Main Documentation
    ├── HACKATHON_SUBMISSION.md       # Submission Details
    ├── IMPROVEMENTS.md               # Enhancement Log
    └── QUICK_START_ENHANCED.md       # Quick Start Guide
```

### 🔑 Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Environment** | RL task orchestration, curriculum learning | Custom Gym-style API |
| **Grader** | 7-signal reward calculation, anti-cheat | Rule-based + heuristics |
| **Tasks** | 5 progressive fraud scenarios | Synthetic log generation |
| **Server** | REST API for training/inference | FastAPI + Uvicorn |
| **Training** | REINFORCE with LoRA fine-tuning | Unsloth + PyTorch |
| **Deployment** | Dockerized API on HF Spaces | Docker + HF Spaces |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode, get task + logs |
| `POST` | `/step` | Submit action, get reward breakdown |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `POST` | `/test/{1-5}` | Test directly against any task |

---

## Stack

- **Model**: Llama-3-8B-Instruct (4-bit quantized via Unsloth)
- **Training**: REINFORCE + LoRA (r=8) + Curriculum Learning
- **Framework**: FastAPI + Docker + HuggingFace Spaces
- **Optimization**: Memory-efficient training for T4 GPUs

---

## Access Points

| Resource | Link |
|----------|------|
| **Live Demo API** | [https://shiva0999-fleet-watch.hf.space](https://shiva0999-fleet-watch.hf.space) |
| **HuggingFace Space** | [https://huggingface.co/spaces/shiva0999/Fleet-Watch](https://huggingface.co/spaces/shiva0999/Fleet-Watch) |
| **Training Notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing) |
| **Source Code** | [https://github.com/shivakewat1/FleetWatch](https://github.com/shivakewat1/FleetWatch) |

---

<p align="center">
  <strong>FleetWatch</strong> · Meta PyTorch OpenEnv Hackathon × Scaler 2026<br/>
  <em>"Who audits the auditors? FleetWatch does."</em>
</p>
