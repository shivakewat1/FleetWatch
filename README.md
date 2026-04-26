---
title: Fleet-Watch
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
<div align="center">

# FleetWatch AI

### AI-Powered Multi-Agent Fleet Fraud Detection System

<br/>

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/shiva0999/Fleet-Watch)
[![Live API](https://img.shields.io/badge/API-Live-brightgreen?style=for-the-badge&logo=fastapi)](https://shiva0999-fleet-watch.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/shivakewat1/FleetWatch)
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Developed for the Meta PyTorch OpenEnv Hackathon × Scaler 2026**

</div>

## Overview

FleetWatch is a reinforcement learning environment built to train Large Language Models (LLMs) to detect sophisticated, coordinated fraud patterns in multi-agent fleet operations. It employs **curriculum learning** and a **multi-signal reward mechanism** to identify deception across complex fleet operation logs.

### Problem Statement

Fleet operations are highly vulnerable to coordinated fraud schemes in which multiple agents collaborate to conceal evidence. Traditional rule-based detection systems consistently fail against these sophisticated patterns. FleetWatch addresses this gap by training LLMs to:

- Analyze noisy, multi-source operational data
- Identify subtle correlations across agent logs
- Detect coordinated deception across multiple actors
- Produce evidence-based fraud assessments with causal reasoning

### Solution Approach

FleetWatch implements a full reinforcement learning pipeline with:

- **Curriculum Learning** — Progressive task difficulty, from single-agent anomalies to multi-agent collusion
- **Multi-Signal Rewards** — A 7-component reward system that prevents gaming and rewards genuine reasoning
- **Anti-Cheat Mechanisms** — Penalties for pattern exploitation without supporting evidence
- **Evidence-Based Evaluation** — Rewards tied to specific log references and causal chains

---

## Features

| Feature | Description |
|---|---|
| **Progressive Task Complexity** | 5 fraud detection scenarios from single-agent GPS tampering to multi-agent financial collusion |
| **Robust Reward System** | 7-signal evaluation mechanism with anti-gaming penalties |
| **Curriculum Learning** | Adaptive task selection based on real-time agent performance |
| **Production-Ready API** | RESTful endpoints for real-time fraud detection with full documentation |
| **Efficient Training** | 4-bit quantized Llama-3-8B with LoRA fine-tuning (~30 min on a T4 GPU) |
| **Comprehensive Evaluation** | Evidence-based scoring with per-signal reward breakdown |

---

## Performance

Training using **REINFORCE policy gradient optimization** delivers significant improvement across all five detection tasks. The full training analysis — baseline vs. enhanced — is shown below.

![FleetWatch Training Analysis — Baseline vs Enhanced](./fleetwatch.png)

---

### Training Summary

| Metric | Baseline (50 eps) | Enhanced (75 eps) |
|---|:---:|:---:|
| Mean Reward | `0.047` | `0.521` |
| Best Reward | `0.733` | `0.800` |
| Final 20% Avg | `0.001` | `0.377` |
| Std Dev *(lower = better)* | `0.238` | `0.156` |
| % Episodes > 0.6 | `0.040` | `0.440` |

> The Enhanced model achieves **11× higher mean reward**, **10× more episodes exceeding 0.6**, and a tighter reward distribution — demonstrating stable, consistent fraud detection rather than occasional lucky guesses.

---

### Per-Task Reward — Before vs After Training

| Task | Label | Avg Reward (Before) | Avg Reward (After) | Delta |
|:----:|---|:---:|:---:|:---:|
| T1 | Obvious | `0.065` | `0.528` | **+0.463** |
| T2 | Pattern | `0.074` | `0.550` | **+0.476** |
| T3 | Adversarial | `0.045` | `0.587` | **+0.543** |
| T4 | Cascade | `0.009` | `0.563` | **+0.553** |
| T5 | Collusion | `0.043` | `0.377` | **+0.334** |

> T4 (Cascade Negligence) shows the largest absolute gain `(+0.553)` from the lowest baseline `(0.009)`, demonstrating the model's ability to master complex causal chain analysis. T5 (Financial Collusion) has the lowest post-training score `(0.377)`, reflecting the inherent difficulty of multi-agent phantom transaction detection.

---

### Reward Distribution

| | Baseline | Enhanced |
|---|:---:|:---:|
| Mean (B mean) | `0.047` | — |
| Mean (E mean) | — | `0.521` |
| Mean Shift (Δ) | — | **+0.474** |

The Baseline reward distribution is heavily concentrated near `0` (peak frequency ~47 episodes at reward < 0.1), while the Enhanced model's distribution spreads broadly toward higher reward values — mean shift of **+0.474**.

---

### Cumulative Best Reward

| Model | Best Reward Achieved |
|---|:---:|
| Baseline | `0.733` |
| Enhanced | `0.800` |

The Enhanced model reaches its peak best reward of **0.800** early in training and sustains it, while the Baseline plateaus at **0.733** — a gain of **+0.067** in peak performance.

---

### Key Metrics Delta (Baseline → Enhanced)

| Metric | Δ Change | Direction |
|---|:---:|:---:|
| Mean Reward | `+0.474` | ✅ Higher is better |
| Best Reward | `+0.067` | ✅ Higher is better |
| Final 20% Avg | `+0.376` | ✅ Higher is better |
| Std Dev | `−0.082` | ✅ Lower is better |
| % eps > 0.6 | `+0.400` | ✅ Higher is better |

> **Note:** Scores are normalized to the `[0.001, 0.999]` range using the 7-component reward function. Baseline represents random-initialization performance (50 episodes). Enhanced represents the trained model (75 episodes, REINFORCE + LoRA).

---

## Detection Tasks

FleetWatch implements five progressively complex fraud detection scenarios designed to stress-test multi-agent reasoning capabilities.

| Task | Fraud Type | Agents | Complexity | Key Challenges |
|:----:|---|:---:|:---:|---|
| T1 | GPS Tampering | 1 | Low | Route deviation detection, tracker manipulation |
| T2 | Timesheet Fraud | 1 | Medium | Pattern recognition, odometer correlation |
| T3 | Collision Cover-up | 2 | High | Log tampering, witness coercion, cross-agent analysis |
| T4 | Cascade Negligence | 3 | High | Causal chain reconstruction, inspection records |
| T5 | Financial Collusion | 3 | Critical | Shell vendor identification, phantom transactions |

**Design Philosophy:** Each task requires deeper reasoning than the last, progressing from single-agent pattern detection to full multi-agent coordination analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            FleetWatch Environment (HuggingFace Spaces)       │
│                                                              │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ 5 Tasks  │───▶│ Adaptive         │───▶│ Master Grader │  │
│  │  T1→T5   │    │ Curriculum       │    │  (7-signal)   │  │
│  └──────────┘    └──────────────────┘    └───────────────┘  │
│                                                              │
│       POST /reset    POST /step    GET /state    GET /health │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Training Pipeline  (Google Colab — T4 GPU)          │
│                                                              │
│  • Llama-3-8B-Instruct (4-bit quantized)                    │
│  • LoRA Fine-tuning  (r=8, alpha=16)                        │
│  • REINFORCE + Advantage Baseline + Entropy Bonus           │
│  • Curriculum Learning + Adaptive LR + Gradient Clipping    │
└─────────────────────────────────────────────────────────────┘
```

### System Components

| Component | Description |
|---|---|
| **Environment** | OpenEnv-compliant RL environment with 5 fraud detection tasks |
| **Curriculum** | Adaptive task selection based on rolling agent performance |
| **Grader** | Multi-signal reward function with anti-cheat mechanisms |
| **Model** | 4-bit quantized Llama-3-8B-Instruct with LoRA adapters |
| **Trainer** | REINFORCE policy gradient with variance reduction and entropy bonus |

---

## Reward System

Seven independent reward signals are normalized to `[0.001, 0.999]`. The system is designed to reward genuine, evidence-backed reasoning and penalize shortcut strategies.

| Signal | Weight | Description |
|---|:---:|---|
| Valid JSON Format | `+0.4` | Proper structured response formatting |
| Anomaly Detection | `+1.5` | Accurate fraud identification |
| Agent Identification | `+0.8` | Correct perpetrator(s) identified |
| Severity Classification | `+0.4` | Appropriate severity level assigned |
| Keyword Coverage | `+0.8` | Evidence-based language used |
| Contextual Reasoning | `+0.4` | Logical connections drawn |
| Evidence Integration | `+0.3` | Specific log references cited |
| Task Complexity Bonus | `+0.2` | Extra credit for Tasks 3–5 |
| Anti-Cheat Penalty | `−0.2` | Applied when gaming is detected |

**Why it works:**
- Always claiming fraud → low agent identification score
- Always claiming no fraud → missed anomaly penalty
- Only genuine, evidence-grounded reasoning consistently scores high

---

## Getting Started

### API Usage

Test the deployed model directly against a fraud detection task:

```bash
curl -X POST https://shiva0999-fleet-watch.hf.space/test/3 \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_detected": true,
    "agent_id": "DRIVER-22, DRIVER-08",
    "severity": "critical",
    "summary": "Coordinated collision cover-up with log tampering and witness coercion."
  }'
```

**Response:** Returns a normalized score `(0–1)` with a full 7-signal reward breakdown.

---

### Training

Train a custom model using Google Colab with a free T4 GPU (~30 minutes):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing)

**Step 1 — Install dependencies** *(restart runtime after)*

```python
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes
```

**Step 2 — Upload and run the training script**

```python
# Upload FleetWatch_Colab_Train.py to Colab, then execute:
exec(open("FleetWatch_Colab_Train.py").read())
```

**Training Configuration**

| Parameter | Value |
|---|---|
| Base Model | Llama-3-8B-Instruct (4-bit quantized) |
| Fine-tuning Method | LoRA (rank=8, alpha=16) |
| Training Algorithm | REINFORCE with advantage baseline |
| Optimizer | AdamW with gradient clipping |
| Task Sampling | Adaptive curriculum based on performance |

---

## API Reference

**Base URL:** `https://shiva0999-fleet-watch.hf.space`

| Method | Endpoint | Description |
|:---:|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit a fraud detection action |
| `GET` | `/state` | Retrieve the current episode state |
| `GET` | `/health` | Health check |
| `POST` | `/test/{task_id}` | Test a specific task (1–5) |

**Python Example**

```python
import requests

BASE_URL = "https://shiva0999-fleet-watch.hf.space"

# Start a new episode
task_data = requests.post(f"{BASE_URL}/reset").json()

# Submit a detection action
action = {
    "anomaly_detected": True,
    "agent_id": "DRIVER-22",
    "severity": "high",
    "summary": "GPS tampering detected with route deviation exceeding 40km."
}

response = requests.post(f"{BASE_URL}/step", json=action).json()
print(f"Reward: {response['reward']}")
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Base Model** | Llama-3-8B-Instruct (4-bit quantized) |
| **Training** | Unsloth · REINFORCE · LoRA (r=8) |
| **Framework** | FastAPI · Docker |
| **Deployment** | HuggingFace Spaces |
| **Libraries** | `transformers` · `peft` · `bitsandbytes` · `accelerate` · `trl` |

---

## Resources

### Live Deployments
- **Live API:** https://shiva0999-fleet-watch.hf.space
- **HuggingFace Space:** https://huggingface.co/spaces/shiva0999/Fleet-Watch

### Repository & Code
- **GitHub Repository:** https://github.com/shivakewat1/FleetWatch
- **Colab Training Notebook:** https://colab.research.google.com/drive/1ZYWRl3NI86Cz8VrrxGm3vOGrKxpqHKp1?usp=sharing

### Documentation & Frameworks
- **OpenEnv Framework:** https://openenv.dev
- **Unsloth:** https://github.com/unslothai/unsloth
- **Meta Llama 3:** https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

---

<div align="center">

**Meta PyTorch OpenEnv Hackathon × Scaler 2026**

Made with ❤️ for better fleet safety and accountability.

</div>