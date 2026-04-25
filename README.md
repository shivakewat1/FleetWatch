---
title: FleetWatch
emoji: 🚛
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# FleetWatch

An OpenEnv-compliant agentic RL environment for fleet risk assessment.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Reset the environment, returns observation + input logs |
| POST | `/step` | Submit an action, returns reward and step info |
| GET | `/state` | Current environment state |
| GET | `/health` | Health check |

## Quick Start

```bash
# Reset
curl -X POST http://localhost:7860/reset

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"summary": "Fleet normal", "overall_risk": "low"}'

# State
curl http://localhost:7860/state

# Health
curl http://localhost:7860/health
```
