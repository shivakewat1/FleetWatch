from pydantic import BaseModel
from typing import Optional


class Action(BaseModel):
    anomaly_detected: bool
    agent_id: str
    severity: str
    summary: str


class Reward(BaseModel):
    score: float
    breakdown: dict
    feedback: str
    raw_score: Optional[float] = None
