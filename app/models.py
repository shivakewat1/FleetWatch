from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    task_id: str
    step_count: int


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
