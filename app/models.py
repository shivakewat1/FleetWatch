from pydantic import BaseModel


class Observation(BaseModel):
    task_id: str
    step_count: int


class Action(BaseModel):
    summary: str
    overall_risk: str


class Reward(BaseModel):
    score: float
    feedback: str
