from app.models import Action, Observation, Reward
from app.tasks.task1_basic import TASK
from app.graders.basic_grader import calculate_reward


class FleetWatchEnv:
    TASK_ID = "basic-test"

    def __init__(self) -> None:
        self._step_count: int = 0
        self._last_reward: Reward | None = None

    def reset(self) -> dict:
        self._step_count = 0
        self._last_reward = None
        obs = Observation(task_id=self.TASK_ID, step_count=self._step_count)
        return {
            "observation": obs.model_dump(),
            "input_logs": TASK["input_logs"],
        }

    def step(self, action: Action) -> dict:
        self._step_count += 1
        reward = calculate_reward(action)
        self._last_reward = reward
        return {
            "reward": reward.model_dump(),
            "step_count": self._step_count,
            "done": True,
        }

    def state(self) -> dict:
        return {
            "task_id": self.TASK_ID,
            "step_count": self._step_count,
            "last_reward": self._last_reward.model_dump() if self._last_reward else None,
        }
