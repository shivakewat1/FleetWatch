from app.models import Observation
from app.graders.master_grader import calculate_master_reward
from app.tasks.task1_obvious import TASK as TASK1
from app.tasks.task2_pattern import TASK as TASK2
from app.tasks.task3_adversarial import TASK as TASK3
from app.tasks.task4_cascade import TASK as TASK4
from app.tasks.task5_collusion import TASK as TASK5

TASKS: dict[str, dict] = {
    "task1-obvious":     TASK1,
    "task2-pattern":     TASK2,
    "task3-adversarial": TASK3,
    "task4-cascade":     TASK4,
    "task5-collusion":   TASK5,
}

# Adaptive curriculum: episode range → task id
CURRICULUM: list[tuple[int, int, str]] = [
    (1,  20, "task1-obvious"),
    (21, 40, "task2-pattern"),
    (41, 60, "task3-adversarial"),
    (61, 80, "task4-cascade"),
    (81, 99999, "task5-collusion"),
]


def _task_for_episode(episode: int) -> str:
    for start, end, task_id in CURRICULUM:
        if start <= episode <= end:
            return task_id
    return "task5-collusion"


class FleetWatchEnv:

    def __init__(self) -> None:
        self.episode_count: int = 0
        self._task_id: str = "task1-obvious"
        self._step_count: int = 0
        self._last_reward: dict | None = None

    # ------------------------------------------------------------------ #
    # reset                                                                #
    # ------------------------------------------------------------------ #
    def reset(self) -> dict:
        self.episode_count += 1
        self._task_id = _task_for_episode(self.episode_count)
        self._step_count = 0
        self._last_reward = None

        task = TASKS[self._task_id]
        obs = Observation(task_id=self._task_id, step_count=self._step_count)

        return {
            "observation": obs.model_dump(),
            "episode": self.episode_count,
            "task_description": task.get("task_description", ""),
            "input_logs": task.get("input_logs", []),
        }

    # ------------------------------------------------------------------ #
    # step                                                                 #
    # ------------------------------------------------------------------ #
    def step(self, agent_action: dict) -> dict:
        self._step_count += 1
        task = TASKS[self._task_id]
        ground_truth = task.get("ground_truth", {})

        try:
            result = calculate_master_reward(agent_action, ground_truth)
        except Exception as exc:
            result = {
                "score": 0.001,
                "breakdown": {"error": str(exc)},
            }

        self._last_reward = result

        return {
            "reward": result,
            "step_count": self._step_count,
            "episode": self.episode_count,
            "task_id": self._task_id,
            "done": True,
        }

    # ------------------------------------------------------------------ #
    # state                                                                #
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "episode": self.episode_count,
            "step_count": self._step_count,
            "last_reward": self._last_reward,
            "curriculum_stage": _task_for_episode(self.episode_count),
        }
