from app.graders.master_grader import calculate_master_reward
from app.tasks.task1_obvious import TASK as t1
from app.tasks.task2_pattern import TASK as t2
from app.tasks.task3_adversarial import TASK as t3
from app.tasks.task4_cascade import TASK as t4
from app.tasks.task5_collusion import TASK as t5

# Exported for use by server/main
TASKS = {
    t1["task_id"]: t1,
    t2["task_id"]: t2,
    t3["task_id"]: t3,
    t4["task_id"]: t4,
    t5["task_id"]: t5,
}


class FleetWatchEnv:
    """
    OpenEnv environment for FleetWatch with:
      - Multi-Dimensional Reward System (via calculate_master_reward)
      - Adaptive Curriculum (episodes 1-20 → t1, 21-40 → t2, 41-60 → t3,
                             61-80 → t4, 81+ → t5)
    """

    def __init__(self) -> None:
        self.episode_count: int = 0
        self._current_task: dict = {}

    # ------------------------------------------------------------------
    # Adaptive Curriculum
    # ------------------------------------------------------------------
    def get_task(self) -> dict:
        # Use modulo so curriculum cycles: 1-20→t1, 21-40→t2 ... 81-100→t1 again
        ep = ((self.episode_count - 1) % 100) + 1
        if ep <= 20:
            return t1
        elif ep <= 40:
            return t2
        elif ep <= 60:
            return t3
        elif ep <= 80:
            return t4
        else:
            return t5

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        self.episode_count += 1
        self._current_task = self.get_task()

        return {
            "observation":      {"task_id": self._current_task["task_id"], "step_count": 0},
            "episode":          self.episode_count,
            "task_id":          self._current_task["task_id"],
            "task_description": self._current_task["task_description"],
            "input_logs":       self._current_task["input_logs"],
            "step_count":       0,
        }

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: dict) -> dict:
        ground_truth = self._current_task.get("ground_truth", {})
        reward_dict  = calculate_master_reward(action, ground_truth)

        return {
            "reward":     reward_dict,
            "done":       True,
            "step_count": 1,
            "episode":    self.episode_count,
            "task_id":    self._current_task.get("task_id", "unknown"),
        }

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "episode_count":      self.episode_count,
            "current_task_id":    self._current_task.get("task_id", "none"),
            "curriculum_stage":   self.get_task()["task_id"] if self.episode_count > 0 else "not started",
        }
