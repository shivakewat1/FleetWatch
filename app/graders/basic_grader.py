from app.models import Action, Reward


def calculate_reward(action: Action) -> Reward:
    # Placeholder — complex scoring logic will replace this later
    return Reward(score=0.5, feedback="Basic grader working")
