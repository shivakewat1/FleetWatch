from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.env import FleetWatchEnv, TASKS

app = FastAPI(title="FleetWatch", version="1.0.0")
env = FleetWatchEnv()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.post("/reset")
async def reset():
    try:
        return env.reset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step")
async def step(agent_action: dict):
    """
    Step endpoint - evaluates agent action against current task.
    If no task is active, automatically resets to task1.
    """
    try:
        # If no current task, auto-reset to task1
        if not env._current_task:
            env.reset()
        
        return env.step(agent_action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
async def state():
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "env": "fleetwatch", "tasks": list(TASKS.keys())}


@app.post("/test")
async def test_step(agent_action: dict):
    """
    Test endpoint - directly evaluates action against task1 without needing reset.
    Useful for quick testing in the Space UI.
    """
    try:
        from app.tasks.task1_obvious import TASK as task1
        from app.graders.master_grader import calculate_master_reward
        
        ground_truth = task1.get("ground_truth", {})
        reward_dict = calculate_master_reward(agent_action, ground_truth)
        
        return {
            "task_id": task1["task_id"],
            "task_description": task1["task_description"],
            "input_logs": task1["input_logs"],
            "agent_action": agent_action,
            "reward": reward_dict,
            "ground_truth": ground_truth,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
