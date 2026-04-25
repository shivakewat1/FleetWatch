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
    try:
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
