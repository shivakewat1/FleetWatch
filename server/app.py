import uvicorn

from app.main import app  # noqa: F401 — imported so uvicorn can resolve it

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
