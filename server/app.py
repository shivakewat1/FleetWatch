import sys
import os

# Ensure the fleetwatch root is on the path regardless of where this is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from app.main import app  # noqa: F401


def main():
    """Entry point for the FleetWatch server (used by [project.scripts])."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
