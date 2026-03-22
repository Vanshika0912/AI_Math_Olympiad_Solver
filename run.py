"""
Convenience entry point for the AI Math Olympiad Solver.
"""

import uvicorn
from app.main import app
from src.utils.common import load_config

if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "app.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
        workers=api_cfg.get("workers", 1),
    )
