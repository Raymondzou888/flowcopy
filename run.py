"""FlowCopy - Production startup script for Railway deployment."""

import sys
import os
import logging

# Ensure the backend directory is on the path
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flowcopy.run")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"

    logger.info(f"Starting FlowCopy on {host}:{port}")
    logger.info(f"OPENROUTER_API_KEY: {'SET' if os.environ.get('OPENROUTER_API_KEY') else 'NOT SET - will fail!'}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
