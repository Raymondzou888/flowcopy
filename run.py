import sys
import os

# Ensure the backend directory is on the path
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
