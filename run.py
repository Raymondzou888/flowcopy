import sys
import os

# Ensure the backend directory is on the path
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
