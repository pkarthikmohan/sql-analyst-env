"""
server/app.py — Entry point for OpenEnv server discovery
This imports and exposes the main FastAPI app from main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()