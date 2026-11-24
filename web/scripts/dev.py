"""Start FastAPI and Vite together; Ctrl+C stops both."""
import os, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
backend=subprocess.Popen([sys.executable,"-m","uvicorn","backend.main:app","--app-dir",str(ROOT),"--host","127.0.0.1","--port","8000"],cwd=ROOT)
frontend=subprocess.Popen("npm run dev",cwd=ROOT/"frontend",shell=True)
try:
    while backend.poll() is None and frontend.poll() is None: time.sleep(.4)
except KeyboardInterrupt: pass
finally:
    for proc in (frontend,backend):
        if proc.poll() is None:
            proc.terminate()
    for proc in (frontend,backend):
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
