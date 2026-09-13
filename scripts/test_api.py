import sys
import time
import json
import threading
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_USER = "A1004703RC79J9"
SAMPLE_ASIN = "B0088CJT4U"       # their top recommendation
BASE = "http://localhost:8001"

def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def _pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

print("Starting uvicorn on port 8001…")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app",
     "--host", "0.0.0.0", "--port", "8001"],
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)

def _stream():
    for line in proc.stdout:
        print("[uvicorn]", line, end="")

threading.Thread(target=_stream, daemon=True).start()

print("Waiting for server to be ready…")
for attempt in range(60):
    time.sleep(3)
    result = _get(f"{BASE}/health")
    if "status" in result:
        print(f"Server ready after {(attempt+1)*3}s\n")
        break
else:
    print("ERROR: server did not start within 180s")
    proc.terminate()
    sys.exit(1)

print("=" * 60)
print(f"GET {BASE}/health")
print("=" * 60)
_pretty(_get(f"{BASE}/health"))

print()
print("=" * 60)
print(f"GET {BASE}/recommend/{SAMPLE_USER}?top_k=5")
print("=" * 60)
_pretty(_get(f"{BASE}/recommend/{SAMPLE_USER}?top_k=5"))

print()
print("=" * 60)
print(f"GET {BASE}/explain/{SAMPLE_USER}/{SAMPLE_ASIN}")
print("=" * 60)
_pretty(_get(f"{BASE}/explain/{SAMPLE_USER}/{SAMPLE_ASIN}"))

print()
print("=" * 60)
print(f"GET {BASE}/recommend/UNKNOWN_USER_XYZ?top_k=5  (cold-start)")
print("=" * 60)
_pretty(_get(f"{BASE}/recommend/UNKNOWN_USER_XYZ?top_k=5"))

proc.terminate()
print("\nDone. Server stopped.")
