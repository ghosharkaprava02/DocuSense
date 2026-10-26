"""Read-only environment check. Does not download models."""
import importlib.util
import json
from pathlib import Path
import platform
import sys

import requests

if __name__ == "__main__":
    report = {"python": sys.version, "platform": platform.platform(),
              "interpreter": sys.executable,
              "packages": {x: bool(importlib.util.find_spec(x)) for x in
                           ["streamlit", "pymupdf", "sentence_transformers", "sklearn", "pytest"]}}
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        response.raise_for_status()
        report["ollama"] = response.json()
    except requests.RequestException as exc:
        report["ollama"] = f"Unavailable: {type(exc).__name__}"
    print(json.dumps(report, indent=2))
