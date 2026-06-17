from __future__ import annotations

import json
import os

import httpx


def main() -> None:
    base_url = os.environ.get("NANOSERVE_BASE_URL", "http://127.0.0.1:8000")
    payload = {
        "model": os.environ.get("NANOSERVE_MODEL", "tinyllama-1.1b-chat"),
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 16,
        "stream": False,
    }
    response = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30.0)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

