"""Model caller: one interface for every model, with on-disk caching, retries,
and token logging.

Provider is OpenAI-compatible. This covers:
  - OpenAI models directly (set OPENAI_API_KEY).
  - Llama / Qwen / etc. via any OpenAI-compatible endpoint (Together, Fireworks,
    a local vLLM server, ...) by passing base_url and the matching API key.

So the whole fleet goes through one code path; only the model string and
base_url change.

Caching is keyed on (model, prompt, system, temperature, max_tokens), so a
rerun after a crash costs nothing for calls already made. Delete the cache dir
to force fresh calls.
"""

from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass

# Load a local .env if present. override=True so the file wins over any stale
# variable left in the shell (e.g. a placeholder from an earlier export).
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # pip install python-dotenv to use a .env file

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
TOKENS_LOG = Path(__file__).resolve().parent.parent / "data" / "tokens.jsonl"


@dataclass
class Response:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool


def _key(model, prompt, system, temperature, max_tokens) -> str:
    blob = json.dumps(
        [model, prompt, system, temperature, max_tokens], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _client(base_url: str | None, api_key_env: str):
    # Imported lazily so the no-API modules (generate/filler/conditions) import
    # without the openai package installed.
    from openai import OpenAI
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"environment variable {api_key_env} is not set")
    return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)


def call_model(model: str, prompt: str, temperature: float = 0.0,
               max_tokens: int = 512, system: str | None = None,
               base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
               retries: int = 5) -> Response:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / (_key(model, prompt, system, temperature, max_tokens) + ".json")
    if cache_path.exists():
        d = json.loads(cache_path.read_text())
        return Response(d["text"], d["prompt_tokens"], d["completion_tokens"], cached=True)

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    client = _client(base_url, api_key_env)

    last_err = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            text = r.choices[0].message.content or ""
            pt = r.usage.prompt_tokens if r.usage else 0
            ct = r.usage.completion_tokens if r.usage else 0
            cache_path.write_text(json.dumps(
                {"text": text, "prompt_tokens": pt, "completion_tokens": ct}))
            TOKENS_LOG.parent.mkdir(exist_ok=True)
            with TOKENS_LOG.open("a") as fh:
                fh.write(json.dumps({"model": model, "prompt_tokens": pt,
                                     "completion_tokens": ct}) + "\n")
            return Response(text, pt, ct, cached=False)
        except Exception as e:  # noqa: BLE001 - backoff on any transient error
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_model failed after {retries} retries: {last_err}")
