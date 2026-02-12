import os
import random
import time

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()

try:
    from groq import Groq
except Exception as exc:  # pragma: no cover - import guard
    Groq = None
    _groq_import_error = exc
else:
    _groq_import_error = None

_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-instant")
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_LLM_MAX_ATTEMPTS = max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "3")))
_LLM_BACKOFF_BASE_SECONDS = max(0.0, float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1.0")))
_LLM_BACKOFF_MAX_SECONDS = max(0.0, float(os.getenv("LLM_BACKOFF_MAX_SECONDS", "8.0")))
_LLM_BACKOFF_JITTER_SECONDS = max(0.0, float(os.getenv("LLM_BACKOFF_JITTER_SECONDS", "0.25")))
_LLM_MIN_INTERVAL_SECONDS = max(0.0, float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0.5")))
_last_call_monotonic = 0.0

def _get_client():
    if Groq is None:
        raise RuntimeError(
            "Groq client not installed. Run: pip install groq"
        ) from _groq_import_error
    if not _GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in your environment."
        )
    return Groq(api_key=_GROQ_API_KEY)

def _throttle() -> None:
    global _last_call_monotonic
    now = time.monotonic()
    wait_for = _LLM_MIN_INTERVAL_SECONDS - (now - _last_call_monotonic)
    if wait_for > 0:
        time.sleep(wait_for)
    _last_call_monotonic = time.monotonic()

def call_llm(system, user, temperature=0):
    client = _get_client()
    last_error = None
    for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
        try:
            _throttle()
            response = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt >= _LLM_MAX_ATTEMPTS:
                break
            delay = min(
                _LLM_BACKOFF_MAX_SECONDS,
                _LLM_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            jitter = random.uniform(0.0, _LLM_BACKOFF_JITTER_SECONDS)
            time.sleep(delay + jitter)
    raise last_error
