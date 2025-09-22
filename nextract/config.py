from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = os.getenv("NEXTRACT_MODEL", "openai:gpt-4o")  # vision-capable by default
DEFAULT_MAX_CONCURRENCY = int(os.getenv("NEXTRACT_MAX_CONCURRENCY", "4"))
DEFAULT_MAX_RUN_RETRIES = int(os.getenv("NEXTRACT_MAX_RUN_RETRIES", "5"))
DEFAULT_PER_CALL_TIMEOUT_SECS = float(os.getenv("NEXTRACT_PER_CALL_TIMEOUT_SECS", "120"))

# JSON (string) mapping of model->pricing, e.g.
# {"openai:gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015}}
NEXTRACT_PRICING_JSON = os.getenv("NEXTRACT_PRICING", "")

@dataclass(frozen=True)
class RuntimeConfig:
    model: str = DEFAULT_MODEL
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_run_retries: int = DEFAULT_MAX_RUN_RETRIES
    per_call_timeout_secs: float = DEFAULT_PER_CALL_TIMEOUT_SECS
    pricing_json: str = NEXTRACT_PRICING_JSON

def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()
