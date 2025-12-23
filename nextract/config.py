from __future__ import annotations

import os
from dataclasses import dataclass

from nextract.core.config import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig

DEFAULT_MODEL = os.getenv("NEXTRACT_MODEL", "openai:gpt-4o")  # vision-capable by default

# Provider-specific default models (using latest as of December 2025)
# Updated with the most current models from each provider
PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-5.1-mini",  
    "anthropic": "claude-haiku-4-5", 
    "google": "gemini-3-flash-preview",  
    "azure": "gpt-5.1-mini",  
    "cohere": "command-a-03-2025",  
    "aws": "anthropic.claude-haiku-4-5-20251001-v1:0",  
    "local": "llama-3.2-vision",
}

DEFAULT_MAX_CONCURRENCY = int(os.getenv("NEXTRACT_MAX_CONCURRENCY", "4"))
DEFAULT_MAX_RUN_RETRIES = int(os.getenv("NEXTRACT_MAX_RUN_RETRIES", "5"))
DEFAULT_PER_CALL_TIMEOUT_SECS = float(os.getenv("NEXTRACT_PER_CALL_TIMEOUT_SECS", "120"))
DEFAULT_MAX_VALIDATION_ROUNDS = int(os.getenv("NEXTRACT_MAX_VALIDATION_ROUNDS", "2"))

# Parallel processing settings
DEFAULT_MAX_WORKERS = int(os.getenv("NEXTRACT_MAX_WORKERS", "10"))

# Multi-pass extraction settings
DEFAULT_ENABLE_MULTIPASS = os.getenv("NEXTRACT_ENABLE_MULTIPASS", "false").lower() == "true"
DEFAULT_NUM_PASSES = int(os.getenv("NEXTRACT_NUM_PASSES", "3"))
DEFAULT_MULTIPASS_MERGE_STRATEGY = os.getenv("NEXTRACT_MULTIPASS_MERGE_STRATEGY", "union")

# Provenance tracking settings
DEFAULT_ENABLE_PROVENANCE = os.getenv("NEXTRACT_ENABLE_PROVENANCE", "false").lower() == "true"

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
    max_validation_rounds: int = DEFAULT_MAX_VALIDATION_ROUNDS

    # Parallel processing
    max_workers: int = DEFAULT_MAX_WORKERS

    # Multi-pass extraction
    enable_multipass: bool = DEFAULT_ENABLE_MULTIPASS
    num_passes: int = DEFAULT_NUM_PASSES
    multipass_merge_strategy: str = DEFAULT_MULTIPASS_MERGE_STRATEGY

    # Provenance tracking
    enable_provenance: bool = DEFAULT_ENABLE_PROVENANCE

def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


def get_default_model_for_provider(provider: str) -> str:
    """Get the default model for a given provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic", "google")

    Returns:
        Default model name for that provider

    Raises:
        ValueError: If provider is not recognized
    """
    if provider not in PROVIDER_DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {', '.join(PROVIDER_DEFAULT_MODELS.keys())}"
        )
    return PROVIDER_DEFAULT_MODELS[provider]


__all__ = [
    "RuntimeConfig",
    "load_runtime_config",
    "ProviderConfig",
    "ExtractorConfig",
    "ChunkerConfig",
    "ExtractionPlan",
    "get_default_model_for_provider",
    "PROVIDER_DEFAULT_MODELS",
]
