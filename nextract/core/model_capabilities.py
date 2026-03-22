"""Model capability registry for determining model features.

This module provides a centralized registry for model capabilities,
replacing fragile string matching with explicit capability declarations.
"""

from __future__ import annotations

from typing import Any

# Model capability registry
# Keys are model name patterns (matched case-insensitively)
# Values are dicts with capability flags
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    # OpenAI GPT-5.x models (current flagship line)
    "gpt-5.4": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gpt-5.4-mini": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gpt-5.4-nano": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gpt-5.2": {"vision": True, "structured_output": True, "context_window": 400000},
    "gpt-5": {"vision": True, "structured_output": True, "context_window": 256000},
    "gpt-5.1": {"vision": True, "structured_output": True, "context_window": 256000},
    "gpt-5.1-mini": {"vision": True, "structured_output": True, "context_window": 256000},
    # OpenAI GPT-4.x models
    "gpt-4.1": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gpt-4.1-mini": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gpt-4.1-nano": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gpt-4o": {"vision": True, "structured_output": True, "context_window": 128000},
    "gpt-4o-mini": {"vision": True, "structured_output": True, "context_window": 128000},
    "gpt-4-turbo": {"vision": True, "structured_output": False, "context_window": 128000},
    "gpt-4": {"vision": False, "structured_output": False, "context_window": 8192},
    # OpenAI o-series reasoning models
    "o4-mini": {"vision": True, "structured_output": True, "context_window": 200000},
    "o3": {"vision": True, "structured_output": True, "context_window": 200000},
    "o3-pro": {"vision": True, "structured_output": True, "context_window": 200000},
    "o3-mini": {"vision": False, "structured_output": True, "context_window": 200000},
    "o1": {"vision": True, "structured_output": True, "context_window": 200000},
    "o1-preview": {"vision": False, "structured_output": True, "context_window": 128000},
    "o1-mini": {"vision": False, "structured_output": True, "context_window": 128000},
    # Anthropic models
    "claude-opus-4-6": {"vision": True, "structured_output": True, "context_window": 1048576},
    "claude-sonnet-4-6": {"vision": True, "structured_output": True, "context_window": 1048576},
    "claude-opus-4-5": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-opus-4": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-sonnet-4": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-haiku-4-5": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-haiku-4": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-3-5-sonnet": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-3-5-haiku": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-3-opus": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-3-sonnet": {"vision": True, "structured_output": True, "context_window": 200000},
    "claude-3-haiku": {"vision": True, "structured_output": True, "context_window": 200000},
    # Google Gemini models
    "gemini-3.1-pro-preview": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gemini-3-flash-preview": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gemini-3.1-flash-lite-preview": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gemini-2.5-pro": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gemini-2.5-flash": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gemini-2.0-flash": {"vision": True, "structured_output": True, "context_window": 1048576},
    "gemini-1.5-pro": {"vision": True, "structured_output": True, "context_window": 2000000},
    "gemini-1.5-flash": {"vision": True, "structured_output": True, "context_window": 1000000},
    # Meta Llama models
    "llama-4-scout": {"vision": True, "structured_output": False, "context_window": 10485760},
    "llama-4-maverick": {"vision": True, "structured_output": False, "context_window": 1048576},
    "llama-3.2-vision": {"vision": True, "structured_output": False, "context_window": 128000},
    "llama-3.2": {"vision": False, "structured_output": False, "context_window": 128000},
    "llama-3.1": {"vision": False, "structured_output": False, "context_window": 128000},
    "llama-3": {"vision": False, "structured_output": False, "context_window": 8192},
    "llava": {"vision": True, "structured_output": False, "context_window": 4096},
    # Mistral models
    "mistral-large-2512": {"vision": True, "structured_output": True, "context_window": 256000},
    "mistral-small-2603": {"vision": True, "structured_output": True, "context_window": 256000},
    "mistral-large": {"vision": True, "structured_output": True, "context_window": 128000},
    "mistral-medium": {"vision": False, "structured_output": True, "context_window": 32000},
    "mistral-small": {"vision": False, "structured_output": True, "context_window": 32000},
    "pixtral-large": {"vision": True, "structured_output": True, "context_window": 128000},
    "pixtral-12b": {"vision": True, "structured_output": True, "context_window": 128000},
    "mistral": {"vision": False, "structured_output": False, "context_window": 32000},
    "mixtral": {"vision": False, "structured_output": False, "context_window": 32000},
    # Cohere models
    "command-a-vision": {"vision": True, "structured_output": True, "context_window": 128000},
    "command-a": {"vision": False, "structured_output": True, "context_window": 256000},
    "command-r-plus": {"vision": False, "structured_output": True, "context_window": 128000},
    "command-r": {"vision": False, "structured_output": True, "context_window": 128000},
    # DeepSeek models
    "deepseek-chat": {"vision": False, "structured_output": True, "context_window": 128000},
    "deepseek-coder": {"vision": False, "structured_output": True, "context_window": 128000},
    "deepseek-reasoner": {"vision": False, "structured_output": True, "context_window": 128000},
    "deepseek-v3": {"vision": False, "structured_output": True, "context_window": 128000},
    # AWS Bedrock model patterns
    "anthropic.claude": {"vision": True, "structured_output": True, "context_window": 200000},
    "amazon.nova-pro": {"vision": True, "structured_output": True, "context_window": 300000},
    "amazon.nova-lite": {"vision": True, "structured_output": True, "context_window": 300000},
    "amazon.nova-micro": {"vision": False, "structured_output": True, "context_window": 128000},
    "amazon.titan": {"vision": False, "structured_output": True, "context_window": 8192},
}

# Providers that generally support vision across their models
# Used as fallback when specific model is not in registry
VISION_CAPABLE_PROVIDERS = {"anthropic", "google", "openai"}


def _normalize_model_name(model: str) -> str:
    """Normalize model name for matching."""
    # Remove provider prefix if present (e.g., "openai:gpt-4o" -> "gpt-4o")
    if ":" in model:
        model = model.split(":", 1)[1]
    return model.lower().strip()


def _find_matching_capability(model: str) -> dict[str, Any] | None:
    """Find capability entry that matches the model name."""
    normalized = _normalize_model_name(model)

    # Exact match first
    if normalized in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[normalized]

    # Prefix match
    for pattern, capabilities in sorted(MODEL_CAPABILITIES.items(), key=lambda x: len(x[0]), reverse=True):
        if normalized.startswith(pattern):
            return capabilities

    # Substring match
    for pattern, capabilities in sorted(MODEL_CAPABILITIES.items(), key=lambda x: len(x[0]), reverse=True):
        if pattern in normalized:
            return capabilities

    return None


def get_model_capability(
    model: str,
    capability: str,
    default: bool = False,
    provider: str | None = None,
) -> bool:
    """Get a specific capability for a model.

    Args:
        model: The model name (e.g., "gpt-5.4", "claude-opus-4-6")
        capability: The capability to check (e.g., "vision", "structured_output")
        default: Default value if model/capability not found
        provider: Optional provider name for fallback heuristics

    Returns:
        Boolean indicating if the model has the capability
    """
    capabilities = _find_matching_capability(model)

    if capabilities is not None:
        return capabilities.get(capability, default)

    # Fallback to provider-based heuristic
    if provider and capability == "vision":
        return provider.lower() in VISION_CAPABLE_PROVIDERS

    return default


def get_model_capabilities(model: str) -> dict[str, Any]:
    """Get all capabilities for a model.

    Args:
        model: The model name

    Returns:
        Dict of capabilities, or empty dict if model not found
    """
    capabilities = _find_matching_capability(model)
    return dict(capabilities) if capabilities else {}


def register_model_capability(
    model: str,
    capabilities: dict[str, Any],
) -> None:
    """Register or update capabilities for a model.

    Args:
        model: The model name pattern
        capabilities: Dict of capability flags
    """
    MODEL_CAPABILITIES[model.lower()] = capabilities
