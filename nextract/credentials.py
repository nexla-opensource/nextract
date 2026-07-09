"""Credential / env-var helpers for early feedback before pipeline work.

These helpers never hard-fail: API keys may be supplied via
``ProviderConfig.api_key`` rather than the environment. Callers should
warn only.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import TextIO


# Maps nextract provider names to required environment variables.
# Providers requiring ALL listed env vars use "all"; others use "any".
# Kept here (in addition to pydantic_ai_provider) so callers can check
# credentials without constructing a provider instance.
PROVIDER_REQUIRED_ENV: dict[str, dict[str, list[str]]] = {
    "openai": {"any": ["OPENAI_API_KEY"]},
    "anthropic": {"any": ["ANTHROPIC_API_KEY"]},
    "google": {"any": ["GOOGLE_API_KEY"]},
    "google-vertex": {"all": ["GOOGLE_API_KEY"]},
    "azure": {"all": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]},
    "cohere": {"any": ["CO_API_KEY"]},
    # AWS/Bedrock static keys (profile/role handled separately below).
    "aws": {"all": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]},
    "bedrock": {"all": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]},
    "local": {"any": []},
}

# Env vars that indicate non-static AWS auth (profile, IRSA, ECS task role, …).
_AWS_ALT_AUTH_ENV = (
    "AWS_PROFILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


def get_required_env_vars_for_provider(provider: str) -> list[str]:
    """Return the flat list of env var names associated with a provider.

    Unknown providers (including local OCR backends without a static map
    entry) return an empty list.
    """
    env_spec = PROVIDER_REQUIRED_ENV.get(provider, {})
    return list(dict.fromkeys(env_spec.get("all", []) + env_spec.get("any", [])))


def get_missing_provider_env_vars(
    provider: str,
    *,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return env var names that appear missing for the given provider.

    Returns an empty list when:
    - ``api_key`` is provided (credentials may come from ProviderConfig)
    - the provider has no known env requirements
    - all required credentials appear present in the environment

    For specs with ``any``, all listed names are returned when *none* of
    them are set. For ``all``, only the unset names are returned.
    """
    if api_key:
        return []

    env_spec = PROVIDER_REQUIRED_ENV.get(provider)
    if not env_spec:
        return []

    env = os.environ if environ is None else environ

    # AWS/Bedrock: profile / role / web-identity counts as configured.
    if provider in {"aws", "bedrock"} and any(env.get(k) for k in _AWS_ALT_AUTH_ENV):
        return []

    missing: list[str] = []

    all_vars = env_spec.get("all", [])
    if all_vars:
        missing.extend(k for k in all_vars if not env.get(k))

    any_vars = env_spec.get("any", [])
    if any_vars and not any(env.get(k) for k in any_vars):
        missing.extend(any_vars)

    # Preserve order, drop duplicates
    return list(dict.fromkeys(missing))


def format_missing_credentials_warning(provider: str, missing: list[str]) -> str:
    """Build a user-facing warning string for missing provider credentials."""
    return (
        f"Warning: credentials for provider '{provider}' may be missing. "
        f"Unset env vars: {', '.join(missing)}. "
        f"Set them or pass api_key in ProviderConfig."
    )


def warn_if_missing_credentials(
    provider: str,
    *,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> list[str]:
    """Warn to stderr if provider credentials look missing.

    Does not raise. Returns the list of missing env var names (empty when
    no warning was emitted).
    """
    missing = get_missing_provider_env_vars(
        provider, api_key=api_key, environ=environ
    )
    if missing:
        msg = format_missing_credentials_warning(provider, missing)
        print(msg, file=stream or sys.stderr)
    return missing


__all__ = [
    "PROVIDER_REQUIRED_ENV",
    "format_missing_credentials_warning",
    "get_missing_provider_env_vars",
    "get_required_env_vars_for_provider",
    "warn_if_missing_credentials",
]
