"""Unit tests for early credential feedback helpers."""

from __future__ import annotations

import io

from nextract.credentials import (
    format_missing_credentials_warning,
    get_missing_provider_env_vars,
    get_required_env_vars_for_provider,
    warn_if_missing_credentials,
)


class TestGetRequiredEnvVars:
    def test_openai_requires_api_key(self):
        assert "OPENAI_API_KEY" in get_required_env_vars_for_provider("openai")

    def test_azure_requires_key_and_endpoint(self):
        required = get_required_env_vars_for_provider("azure")
        assert "AZURE_OPENAI_API_KEY" in required
        assert "AZURE_OPENAI_ENDPOINT" in required

    def test_unknown_provider_returns_empty(self):
        assert get_required_env_vars_for_provider("tesseract") == []
        assert get_required_env_vars_for_provider("not-a-provider") == []

    def test_local_has_no_required_env(self):
        assert get_required_env_vars_for_provider("local") == []


class TestGetMissingProviderEnvVars:
    def test_api_key_in_config_skips_env_check(self):
        missing = get_missing_provider_env_vars(
            "openai",
            api_key="sk-test",
            environ={},
        )
        assert missing == []

    def test_openai_missing_when_env_empty(self):
        missing = get_missing_provider_env_vars("openai", environ={})
        assert missing == ["OPENAI_API_KEY"]

    def test_openai_ok_when_env_set(self):
        missing = get_missing_provider_env_vars(
            "openai",
            environ={"OPENAI_API_KEY": "sk-live"},
        )
        assert missing == []

    def test_azure_all_semantics(self):
        missing = get_missing_provider_env_vars(
            "azure",
            environ={"AZURE_OPENAI_API_KEY": "key-only"},
        )
        assert missing == ["AZURE_OPENAI_ENDPOINT"]

        missing_both = get_missing_provider_env_vars("azure", environ={})
        assert "AZURE_OPENAI_API_KEY" in missing_both
        assert "AZURE_OPENAI_ENDPOINT" in missing_both

        present = get_missing_provider_env_vars(
            "azure",
            environ={
                "AZURE_OPENAI_API_KEY": "k",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            },
        )
        assert present == []

    def test_unknown_provider_no_missing(self):
        assert get_missing_provider_env_vars("tesseract", environ={}) == []

    def test_bedrock_ok_with_aws_profile(self):
        missing = get_missing_provider_env_vars(
            "bedrock",
            environ={"AWS_PROFILE": "dev"},
        )
        assert missing == []

    def test_bedrock_ok_with_static_keys(self):
        missing = get_missing_provider_env_vars(
            "bedrock",
            environ={
                "AWS_ACCESS_KEY_ID": "AKIA...",
                "AWS_SECRET_ACCESS_KEY": "secret",
            },
        )
        assert missing == []

    def test_bedrock_missing_when_no_auth_env(self):
        missing = get_missing_provider_env_vars("bedrock", environ={})
        assert "AWS_ACCESS_KEY_ID" in missing
        assert "AWS_SECRET_ACCESS_KEY" in missing

    def test_bedrock_partial_static_keys_still_missing(self):
        missing = get_missing_provider_env_vars(
            "bedrock",
            environ={"AWS_ACCESS_KEY_ID": "AKIA..."},
        )
        assert missing == ["AWS_SECRET_ACCESS_KEY"]


class TestWarnIfMissingCredentials:
    def test_warns_to_stream_when_missing(self):
        buf = io.StringIO()
        missing = warn_if_missing_credentials(
            "anthropic",
            environ={},
            stream=buf,
        )
        assert missing == ["ANTHROPIC_API_KEY"]
        out = buf.getvalue()
        assert "anthropic" in out
        assert "ANTHROPIC_API_KEY" in out
        assert "Warning:" in out

    def test_no_warn_when_api_key_provided(self):
        buf = io.StringIO()
        missing = warn_if_missing_credentials(
            "openai",
            api_key="sk-from-config",
            environ={},
            stream=buf,
        )
        assert missing == []
        assert buf.getvalue() == ""

    def test_format_message(self):
        msg = format_missing_credentials_warning("openai", ["OPENAI_API_KEY"])
        assert "openai" in msg
        assert "OPENAI_API_KEY" in msg
        assert "ProviderConfig" in msg
