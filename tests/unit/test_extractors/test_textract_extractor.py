from nextract.core import ExtractorConfig, ProviderConfig
from nextract.extractors.textract_extractor import TextractExtractor


def test_textract_resolves_client_kwargs_from_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

    extractor = TextractExtractor()
    extractor.config = ExtractorConfig(
        name="textract",
        provider=ProviderConfig(name="aws", model="anthropic.claude-3-5-sonnet"),
    )

    client_kwargs = extractor._resolve_client_kwargs()

    assert client_kwargs == {
        "aws_access_key_id": "env-access",
        "aws_secret_access_key": "env-secret",
        "region_name": "us-west-2",
    }


def test_textract_prefers_explicit_client_kwargs_over_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

    extractor = TextractExtractor()
    extractor.config = ExtractorConfig(
        name="textract",
        provider=ProviderConfig(name="aws", model="anthropic.claude-3-5-sonnet"),
        extractor_params={
            "aws_access_key": "plan-access",
            "aws_secret_key": "plan-secret",
            "region": "eu-central-1",
        },
    )

    client_kwargs = extractor._resolve_client_kwargs()

    assert client_kwargs == {
        "aws_access_key_id": "plan-access",
        "aws_secret_access_key": "plan-secret",
        "region_name": "eu-central-1",
    }
