from __future__ import annotations

from nextract.providers.bedrock_provider import AWSProvider  # noqa: F401 - legacy alias

# The 'aws' provider registration now lives in bedrock_provider.py
# which registers both 'bedrock' and 'aws' (as a deprecated alias).
