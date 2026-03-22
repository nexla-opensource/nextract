"""Provider implementations and registration."""

from .pydantic_ai_provider import PydanticAIProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .azure_provider import AzureProvider
from .aws_provider import AWSProvider
from .local_provider import LocalProvider
from .cohere_provider import CohereProvider
from .tesseract_provider import TesseractProvider
from .easyocr_provider import EasyOCRProvider
from .paddleocr_provider import PaddleOCRProvider
from .custom_provider_template import CustomProviderTemplate

__all__ = [
    "PydanticAIProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "AzureProvider",
    "AWSProvider",
    "LocalProvider",
    "CohereProvider",
    "TesseractProvider",
    "EasyOCRProvider",
    "PaddleOCRProvider",
    "CustomProviderTemplate",
]
