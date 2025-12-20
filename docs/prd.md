# Nextract V2: Product Requirements Document & Architecture

## Executive Summary

Nextract is an enterprise-grade, open-source Python package for intelligent document ingestion and structured data extraction. It provides a modular, extensible framework that transforms unstructured documents (PDFs, Word, PowerPoint, images) into clean Markdown or structured data conforming to user-defined JSON schemas.

### Key Differentiators

- **Dual-layer abstraction**: Extractors + Providers (LLMs/APIs)
- **Plugin-based architecture** for unlimited extensibility
- **Zero infrastructure setup** - pure Python package
- **Stateless design** with optional async processing
- **Production-ready** with enterprise reliability

## 1. Product Vision & Goals

### 1.1 Vision
To be the first-choice, open-source document ingestion package that enterprises trust for mission-critical document processing workflows.

### 1.2 Goals

- **Simplicity**: pip install nextract and you're ready
- **Flexibility**: Support any document type, any extraction technique, any provider
- **Reliability**: Handle edge cases, provide confidence scores, ensure accuracy
- **Performance**: Process large documents efficiently with parallel execution
- **Extensibility**: Dual-layer plugin architecture for unlimited combinations

## 2. Core Capabilities

### 2.1 Document to Markdown Conversion
Transform documents into LLM-ready Markdown with:
- Configurable output formats (Markdown, HTML)
- Image replacement with VLM descriptions
- Preservation of document structure
- Citation tracking for all content

### 2.2 Structured Data Extraction
Extract structured data from documents:
- AI-powered schema suggestion from samples
- Complex nested schemas with arrays and conditionals
- Multi-pass extraction for accuracy
- Confidence scores and source citations per field
- Validation and quality scoring

### 2.3 Intelligent Extraction
- Multiple extraction techniques (VLM, Text, OCR, Hybrid, Third-party APIs)
- Multiple provider support per extractor
- Smart chunker selection based on modality
- Document type auto-detection
- Batch extraction with schema improvement suggestions

## 3. Technical Architecture

### 3.1 Architectural Principles

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                             │
│                   (CLI / Python SDK)                              │
└─────────────────┬────────────────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                            │
│        (Pipeline Manager, Config Validator, Router)               │
└─────────────────┬────────────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┬──────────────┐
    │             │             │              │              │
┌───▼─────┐  ┌───▼──────┐  ┌──▼───────┐  ┌──▼───────┐  ┌──▼──────┐
│INGEST & │  │ EXTRACTOR│  │SCHEMA    │  │VALIDATION│  │ OUTPUT  │
│ PARSE   │  │ LAYER    │  │ALIGNMENT │  │LAYER     │  │FORMATTER│
└───┬─────┘  └───┬──────┘  └──┬───────┘  └──┬───────┘  └──┬──────┘
    │            │            │              │              │
    │      ┌─────▼──────┐     │              │              │
    │      │  CHUNKER   │     │              │              │
    │      │            │     │              │              │
    │      └─────┬──────┘     │              │              │
    │            │            │              │              │
┌───▼────────────▼────────────▼──────────────▼──────────────▼──────┐
│                     PLUGIN REGISTRY SYSTEM                        │
│   ┌──────────────┐  ┌───────────────┐  ┌──────────────┐         │
│   │   EXTRACTORS    │  │   PROVIDERS   │  │  VALIDATORS  │         │
│   │              │  │ (LLMs/APIs)   │  │   (Rules)    │         │
│   └──────────────┘  └───────────────┘  └──────────────┘         │
└───────────────────────────────────────────────────────────────────┘
```
### 3.2 Key Architectural Concepts

#### Extractors vs Providers: Dual-Layer Abstraction

**EXTRACTORS = Extraction Plugins (Techniques)**
- VLMExtractor: Extract from images using vision-language models
- TextExtractor: Extract from text-only inputs
- OCRExtractor: Extract via OCR engines
- HybridExtractor: Combine multiple extraction techniques
- TextractExtractor: Extract using AWS Textract
- LlamaIndexExtractor: Extract using LlamaIndex
- ChandraModelExtractor: Extract using Chandra Model
- Extensible to any future extraction technique

**PROVIDERS = LLM/API Backends**
- OpenAIProvider: GPT-4, GPT-4 Vision, etc.
- AnthropicProvider: Claude models
- GoogleProvider: Gemini models
- LocalProvider: Ollama, vLLM, etc.
- AWSProvider: Bedrock
- AzureProvider: Azure OpenAI
- Extensible to any LLM or API service

**Relationship:**

- One Extractor → Multiple Providers
  - Example: VLMExtractor → {OpenAIProvider, AnthropicProvider, GoogleProvider}
  
- One Provider → Multiple Extractors (where applicable)
  - Example: OpenAIProvider → {VLMExtractor, TextExtractor}
### 3.3 Layer Responsibilities

#### Ingest & Parse Layer
- File loading and format detection
- Format conversion (Word/PPT/Excel → PDF → Images)
- Validation (corrupted, password-protected)
- Image extraction, text extraction, and layout parsing

#### Chunker Layer
- **Visual Modality:**
  - Page chunker only
  - Number of pages per chunk
  - Page overlap between chunks
  - Image resolution management
- **Text Modality:**
  - Advanced chunkers
  - Semantic chunker
  - Table-aware chunker
  - Section-based chunker
  - Fixed-size chunker
- **Hybrid Modality:**
  - Mixed page + text chunkers
  - Modality-aware routing rules
  - Rule-based capability enabling based on modality

#### Extractor Layer
- Applies extraction techniques to chunks
- Routes to appropriate providers
- Manages extractor-specific configurations
- Handles extractor-level retries and fallbacks

#### Schema Alignment Layer
- Schema suggestion from samples
- Structured extraction with multiple passes
- Schema partitioning for large schemas
- Result merging and verification
- Confidence calculation

#### Validation Layer
- Schema validation
- Business rule validation
- Quality scoring
- Consistency checks

#### Output Formatter Layer
- Format conversion (JSON, Markdown, HTML, CSV)
- Citation formatting
- Metadata enrichment

### 3.4 Architecture Plan: Extensibility, Naming, and Structure

This plan refines Nextract to be extractor-agnostic, provider-agnostic, and easy to extend without breaking core contracts.

#### 3.4.1 Terminology and Naming Conventions
- Use "extractor" for algorithmic extraction techniques. Class suffix: `*Extractor`.
- Use "provider" for model/API backends. Class suffix: `*Provider`.
- Use "chunker" for document splitting. Class suffix: `*Chunker`.
- Use "parser" for OCR and layout parsing. Class suffix: `*Parser`.
- Use "pipeline" for orchestration objects. Class suffix: `*Pipeline`.
- Use "validator" and "formatter" for post-extraction steps. Class suffixes: `*Validator`, `*Formatter`.
- Data objects: `DocumentArtifact`, `DocumentChunk`, `ExtractionResult`, `FieldResult`, `Citation`, `ConfidenceScore`, `ProviderRequest`, `ProviderResponse`.
- Config objects: `ProviderConfig`, `ExtractorConfig`, `ChunkerConfig`, `ExtractionPlan`; CLI flags use `--extractor`, `--provider`, `--chunker`.

#### 3.4.2 Module Structure (Proposed)
- `nextract/ingest`: file loading, format detection, conversion to canonical artifacts.
- `nextract/parse`: OCR, layout parsing, text extraction, table detection.
- `nextract/chunking`: chunkers and chunk selection policies.
- `nextract/extractors`: extractors (VLM, layout-aware, template, rule-based, RAG, hybrid).
- `nextract/providers`: provider implementations and transport clients.
- `nextract/schema`: schema inference, validation helpers, structured output parsing.
- `nextract/validate`: business rules, consistency checks, quality scoring.
- `nextract/output`: formatters, citation renderers, metadata enrichment.
- `nextract/pipeline`: orchestration, routing, execution graph, retries and fallbacks.
- `nextract/registry`: plugin registry, discovery, capability indexing.
- `nextract/telemetry`: structured logging, metrics, trace export.

#### 3.4.3 Extension Points and Capability Contracts
- All plugins declare `Capabilities` and `Requirements` (modalities, inputs, outputs, limitations).
- Routing uses capability matching plus explicit user overrides; no hidden magic.
- Plugins are isolated behind interfaces and never import concrete providers directly.
- Extractor and provider compatibility is negotiated at runtime and validated early.
- Each plugin has deterministic inputs and outputs; no cross-stage side effects.

#### 3.4.4 Pipeline Orchestration Model
- Model the workflow as a DAG of `Stage` nodes: Ingest -> Parse -> Chunk -> Extract -> Validate -> PostProcess -> Export.
- Each stage emits typed artifacts and metadata, enabling caching and re-use.
- Support multi-pass extraction by inserting looped stages or verification stages.
- Enable "ensemble" extraction by running multiple extractors and reconciling results.

#### 3.4.5 Plan Design Principles
- Separate user intent from runtime controls: `ExtractionPlan` plus runtime overrides (timeouts, retries, concurrency).
- Keep configs composable, typed, and validated up front.
- Use explicit error types with actionable suggestions.
- Ensure configs are serializable to JSON/YAML for reproducibility.

### 3.5 Advanced Extractor Coverage (SOTA)

This section captures target extractor families and parsing techniques to keep Nextract extensible and state-of-the-art.

- Vision-first: VLM extractors, layout-aware models, multimodal parsers.
- Text-first: robust PDF text extraction, semantic chunkers, section-aware parsing.
- OCR-first: classical OCR, document OCR with layout detection, handwritten OCR.
- Layout and tables: table detectors, cell extraction, layout segmentation.
- Schema-guided extraction: structured output with JSON Schema, constrained decoding, validator-driven retries.
- Multi-pass verification: self-consistency checks, cross-page validation, field-level re-prompting.
- Hybrid ensembles: combine OCR, text, and vision extractors with voting or confidence weighting.
- Rule and template extraction: regex and template parsers for deterministic fields.
- Retrieval-augmented extraction: RAG on long documents and cross-document extraction.

## 4. Detailed Component Design
### 4.1 Plugin System Architecture

#### Base Interfaces

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class Modality(Enum):
    """Modality determines available features"""
    VISUAL = "visual"      # Documents as images (VLM extractors)
    TEXT = "text"          # Text-based extractors
    HYBRID = "hybrid"      # Combination of both

@dataclass
class ProviderRequest:
    """Normalized provider request across text, vision, and structured outputs"""
    messages: List[Dict[str, Any]]
    images: Optional[List[str]] = None
    schema: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderResponse:
    """Normalized provider response"""
    text: str
    structured_output: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    raw: Any = None

class BaseProvider(ABC):
    """Base interface for all LLM/API providers"""
    
    @abstractmethod
    def initialize(self, config: 'ProviderConfig') -> None:
        """Initialize provider with configuration"""
        pass
    
    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a response for the given request"""
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether provider supports vision inputs"""
        pass
    
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether provider supports structured output"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities"""
        pass

class BaseExtractor(ABC):
    """Base interface for all extractors"""
    
    @abstractmethod
    def initialize(self, config: 'ExtractorConfig') -> None:
        """Initialize extractor with configuration"""
        pass
    
    @abstractmethod
    def run(
        self, 
        input_data: Any, 
        provider: BaseProvider,
        **kwargs
    ) -> 'ExtractorResult':
        """Run extraction using the given provider"""
        pass
    
    @abstractmethod
    def get_modality(self) -> Modality:
        """Return the modality this extractor uses"""
        pass
    
    @abstractmethod
    def get_supported_providers(self) -> List[str]:
        """Return list of compatible provider names"""
        pass
    
    @abstractmethod
    def validate_config(self, config: 'ExtractorConfig') -> bool:
        """Validate extractor configuration"""
        pass

class BaseChunker(ABC):
    """Base interface for chunkers"""
    
    @abstractmethod
    def get_applicable_modalities(self) -> List[Modality]:
        """Return modalities where this chunker is applicable"""
        pass
    
    @abstractmethod
    def chunk(
        self, 
        document: 'Document', 
        config: 'ChunkerConfig'
    ) -> List['DocumentChunk']:
        """Chunk document according to the chunker"""
        pass
    
    @abstractmethod
    def validate_config(self, config: 'ChunkerConfig') -> bool:
        """Validate chunker configuration"""
        pass

class BaseValidator(ABC):
    """Base interface for validators"""
    
    @abstractmethod
    def validate(
        self, 
        data: Dict, 
        schema: Dict,
        **kwargs
    ) -> 'ValidationResult':
        """Validate extracted data"""
        pass
```
### 4.2 Plan & Config System with Validation

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

@dataclass
class ProviderConfig:
    """Configuration for a specific provider"""
    name: str                           # "openai", "anthropic", etc.
    model: str                          # "gpt-4", "claude-3-5-sonnet", etc.
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    # Provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate provider configuration"""
        if not self.name or not self.model:
            raise ValueError("Provider name and model are required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return True

@dataclass
class ExtractorConfig:
    """Configuration for an extractor"""
    name: str                 # "vlm", "text", "ocr", "hybrid", etc.
    provider: ProviderConfig
    fallback_provider: Optional[ProviderConfig] = None
    
    # Extractor-specific settings
    enable_caching: bool = True
    batch_size: int = 1
    modality: Optional[Modality] = None
    
    # Extractor-specific parameters
    extractor_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate extractor configuration"""
        self.provider.validate()
        if self.fallback_provider:
            self.fallback_provider.validate()
        
        # Rule-based validation: ensure provider supports extractor
        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(self.name)
        if extractor_class:
            supported_providers = extractor_class.get_supported_providers()
            if self.provider.name not in supported_providers:
                raise ValueError(
                    f"Extractor '{self.name}' does not support "
                    f"provider '{self.provider.name}'"
                )
        return True

@dataclass
class ChunkerConfig:
    """Configuration for a chunker"""
    name: str                          # "page", "semantic", "table_aware", etc.
    
    # Visual modality chunking (applicable to VLM extractors)
    pages_per_chunk: int = 5
    page_overlap: int = 1
    max_image_dimension: int = 2048
    image_quality: int = 95
    
    # Text modality chunking (applicable to text extractors)
    chunk_size: int = 2000              # tokens
    chunk_overlap: int = 200            # tokens
    preserve_tables: bool = True
    preserve_sections: bool = True
    respect_sentence_boundaries: bool = True
    
    # Common settings
    min_chunk_size: int = 100
    max_chunk_size: int = 10000
    
    def validate(self, modality: Modality) -> bool:
        """Validate chunker config based on modality"""
        chunker_registry = ChunkerRegistry.get_instance()
        chunker_class = chunker_registry.get(self.name)
        
        if chunker_class:
            applicable_modalities = chunker_class.get_applicable_modalities()
            if modality not in applicable_modalities:
                raise ValueError(
                    f"Chunker '{self.name}' is not applicable "
                    f"to modality '{modality.value}'. "
                    f"Applicable modalities: {[m.value for m in applicable_modalities]}"
                )
        
        if modality == Modality.VISUAL:
            if self.pages_per_chunk < 1:
                raise ValueError("pages_per_chunk must be >= 1")
            if self.page_overlap >= self.pages_per_chunk:
                raise ValueError("page_overlap must be < pages_per_chunk")
        
        elif modality == Modality.TEXT:
            if self.chunk_size < self.min_chunk_size:
                raise ValueError(f"chunk_size must be >= {self.min_chunk_size}")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be < chunk_size")
        
        return True

@dataclass
class ExtractionPlan:
    """Complete extraction plan"""
    extractor: ExtractorConfig
    chunker: ChunkerConfig
    
    # Extraction settings
    num_passes: int = 1
    include_confidence: bool = True
    include_citations: bool = True
    include_raw_text: bool = False
    
    # Schema settings
    auto_suggest_schema: bool = False
    schema_validation: bool = True
    
    # Retry and fallback
    retry_on_failure: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0
    
    # Validation
    validation_rules: List[str] = field(default_factory=list)
    strict_validation: bool = False
    
    def validate(self) -> bool:
        """Validate complete extraction plan"""
        self.extractor.validate()
        
        # Get modality from extractor
        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(self.extractor.name)
        if extractor_class:
            modality = extractor_class.get_modality()
            self.chunker.validate(modality)
        
        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")
        
        return True
```
### 4.3 Extractor Implementations

#### VLM Extractor (Vision-Language Model)

```python
from typing import List, Dict, Any
from nextract.core import ProviderRequest
import base64
from io import BytesIO

class VLMExtractor(BaseExtractor):
    """Extract from images using vision-language models"""
    
    SUPPORTED_PROVIDERS = [
        "openai",      # GPT-4V, GPT-4o
        "anthropic",   # Claude 3.5 Sonnet, Claude 4
        "google",      # Gemini Vision
        "azure",       # Azure OpenAI
        "local"        # Local VLM models
    ]
    
    def __init__(self):
        self.config: Optional[ExtractorConfig] = None
        self.provider: Optional[BaseProvider] = None
    
    def initialize(self, config: ExtractorConfig) -> None:
        """Initialize VLM extractor"""
        self.config = config
        self.validate_config(config)
    
    def get_modality(self) -> Modality:
        """VLM operates in VISUAL modality"""
        return Modality.VISUAL
    
    def get_supported_providers(self) -> List[str]:
        """Return supported providers"""
        return self.SUPPORTED_PROVIDERS
    
    def validate_config(self, config: ExtractorConfig) -> bool:
        """Validate VLM extractor configuration"""
        if not config.provider.supports_vision():
            raise ValueError(
                f"Provider '{config.provider.name}' does not support vision"
            )
        return True
    
    def run(
        self, 
        input_data: List['ImageChunk'],
        provider: BaseProvider,
        prompt: str,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> 'ExtractorResult':
        """
        Run extraction on image chunks using a VLM
        
        Args:
            input_data: List of image chunks (PDFs converted to images)
            provider: Provider instance to use
            prompt: Extraction prompt
            schema: Optional schema for structured extraction
            **kwargs: Additional parameters
        """
        results = []
        
        for chunk in input_data:
            # Prepare images for provider
            images_b64 = [
                self._prepare_image(img, self.config.extractor_params)
                for img in chunk.images
            ]
            
            # Build prompt with images
            messages = self._build_vlm_prompt(
                images_b64, 
                prompt, 
                schema,
                chunk.metadata
            )
            
            # Call provider
            try:
                request = ProviderRequest(
                    messages=messages,
                    images=images_b64,
                    schema=schema,
                    options={
                        "temperature": self.config.provider.temperature,
                        "max_tokens": self.config.provider.max_tokens,
                        **kwargs
                    }
                )
                response = provider.generate(request)
                payload = response.structured_output or response.text
                
                results.append({
                    'chunk_id': chunk.id,
                    'response': payload,
                    'page_range': chunk.page_range,
                    'metadata': chunk.metadata
                })
                
            except Exception as e:
                if self.config.fallback_provider:
                    # Try fallback provider
                    response = self._try_fallback(chunk, prompt, schema)
                    results.append(response)
                else:
                    raise
        
        return ExtractorResult(
            name="vlm",
            provider_name=provider.name,
            results=results,
            metadata={
                'modality': 'visual',
                'num_chunks': len(input_data),
                'total_images': sum(len(c.images) for c in input_data)
            }
        )
    
    def _prepare_image(
        self, 
        image: Any, 
        params: Dict
    ) -> str:
        """Prepare image for VLM extraction"""
        # Resize if needed
        max_dimension = params.get('max_image_dimension', 2048)
        quality = params.get('image_quality', 95)
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", quality=quality)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return img_b64
    
    def _build_vlm_prompt(
        self,
        images: List[str],
        prompt: str,
        schema: Optional[Dict],
        metadata: Dict
    ) -> List[Dict]:
        """Build VLM-specific prompt with images"""
        messages = []
        
        # Add context
        system_msg = "You are an expert document analyzer."
        if schema:
            system_msg += f"\n\nExtract information according to this schema:\n{schema}"
        
        # Add images and prompt
        content = []
        for img_b64 in images:
            content.append({
                "type": "image",
                "image": img_b64
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({
            "role": "system",
            "content": system_msg
        })
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
```
#### Text Extractor

```python
class TextExtractor(BaseExtractor):
    """Extract from text-only inputs"""
    
    SUPPORTED_PROVIDERS = [
        "openai",
        "anthropic",
        "google",
        "azure",
        "local",
        "cohere"
    ]
    
    def get_modality(self) -> Modality:
        """Text operates in TEXT modality"""
        return Modality.TEXT
    
    def get_supported_providers(self) -> List[str]:
        return self.SUPPORTED_PROVIDERS
    
    def validate_config(self, config: ExtractorConfig) -> bool:
        """Validate text extractor configuration"""
        # Text extractor works with any text-capable LLM
        return True
    
    def run(
        self,
        input_data: List['TextChunk'],
        provider: BaseProvider,
        prompt: str,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> 'ExtractorResult':
        """Run extraction on text chunks"""
        results = []
        
        for chunk in input_data:
            # Build text-only prompt
            messages = self._build_text_prompt(
                chunk.text,
                prompt,
                schema,
                chunk.metadata
            )
            
            # Call provider
            request = ProviderRequest(
                messages=messages,
                schema=schema,
                options={
                    "temperature": self.config.provider.temperature,
                    **kwargs
                }
            )
            response = provider.generate(request)
            payload = response.structured_output or response.text
            
            results.append({
                'chunk_id': chunk.id,
                'response': payload,
                'metadata': chunk.metadata
            })
        
        return ExtractorResult(
            name="text",
            provider_name=provider.name,
            results=results,
            metadata={'modality': 'text'}
        )
```

#### Third-Party Extractor (AWS Textract Example)

```python
class TextractExtractor(BaseExtractor):
    """Extract using AWS Textract"""
    
    SUPPORTED_PROVIDERS = ["aws"]  # Only works with AWS
    
    def get_modality(self) -> Modality:
        return Modality.VISUAL
    
    def get_supported_providers(self) -> List[str]:
        return self.SUPPORTED_PROVIDERS
    
    def validate_config(self, config: ExtractorConfig) -> bool:
        """Validate Textract configuration"""
        required_params = ['aws_access_key', 'aws_secret_key', 'region']
        for param in required_params:
            if param not in config.extractor_params:
                raise ValueError(f"Textract requires '{param}' in extractor_params")
        return True
    
    def run(
        self,
        input_data: List['ImageChunk'],
        provider: BaseProvider,  # AWSProvider
        **kwargs
    ) -> 'ExtractorResult':
        """Run extraction using Textract API"""
        import boto3
        
        textract_client = boto3.client(
            'textract',
            aws_access_key_id=self.config.extractor_params['aws_access_key'],
            aws_secret_key=self.config.extractor_params['aws_secret_key'],
            region_name=self.config.extractor_params['region']
        )
        
        results = []
        for chunk in input_data:
            # Call Textract
            response = textract_client.analyze_document(
                Document={'Bytes': chunk.images[0]},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            # Parse Textract response
            parsed = self._parse_textract_response(response)
            results.append(parsed)
        
        return ExtractorResult(
            name="textract",
            provider_name="aws",
            results=results
        )
```
### 4.4 Chunkers with Modality Awareness

#### Page Chunker (VISUAL Modality)

```python
class PageBasedChunker(BaseChunker):
    """Chunk PDF pages for visual extractors"""
    
    def get_applicable_modalities(self) -> List[Modality]:
        """Only applicable to VISUAL modality"""
        return [Modality.VISUAL]
    
    def validate_config(self, config: ChunkerConfig) -> bool:
        """Validate page chunker config"""
        if config.pages_per_chunk < 1:
            raise ValueError("pages_per_chunk must be >= 1")
        if config.page_overlap >= config.pages_per_chunk:
            raise ValueError("page_overlap must be < pages_per_chunk")
        return True
    
    def chunk(
        self,
        document: 'Document',
        config: ChunkerConfig
    ) -> List['ImageChunk']:
        """
        Chunk document by pages
        
        Args:
            document: Document with PDF/images
            config: Chunker configuration
            
        Returns:
            List of ImageChunk objects
        """
        chunks = []
        total_pages = document.get_page_count()
        
        start_page = 0
        chunk_id = 0
        
        while start_page < total_pages:
            end_page = min(
                start_page + config.pages_per_chunk,
                total_pages
            )
            
            # Extract pages as images
            images = document.extract_pages_as_images(
                start_page=start_page,
                end_page=end_page,
                max_dimension=config.max_image_dimension,
                quality=config.image_quality
            )
            
            chunk = ImageChunk(
                id=f"chunk_{chunk_id}",
                images=images,
                page_range=(start_page, end_page),
                metadata={
                    'total_pages': end_page - start_page,
                    'overlap_with_previous': config.page_overlap if chunk_id > 0 else 0
                }
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_page = end_page - config.page_overlap
            chunk_id += 1
        
        return chunks
```

#### Semantic Chunker (TEXT Modality)

```python
class SemanticChunker(BaseChunker):
    """Semantic chunker for text extractors"""
    
    def get_applicable_modalities(self) -> List[Modality]:
        """Only applicable to TEXT modality"""
        return [Modality.TEXT, Modality.HYBRID]
    
    def validate_config(self, config: ChunkerConfig) -> bool:
        """Validate semantic chunker config"""
        if config.chunk_size < config.min_chunk_size:
            raise ValueError(f"chunk_size must be >= {config.min_chunk_size}")
        return True
    
    def chunk(
        self,
        document: 'Document',
        config: ChunkerConfig
    ) -> List['TextChunk']:
        """
        Chunk by semantic boundaries
        
        Respects:
        - Paragraph boundaries
        - Section headings
        - Sentence boundaries
        """
        text = document.extract_text()
        
        # Detect semantic boundaries
        paragraphs = self._detect_paragraphs(text)
        sections = self._detect_sections(paragraphs)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section_size = self._estimate_tokens(section)
            
            if current_size + section_size > config.chunk_size:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(self._create_text_chunk(
                        current_chunk, 
                        config
                    ))
                current_chunk = [section]
                current_size = section_size
            else:
                current_chunk.append(section)
                current_size += section_size
        
        # Add remaining
        if current_chunk:
            chunks.append(self._create_text_chunk(current_chunk, config))
        
        return chunks
```

#### Table-Aware Chunker (TEXT Modality)

```python
class TableAwareChunker(BaseChunker):
    """Never split tables across chunks"""
    
    def get_applicable_modalities(self) -> List[Modality]:
        """Only applicable to TEXT modality"""
        return [Modality.TEXT, Modality.HYBRID]
    
    def chunk(
        self,
        document: 'Document',
        config: ChunkerConfig
    ) -> List['TextChunk']:
        """Chunk while preserving table integrity"""
        text = document.extract_text()
        
        # Detect tables
        tables = self._detect_tables(text)
        
        # Create chunks ensuring tables are never split
        chunks = self._chunk_preserving_tables(
            text, 
            tables, 
            config
        )
        
        return chunks
```
### 4.5 Registry System

```python
from typing import Dict, Type, Optional

class ExtractorRegistry:
    """Registry for all extractor implementations"""
    
    _instance: Optional['ExtractorRegistry'] = None
    _extractors: Dict[str, Type[BaseExtractor]] = {}
    
    @classmethod
    def get_instance(cls) -> 'ExtractorRegistry':
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(
        self, 
        name: str, 
        extractor_class: Type[BaseExtractor]
    ) -> None:
        """Register an extractor"""
        self._extractors[name] = extractor_class
    
    def get(self, name: str) -> Optional[Type[BaseExtractor]]:
        """Get extractor class by name"""
        return self._extractors.get(name)
    
    def list_extractors(self) -> List[str]:
        """List all registered extractors"""
        return list(self._extractors.keys())
    
    def get_compatible_providers(self, extractor_name: str) -> List[str]:
        """Get providers compatible with an extractor"""
        extractor_class = self.get(extractor_name)
        if extractor_class:
            return extractor_class.get_supported_providers()
        return []

class ProviderRegistry:
    """Registry for all provider implementations"""
    
    _instance: Optional['ProviderRegistry'] = None
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def get_instance(cls) -> 'ProviderRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(
        self, 
        name: str, 
        provider_class: Type[BaseProvider]
    ) -> None:
        """Register a provider"""
        self._providers[name] = provider_class
    
    def get(self, name: str) -> Optional[Type[BaseProvider]]:
        """Get provider class by name"""
        return self._providers.get(name)

class ChunkerRegistry:
    """Registry for chunkers"""
    
    _instance: Optional['ChunkerRegistry'] = None
    _chunkers: Dict[str, Type[BaseChunker]] = {}
    
    @classmethod
    def get_instance(cls) -> 'ChunkerRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(
        self, 
        name: str, 
        chunker_class: Type[BaseChunker]
    ) -> None:
        """Register a chunker"""
        self._chunkers[name] = chunker_class
    
    def get(self, name: str) -> Optional[Type[BaseChunker]]:
        """Get chunker class by name"""
        return self._chunkers.get(name)
    
    def get_chunkers_for_modality(
        self, 
        modality: Modality
    ) -> List[str]:
        """Get chunkers applicable to a modality"""
        applicable = []
        for name, chunker_class in self._chunkers.items():
            if modality in chunker_class.get_applicable_modalities():
                applicable.append(name)
        return applicable

# Decorator for easy registration
def register_extractor(name: str):
    """Decorator to register an extractor"""
    def decorator(cls: Type[BaseExtractor]):
        ExtractorRegistry.get_instance().register(name, cls)
        return cls
    return decorator

def register_provider(name: str):
    """Decorator to register a provider"""
    def decorator(cls: Type[BaseProvider]):
        ProviderRegistry.get_instance().register(name, cls)
        return cls
    return decorator

def register_chunker(name: str):
    """Decorator to register a chunker"""
    def decorator(cls: Type[BaseChunker]):
        ChunkerRegistry.get_instance().register(name, cls)
        return cls
    return decorator
```
### 4.6 Usage Examples

#### Example 1: VLM Extractor with Page Chunker

```python
from nextract import ExtractionPipeline, ExtractionPlan, ExtractorConfig, ChunkerConfig, ProviderConfig

# Configure provider
provider = ProviderConfig(
    name="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",
    temperature=0.0
)

# Configure extractor
extractor_config = ExtractorConfig(
    name="vlm",  # Vision-Language Model
    provider=provider,
    extractor_params={
        'max_image_dimension': 2048,
        'image_quality': 95
    }
)

# Configure chunker (page-based for VLM)
chunker_config = ChunkerConfig(
    name="page",  # Only applicable chunker for VLM
    pages_per_chunk=3,  # Chunk 3 pages at a time
    page_overlap=1,  # 1 page overlap between chunks
    max_image_dimension=2048
)

# Create extraction plan
plan = ExtractionPlan(
    extractor=extractor_config,
    chunker=chunker_config,
    num_passes=2,
    include_confidence=True,
    include_citations=True
)

# Extract data
pipeline = ExtractionPipeline(plan)
result = pipeline.extract(
    document="long_contract.pdf",
    schema=my_schema,
    prompt="Extract contract details"
)
```

#### Example 2: Text Extractor with Semantic Chunker

```python
# Configure for text extraction
provider = ProviderConfig(
    name="openai",
    model="gpt-4o",
    api_key="your-key"
)

extractor_config = ExtractorConfig(
    name="text",  # Text-only extractor
    provider=provider
)

chunker_config = ChunkerConfig(
    name="semantic",  # Applicable to TEXT modality
    chunk_size=2000,
    chunk_overlap=200,
    preserve_sections=True,
    respect_sentence_boundaries=True
)

plan = ExtractionPlan(
    extractor=extractor_config,
    chunker=chunker_config,
    num_passes=1
)

pipeline = ExtractionPipeline(plan)
result = pipeline.extract(
    document="research_paper.pdf",
    schema=paper_schema
)
```

#### Example 3: Third-Party Extractor (Textract)

```python
provider = ProviderConfig(
    name="aws",
    model="textract"
)

extractor_config = ExtractorConfig(
    name="textract",
    provider=provider,
    extractor_params={
        'aws_access_key': 'your-key',
        'aws_secret_key': 'your-secret',
        'region': 'us-east-1'
    }
)

chunker_config = ChunkerConfig(
    name="page",
    pages_per_chunk=10
)

plan = ExtractionPlan(
    extractor=extractor_config,
    chunker=chunker_config
)

pipeline = ExtractionPipeline(plan)
result = pipeline.extract(document="form.pdf")
```

#### Example 4: Custom Extractor

```python
from nextract.extractors import BaseExtractor, register_extractor, Modality

@register_extractor("custom_llama_index")
class LlamaIndexExtractor(BaseExtractor):
    """Custom extractor using LlamaIndex"""
    
    SUPPORTED_PROVIDERS = ["openai", "anthropic", "local"]
    
    def get_modality(self) -> Modality:
        return Modality.TEXT
    
    def get_supported_providers(self) -> List[str]:
        return self.SUPPORTED_PROVIDERS
    
    def validate_config(self, config: ExtractorConfig) -> bool:
        required = ['index_path', 'retriever_mode']
        for param in required:
            if param not in config.extractor_params:
                raise ValueError(f"LlamaIndex requires '{param}'")
        return True
    
    def run(self, input_data, provider, **kwargs):
        # Custom LlamaIndex implementation
        from llama_index import VectorStoreIndex, SimpleDirectoryReader
        
        index_path = self.config.extractor_params['index_path']
        # ... implementation
        pass

# Use custom extractor
extractor_config = ExtractorConfig(
    name="custom_llama_index",
    provider=ProviderConfig(name="openai", model="gpt-4"),
    extractor_params={
        'index_path': './my_index',
        'retriever_mode': 'similarity'
    }
)
```

## 5. Naming Conventions & Standards

### 5.1 Module Naming

```
nextract/
├── core/                    # Core abstractions and public API
├── ingest/                  # File loading and conversions
├── parse/                   # OCR and layout parsing
├── chunking/                # Chunkers
│   ├── page_chunker.py
│   ├── semantic_chunker.py
│   ├── table_aware_chunker.py
│   └── section_chunker.py
├── extractors/                 # Extractor implementations
│   ├── vlm_extractor.py
│   ├── text_extractor.py
│   ├── ocr_extractor.py
│   ├── textract_extractor.py
│   └── llamaindex_extractor.py
├── providers/               # Provider implementations
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── google_provider.py
│   └── aws_provider.py
├── schema/                  # Schema utilities
├── validate/                # Validation logic
├── output/                  # Output formatting
├── pipeline/                # Orchestration and routing
├── registry/                # Plugin registry
└── telemetry/               # Logging, metrics, tracing
```

### 5.2 Class Naming

```python
# Extractors: [Purpose]Extractor
VLMExtractor
TextExtractor
OCRExtractor
HybridExtractor
TextractExtractor

# Providers: [Service]Provider
OpenAIProvider
AnthropicProvider
GoogleProvider
AWSProvider
LocalProvider

# Chunkers: [Purpose]Chunker
PageBasedChunker  # or PageChunker
SemanticChunker   # or SemanticChunker
TableAwareChunker # or TableAwareChunker

# Pipelines: [Purpose]Pipeline
ExtractionPipeline
BatchPipeline

# Config: [Component]Config
ProviderConfig
ExtractorConfig
ChunkerConfig
ExtractionPlan

# Results: [Component]Result
ExtractorResult
ExtractionResult
ValidationResult
```

### 5.3 Extractor Naming

```python
# Initialization
def initialize(config) -> None

# Execution
def run(input_data, provider, **kwargs) -> Result

# Validation
def validate(data, schema) -> ValidationResult
def validate_config(config) -> bool

# Provider interface
def generate(request: ProviderRequest) -> ProviderResponse
def supports_vision() -> bool
def supports_structured_output() -> bool

# Capabilities
def get_modality() -> Modality
def get_supported_providers() -> List[str]
def get_applicable_modalities() -> List[Modality]

# Registry
def register(name, class) -> None
def get(name) -> Optional[Type]
def list_extractors() -> List[str]
```

### 5.4 Configuration Parameter Naming

```python
# Provider config
name: str                    # Provider identifier
model: str                   # Model identifier
api_key: str                # Authentication
api_base: str               # Base URL
timeout: int                # Request timeout
max_retries: int            # Retry attempts
temperature: float          # Model temperature
max_tokens: int             # Max output tokens

# Extractor config
name: str         # Extractor identifier
provider: ProviderConfig    # Provider configuration
fallback_provider: ...      # Fallback provider
extractor_params: dict      # Extractor-specific params
enable_caching: bool     # Cache per-chunk outputs
batch_size: int          # Provider batch size
modality: Optional[Modality]  # Optional override for custom extractors

# Chunker config (Visual Modality)
name: str                   # Chunker identifier
pages_per_chunk: int        # Pages per chunk
page_overlap: int           # Overlap in pages
max_image_dimension: int    # Max image size
image_quality: int          # Image quality (0-100)

# Chunker config (Text Modality)
chunk_size: int             # Size in tokens
chunk_overlap: int          # Overlap in tokens
preserve_tables: bool       # Don't split tables
preserve_sections: bool     # Don't split sections
respect_sentence_boundaries: bool
```

## 6. Project Structure

```
nextract/
├── __init__.py
├── version.py
│
├── core/
│   ├── __init__.py
│   ├── base.py                    # Base interfaces and shared types
│   ├── config.py                  # Extractor/Provider/Chunker config + ExtractionPlan
│   ├── artifacts.py               # DocumentArtifact, DocumentChunk, FieldResult
│   └── exceptions.py              # Custom exceptions
│
├── ingest/
│   ├── __init__.py
│   ├── loaders/
│   │   ├── pdf_loader.py
│   │   ├── docx_loader.py
│   │   ├── pptx_loader.py
│   │   └── image_loader.py
│   ├── converters/
│   │   ├── docx_to_pdf.py
│   │   ├── pptx_to_pdf.py
│   │   └── image_converter.py
│   ├── classifiers/
│   │   └── document_classifier.py
│   └── validators/
│       └── document_validator.py
│
├── parse/
│   ├── __init__.py
│   ├── ocr/
│   ├── layout/
│   └── text/
│
├── chunking/
│   ├── __init__.py
│   ├── chunkers/
│   │   ├── page_chunker.py        # VISUAL modality
│   │   ├── semantic_chunker.py    # TEXT modality
│   │   ├── table_aware_chunker.py # TEXT modality
│   │   └── hybrid_chunker.py      # HYBRID modality
│   └── policies.py
│
├── extractors/
│   ├── __init__.py
│   ├── vlm_extractor.py         # Vision-Language Model
│   ├── text_extractor.py        # Text-only extractor
│   ├── ocr_extractor.py         # OCR extractor
│   ├── hybrid_extractor.py      # Hybrid approach
│   ├── textract_extractor.py    # AWS Textract
│   ├── llamaindex_extractor.py  # LlamaIndex integration
│   └── custom_extractor_template.py
│
├── providers/
│   ├── __init__.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── google_provider.py
│   ├── azure_provider.py
│   ├── aws_provider.py
│   ├── local_provider.py          # Ollama, vLLM
│   └── custom_provider_template.py
│
├── schema/
│   ├── __init__.py
│   ├── inference.py
│   └── validation.py
│
├── validate/
│   ├── __init__.py
│   ├── rules.py
│   └── scoring.py
│
├── output/
│   ├── __init__.py
│   ├── formatters.py
│   └── citations.py
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── router.py
│   └── retries.py
│
├── registry/
│   ├── __init__.py
│   ├── provider_registry.py
│   ├── extractor_registry.py
│   └── chunker_registry.py
│
├── telemetry/
│   ├── __init__.py
│   ├── logging.py
│   └── metrics.py
│
├── cli/
│   ├── __init__.py
│   ├── main.py
│   └── commands/
│       ├── extract.py
│       ├── convert.py
│       ├── suggest_schema.py
│       └── batch.py
│
├── templates/
│   ├── schemas/
│   └── prompts/
│
└── tests/
    ├── unit/
    └── integration/
```

## 7. Feature Matrix

### 7.1 Extractor Compatibility Matrix

| Extractor | Visual Modality | Text Modality | Hybrid Modality | Supported Providers |
|-----------|-------------|-----------|-------------|---------------------|
| VLMExtractor | ✅ | ❌ | ✅ | OpenAI, Anthropic, Google, Azure, Local |
| TextExtractor | ❌ | ✅ | ✅ | OpenAI, Anthropic, Google, Azure, Local, Cohere |
| OCRExtractor | ✅ | ❌ | ✅ | Tesseract, EasyOCR, PaddleOCR |
| HybridExtractor | ✅ | ✅ | ✅ | Any combination |
| TextractExtractor | ✅ | ❌ | ❌ | AWS only |
| LlamaIndexExtractor | ❌ | ✅ | ❌ | OpenAI, Anthropic, Local |

### 7.2 Chunker Matrix

| Chunker | Visual Modality | Text Modality | Hybrid Modality | Best For |
|----------|-------------|-----------|-------------|-----------|
| Page | ✅ | ❌ | ✅ | VLM extractors, scanned docs |
| Fixed-size | ❌ | ✅ | ✅ | General text extraction |
| Semantic | ❌ | ✅ | ✅ | Preserving context |
| Table-aware | ❌ | ✅ | ✅ | Documents with tables |
| Section-based | ❌ | ✅ | ✅ | Structured documents |

### 7.3 Provider Capabilities Matrix

| Provider | Vision | Structured Output | Streaming | Local | Max Tokens |
|----------|--------|-------------------|-----------|-------|------------|
| OpenAI | ✅ | ✅ | ✅ | ❌ | 128k |
| Anthropic | ✅ | ✅ | ✅ | ❌ | 200k |
| Google | ✅ | ✅ | ✅ | ❌ | 1M |
| Azure | ✅ | ✅ | ✅ | ❌ | 128k |
| Local (Ollama) | ✅* | ✅ | ✅ | ✅ | Varies |
| AWS Textract | ✅ | ✅ | ❌ | ❌ | N/A |

*Depends on model

## 8. API Examples

### 8.1 Simple Extraction

```python
from nextract import extract_simple

# Simplest usage
result = extract_simple(
    document="invoice.pdf",
    schema={"vendor": "string", "total": "number"},
    provider="anthropic"
)
```

### 8.2 Advanced Extraction with Full Control

```python
from nextract import ExtractionPipeline
from nextract.config import ExtractionPlan, ExtractorConfig, ChunkerConfig, ProviderConfig

# Full control over extraction
plan = ExtractionPlan(
    extractor=ExtractorConfig(
        name="vlm",
        provider=ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="sk-...",
            temperature=0.0,
            max_tokens=4096
        ),
        fallback_provider=ProviderConfig(
            name="openai",
            model="gpt-4o"
        ),
        extractor_params={
            'enable_ocr_fallback': True,
            'image_preprocessing': 'enhance'
        }
    ),
    chunker=ChunkerConfig(
        name="page",
        pages_per_chunk=5,
        page_overlap=1,
        max_image_dimension=2048
    ),
    num_passes=2,
    include_confidence=True,
    include_citations=True,
    retry_on_failure=True,
    max_retries=3
)

pipeline = ExtractionPipeline(plan)
result = pipeline.extract(
    document="complex_document.pdf",
    schema=complex_schema,
    prompt="Extract all contract terms and conditions"
)

# Access results
print(f"Extracted data: {result.data}")
print(f"Overall confidence: {result.metadata.confidence}")
print(f"Field metadata: {result.field_metadata}")
```

### 8.3 Batch Extraction

```python
from nextract import BatchPipeline, ExtractionPlan

plan = ExtractionPlan(...)  # See earlier examples

batch_pipeline = BatchPipeline(
    plan=plan,
    max_workers=4,
    enable_suggestions=True,
    progress_callback=lambda p: print(f"Progress: {p}%")
)

results = batch_pipeline.extract_batch(
    documents=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    schema=schema
)

# Review suggestions for schema improvement
if results.suggestions:
    for suggestion in results.suggestions:
        print(f"Suggestion: {suggestion.description}")
        print(f"Confidence improvement: +{suggestion.impact}%")
```

### 8.4 Schema Suggestion

```python
from nextract import SchemaGenerator

generator = SchemaGenerator(
    provider=ProviderConfig(name="anthropic", model="claude-3-5-sonnet")
)

suggested_schema = generator.suggest_schema(
    sample_documents=["sample1.pdf", "sample2.pdf", "sample3.pdf"],
    prompt="Extract vendor information, line items with quantities and prices, and total amount",
    examples=[
        {"vendor": "Acme Corp", "items": [{"name": "Widget", "qty": 5}]}
    ]
)

# Refine and save
generator.save_schema(suggested_schema, "invoice_schema.json")
```

### 8.5 Custom Extractor Implementation

```python
from nextract.extractors import BaseExtractor, register_extractor
from nextract.core import Modality, ExtractorResult

@register_extractor("chandra_model")
class ChandraModelExtractor(BaseExtractor):
    """Custom extractor for Chandra Model API"""
    
    SUPPORTED_PROVIDERS = ["chandra"]
    
    def get_modality(self) -> Modality:
        return Modality.VISUAL
    
    def get_supported_providers(self) -> List[str]:
        return self.SUPPORTED_PROVIDERS
    
    def validate_config(self, config: ExtractorConfig) -> bool:
        required_params = ['api_endpoint', 'api_key']
        for param in required_params:
            if param not in config.extractor_params:
                raise ValueError(f"Chandra requires '{param}' in extractor_params")
        return True
    
    def run(self, input_data, provider, **kwargs) -> ExtractorResult:
        """Run extraction using Chandra Model API"""
        import requests
        
        endpoint = self.config.extractor_params['api_endpoint']
        api_key = self.config.extractor_params['api_key']
        
        results = []
        for chunk in input_data:
            response = requests.post(
                f"{endpoint}/analyze",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "images": chunk.images,
                    "prompt": kwargs.get('prompt'),
                    "schema": kwargs.get('schema')
                }
            )
            results.append(response.json())
        
        return ExtractorResult(
            name="chandra_model",
            provider_name="chandra",
            results=results,
            metadata={'custom_extractor': True}
        )

# Use custom extractor
plan = ExtractionPlan(
    extractor=ExtractorConfig(
        name="chandra_model",
        provider=ProviderConfig(name="chandra", model="chandra-v1"),
        extractor_params={
            'api_endpoint': 'https://api.chandra.ai',
            'api_key': 'your-key'
        }
    ),
    chunker=ChunkerConfig(name="page", pages_per_chunk=10)
)
```

### 8.6 Custom Provider Implementation

```python
from nextract.providers import BaseProvider, register_provider
from nextract.core import ProviderRequest, ProviderResponse

@register_provider("my_custom_llm")
class MyCustomProvider(BaseProvider):
    """Custom provider for proprietary LLM"""
    
    def initialize(self, config: ProviderConfig) -> None:
        self.endpoint = config.api_base
        self.api_key = config.api_key
        self.model = config.model
    
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Call custom LLM API"""
        import requests
        
        response = requests.post(
            f"{self.endpoint}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": request.messages,
                "temperature": request.options.get('temperature', 0.0),
                "max_tokens": request.options.get('max_tokens', 1000)
            }
        )
        return ProviderResponse(text=response.json()['text'], raw=response.json())
    
    def supports_vision(self) -> bool:
        return self.model.startswith("vision-")
    
    def supports_structured_output(self) -> bool:
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'vision': self.supports_vision(),
            'structured_output': True,
            'streaming': False,
            'max_tokens': 4096
        }

# Register and use
provider_config = ProviderConfig(
    name="my_custom_llm",
    model="vision-model-v2",
    api_base="https://api.mycompany.com",
    api_key="secret-key"
)
```

### 8.7 Modality-Aware Configuration

```python
from nextract import get_available_chunkers
from nextract.core import Modality

# Check available chunkers for an extractor
extractor_name = "vlm"
extractor_class = ExtractorRegistry.get_instance().get(extractor_name)
modality = extractor_class.get_modality()

# Get applicable chunkers
chunkers = ChunkerRegistry.get_instance().get_chunkers_for_modality(modality)
print(f"Available chunkers for {modality.value} modality: {chunkers}")
# Output: ['page']

# For text modality
text_extractor = ExtractorRegistry.get_instance().get("text")
text_modality = text_extractor.get_modality()
text_chunkers = ChunkerRegistry.get_instance().get_chunkers_for_modality(text_modality)
print(f"Available chunkers for {text_modality.value} modality: {text_chunkers}")
# Output: ['semantic', 'table_aware', 'section', 'fixed_size']
```

## 9. Plan Validation Rules

### 9.1 Rule-Based Validation

The system enforces strict validation rules based on modalities:

```python
class PlanValidator:
    """Validates extraction plans based on modalities and compatibility"""
    
    @staticmethod
    def validate_extraction_plan(plan: ExtractionPlan) -> ValidationResult:
        """Comprehensive plan validation"""
        
        # 1. Validate extractor config
        plan.extractor.validate()
        
        # 2. Get extractor's modality
        extractor_registry = ExtractorRegistry.get_instance()
        extractor_class = extractor_registry.get(plan.extractor.name)
        
        if not extractor_class:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown extractor: {plan.extractor.name}"]
            )
        
        modality = extractor_class.get_modality()
        
        # 3. Validate chunker is applicable to modality
        chunker_registry = ChunkerRegistry.get_instance()
        chunker_class = chunker_registry.get(plan.chunker.name)
        
        if chunker_class:
            applicable_modalities = chunker_class.get_applicable_modalities()
            if modality not in applicable_modalities:
                return ValidationResult(
                    valid=False,
                    errors=[
                        f"Chunker '{plan.chunker.name}' is not "
                        f"applicable to modality '{modality.value}'. "
                        f"Available chunkers: "
                        f"{chunker_registry.get_chunkers_for_modality(modality)}"
                    ]
                )
        
        # 4. Validate provider supports extractor
        supported_providers = extractor_class.get_supported_providers()
        if plan.extractor.provider.name not in supported_providers:
            return ValidationResult(
                valid=False,
                errors=[
                    f"Extractor '{plan.extractor.name}' does not support "
                    f"provider '{plan.extractor.provider.name}'. "
                    f"Supported providers: {supported_providers}"
                ]
            )
        
        # 5. Validate provider capabilities
        provider_registry = ProviderRegistry.get_instance()
        provider_class = provider_registry.get(plan.extractor.provider.name)
        
        if provider_class and modality == Modality.VISUAL:
            if not provider_class.supports_vision():
                return ValidationResult(
                    valid=False,
                    errors=[
                        f"Provider '{plan.extractor.provider.name}' does not "
                        f"support vision, but extractor requires VISUAL modality"
                    ]
                )
        
        # 6. Validate chunker config parameters
        plan.chunker.validate(modality)
        
        return ValidationResult(valid=True, errors=[])
```

### 9.2 Automatic Capability Detection

```python
class CapabilityDetector:
    """Detects and reports available capabilities based on a plan"""
    
    @staticmethod
    def detect_capabilities(plan: ExtractionPlan) -> Dict[str, Any]:
        """Detect what the system can do with a given plan"""
        
        extractor_class = ExtractorRegistry.get_instance().get(
            plan.extractor.name
        )
        provider_class = ProviderRegistry.get_instance().get(
            plan.extractor.provider.name
        )
        
        capabilities = {
            'modality': extractor_class.get_modality().value,
            'supported_chunkers': ChunkerRegistry
                .get_instance()
                .get_chunkers_for_modality(extractor_class.get_modality()),
            'provider_capabilities': provider_class.get_capabilities() if provider_class else {},
            'multi_pass_extraction': plan.num_passes > 1,
            'has_fallback': plan.extractor.fallback_provider is not None,
            'confidence_scoring': plan.include_confidence,
            'citation_tracking': plan.include_citations
        }
        
        return capabilities
```

## 10. CLI Examples

### 10.1 Basic CLI Usage

```bash
# Extract with auto-detection
nextract extract invoice.pdf \
    --schema schema.json \
    --provider anthropic \
    --output result.json

# Extract with explicit extractor and chunker
nextract extract contract.pdf \
    --extractor vlm \
    --provider anthropic \
    --model claude-3-5-sonnet \
    --chunker page \
    --pages-per-chunk 3 \
    --schema contract_schema.json \
    --output contract_data.json

# Text-based extraction with semantic chunker
nextract extract research_paper.pdf \
    --extractor text \
    --provider openai \
    --chunker semantic \
    --chunk-size 2000 \
    --schema paper_schema.json

# Use custom extractor
nextract extract form.pdf \
    --extractor textract \
    --provider aws \
    --extractor-params aws_access_key=XXX aws_secret_key=YYY region=us-east-1 \
    --chunker page \
    --pages-per-chunk 10
```
### 10.2 List Available Options

```bash
# List all extractors
nextract list extractors
# Output:
# Available extractors:
# - vlm (Visual modality, providers: openai, anthropic, google, azure, local)
# - text (Text modality, providers: openai, anthropic, google, azure, local, cohere)
# - ocr (Visual modality, providers: tesseract, easyocr, paddleocr)
# - textract (Visual modality, providers: aws)
# - llamaindex (Text modality, providers: openai, anthropic, local)

# List chunkers for an extractor
nextract list chunkers --extractor vlm
# Output:
# Available chunkers for 'vlm' extractor (Visual modality):
# - page: Chunk by PDF pages (pages_per_chunk, page_overlap)

nextract list chunkers --extractor text
# Output:
# Available chunkers for 'text' extractor (Text modality):
# - semantic: Semantic boundary-aware chunking
# - table_aware: Preserve table integrity
# - section: Section-based chunking
# - fixed_size: Fixed token-size chunks

# Check provider capabilities
nextract check-provider anthropic
# Output:
# Provider: anthropic
# Supports vision: Yes
# Supports structured output: Yes
# Max tokens: 200,000
# Compatible extractors: vlm, text, hybrid
```
### 10.3 Plan Validation

```bash
# Validate a plan file before running
nextract validate-config plan.yaml
# Output:
# ✓ Extractor 'vlm' is valid
# ✓ Provider 'anthropic' supports extractor 'vlm'
# ✓ Chunker 'page' is applicable to Visual modality
# ✓ All plan settings are valid

# Validate with detailed output
nextract validate-config plan.yaml --verbose
# Output:
# Plan Summary:
# - Extractor: vlm (Visual modality)
# - Provider: anthropic (claude-3-5-sonnet)
# - Chunker: page (3 pages per chunk, 1 page overlap)
# - Available chunkers: ['page']
# - Provider capabilities: vision=True, structured_output=True
# - Multi-pass extraction: No (1 pass)
# ✓ Plan is valid and ready to use
```

## 11. Error Handling & Messages

### 11.1 Clear Error Messages

#### Example 1: Invalid chunker for extractor

```
PlanError: Chunker 'semantic' is not applicable to modality 'visual'.

The 'vlm' extractor operates in VISUAL modality, which only supports:
- page: Chunk by PDF pages

Available text-modality chunkers like 'semantic', 'table_aware', and 'section'
can only be used with extractors that operate in TEXT modality, such as:
- text
- llamaindex

Suggestion: Change chunker to 'page' or use a text-modality extractor.
```

#### Example 2: Provider doesn't support extractor

```
PlanError: Extractor 'vlm' does not support provider 'cohere'.

The 'vlm' extractor requires vision-capable providers. Supported providers:
- openai (gpt-4o, gpt-4-vision)
- anthropic (claude-3-5-sonnet, claude-4)
- google (gemini-pro-vision)
- azure (gpt-4o via Azure OpenAI)
- local (vision-capable models via Ollama)

Suggestion: Change provider to one of the supported options above.
```

#### Example 3: Provider doesn't support vision

```
PlanError: Provider 'cohere' does not support vision capabilities.

The 'vlm' extractor requires VISUAL modality, but the configured 
provider 'cohere' (command-r-plus) does not support image inputs.

Suggestions:
1. Use a vision-capable provider: openai, anthropic, google
2. Switch to 'text' extractor if you want to use Cohere
3. Use 'hybrid' extractor with a different provider for vision tasks
```
### 11.2 Helpful Suggestions

```python
class PlanAssistant:
    """Provides helpful suggestions when plan errors occur"""
    
    @staticmethod
    def suggest_alternatives(error: PlanError) -> List[str]:
        """Provide actionable suggestions based on error"""
        
        suggestions = []
        
        if error.type == "incompatible_chunker":
            extractor = error.context['extractor']
            current_chunker = error.context['chunker']
            
            # Get applicable chunkers
            applicable = ChunkerRegistry.get_instance() \
                .get_chunkers_for_modality(extractor.get_modality())
            
            suggestions.append(
                f"Change chunker to one of: {', '.join(applicable)}"
            )
            
            # Suggest alternative extractor
            if current_chunker in ["semantic", "table_aware"]:
                suggestions.append(
                    "Or switch to a TEXT modality extractor: 'text', 'llamaindex'"
                )
        
        elif error.type == "unsupported_provider":
            extractor = error.context['extractor']
            current_provider = error.context['provider']
            
            supported = extractor.get_supported_providers()
            suggestions.append(
                f"Use one of these supported providers: {', '.join(supported)}"
            )
        
        return suggestions
```

## 12. Testing Strategy

### 12.1 Unit Tests

```python
# tests/unit/test_plan_validation.py
def test_vlm_extractor_with_page_chunker_valid():
    """VLM extractor with page chunker should be valid"""
    config = ExtractionPlan(
        extractor=ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(name="anthropic", model="claude-3-5-sonnet")
        ),
        chunker=ChunkerConfig(name="page", pages_per_chunk=3)
    )
    assert config.validate() == True

def test_vlm_extractor_with_semantic_chunker_invalid():
    """VLM extractor with semantic chunker should raise error"""
    config = ExtractionPlan(
        extractor=ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(name="anthropic", model="claude-3-5-sonnet")
        ),
        chunker=ChunkerConfig(name="semantic")  # Invalid for visual modality
    )
    with pytest.raises(PlanError) as exc_info:
        config.validate()
    
    assert "not applicable to modality 'visual'" in str(exc_info.value)

def test_text_extractor_with_semantic_chunker_valid():
    """Text extractor with semantic chunker should be valid"""
    config = ExtractionPlan(
        extractor=ExtractorConfig(
            name="text",
            provider=ProviderConfig(name="openai", model="gpt-4")
        ),
        chunker=ChunkerConfig(name="semantic", chunk_size=2000)
    )
    assert config.validate() == True

def test_extractor_provider_compatibility():
    """Extractor and provider must be compatible"""
    config = ExtractionPlan(
        extractor=ExtractorConfig(
            name="textract",
            provider=ProviderConfig(name="openai", model="gpt-4")  # Invalid
        ),
        chunker=ChunkerConfig(name="page")
    )
    with pytest.raises(PlanError) as exc_info:
        config.validate()
    
    assert "does not support provider" in str(exc_info.value)
```
### 12.2 Integration Tests

```python
# tests/integration/test_extractor_provider_integration.py
@pytest.mark.integration
def test_vlm_extractor_with_anthropic_provider():
    """Test VLM extractor with Anthropic provider end-to-end"""
    config = ExtractionPlan(
        extractor=ExtractorConfig(
            name="vlm",
            provider=ProviderConfig(
                name="anthropic",
                model="claude-3-5-sonnet",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        ),
        chunker=ChunkerConfig(name="page", pages_per_chunk=2)
    )
    
    pipeline = ExtractionPipeline(config)
    result = pipeline.extract(
        document="tests/fixtures/sample_invoice.pdf",
        schema=INVOICE_SCHEMA
    )
    
    assert result.data is not None
    assert result.metadata.confidence > 0.7
    assert result.metadata.modality == "visual"

@pytest.mark.integration
def test_text_extractor_with_multiple_chunkers():
    """Test text extractor with different chunkers"""
    chunkers = ["semantic", "table_aware", "section", "fixed_size"]
    
    for chunker in chunkers:
        config = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4")
            ),
            chunker=ChunkerConfig(name=chunker, chunk_size=2000)
        )
        
        pipeline = ExtractionPipeline(config)
        result = pipeline.extract(
            document="tests/fixtures/research_paper.pdf",
            schema=PAPER_SCHEMA
        )
        
        assert result.data is not None
        print(f"Chunker '{chunker}': {len(result.metadata.chunks)} chunks")
```

## 13. Documentation Structure
### 13.1 Getting Started Guide
# Getting Started with Nextract

## Understanding Modalities

Nextract operates in three modalities:

### Visual Modality
- **Used by**: VLM extractors, OCR extractors, Textract
- **Input**: Documents converted to images (PDF pages → images)
- **Chunkers**: Page-based only (how many pages per chunk, overlap)
- **Best for**: Scanned documents, complex layouts, forms, images

### Text Modality
- **Used by**: Text extractors, LlamaIndex extractors
- **Input**: Extracted text from documents
- **Chunkers**: Semantic, table-aware, section-based, fixed-size
- **Best for**: Text-heavy documents, articles, reports

### Hybrid Modality
- **Used by**: Hybrid extractors
- **Input**: Both images and text
- **Chunkers**: Combination of chunkers
- **Best for**: Documents with mixed content

## Choosing the Right Plan

### For Invoices and Forms → Use VLM
```python
config = ExtractionPlan(
    extractor=ExtractorConfig(name="vlm", ...),
    chunker=ChunkerConfig(name="page", pages_per_chunk=5)
)
```

### For Research Papers → Use Text
```python
config = ExtractionPlan(
    extractor=ExtractorConfig(name="text", ...),
    chunker=ChunkerConfig(name="semantic", chunk_size=2000)
)
```

### For Complex Documents → Use Hybrid
```python
config = ExtractionPlan(
    extractor=ExtractorConfig(name="hybrid", ...),
    chunker=ChunkerConfig(name="hybrid", ...)
)
```
