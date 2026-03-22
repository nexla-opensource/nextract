"""
Unit tests for core artifact classes.

Tests:
- DocumentArtifact
- DocumentChunk
- TextChunk
- ImageChunk
- ExtractionResult
- ValidationResult
"""


from nextract.core import (
    CharInterval,
    Citation,
    ConfidenceScore,
    DocumentArtifact,
    DocumentChunk,
    ExtractionResult,
    ExtractorResult,
    FieldResult,
    ImageChunk,
    Modality,
    TextChunk,
    ValidationResult,
)


class TestDocumentArtifact:
    """Tests for DocumentArtifact."""

    def test_basic_creation(self):
        """Create basic document artifact."""
        artifact = DocumentArtifact(
            source_path="/path/to/doc.pdf",
            mime_type="application/pdf",
        )
        assert artifact.source_path == "/path/to/doc.pdf"
        assert artifact.mime_type == "application/pdf"

    def test_with_content(self):
        """Create artifact with content."""
        artifact = DocumentArtifact(
            source_path="/path/to/doc.pdf",
            mime_type="application/pdf",
            content=b"PDF content bytes",
            text="Extracted text",
        )
        assert artifact.content == b"PDF content bytes"
        assert artifact.text == "Extracted text"

    def test_with_metadata(self):
        """Create artifact with metadata."""
        artifact = DocumentArtifact(
            source_path="/path/to/doc.pdf",
            mime_type="application/pdf",
            metadata={"pages": 10, "author": "Test"},
        )
        assert artifact.metadata["pages"] == 10
        assert artifact.metadata["author"] == "Test"

    def test_default_metadata_is_empty_dict(self):
        """Default metadata should be empty dict."""
        artifact = DocumentArtifact(
            source_path="/path/to/doc.pdf",
            mime_type="application/pdf",
        )
        assert artifact.metadata == {}


class TestCharInterval:
    """Tests for CharInterval."""

    def test_length(self):
        """Test interval length."""
        interval = CharInterval(start_pos=10, end_pos=50)
        assert len(interval) == 40

    def test_zero_length(self):
        """Test zero-length interval."""
        interval = CharInterval(start_pos=10, end_pos=10)
        assert len(interval) == 0


class TestDocumentChunk:
    """Tests for DocumentChunk."""

    def test_basic_creation(self):
        """Create basic document chunk."""
        chunk = DocumentChunk(
            id="chunk_0",
            content="Chunk content",
            source_path="/path/to/doc.pdf",
            modality=Modality.TEXT,
        )
        assert chunk.id == "chunk_0"
        assert chunk.content == "Chunk content"
        assert chunk.modality == Modality.TEXT

    def test_with_metadata(self):
        """Create chunk with metadata."""
        chunk = DocumentChunk(
            id="chunk_0",
            content="Content",
            source_path="/path/to/doc.pdf",
            modality=Modality.TEXT,
            metadata={"page": 1},
        )
        assert chunk.metadata["page"] == 1

    def test_with_char_interval(self):
        """Create chunk with character interval."""
        interval = CharInterval(start_pos=0, end_pos=100)
        chunk = DocumentChunk(
            id="chunk_0",
            content="Content",
            source_path="/path/to/doc.pdf",
            modality=Modality.TEXT,
            char_interval=interval,
        )
        assert chunk.char_interval.start_pos == 0
        assert chunk.char_interval.end_pos == 100


class TestTextChunk:
    """Tests for TextChunk."""

    def test_basic_creation(self):
        """Create basic text chunk."""
        chunk = TextChunk(
            id="text_0",
            text="This is the text content.",
            source_path="/path/to/doc.txt",
        )
        assert chunk.id == "text_0"
        assert chunk.text == "This is the text content."

    def test_with_metadata(self):
        """Create text chunk with metadata."""
        chunk = TextChunk(
            id="text_0",
            text="Content",
            source_path="/path/to/doc.txt",
            metadata={"section": "intro"},
        )
        assert chunk.metadata["section"] == "intro"


class TestImageChunk:
    """Tests for ImageChunk."""

    def test_basic_creation(self):
        """Create basic image chunk."""
        chunk = ImageChunk(
            id="img_0",
            images=[b"image1", b"image2"],
            source_path="/path/to/doc.pdf",
            page_range=(1, 3),
        )
        assert chunk.id == "img_0"
        assert len(chunk.images) == 2
        assert chunk.page_range == (1, 3)


class TestCitation:
    """Tests for Citation."""

    def test_basic_creation(self):
        """Create basic citation."""
        citation = Citation(
            source_path="/path/to/doc.pdf",
            chunk_id="chunk_0",
        )
        assert citation.source_path == "/path/to/doc.pdf"
        assert citation.chunk_id == "chunk_0"

    def test_with_span_and_snippet(self):
        """Create citation with span and snippet."""
        citation = Citation(
            source_path="/path/to/doc.pdf",
            chunk_id="chunk_0",
            span=CharInterval(10, 50),
            snippet="...relevant text...",
        )
        assert citation.span.start_pos == 10
        assert citation.snippet == "...relevant text..."


class TestConfidenceScore:
    """Tests for ConfidenceScore."""

    def test_basic_creation(self):
        """Create basic confidence score."""
        score = ConfidenceScore(value=0.95)
        assert score.value == 0.95
        assert score.rationale is None

    def test_with_rationale(self):
        """Create confidence score with rationale."""
        score = ConfidenceScore(
            value=0.85,
            rationale="Field was extracted from clear text",
        )
        assert score.rationale == "Field was extracted from clear text"


class TestFieldResult:
    """Tests for FieldResult."""

    def test_basic_creation(self):
        """Create basic field result."""
        result = FieldResult(name="invoice_number", value="INV-001")
        assert result.name == "invoice_number"
        assert result.value == "INV-001"

    def test_with_confidence(self):
        """Create field result with confidence."""
        result = FieldResult(
            name="total",
            value=500.00,
            confidence=ConfidenceScore(value=0.98),
        )
        assert result.confidence.value == 0.98

    def test_with_citations(self):
        """Create field result with citations."""
        citation = Citation(source_path="/doc.pdf", chunk_id="chunk_0")
        result = FieldResult(
            name="vendor",
            value="Acme Corp",
            citations=[citation],
        )
        assert len(result.citations) == 1


class TestExtractorResult:
    """Tests for ExtractorResult."""

    def test_basic_creation(self):
        """Create basic extractor result."""
        result = ExtractorResult(
            name="text",
            provider_name="openai",
            results=[{"chunk_id": "chunk_0", "response": {"field": "value"}}],
        )
        assert result.name == "text"
        assert result.provider_name == "openai"
        assert len(result.results) == 1


class TestExtractionResult:
    """Tests for ExtractionResult."""

    def test_basic_creation(self):
        """Create basic extraction result."""
        result = ExtractionResult(data={"field": "value"})
        assert result.data == {"field": "value"}

    def test_with_metadata(self):
        """Create extraction result with metadata."""
        result = ExtractionResult(
            data={"field": "value"},
            metadata={"chunks": 5, "provider": "openai"},
        )
        assert result.metadata["chunks"] == 5


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self):
        """Create valid validation result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []

    def test_invalid_result(self):
        """Create invalid validation result."""
        result = ValidationResult(
            valid=False,
            errors=["Missing required field: name"],
        )
        assert result.valid is False
        assert len(result.errors) == 1

    def test_with_warnings(self):
        """Create result with warnings."""
        result = ValidationResult(
            valid=True,
            warnings=["Optional field 'notes' is empty"],
        )
        assert len(result.warnings) == 1
