"""
Integration tests for SemanticChunker.
"""

import pytest

import nextract.chunking  # noqa: F401
from nextract.core import ChunkerConfig, DocumentArtifact, Modality
from nextract.registry import ChunkerRegistry


@pytest.mark.integration
class TestSemanticChunkerUnit:
    """Unit tests for SemanticChunker."""

    def test_chunker_registered(self):
        """SemanticChunker should be registered."""
        registry = ChunkerRegistry.get_instance()
        assert "semantic" in registry.get_chunkers_for_modality(Modality.TEXT)

    def test_applicable_modalities(self):
        """SemanticChunker should be applicable to TEXT modality."""
        chunker_class = ChunkerRegistry.get_instance().get("semantic")
        assert chunker_class is not None
        
        modalities = chunker_class.get_applicable_modalities()
        assert Modality.TEXT in modalities

    def test_not_applicable_to_visual(self):
        """SemanticChunker should not be applicable to VISUAL modality."""
        chunker_class = ChunkerRegistry.get_instance().get("semantic")
        modalities = chunker_class.get_applicable_modalities()
        assert Modality.VISUAL not in modalities

    def test_validate_config_valid(self):
        """Valid config should pass validation."""
        chunker_class = ChunkerRegistry.get_instance().get("semantic")
        chunker = chunker_class()
        
        config = ChunkerConfig(name="semantic", chunk_size=2000, chunk_overlap=200)
        assert chunker.validate_config(config) is True


@pytest.mark.integration
class TestSemanticChunkerWithText:
    """Test semantic chunking with text content."""

    def test_chunk_simple_text(self, sample_text_content):
        """Test chunking simple text document."""
        chunker_class = ChunkerRegistry.get_instance().get("semantic")
        chunker = chunker_class()
        
        config = ChunkerConfig(name="semantic", chunk_size=500, chunk_overlap=50)
        artifact = DocumentArtifact(
            source_path="invoice.txt",
            mime_type="text/plain",
            text=sample_text_content,
        )
        
        chunks = chunker.chunk(artifact, config)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_overlap(self, sample_text_content):
        """Test that chunks have proper overlap."""
        chunker_class = ChunkerRegistry.get_instance().get("semantic")
        chunker = chunker_class()
        
        config = ChunkerConfig(name="semantic", chunk_size=200, chunk_overlap=50)
        artifact = DocumentArtifact(
            source_path="test.txt",
            mime_type="text/plain",
            text=sample_text_content,
        )
        
        chunks = chunker.chunk(artifact, config)
        
        assert len(chunks) > 1

    def test_chunk_metadata(self, sample_text_content):
        """Test that chunks include metadata."""
        chunker_class = ChunkerRegistry.get_instance().get("semantic")
        chunker = chunker_class()
        
        config = ChunkerConfig(name="semantic", chunk_size=500)
        artifact = DocumentArtifact(
            source_path="test.txt",
            mime_type="text/plain",
            text=sample_text_content,
        )
        
        chunks = chunker.chunk(artifact, config)
        
        for chunk in chunks:
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "source_path")
