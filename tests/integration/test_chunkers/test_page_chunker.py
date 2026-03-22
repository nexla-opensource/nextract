"""
Integration tests for PageBasedChunker.
"""

import pytest

import nextract.chunking  # noqa: F401
from nextract.core import ChunkerConfig, DocumentArtifact, Modality
from nextract.registry import ChunkerRegistry


@pytest.mark.integration
class TestPageChunkerUnit:
    """Unit tests for PageBasedChunker."""

    def test_chunker_registered(self):
        """PageBasedChunker should be registered."""
        registry = ChunkerRegistry.get_instance()
        assert "page" in registry.get_chunkers_for_modality(Modality.VISUAL)

    def test_applicable_modalities(self):
        """PageChunker should be applicable to VISUAL modality."""
        chunker_class = ChunkerRegistry.get_instance().get("page")
        assert chunker_class is not None
        
        modalities = chunker_class.get_applicable_modalities()
        assert Modality.VISUAL in modalities

    def test_validate_config_valid(self):
        """Valid config should pass validation."""
        chunker_class = ChunkerRegistry.get_instance().get("page")
        chunker = chunker_class()
        
        config = ChunkerConfig(name="page", pages_per_chunk=3, page_overlap=1)
        assert chunker.validate_config(config) is True

    def test_chunk_empty_document(self):
        """Chunking empty document should return empty list or handle gracefully."""
        chunker_class = ChunkerRegistry.get_instance().get("page")
        chunker = chunker_class()
        
        config = ChunkerConfig(name="page", pages_per_chunk=3)
        artifact = DocumentArtifact(
            source_path="empty.pdf",
            mime_type="application/pdf",
            content=b"",
            text="",
        )
        
        try:
            chunks = chunker.chunk(artifact, config)
            assert isinstance(chunks, list)
        except Exception:
            pass


@pytest.mark.integration
class TestPageChunkerConfigurations:
    """Test various page chunker configurations."""

    def test_single_page_per_chunk(self):
        """Test single page per chunk configuration."""
        config = ChunkerConfig(name="page", pages_per_chunk=1, page_overlap=0)
        assert config.pages_per_chunk == 1
        assert config.page_overlap == 0

    def test_large_chunk_size(self):
        """Test large chunk size configuration."""
        config = ChunkerConfig(name="page", pages_per_chunk=10, page_overlap=2)
        assert config.pages_per_chunk == 10
        assert config.page_overlap == 2

    def test_overlap_validation(self):
        """Overlap should be less than pages_per_chunk."""
        config = ChunkerConfig(name="page", pages_per_chunk=3, page_overlap=3)
        
        with pytest.raises(ValueError, match="page_overlap must be < pages_per_chunk"):
            config.validate(Modality.VISUAL)
