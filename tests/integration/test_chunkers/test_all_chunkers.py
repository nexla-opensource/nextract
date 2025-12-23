"""
Integration tests for remaining chunkers (FixedSize, Section, TableAware, Hybrid).
"""

import pytest

import nextract.chunking  # noqa: F401
from nextract.core import Modality
from nextract.registry import ChunkerRegistry


@pytest.mark.integration
class TestFixedSizeChunkerUnit:
    """Unit tests for FixedSizeChunker."""

    def test_chunker_registered(self):
        """FixedSizeChunker should be registered."""
        registry = ChunkerRegistry.get_instance()
        chunker_class = registry.get("fixed_size")
        assert chunker_class is not None

    def test_applicable_modalities(self):
        """FixedSizeChunker should be applicable to TEXT modality."""
        chunker_class = ChunkerRegistry.get_instance().get("fixed_size")
        if chunker_class:
            modalities = chunker_class.get_applicable_modalities()
            assert Modality.TEXT in modalities


@pytest.mark.integration
class TestSectionChunkerUnit:
    """Unit tests for SectionChunker."""

    def test_chunker_registered(self):
        """SectionChunker should be registered."""
        registry = ChunkerRegistry.get_instance()
        chunker_class = registry.get("section")
        assert chunker_class is not None

    def test_applicable_modalities(self):
        """SectionChunker should be applicable to TEXT modality."""
        chunker_class = ChunkerRegistry.get_instance().get("section")
        if chunker_class:
            modalities = chunker_class.get_applicable_modalities()
            assert Modality.TEXT in modalities


@pytest.mark.integration
class TestTableAwareChunkerUnit:
    """Unit tests for TableAwareChunker."""

    def test_chunker_registered(self):
        """TableAwareChunker should be registered."""
        registry = ChunkerRegistry.get_instance()
        chunker_class = registry.get("table_aware")
        assert chunker_class is not None

    def test_applicable_modalities(self):
        """TableAwareChunker should be applicable to TEXT modality."""
        chunker_class = ChunkerRegistry.get_instance().get("table_aware")
        if chunker_class:
            modalities = chunker_class.get_applicable_modalities()
            assert Modality.TEXT in modalities


@pytest.mark.integration
class TestHybridChunkerUnit:
    """Unit tests for HybridChunker."""

    def test_chunker_registered(self):
        """HybridChunker should be registered."""
        registry = ChunkerRegistry.get_instance()
        chunker_class = registry.get("hybrid")
        assert chunker_class is not None

    def test_applicable_modalities(self):
        """HybridChunker should be applicable to HYBRID modality."""
        chunker_class = ChunkerRegistry.get_instance().get("hybrid")
        if chunker_class:
            modalities = chunker_class.get_applicable_modalities()
            assert Modality.HYBRID in modalities


@pytest.mark.integration
class TestChunkerRegistry:
    """Tests for chunker registry functionality."""

    def test_get_chunkers_for_text_modality(self):
        """Should return text-compatible chunkers."""
        registry = ChunkerRegistry.get_instance()
        chunkers = registry.get_chunkers_for_modality(Modality.TEXT)
        
        assert "semantic" in chunkers
        assert isinstance(chunkers, list)

    def test_get_chunkers_for_visual_modality(self):
        """Should return visual-compatible chunkers."""
        registry = ChunkerRegistry.get_instance()
        chunkers = registry.get_chunkers_for_modality(Modality.VISUAL)
        
        assert "page" in chunkers
        assert isinstance(chunkers, list)

    def test_get_chunkers_for_hybrid_modality(self):
        """Should return hybrid-compatible chunkers."""
        registry = ChunkerRegistry.get_instance()
        chunkers = registry.get_chunkers_for_modality(Modality.HYBRID)
        
        assert isinstance(chunkers, list)

    def test_get_nonexistent_chunker(self):
        """Should return None for unknown chunker."""
        registry = ChunkerRegistry.get_instance()
        assert registry.get("nonexistent") is None
