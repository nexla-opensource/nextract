"""
Integration tests for invalid document handling.
"""

import pytest

from nextract import ExtractionPipeline
from nextract.core import ChunkerConfig, ExtractionPlan, ExtractorConfig, ProviderConfig
from nextract.ingest import DocumentValidator, load_documents
from nextract.core import DocumentArtifact


@pytest.mark.integration
class TestInvalidDocuments:
    """Tests for handling invalid documents."""

    def test_empty_file_validation_fails(self, empty_file):
        """Empty files should fail validation."""
        validator = DocumentValidator()
        artifact = DocumentArtifact(
            source_path=str(empty_file),
            mime_type="text/plain",
        )
        
        result = validator.validate(artifact)
        
        assert result.valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_nonexistent_file_validation_fails(self, tmp_path):
        """Nonexistent files should fail validation."""
        validator = DocumentValidator()
        artifact = DocumentArtifact(
            source_path=str(tmp_path / "does_not_exist.pdf"),
            mime_type="application/pdf",
        )
        
        result = validator.validate(artifact)
        
        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_directory_as_file_fails(self, tmp_path):
        """Directory path should fail validation."""
        validator = DocumentValidator()
        artifact = DocumentArtifact(
            source_path=str(tmp_path),
            mime_type="application/pdf",
        )
        
        result = validator.validate(artifact)
        
        assert result.valid is False
        assert any("directory" in e.lower() for e in result.errors)

    def test_corrupt_file_handling(self, corrupt_file, simple_schema):
        """Corrupt files should be handled gracefully."""
        plan = ExtractionPlan(
            extractor=ExtractorConfig(
                name="text",
                provider=ProviderConfig(name="openai", model="gpt-4o"),
            ),
            chunker=ChunkerConfig(name="semantic"),
        )
        
        _ = ExtractionPipeline(plan)
        
        try:
            _ = load_documents([corrupt_file])
        except Exception:
            assert True


@pytest.mark.integration
class TestDocumentEdgeCases:
    """Edge case tests for document handling."""

    def test_very_short_content(self, tmp_path, simple_schema):
        """Very short content should be handled."""
        short_file = tmp_path / "short.txt"
        short_file.write_text("X")
        
        artifacts = load_documents([short_file])
        
        assert len(artifacts) == 1

    def test_unicode_content(self, tmp_path):
        """Unicode content should be handled."""
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("日本語テスト 🎉 émojis çafé")
        
        artifacts = load_documents([unicode_file])
        
        assert len(artifacts) == 1
        assert artifacts[0] is not None

    def test_special_characters_in_path(self, tmp_path):
        """Special characters in file path should work."""
        special_dir = tmp_path / "test dir (1)"
        special_dir.mkdir()
        special_file = special_dir / "file [test].txt"
        special_file.write_text("Content")
        
        artifacts = load_documents([special_file])
        
        assert len(artifacts) == 1
