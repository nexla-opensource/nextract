"""Tests for audio/video passthrough chunking."""

from pathlib import Path

import pytest

from nextract.core import ChunkerConfig, DocumentArtifact, DocumentChunk, Modality
from nextract.mimetypes_map import is_audio, is_video, is_media


# -- mimetypes_map tests --


class TestMediaTypeDetection:
    @pytest.mark.parametrize("ext", [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus"])
    def test_is_audio(self, ext):
        assert is_audio(Path(f"file{ext}"))

    @pytest.mark.parametrize("ext", [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".m4v"])
    def test_is_video(self, ext):
        assert is_video(Path(f"file{ext}"))

    def test_is_media_audio(self):
        assert is_media(Path("track.mp3"))

    def test_is_media_video(self):
        assert is_media(Path("clip.mp4"))

    def test_not_audio(self):
        assert not is_audio(Path("file.pdf"))
        assert not is_audio(Path("file.txt"))
        assert not is_audio(Path("file.mp4"))

    def test_not_video(self):
        assert not is_video(Path("file.pdf"))
        assert not is_video(Path("file.mp3"))

    def test_case_insensitive(self):
        assert is_audio(Path("FILE.MP3"))
        assert is_video(Path("FILE.MP4"))


# -- Chunker passthrough tests --


def _make_media_artifact(tmp_path: Path, filename: str) -> DocumentArtifact:
    """Create a dummy media file and matching DocumentArtifact."""
    media_path = tmp_path / filename
    media_path.write_bytes(b"\x00\x01\x02\x03")
    return DocumentArtifact(
        source_path=str(media_path),
        mime_type="audio/mpeg" if filename.endswith(".mp3") else "video/mp4",
    )


def _assert_passthrough_chunk(chunks: list, source_path: str):
    """Verify a single passthrough DocumentChunk was produced."""
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.id == "chunk_0"
    assert chunk.source_path == source_path
    assert chunk.modality == Modality.VISUAL
    assert chunk.metadata.get("passthrough") is True
    assert "media_type" in chunk.metadata


class TestPageChunkerPassthrough:
    def test_audio_passthrough(self, tmp_path):
        from nextract.chunking.page_chunker import PageBasedChunker

        doc = _make_media_artifact(tmp_path, "track.mp3")
        chunker = PageBasedChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="page"))
        _assert_passthrough_chunk(chunks, doc.source_path)

    def test_video_passthrough(self, tmp_path):
        from nextract.chunking.page_chunker import PageBasedChunker

        doc = _make_media_artifact(tmp_path, "clip.mp4")
        chunker = PageBasedChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="page"))
        _assert_passthrough_chunk(chunks, doc.source_path)


class TestSemanticChunkerPassthrough:
    def test_audio_passthrough(self, tmp_path):
        from nextract.chunking.semantic_chunker import SemanticChunker

        doc = _make_media_artifact(tmp_path, "track.mp3")
        chunker = SemanticChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="semantic"))
        _assert_passthrough_chunk(chunks, doc.source_path)

    def test_video_passthrough(self, tmp_path):
        from nextract.chunking.semantic_chunker import SemanticChunker

        doc = _make_media_artifact(tmp_path, "clip.mp4")
        chunker = SemanticChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="semantic"))
        _assert_passthrough_chunk(chunks, doc.source_path)


class TestFixedSizeChunkerPassthrough:
    def test_audio_passthrough(self, tmp_path):
        from nextract.chunking.fixed_size_chunker import FixedSizeChunker

        doc = _make_media_artifact(tmp_path, "track.mp3")
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="fixed_size"))
        _assert_passthrough_chunk(chunks, doc.source_path)

    def test_video_passthrough(self, tmp_path):
        from nextract.chunking.fixed_size_chunker import FixedSizeChunker

        doc = _make_media_artifact(tmp_path, "clip.mp4")
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="fixed_size"))
        _assert_passthrough_chunk(chunks, doc.source_path)


class TestHybridChunkerPassthrough:
    def test_audio_passthrough(self, tmp_path):
        from nextract.chunking.hybrid_chunker import HybridChunker

        doc = _make_media_artifact(tmp_path, "track.mp3")
        chunker = HybridChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="hybrid"))
        _assert_passthrough_chunk(chunks, doc.source_path)

    def test_video_passthrough(self, tmp_path):
        from nextract.chunking.hybrid_chunker import HybridChunker

        doc = _make_media_artifact(tmp_path, "clip.mp4")
        chunker = HybridChunker()
        chunks = chunker.chunk(doc, ChunkerConfig(name="hybrid"))
        _assert_passthrough_chunk(chunks, doc.source_path)
