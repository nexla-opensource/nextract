from nextract.chunking.hybrid_chunker import HybridChunker
from nextract.core import ChunkerConfig, DocumentArtifact, DocumentChunk, Modality, TextChunk


def test_hybrid_chunker_combines_visual_and_text_chunks_for_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    document = DocumentArtifact(
        source_path=str(pdf_path),
        mime_type="application/pdf",
    )
    chunker = HybridChunker()

    visual_chunk = DocumentChunk(
        id="chunk_0",
        content=b"%PDF-1.4\n",
        source_path=str(pdf_path),
        modality=Modality.VISUAL,
        metadata={"page_range": (1, 1)},
    )
    text_chunk = TextChunk(
        id="chunk_0",
        text="Hello from the PDF text layer",
        source_path=str(pdf_path),
        metadata={"char_length": 29},
    )

    chunker._page_chunker.chunk = lambda document, config: [visual_chunk]  # type: ignore[method-assign]
    chunker._semantic_chunker.chunk = lambda document, config: [text_chunk]  # type: ignore[method-assign]

    chunks = chunker.chunk(document, ChunkerConfig(name="hybrid"))

    assert len(chunks) == 2
    assert {chunk.id for chunk in chunks} == {"chunk_0_visual", "chunk_0_text"}
    assert [chunk.metadata["hybrid_source"] for chunk in chunks] == ["visual", "text"]
    assert [chunk.metadata["hybrid_order"] for chunk in chunks] == [0, 1]


def test_hybrid_chunker_tags_text_only_documents(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("plain text")

    document = DocumentArtifact(
        source_path=str(text_path),
        mime_type="text/plain",
    )
    chunker = HybridChunker()
    text_chunk = TextChunk(
        id="chunk_3",
        text="plain text",
        source_path=str(text_path),
        metadata={},
    )

    chunker._semantic_chunker.chunk = lambda document, config: [text_chunk]  # type: ignore[method-assign]

    chunks = chunker.chunk(document, ChunkerConfig(name="hybrid"))

    assert len(chunks) == 1
    assert chunks[0].id == "chunk_3_text"
    assert chunks[0].metadata["hybrid_source"] == "text"
