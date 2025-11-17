"""
Document chunking and token estimation for handling large files.

This module provides:
- TokenEstimator: Estimate token count before sending to LLM
- DocumentChunker: Split large documents into manageable chunks
- ChunkExtractor: Extract from chunks and merge results
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import structlog

from .config import RuntimeConfig
from .mimetypes_map import is_pdf, is_textual, is_image
from .schema import JsonSchema
from .parallel import ParallelProcessor
from .provenance import ProvenanceTracker

log = structlog.get_logger(__name__)

ChunkingStrategy = Literal["none", "page_based", "semantic", "hybrid", "auto"]


@dataclass
class TokenEstimate:
    """Estimate of token usage before extraction"""
    file_tokens: int
    schema_tokens: int
    prompt_tokens: int
    total_tokens: int
    model_limit: int
    utilization: float  # 0.0 to 1.0
    needs_chunking: bool
    recommended_chunks: int


@dataclass
class CharInterval:
    """Character position interval in source document"""
    start_pos: int
    end_pos: int

    def __len__(self) -> int:
        return self.end_pos - self.start_pos


@dataclass
class DocumentChunk:
    """A chunk of a document for extraction"""
    chunk_id: int
    content: str | bytes
    source_file: str
    chunk_type: str  # "text", "pdf_pages", "semantic", "sentence_aware"
    metadata: dict[str, Any]  # page_range, byte_range, etc.
    char_interval: Optional[CharInterval] = None  # Position in source document


class TokenEstimator:
    """Estimate token count before sending to LLM"""
    
    def __init__(self, model: str):
        self.model = model
        self.encoding = None
        
        # Try to import tiktoken for accurate token counting
        try:
            import tiktoken
            # Get appropriate tokenizer
            if "gpt" in model.lower():
                try:
                    self.encoding = tiktoken.encoding_for_model("gpt-4")
                except KeyError:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
            elif "claude" in model.lower():
                # Claude uses similar tokenization to GPT
                self.encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # Fallback
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            log.warning(
                "tiktoken_not_available",
                message="tiktoken not installed, using character-based estimation (less accurate)"
            )
            self.encoding = None
    
    def estimate_tokens(
        self,
        files: list[str],
        schema: JsonSchema,
        user_prompt: str | None,
        examples: list[Any] | None
    ) -> TokenEstimate:
        """
        Estimate total tokens needed for extraction
        
        Returns TokenEstimate with:
        - Breakdown of token usage
        - Whether chunking is needed
        - Recommended number of chunks
        """
        
        try:
            # 1. Estimate file content tokens
            file_tokens = 0
            for file_path in files:
                file_tokens += self._estimate_file_tokens(file_path)
            
            # 2. Estimate schema tokens
            schema_str = json.dumps(schema, indent=2)
            schema_tokens = self._count_tokens(schema_str)
            
            # 3. Estimate prompt tokens
            prompt_tokens = 0
            if user_prompt:
                prompt_tokens += self._count_tokens(user_prompt)
            
            # System prompt overhead (~500-1000 tokens)
            prompt_tokens += 1000
            
            # 4. Estimate examples tokens
            if examples:
                examples_str = json.dumps(examples, indent=2)
                prompt_tokens += self._count_tokens(examples_str)
            
            # 5. Total tokens
            total_tokens = file_tokens + schema_tokens + prompt_tokens
            
            # 6. Model limits
            model_limit = self._get_model_limit(self.model)

            # Reserve 30% for output and safety margin (use 70% of context)
            # For multi-record extraction (arrays), use lower threshold to ensure chunking
            # This helps find ALL records across large documents
            effective_limit = int(model_limit * 0.3)  # Changed from 0.7 to 0.3 for better multi-record extraction
            
            utilization = total_tokens / effective_limit if effective_limit > 0 else 0.0
            needs_chunking = total_tokens > effective_limit
            
            # Calculate recommended chunks
            if needs_chunking:
                # Chunk based on file content (largest component)
                # Each chunk needs schema + prompt overhead
                overhead = schema_tokens + prompt_tokens
                available_per_chunk = effective_limit - overhead
                
                if available_per_chunk > 0:
                    recommended_chunks = max(2, (file_tokens // available_per_chunk) + 1)
                else:
                    # Schema + prompt alone exceed limit - this is a problem
                    recommended_chunks = 1
                    log.error(
                        "schema_prompt_too_large",
                        schema_tokens=schema_tokens,
                        prompt_tokens=prompt_tokens,
                        overhead=overhead,
                        effective_limit=effective_limit
                    )
            else:
                recommended_chunks = 1
            
            return TokenEstimate(
                file_tokens=file_tokens,
                schema_tokens=schema_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                model_limit=model_limit,
                utilization=utilization,
                needs_chunking=needs_chunking,
                recommended_chunks=recommended_chunks
            )
        
        except Exception as e:
            log.error("token_estimation_failed", error=str(e), error_type=type(e).__name__)
            # Return conservative estimate that triggers chunking for safety
            return TokenEstimate(
                file_tokens=100000,
                schema_tokens=1000,
                prompt_tokens=1000,
                total_tokens=102000,
                model_limit=100000,
                utilization=1.02,
                needs_chunking=True,
                recommended_chunks=2
            )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                log.debug("tiktoken_encoding_failed", error=str(e))
                # Fallback to character-based estimation
                return len(text) // 4
        else:
            # Character-based estimation: ~4 chars per token
            return len(text) // 4
    
    def _estimate_file_tokens(self, file_path: str) -> int:
        """Estimate tokens for a single file"""
        try:
            path = Path(file_path).expanduser().resolve()
            
            if not path.exists():
                log.warning("file_not_found_for_estimation", file=str(path))
                return 0
            
            if is_textual(path):
                # Read text and count tokens
                try:
                    text = path.read_text(errors='ignore')
                    return self._count_tokens(text)
                except Exception as e:
                    log.warning("text_file_read_failed", file=str(path), error=str(e))
                    # Fallback: estimate by file size
                    return path.stat().st_size // 4
            
            elif is_pdf(path):
                # Try to extract text and count tokens
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(path)
                    total_text = ""
                    for page in doc:
                        total_text += page.get_text()
                    doc.close()
                    return self._count_tokens(total_text)
                except ImportError:
                    log.debug("pymupdf_not_available", message="Using file size estimation for PDF")
                    # Fallback: estimate by file size (rough: 1 token per 4 bytes)
                    return path.stat().st_size // 4
                except Exception as e:
                    log.warning("pdf_text_extraction_failed", file=str(path), error=str(e))
                    return path.stat().st_size // 4
            
            elif is_image(path):
                # Images use fixed tokens in vision models
                # GPT-4V: ~765 tokens per image (high detail)
                # Claude: ~1600 tokens per image
                if "claude" in self.model.lower():
                    return 1600
                else:
                    return 765
            
            else:
                # Unknown: estimate by file size
                return path.stat().st_size // 4
        
        except Exception as e:
            log.error("file_token_estimation_failed", file=file_path, error=str(e))
            # Conservative estimate
            return 10000
    
    def _get_model_limit(self, model: str) -> int:
        """Get context window limit for model"""
        model_lower = model.lower()
        
        limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "claude-3-5-sonnet": 200000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "gemini-1.5-pro": 1000000,
            "gemini-1.5-flash": 1000000,
            "gemini-pro": 32000,
        }
        
        for key, limit in limits.items():
            if key in model_lower:
                return limit
        
        # Default conservative limit
        log.warning("unknown_model_using_default_limit", model=model, default_limit=100000)
        return 100000


class SentenceAwareChunker:
    """
    LangExtract-inspired sentence-aware text chunking.

    Implements 3 intelligent rules:
    A. Long sentences → Break at newlines
    B. Oversized tokens → Standalone chunks
    C. Multiple sentences → Pack together
    """

    def __init__(self, max_char_buffer: int = 10000):
        """
        Initialize sentence-aware chunker.

        Args:
            max_char_buffer: Maximum characters per chunk (default: 10000)
        """
        self.max_char_buffer = max_char_buffer

    def chunk_text(
        self,
        text: str,
        source_file: str,
        num_chunks: Optional[int] = None
    ) -> list[DocumentChunk]:
        """
        Chunk text with sentence and newline awareness.

        Args:
            text: Text to chunk
            source_file: Source file path
            num_chunks: Target number of chunks (optional, will calculate optimal)

        Returns:
            List of DocumentChunk objects with char_interval tracking
        """
        if not text.strip():
            log.warning("empty_text_for_chunking", source_file=source_file)
            return []

        # Detect sentences
        sentences = self._detect_sentences(text)

        if len(sentences) == 0:
            # Fallback: treat entire text as one sentence
            sentences = [text]

        log.info(
            "sentence_detection_complete",
            source_file=source_file,
            total_sentences=len(sentences),
            text_length=len(text)
        )

        # Chunk sentences with 3 rules
        chunks = []
        current_chunk_text = []
        current_chunk_start = 0
        current_length = 0

        for sentence_idx, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # Rule B: Oversized sentence
            if sentence_length > self.max_char_buffer:
                # Flush current chunk first
                if current_chunk_text:
                    chunk_text = ''.join(current_chunk_text)
                    chunks.append(self._create_chunk(
                        chunk_id=len(chunks),
                        content=chunk_text,
                        source_file=source_file,
                        char_start=current_chunk_start,
                        char_end=current_chunk_start + len(chunk_text)
                    ))
                    current_chunk_text = []
                    current_length = 0

                # Rule A: Try to break at newlines
                if '\n' in sentence:
                    sub_chunks = self._break_at_newlines(
                        sentence,
                        source_file,
                        start_chunk_id=len(chunks),
                        char_offset=current_chunk_start + current_length
                    )
                    chunks.extend(sub_chunks)
                    current_chunk_start += sentence_length
                else:
                    # Standalone chunk (no newlines to break on)
                    chunks.append(self._create_chunk(
                        chunk_id=len(chunks),
                        content=sentence,
                        source_file=source_file,
                        char_start=current_chunk_start + current_length,
                        char_end=current_chunk_start + current_length + sentence_length
                    ))
                    current_chunk_start += sentence_length
                continue

            # Rule C: Pack sentences together
            if current_length + sentence_length <= self.max_char_buffer:
                current_chunk_text.append(sentence)
                current_length += sentence_length
            else:
                # Flush current chunk
                if current_chunk_text:
                    chunk_text = ''.join(current_chunk_text)
                    chunks.append(self._create_chunk(
                        chunk_id=len(chunks),
                        content=chunk_text,
                        source_file=source_file,
                        char_start=current_chunk_start,
                        char_end=current_chunk_start + len(chunk_text)
                    ))
                    current_chunk_start += len(chunk_text)

                # Start new chunk
                current_chunk_text = [sentence]
                current_length = sentence_length

        # Flush remaining
        if current_chunk_text:
            chunk_text = ''.join(current_chunk_text)
            chunks.append(self._create_chunk(
                chunk_id=len(chunks),
                content=chunk_text,
                source_file=source_file,
                char_start=current_chunk_start,
                char_end=current_chunk_start + len(chunk_text)
            ))

        log.info(
            "sentence_aware_chunking_complete",
            source_file=source_file,
            total_chunks=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
        )

        return chunks

    def _detect_sentences(self, text: str) -> list[str]:
        """
        Detect sentence boundaries in text.

        Simple approach: Split on .!? followed by space/newline.
        Preserves the delimiter with the sentence.
        """
        import re

        # Pattern: sentence ending punctuation followed by space/newline or end of string
        # Keep the punctuation with the sentence
        pattern = r'([^.!?]*[.!?]+(?:\s+|$))'

        sentences = []
        matches = re.finditer(pattern, text)

        for match in matches:
            sentence = match.group(1)
            if sentence.strip():
                sentences.append(sentence)

        # Handle any remaining text without sentence-ending punctuation
        last_match_end = 0
        for match in re.finditer(pattern, text):
            last_match_end = match.end()

        if last_match_end < len(text):
            remaining = text[last_match_end:]
            if remaining.strip():
                sentences.append(remaining)

        return sentences

    def _break_at_newlines(
        self,
        text: str,
        source_file: str,
        start_chunk_id: int,
        char_offset: int
    ) -> list[DocumentChunk]:
        """
        Break oversized text at newline boundaries (Rule A).
        """
        chunks = []
        lines = text.split('\n')

        current_lines = []
        current_length = 0
        current_start = char_offset

        for line_idx, line in enumerate(lines):
            line_with_newline = line + ('\n' if line_idx < len(lines) - 1 else '')
            line_length = len(line_with_newline)

            if current_length + line_length <= self.max_char_buffer:
                current_lines.append(line_with_newline)
                current_length += line_length
            else:
                # Flush current chunk
                if current_lines:
                    chunk_text = ''.join(current_lines)
                    chunks.append(self._create_chunk(
                        chunk_id=start_chunk_id + len(chunks),
                        content=chunk_text,
                        source_file=source_file,
                        char_start=current_start,
                        char_end=current_start + len(chunk_text)
                    ))
                    current_start += len(chunk_text)

                # Start new chunk
                current_lines = [line_with_newline]
                current_length = line_length

        # Flush remaining
        if current_lines:
            chunk_text = ''.join(current_lines)
            chunks.append(self._create_chunk(
                chunk_id=start_chunk_id + len(chunks),
                content=chunk_text,
                source_file=source_file,
                char_start=current_start,
                char_end=current_start + len(chunk_text)
            ))

        return chunks

    def _create_chunk(
        self,
        chunk_id: int,
        content: str,
        source_file: str,
        char_start: int,
        char_end: int
    ) -> DocumentChunk:
        """Create a DocumentChunk with character interval tracking."""
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            source_file=source_file,
            chunk_type="sentence_aware",
            metadata={
                "char_length": len(content),
                "sentence_count": content.count('.') + content.count('!') + content.count('?')
            },
            char_interval=CharInterval(start_pos=char_start, end_pos=char_end)
        )


class DocumentChunker:
    """Intelligently chunk documents for extraction"""

    def chunk_documents(
        self,
        file_paths: list[str],
        num_chunks: int,
        strategy: ChunkingStrategy = "auto"
    ) -> list[DocumentChunk]:
        """
        Chunk documents into smaller pieces

        For multiple files, distributes them across chunks.
        For single large file, splits it into chunks.

        Strategies:
        - page_based: Split PDFs by page ranges
        - semantic: Split text by paragraphs/sections
        - hybrid: Combine both approaches
        - auto: Choose best strategy based on file type
        """

        if len(file_paths) == 0:
            return []

        # If multiple files, distribute them across chunks
        if len(file_paths) > 1:
            return self._chunk_multiple_files(file_paths, num_chunks)

        # Single file - split it
        return self._chunk_single_file(file_paths[0], num_chunks, strategy)

    def _chunk_multiple_files(
        self,
        file_paths: list[str],
        num_chunks: int
    ) -> list[DocumentChunk]:
        """Distribute multiple files across chunks"""
        chunks = []
        files_per_chunk = max(1, len(file_paths) // num_chunks)

        for i in range(num_chunks):
            start_idx = i * files_per_chunk
            end_idx = min((i + 1) * files_per_chunk, len(file_paths))

            if start_idx >= len(file_paths):
                break

            chunk_files = file_paths[start_idx:end_idx]

            # Create a chunk that references multiple files
            chunks.append(DocumentChunk(
                chunk_id=i,
                content=json.dumps(chunk_files),  # Store file paths as JSON
                source_file=",".join(chunk_files),
                chunk_type="multi_file",
                metadata={
                    "file_paths": chunk_files,
                    "file_range": (start_idx, end_idx),
                    "total_files": len(file_paths)
                }
            ))

        return chunks

    def _chunk_single_file(
        self,
        file_path: str,
        num_chunks: int,
        strategy: ChunkingStrategy
    ) -> list[DocumentChunk]:
        """Chunk a single document"""
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if strategy == "auto":
            if is_pdf(path):
                # For PDFs, try intelligent text extraction first
                return self._chunk_pdf_intelligent(path, num_chunks)
            elif is_textual(path):
                strategy = "semantic"
            else:
                strategy = "page_based"

        if strategy == "page_based":
            return self._chunk_pdf_by_pages(path, num_chunks)
        elif strategy == "semantic":
            return self._chunk_text_semantic(path, num_chunks)
        else:
            return self._chunk_hybrid(path, num_chunks)

    def _chunk_pdf_intelligent(self, path: Path, num_chunks: int) -> list[DocumentChunk]:
        """
        Intelligently chunk PDF based on type.

        For text-rich PDFs: Extract text and use sentence-aware chunking
        For scanned PDFs: Use page-based chunking
        """
        try:
            from .pdf_extractor import PDFTextExtractor
            from .pdf_analyzer import PDFType
        except ImportError:
            log.warning(
                "pdf_extractor_not_available",
                message="PDF extractor not available, falling back to page-based chunking"
            )
            return self._chunk_pdf_by_pages(path, num_chunks)

        try:
            # Extract text and analyze PDF
            extractor = PDFTextExtractor()
            text, analysis = extractor.extract(path)

            log.info(
                "pdf_analyzed_for_chunking",
                file=str(path),
                pdf_type=analysis.pdf_type.value,
                text_length=len(text),
                recommended_method=analysis.recommended_method.value
            )

            # For text-rich PDFs, use sentence-aware chunking
            if analysis.pdf_type == PDFType.TEXT_RICH and text.strip():
                log.info(
                    "using_sentence_aware_chunking",
                    file=str(path),
                    reason="text_rich_pdf"
                )

                # Calculate optimal chunk size based on text length and num_chunks
                text_length = len(text)
                target_chunk_size = max(5000, text_length // num_chunks)

                # Use sentence-aware chunker
                chunker = SentenceAwareChunker(max_char_buffer=target_chunk_size)
                return chunker.chunk_text(text, str(path), num_chunks)
            else:
                # For scanned/hybrid PDFs, use page-based chunking
                log.info(
                    "using_page_based_chunking",
                    file=str(path),
                    reason=f"pdf_type_{analysis.pdf_type.value}"
                )
                return self._chunk_pdf_by_pages(path, num_chunks)

        except Exception as e:
            log.error(
                "pdf_intelligent_chunking_failed",
                file=str(path),
                error=str(e),
                error_type=type(e).__name__
            )
            # Fallback to page-based chunking
            return self._chunk_pdf_by_pages(path, num_chunks)

    def _chunk_pdf_by_pages(self, path: Path, num_chunks: int) -> list[DocumentChunk]:
        """Split PDF into page-based chunks"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            log.error(
                "pymupdf_not_installed",
                message="PyMuPDF (fitz) required for PDF chunking. Install with: pip install PyMuPDF"
            )
            raise ImportError("PyMuPDF required for PDF chunking. Install with: pip install PyMuPDF")

        try:
            doc = fitz.open(path)
            total_pages = len(doc)
            doc.close()

            if total_pages == 0:
                log.warning("empty_pdf", file=str(path))
                return []

            pages_per_chunk = max(1, total_pages // num_chunks)

            chunks = []
            for i in range(num_chunks):
                start_page = i * pages_per_chunk
                end_page = min((i + 1) * pages_per_chunk, total_pages)

                if start_page >= total_pages:
                    break

                # Extract pages to new PDF
                doc = fitz.open(path)
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

                # Convert to bytes
                chunk_bytes = chunk_doc.tobytes()

                chunk_doc.close()
                doc.close()

                chunks.append(DocumentChunk(
                    chunk_id=i,
                    content=chunk_bytes,
                    source_file=str(path),
                    chunk_type="pdf_pages",
                    metadata={
                        "page_range": (start_page + 1, end_page),  # 1-indexed for user display
                        "total_pages": total_pages,
                        "pages_in_chunk": end_page - start_page
                    }
                ))

            log.info(
                "pdf_chunked",
                file=str(path),
                total_pages=total_pages,
                num_chunks=len(chunks),
                pages_per_chunk=pages_per_chunk
            )

            return chunks

        except Exception as e:
            log.error("pdf_chunking_failed", file=str(path), error=str(e), error_type=type(e).__name__)
            raise

    def _chunk_text_semantic(self, path: Path, num_chunks: int) -> list[DocumentChunk]:
        """Split text by semantic boundaries using sentence-aware chunking"""
        try:
            text = path.read_text(errors='ignore')

            if not text.strip():
                log.warning("empty_text_file", file=str(path))
                return []

            # Calculate optimal chunk size based on text length
            text_length = len(text)
            target_chunk_size = max(5000, text_length // num_chunks)

            log.info(
                "text_chunking_started",
                file=str(path),
                text_length=text_length,
                num_chunks=num_chunks,
                target_chunk_size=target_chunk_size
            )

            # Use sentence-aware chunker
            chunker = SentenceAwareChunker(max_char_buffer=target_chunk_size)
            return chunker.chunk_text(text, str(path), num_chunks)

        except Exception as e:
            log.error("text_chunking_failed", file=str(path), error=str(e), error_type=type(e).__name__)
            raise

    def _chunk_hybrid(self, path: Path, num_chunks: int) -> list[DocumentChunk]:
        """Hybrid chunking strategy"""
        # For now, delegate to appropriate method
        if is_pdf(path):
            return self._chunk_pdf_by_pages(path, num_chunks)
        else:
            return self._chunk_text_semantic(path, num_chunks)


class ChunkExtractor:
    """Extract from chunks and merge results with optional parallel processing"""

    def __init__(self, max_workers: int = 10, enable_provenance: bool = False):
        """
        Initialize chunk extractor.

        Args:
            max_workers: Maximum number of parallel workers (default: 10)
            enable_provenance: Whether to track provenance (default: False)
        """
        self.max_workers = max_workers
        self.enable_provenance = enable_provenance
        self.processor = ParallelProcessor(max_workers=max_workers) if max_workers > 1 else None

        log.info(
            "chunk_extractor_initialized",
            max_workers=max_workers,
            enable_provenance=enable_provenance,
            parallel_enabled=self.processor is not None
        )

    async def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
        schema: JsonSchema,
        config: RuntimeConfig,
        user_prompt: str | None,
        examples: list[Any] | None,
        include_extra: bool = False
    ) -> tuple[dict[str, Any] | list[Any], dict[str, Any]]:
        """
        Extract from each chunk and merge results

        Strategy:
        1. Make all schema fields optional for chunk extraction
           - For array schemas: wrap in object so each chunk returns array
           - For object schemas: make fields optional
        2. Extract from each chunk independently
        3. Merge results with conflict resolution
           - For array schemas: concatenate all items
           - For object schemas: merge fields with first-non-empty strategy
        4. Validate merged result against original schema

        Returns: (merged_data, report_dict)
            - merged_data: dict for object schemas, list for array schemas
            - report_dict: extraction metadata and statistics
        """

        from .agent_runner import run_extraction_async
        from .pricing import estimate_cost_usd, parse_pricing_json
        from jsonschema import Draft202012Validator, ValidationError

        overall_start = time.time()

        log.info(
            "chunk_extraction_started",
            num_chunks=len(chunks),
            schema_keys=len(schema.get("properties", {}))
        )

        # 1. Create optional schema for chunks
        schema_start = time.time()
        optional_schema = self._make_schema_optional(schema)
        log.debug(
            "schema_preparation_complete",
            duration_ms=int((time.time() - schema_start) * 1000)
        )

        # 2. Extract from each chunk (parallel or sequential)
        is_array = self._is_array_schema(schema)

        extraction_start = time.time()
        if self.processor and len(chunks) > 1:
            # Parallel processing
            log.info(
                "using_parallel_extraction",
                num_chunks=len(chunks),
                max_workers=self.max_workers
            )
            chunk_results, all_usage, chunk_errors = await self._extract_chunks_parallel(
                chunks=chunks,
                optional_schema=optional_schema,
                config=config,
                user_prompt=user_prompt,
                examples=examples,
                include_extra=include_extra,
                is_array_schema=is_array
            )
        else:
            # Sequential processing
            log.info("using_sequential_extraction", num_chunks=len(chunks))
            chunk_results, all_usage, chunk_errors = await self._extract_chunks_sequential(
                chunks=chunks,
                optional_schema=optional_schema,
                config=config,
                user_prompt=user_prompt,
                examples=examples,
                include_extra=include_extra,
                is_array_schema=is_array
            )

        extraction_duration = time.time() - extraction_start
        log.info(
            "chunk_extraction_complete",
            duration_seconds=round(extraction_duration, 2),
            chunks_processed=len(chunks),
            successful=len(chunk_results) - len(chunk_errors),
            failed=len(chunk_errors)
        )

        # 3. Merge results
        merge_start = time.time()
        merged_data, provenance = self._merge_chunk_results(chunk_results, schema)

        # Calculate merged_fields count (works for both arrays and objects)
        if isinstance(merged_data, list):
            merged_fields_count = len(merged_data)  # Number of items in array
        elif isinstance(merged_data, dict):
            merged_fields_count = len(merged_data)  # Number of fields in object
        else:
            merged_fields_count = 0

        merge_duration = time.time() - merge_start
        log.info(
            "chunks_merged",
            total_chunks=len(chunks),
            successful_chunks=len(chunk_results) - len(chunk_errors),
            failed_chunks=len(chunk_errors),
            merged_fields=merged_fields_count,
            result_type="array" if isinstance(merged_data, list) else "object",
            duration_ms=int(merge_duration * 1000)
        )

        # 4. Validate against original schema
        validation_errors = []
        try:
            Draft202012Validator(schema).validate(merged_data)
            log.info("merged_data_validated", status="success")
        except ValidationError as e:
            log.warning(
                "merged_data_validation_failed",
                error=str(e),
                path=list(e.path) if hasattr(e, 'path') else []
            )
            validation_errors.append({
                "message": str(e),
                "path": list(e.path) if hasattr(e, 'path') else []
            })

        # 5. Aggregate usage
        total_usage = self._aggregate_usage(all_usage)

        # 6. Estimate cost
        cost = None
        try:
            pricing_data = parse_pricing_json(config.pricing_json)
            # Create a RunUsage object for cost estimation
            from pydantic_ai.usage import RunUsage
            usage_obj = RunUsage(
                requests=total_usage.get("requests", 0),
                input_tokens=total_usage.get("input_tokens", 0),
                output_tokens=total_usage.get("output_tokens", 0),
                tool_calls=total_usage.get("tool_calls", 0)
            )
            cost = estimate_cost_usd(usage_obj, config.model, pricing_data)
        except Exception as e:
            log.warning("cost_estimation_failed", error=str(e))

        # 7. Create report
        report = {
            "model": config.model,
            "files": [c.source_file for c in chunks],
            "usage": {
                **total_usage,
                "chunks": len(chunks),
                "successful_chunks": len(chunk_results) - len(chunk_errors),
                "failed_chunks": len(chunk_errors),
                "chunk_metadata": [c.metadata for c in chunks],
                "field_provenance": provenance
            },
            "cost_estimate_usd": cost,
            "warnings": [],
            "chunk_errors": chunk_errors,
            "validation_errors": validation_errors
        }

        if chunk_errors:
            report["warnings"].append(
                f"{len(chunk_errors)} chunk(s) failed extraction - partial results returned"
            )

        if validation_errors:
            report["warnings"].append(
                f"Merged data failed validation against schema - {len(validation_errors)} error(s)"
            )

        # Add overall timing
        overall_duration = time.time() - overall_start
        log.info(
            "chunk_extraction_pipeline_complete",
            total_duration_seconds=round(overall_duration, 2),
            extraction_duration_seconds=round(extraction_duration, 2),
            merge_duration_ms=int(merge_duration * 1000),
            extraction_percentage=round(extraction_duration / overall_duration * 100, 1)
        )

        return merged_data, report

    async def _extract_chunks_sequential(
        self,
        chunks: list[DocumentChunk],
        optional_schema: JsonSchema,
        config: RuntimeConfig,
        user_prompt: str | None,
        examples: list[Any] | None,
        include_extra: bool,
        is_array_schema: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract from chunks sequentially (original implementation)"""
        from .agent_runner import run_extraction_async

        chunk_results = []
        all_usage = []
        chunk_errors = []

        for chunk in chunks:
            log.info(
                "extracting_chunk",
                chunk_id=chunk.chunk_id,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata
            )

            # Save chunk to temp file
            temp_file = None
            try:
                temp_file = self._save_chunk_to_temp(chunk)

                # Extract from this chunk
                result = await run_extraction_async(
                    config=config,
                    files=[temp_file],
                    schema_or_model=optional_schema,
                    user_prompt=self._create_chunk_prompt(user_prompt, chunk, is_array_schema),
                    examples=examples,
                    include_extra=include_extra,
                    return_pydantic=False
                )

                chunk_results.append(result[0])
                all_usage.append(result[1].usage)

                log.info(
                    "chunk_extracted",
                    chunk_id=chunk.chunk_id,
                    fields_extracted=len(result[0]) if isinstance(result[0], dict) else 0
                )

            except Exception as e:
                log.error(
                    "chunk_extraction_failed",
                    chunk_id=chunk.chunk_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc()
                )
                chunk_errors.append({
                    "chunk_id": chunk.chunk_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                })
                # Continue with other chunks - don't fail entire extraction
                chunk_results.append({})
                all_usage.append({})

            finally:
                # Clean up temp file
                if temp_file:
                    try:
                        Path(temp_file).unlink(missing_ok=True)
                    except Exception as e:
                        log.debug("temp_file_cleanup_failed", file=temp_file, error=str(e))

        return chunk_results, all_usage, chunk_errors

    async def _extract_chunks_parallel(
        self,
        chunks: list[DocumentChunk],
        optional_schema: JsonSchema,
        config: RuntimeConfig,
        user_prompt: str | None,
        examples: list[Any] | None,
        include_extra: bool,
        is_array_schema: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract from chunks in parallel using ParallelProcessor"""
        from .agent_runner import run_extraction_async

        # Define extraction function for a single chunk
        def extract_single_chunk(chunk: DocumentChunk) -> dict[str, Any]:
            """Extract from a single chunk (sync wrapper for async)"""
            temp_file = None
            try:
                # Save chunk to temp file
                temp_file = self._save_chunk_to_temp(chunk)

                # Run extraction (need to run async in sync context)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        run_extraction_async(
                            config=config,
                            files=[temp_file],
                            schema_or_model=optional_schema,
                            user_prompt=self._create_chunk_prompt(user_prompt, chunk, is_array_schema),
                            examples=examples,
                            include_extra=include_extra,
                            return_pydantic=False
                        )
                    )

                    return {
                        "success": True,
                        "chunk_id": chunk.chunk_id,
                        "data": result[0],
                        "usage": result[1].usage,
                        "error": None
                    }
                finally:
                    loop.close()

            except Exception as e:
                log.error(
                    "chunk_extraction_failed_parallel",
                    chunk_id=chunk.chunk_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc()
                )
                return {
                    "success": False,
                    "chunk_id": chunk.chunk_id,
                    "data": {},
                    "usage": {},
                    "error": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }
                }

            finally:
                # Clean up temp file
                if temp_file:
                    try:
                        Path(temp_file).unlink(missing_ok=True)
                    except Exception as e:
                        log.debug("temp_file_cleanup_failed", file=temp_file, error=str(e))

        # Process chunks in parallel
        batch_result = self.processor.process_batch(
            items=chunks,
            process_fn=extract_single_chunk,
            batch_size=None,  # Process all at once
            fail_fast=False,  # Continue on errors
            return_errors=True  # Get error details
        )

        # Separate results, usage, and errors
        chunk_results = []
        all_usage = []
        chunk_errors = []

        for result in batch_result.results:
            if result and result["success"]:
                chunk_results.append(result["data"])
                all_usage.append(result["usage"])
            else:
                chunk_results.append({})
                all_usage.append({})
                if result and result["error"]:
                    chunk_errors.append({
                        "chunk_id": result["chunk_id"],
                        **result["error"]
                    })

        return chunk_results, all_usage, chunk_errors

    def _is_array_schema(self, schema: JsonSchema) -> bool:
        """Check if schema is an array schema"""
        return isinstance(schema, dict) and schema.get("type") == "array"

    def _wrap_array_schema_for_chunk(self, schema: JsonSchema) -> JsonSchema:
        """
        Wrap array schema in an object for chunk extraction.

        For array schemas like:
            {"type": "array", "items": {...}}

        We wrap it as:
            {"type": "object", "properties": {"items": {"type": "array", "items": {...}}}}

        This allows each chunk to extract an array of items, which we then concatenate.
        """
        import copy

        if not self._is_array_schema(schema):
            return schema

        # Wrap array schema in an object
        wrapped = {
            "type": "object",
            "properties": {
                "items": copy.deepcopy(schema)
            },
            "title": schema.get("title", "ChunkItems"),
            "description": f"Chunk extraction wrapper for array schema. Extract all items found in this chunk."
        }

        return wrapped

    def _unwrap_array_result(self, result: dict[str, Any]) -> list[Any]:
        """
        Unwrap array result from chunk extraction.

        Extracts the "items" array from the wrapped object.
        """
        if isinstance(result, dict) and "items" in result:
            items = result["items"]
            if isinstance(items, list):
                return items
        return []

    def _make_schema_optional(self, schema: JsonSchema) -> JsonSchema:
        """Make all fields optional for chunk extraction"""
        import copy
        optional_schema = copy.deepcopy(schema)

        # Handle array schemas - wrap them first
        if self._is_array_schema(optional_schema):
            # For array schemas, we don't make them optional
            # Instead, we wrap them so each chunk can return an array
            return self._wrap_array_schema_for_chunk(optional_schema)

        # Remove required fields at top level
        if "required" in optional_schema:
            del optional_schema["required"]

        # Recursively make nested fields optional
        if "properties" in optional_schema:
            for prop_name, prop_schema in optional_schema["properties"].items():
                if isinstance(prop_schema, dict):
                    if "type" in prop_schema and prop_schema["type"] == "object":
                        optional_schema["properties"][prop_name] = self._make_schema_optional(prop_schema)
                    # Also remove required from nested objects
                    if "required" in prop_schema:
                        del optional_schema["properties"][prop_name]["required"]

        return optional_schema

    def _save_chunk_to_temp(self, chunk: DocumentChunk) -> str:
        """Save chunk content to temporary file"""

        if chunk.chunk_type == "multi_file":
            # For multi-file chunks, return the first file path
            file_paths = chunk.metadata.get("file_paths", [])
            if file_paths:
                return file_paths[0]
            raise ValueError(f"Multi-file chunk {chunk.chunk_id} has no file paths")

        # Create temp file
        suffix = ".pdf" if chunk.chunk_type == "pdf_pages" else ".txt"
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=f"nextract_chunk_{chunk.chunk_id}_")

        try:
            if isinstance(chunk.content, bytes):
                # Binary content (PDF)
                with open(temp_fd, 'wb') as f:
                    f.write(chunk.content)
            else:
                # Text content
                with open(temp_fd, 'w', encoding='utf-8') as f:
                    f.write(str(chunk.content))
        except Exception as e:
            # Clean up on error
            try:
                Path(temp_path).unlink(missing_ok=True)
            except:
                pass
            raise e

        return temp_path

    def _create_chunk_prompt(
        self,
        base_prompt: str | None,
        chunk: DocumentChunk,
        is_array_schema: bool = False
    ) -> str:
        """Create chunk-specific prompt"""

        chunk_info = f"Chunk {chunk.chunk_id + 1}"
        if chunk.chunk_type == "pdf_pages":
            page_range = chunk.metadata.get("page_range", (0, 0))
            total_pages = chunk.metadata.get("total_pages", 0)
            chunk_info = f"Pages {page_range[0]}-{page_range[1]} of {total_pages}"
        elif chunk.chunk_type == "semantic":
            para_range = chunk.metadata.get("paragraph_range", (0, 0))
            total_paras = chunk.metadata.get("total_paragraphs", 0)
            chunk_info = f"Paragraphs {para_range[0]}-{para_range[1]} of {total_paras}"

        if is_array_schema:
            # For array schemas, emphasize extracting ALL items
            chunk_context = f"""
You are extracting data from {chunk_info} of a larger document.

IMPORTANT INSTRUCTIONS FOR ARRAY EXTRACTION:
- Extract ALL items/records found in THIS chunk
- Each item should be a complete record with all available fields
- Do NOT skip any items - extract every single one you find
- If a field is not available for an item, leave it empty/null
- Do NOT hallucinate or infer information not present in the chunk
- Be thorough - scan the entire chunk for all items

{base_prompt or "Extract all items according to the schema."}
"""
        else:
            # For object schemas, emphasize partial extraction
            chunk_context = f"""
You are extracting data from {chunk_info} of a larger document.

IMPORTANT INSTRUCTIONS:
- Extract ONLY information present in THIS chunk
- Leave fields empty/null if information is not in this chunk
- Do NOT hallucinate or infer information from other chunks
- It's OK to have partial data - other chunks will fill in missing fields
- Be accurate and only extract what you can see in this chunk

{base_prompt or "Extract structured data according to the schema."}
"""
        return chunk_context.strip()

    def _merge_chunk_results(
        self,
        chunk_results: list[dict[str, Any]],
        schema: JsonSchema
    ) -> tuple[dict[str, Any] | list[Any], dict[str, str]]:
        """
        Merge results from multiple chunks

        Strategy:
        - For array schemas: Concatenate all items from all chunks
        - For object schemas:
          - Simple fields: First non-empty value wins
          - Arrays: Concatenate and deduplicate
          - Objects: Recursive merge
        - Track provenance (which chunk provided each value)

        Returns: (merged_data, provenance)
        """

        # Check if this is an array schema
        is_array = self._is_array_schema(schema)

        if is_array:
            # For array schemas, concatenate all items from all chunks
            all_items = []
            provenance: dict[str, str] = {}

            for chunk_idx, chunk_data in enumerate(chunk_results):
                # Unwrap the array from the wrapped object
                items = self._unwrap_array_result(chunk_data)

                if items:
                    # Track which chunk each item came from
                    start_idx = len(all_items)
                    all_items.extend(items)

                    # Record provenance for each item
                    for i in range(len(items)):
                        provenance[f"item_{start_idx + i}"] = f"chunk_{chunk_idx}"

            log.info(
                "array_chunks_merged",
                total_chunks=len(chunk_results),
                total_items=len(all_items),
                items_per_chunk=[len(self._unwrap_array_result(r)) for r in chunk_results]
            )

            return all_items, provenance

        # For object schemas, use the original merge logic
        merged: dict[str, Any] = {}
        provenance = {}

        for chunk_idx, chunk_data in enumerate(chunk_results):
            if not isinstance(chunk_data, dict):
                continue

            for key, value in chunk_data.items():
                if self._is_empty(value):
                    # Skip empty values
                    continue

                if key not in merged or self._is_empty(merged[key]):
                    # First non-empty value
                    merged[key] = value
                    provenance[key] = f"chunk_{chunk_idx}"

                elif isinstance(value, list) and isinstance(merged[key], list):
                    # Concatenate arrays
                    merged[key].extend(value)
                    provenance[key] = f"{provenance.get(key, '')},chunk_{chunk_idx}"

                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    # Recursive merge for objects
                    merged[key] = {**merged[key], **value}
                    provenance[key] = f"{provenance.get(key, '')},chunk_{chunk_idx}"

                # For other types, keep first value (already set)

        return merged, provenance

    def _is_empty(self, value: Any) -> bool:
        """Check if a value is considered empty"""
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

    def _aggregate_usage(self, all_usage: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate usage statistics from multiple chunks"""

        total = {
            "requests": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "details": {}
        }

        for usage in all_usage:
            total["requests"] += usage.get("requests", 0)
            total["tool_calls"] += usage.get("tool_calls", 0)
            total["input_tokens"] += usage.get("input_tokens", 0)
            total["output_tokens"] += usage.get("output_tokens", 0)

        return total

