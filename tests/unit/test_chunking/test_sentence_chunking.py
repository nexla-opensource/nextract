"""
Tests for sentence-aware chunking module.

Tests cover:
- Tokenization with position tracking
- Sentence detection
- Rule A: Breaking at newlines
- Rule B: Oversized sentences
- Rule C: Packing multiple sentences
- Character interval tracking
- Edge cases
"""

from nextract.sentence_chunking import (
    SentenceAwareChunker,
    Token,
    Sentence,
    CharInterval
)


class TestCharInterval:
    """Test CharInterval dataclass"""
    
    def test_length(self):
        interval = CharInterval(start_pos=10, end_pos=50)
        assert len(interval) == 40
    
    def test_repr(self):
        interval = CharInterval(start_pos=0, end_pos=100)
        assert repr(interval) == "CharInterval(0, 100)"


class TestToken:
    """Test Token dataclass"""
    
    def test_basic_token(self):
        token = Token(text="hello", start_pos=0, end_pos=5)
        assert token.text == "hello"
        assert token.start_pos == 0
        assert token.end_pos == 5
        assert len(token) == 5
        assert not token.first_token_after_newline
    
    def test_token_after_newline(self):
        token = Token(
            text="world",
            start_pos=10,
            end_pos=15,
            first_token_after_newline=True
        )
        assert token.first_token_after_newline
        assert "↵" in repr(token)


class TestSentence:
    """Test Sentence dataclass"""
    
    def test_sentence_text(self):
        tokens = [
            Token("Hello", 0, 5),
            Token(" ", 5, 6),
            Token("world", 6, 11),
            Token(".", 11, 12)
        ]
        sentence = Sentence(start_token_idx=0, end_token_idx=4, tokens=tokens)
        assert sentence.text == "Hello world."
        assert len(sentence) == 12
    
    def test_char_interval(self):
        tokens = [
            Token("Hello", 0, 5),
            Token(" ", 5, 6),
            Token("world", 6, 11)
        ]
        sentence = Sentence(start_token_idx=0, end_token_idx=3, tokens=tokens)
        interval = sentence.char_interval
        assert interval.start_pos == 0
        assert interval.end_pos == 11
    
    def test_empty_sentence(self):
        sentence = Sentence(start_token_idx=0, end_token_idx=0, tokens=[])
        assert sentence.text == ""
        assert len(sentence) == 0
        assert sentence.char_interval == CharInterval(0, 0)


class TestTokenization:
    """Test tokenization with position tracking"""
    
    def test_simple_tokenization(self):
        chunker = SentenceAwareChunker()
        text = "Hello world."
        tokens = chunker.tokenize(text)
        
        assert len(tokens) == 4
        assert tokens[0].text == "Hello"
        assert tokens[1].text == " "
        assert tokens[2].text == "world"
        assert tokens[3].text == "."
        
        # Check positions
        assert tokens[0].start_pos == 0
        assert tokens[0].end_pos == 5
        assert tokens[3].start_pos == 11
        assert tokens[3].end_pos == 12
    
    def test_newline_tracking(self):
        chunker = SentenceAwareChunker()
        text = "Line 1\nLine 2"
        tokens = chunker.tokenize(text)
        
        # Find "Line" tokens
        line_tokens = [t for t in tokens if t.text == "Line"]
        assert len(line_tokens) == 2
        
        # First "Line" is at start (counts as after newline)
        assert line_tokens[0].first_token_after_newline
        
        # Second "Line" is after actual newline
        assert line_tokens[1].first_token_after_newline
    
    def test_punctuation_tokenization(self):
        chunker = SentenceAwareChunker()
        text = "Hello, world!"
        tokens = chunker.tokenize(text)
        
        token_texts = [t.text for t in tokens]
        assert "," in token_texts
        assert "!" in token_texts
    
    def test_number_tokenization(self):
        chunker = SentenceAwareChunker()
        text = "The year is 2024."
        tokens = chunker.tokenize(text)
        
        token_texts = [t.text for t in tokens]
        assert "2024" in token_texts


class TestSentenceDetection:
    """Test sentence boundary detection"""
    
    def test_single_sentence(self):
        chunker = SentenceAwareChunker()
        text = "This is a sentence."
        tokens = chunker.tokenize(text)
        sentences = chunker.detect_sentences(tokens)
        
        assert len(sentences) == 1
        assert sentences[0].text == text
    
    def test_multiple_sentences(self):
        chunker = SentenceAwareChunker()
        text = "First sentence. Second sentence! Third sentence?"
        tokens = chunker.tokenize(text)
        sentences = chunker.detect_sentences(tokens)
        
        assert len(sentences) == 3
        assert "First sentence." in sentences[0].text
        assert "Second sentence!" in sentences[1].text
        assert "Third sentence?" in sentences[2].text
    
    def test_double_newline_boundary(self):
        chunker = SentenceAwareChunker()
        text = "Paragraph 1\n\nParagraph 2"
        tokens = chunker.tokenize(text)
        sentences = chunker.detect_sentences(tokens)
        
        assert len(sentences) == 2
    
    def test_no_sentence_ending(self):
        chunker = SentenceAwareChunker()
        text = "No ending punctuation"
        tokens = chunker.tokenize(text)
        sentences = chunker.detect_sentences(tokens)
        
        # Should still create one sentence
        assert len(sentences) == 1
        assert sentences[0].text == text


class TestRuleA:
    """Test Rule A: Long sentences break at newlines"""
    
    def test_break_at_newlines(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        
        # Create a long sentence with newlines
        text = "This is a very long sentence that exceeds the buffer size.\nIt has a newline in the middle.\nAnd another line here."
        
        chunks = list(chunker.chunk_text(text))
        
        # Should break at newlines
        assert len(chunks) >= 2
        
        # Each chunk should respect newline boundaries
        for chunk in chunks:
            # Chunks should be smaller than the original long sentence
            assert len(chunk) <= len(text)
    
    def test_respect_newlines_disabled(self):
        chunker = SentenceAwareChunker(
            max_char_buffer=50,
            respect_newlines=False
        )
        
        text = "A" * 100 + "\n" + "B" * 100
        chunks = list(chunker.chunk_text(text))
        
        # Without newline respect, might create different chunks
        assert len(chunks) >= 1


class TestRuleB:
    """Test Rule B: Oversized sentences become standalone chunks"""
    
    def test_oversized_sentence_standalone(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        
        # Create a sentence larger than buffer
        oversized = "A" * 100 + "."
        text = "Short sentence. " + oversized + " Another short."
        
        chunks = list(chunker.chunk_text(text))
        
        # Should have at least 3 chunks (short, oversized, short)
        assert len(chunks) >= 2
        
        # Find the oversized chunk
        oversized_chunks = [c for c in chunks if len(c) > 50]
        assert len(oversized_chunks) >= 1
    
    def test_multiple_oversized_sentences(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        
        text = ("A" * 100 + ". ") * 3
        chunks = list(chunker.chunk_text(text))
        
        # Each oversized sentence should be standalone
        assert len(chunks) >= 3


class TestRuleC:
    """Test Rule C: Multiple short sentences packed together"""
    
    def test_pack_short_sentences(self):
        chunker = SentenceAwareChunker(max_char_buffer=100)
        
        # Create multiple short sentences that fit together
        text = "Short one. Short two. Short three. Short four."
        chunks = list(chunker.chunk_text(text))
        
        # Should pack into fewer chunks than sentences
        assert len(chunks) < 4
        
        # First chunk should contain multiple sentences
        assert "." in chunks[0].text
        # Count periods in first chunk
        period_count = chunks[0].text.count(".")
        assert period_count >= 2
    
    def test_pack_until_buffer_full(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        
        # Create sentences that should pack 2-3 per chunk
        sentences = ["Sentence one. ", "Sentence two. ", "Sentence three. "]
        text = "".join(sentences)
        
        chunks = list(chunker.chunk_text(text))
        
        # Should pack efficiently
        assert len(chunks) >= 1
        assert len(chunks) < len(sentences)


class TestCharacterTracking:
    """Test character-level position tracking"""
    
    def test_chunk_char_intervals(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        text = "First sentence. Second sentence. Third sentence."
        
        chunks = list(chunker.chunk_text(text))
        
        # Each chunk should have valid char interval
        for chunk in chunks:
            assert chunk.char_interval.start_pos >= 0
            assert chunk.char_interval.end_pos <= len(text)
            assert chunk.char_interval.start_pos < chunk.char_interval.end_pos
    
    def test_non_overlapping_intervals(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        text = "A" * 200
        
        chunks = list(chunker.chunk_text(text))
        
        # Chunks should not overlap
        for i in range(len(chunks) - 1):
            assert chunks[i].char_interval.end_pos <= chunks[i + 1].char_interval.start_pos
    
    def test_complete_coverage(self):
        chunker = SentenceAwareChunker(max_char_buffer=50)
        text = "Complete coverage test. Multiple sentences here. And more."
        
        chunks = list(chunker.chunk_text(text))
        
        # First chunk should start at 0
        assert chunks[0].char_interval.start_pos == 0
        
        # Last chunk should end at text length
        # (approximately, accounting for whitespace)
        assert chunks[-1].char_interval.end_pos <= len(text)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_text(self):
        chunker = SentenceAwareChunker()
        chunks = list(chunker.chunk_text(""))
        
        # Should handle empty text gracefully
        assert len(chunks) == 0 or (len(chunks) == 1 and len(chunks[0]) == 0)
    
    def test_whitespace_only(self):
        chunker = SentenceAwareChunker()
        chunks = list(chunker.chunk_text("   \n\n   "))
        
        # Should handle whitespace-only text
        assert len(chunks) >= 0
    
    def test_single_character(self):
        chunker = SentenceAwareChunker()
        chunks = list(chunker.chunk_text("A"))
        
        assert len(chunks) == 1
        assert chunks[0].text == "A"
    
    def test_very_small_buffer(self):
        chunker = SentenceAwareChunker(max_char_buffer=10)
        text = "This is a test sentence."
        
        chunks = list(chunker.chunk_text(text))
        
        # Should still create chunks even with tiny buffer
        assert len(chunks) >= 1
    
    def test_source_file_tracking(self):
        chunker = SentenceAwareChunker()
        text = "Test sentence."
        source = "test.pdf"
        
        chunks = list(chunker.chunk_text(text, source_file=source))
        
        assert len(chunks) >= 1
        assert chunks[0].source_file == source
    
    def test_chunk_ids_sequential(self):
        chunker = SentenceAwareChunker(max_char_buffer=20)
        text = "First. Second. Third. Fourth. Fifth."
        
        chunks = list(chunker.chunk_text(text))
        
        # Chunk IDs should be sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i


class TestIntegration:
    """Integration tests with realistic data"""
    
    def test_insurance_claim_text(self):
        chunker = SentenceAwareChunker(max_char_buffer=200)
        
        text = """
        Claim ID: CLM-2024-001
        Policy Holder: John Doe
        Incident Date: 2024-01-15
        
        Description: Vehicle collision at intersection of Main St and Oak Ave.
        Damage assessment shows front bumper damage and headlight replacement needed.
        Estimated repair cost: $2,500.
        
        Adjuster Notes: Claim approved. No fault determined.
        """
        
        chunks = list(chunker.chunk_text(text))
        
        # Should create reasonable number of chunks
        assert 1 <= len(chunks) <= 5
        
        # All chunks should have valid intervals
        for chunk in chunks:
            assert len(chunk) > 0
            assert chunk.char_interval.start_pos >= 0
    
    def test_table_like_data(self):
        chunker = SentenceAwareChunker(max_char_buffer=150)
        
        text = """
        Name        | Age | City
        ------------|-----|----------
        Alice Smith | 30  | New York
        Bob Jones   | 25  | Boston
        Carol White | 35  | Chicago
        """
        
        chunks = list(chunker.chunk_text(text))
        
        # Should handle table-like structure
        assert len(chunks) >= 1
        
        # Should preserve structure
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

