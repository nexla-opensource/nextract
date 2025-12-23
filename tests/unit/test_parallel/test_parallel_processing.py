"""
Tests for parallel processing functionality.

Tests:
- ParallelProcessor basic functionality
- Error handling in parallel processing
- Batch processing
- Integration with ChunkExtractor
"""

import pytest
from nextract.parallel import ParallelProcessor, ParallelProcessingError, ProcessingError, BatchResult


class TestParallelProcessor:
    """Test ParallelProcessor class"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = ParallelProcessor(max_workers=5)
        assert processor.max_workers == 5
    
    def test_initialization_invalid_workers(self):
        """Test initialization with invalid max_workers"""
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            ParallelProcessor(max_workers=0)
    
    def test_process_batch_simple(self):
        """Test simple batch processing"""
        processor = ParallelProcessor(max_workers=4)
        
        items = [1, 2, 3, 4, 5]
        results = processor.process_batch(
            items=items,
            process_fn=lambda x: x * 2
        )
        
        assert results == [2, 4, 6, 8, 10]
    
    def test_process_batch_empty_list(self):
        """Test processing empty list raises error"""
        processor = ParallelProcessor(max_workers=4)
        
        with pytest.raises(ValueError, match="Cannot process empty list"):
            processor.process_batch(items=[], process_fn=lambda x: x)
    
    def test_process_batch_with_errors_continue(self):
        """Test batch processing continues on errors"""
        processor = ParallelProcessor(max_workers=4)
        
        def process_fn(x):
            if x == 3:
                raise ValueError(f"Error processing {x}")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        results = processor.process_batch(
            items=items,
            process_fn=process_fn,
            fail_fast=False
        )
        
        # Should have None for failed item
        assert results[0] == 2
        assert results[1] == 4
        assert results[2] is None  # Failed
        assert results[3] == 8
        assert results[4] == 10
    
    def test_process_batch_fail_fast(self):
        """Test batch processing with fail_fast=True"""
        processor = ParallelProcessor(max_workers=4)
        
        def process_fn(x):
            if x == 3:
                raise ValueError(f"Error processing {x}")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        
        with pytest.raises(ParallelProcessingError) as exc_info:
            processor.process_batch(
                items=items,
                process_fn=process_fn,
                fail_fast=True
            )
        
        assert "Processing failed for item" in str(exc_info.value)
        assert len(exc_info.value.errors) > 0
    
    def test_process_batch_return_errors(self):
        """Test batch processing with return_errors=True"""
        processor = ParallelProcessor(max_workers=4)
        
        def process_fn(x):
            if x == 3:
                raise ValueError(f"Error processing {x}")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        batch_result = processor.process_batch(
            items=items,
            process_fn=process_fn,
            fail_fast=False,
            return_errors=True
        )
        
        assert isinstance(batch_result, BatchResult)
        assert batch_result.total_count == 5
        assert batch_result.successful_count == 4
        assert batch_result.failed_count == 1
        assert len(batch_result.errors) == 1
        assert batch_result.errors[0].error_type == "ValueError"
    
    def test_process_batch_with_batching(self):
        """Test batch processing with batch_size"""
        processor = ParallelProcessor(max_workers=2)
        
        items = list(range(10))
        results = processor.process_batch(
            items=items,
            process_fn=lambda x: x * 2,
            batch_size=3
        )
        
        assert results == [x * 2 for x in items]
    
    def test_process_batch_preserves_order(self):
        """Test that results maintain original order"""
        processor = ParallelProcessor(max_workers=10)
        
        import time
        
        def process_fn(x):
            # Simulate variable processing time
            time.sleep(0.01 * (10 - x))  # Later items finish faster
            return x * 2
        
        items = list(range(10))
        results = processor.process_batch(items=items, process_fn=process_fn)
        
        # Results should still be in order despite variable timing
        assert results == [x * 2 for x in items]
    
    def test_processing_error_details(self):
        """Test that ProcessingError captures full details"""
        processor = ParallelProcessor(max_workers=2)

        def process_fn(x):
            if x == 2:
                raise RuntimeError("Test error with traceback")
            return x

        batch_result = processor.process_batch(
            items=[1, 2, 3],
            process_fn=process_fn,
            return_errors=True
        )

        assert len(batch_result.errors) == 1
        error = batch_result.errors[0]

        # item_index is the position in the list (1), not the value (2)
        assert error.item_index == 1
        assert error.error_type == "RuntimeError"
        assert "Test error with traceback" in error.error_message
        assert error.traceback is not None
        assert len(error.traceback) > 0
    
    def test_parallel_speedup(self):
        """Test that parallel processing is actually faster"""
        import time
        
        def slow_process(x):
            time.sleep(0.1)
            return x * 2
        
        items = list(range(5))
        
        # Sequential (max_workers=1)
        processor_seq = ParallelProcessor(max_workers=1)
        start = time.time()
        processor_seq.process_batch(items=items, process_fn=slow_process)
        time_seq = time.time() - start
        
        # Parallel (max_workers=5)
        processor_par = ParallelProcessor(max_workers=5)
        start = time.time()
        processor_par.process_batch(items=items, process_fn=slow_process)
        time_par = time.time() - start
        
        # Parallel should be significantly faster
        # With 5 items and 5 workers, should be ~5x faster
        assert time_par < time_seq / 2, f"Parallel ({time_par:.2f}s) not faster than sequential ({time_seq:.2f}s)"


class TestParallelProcessingError:
    """Test ParallelProcessingError exception"""
    
    def test_error_string_representation(self):
        """Test error string formatting"""
        errors = [
            ProcessingError(
                item_index=i,
                error_type="ValueError",
                error_message=f"Error {i}",
                traceback="traceback here",
                item_repr=f"item_{i}"
            )
            for i in range(10)
        ]
        
        exc = ParallelProcessingError("Test error", errors=errors)
        error_str = str(exc)
        
        # Should show first 5 errors
        assert "Error 0" in error_str
        assert "Error 4" in error_str
        
        # Should indicate there are more
        assert "and 5 more errors" in error_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

