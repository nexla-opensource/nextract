"""
Tests for comprehensive error handling and traceback capture.

Tests:
- Error handling in parallel processing
- Error handling in multi-pass extraction
- Traceback capture and formatting
- Informative error messages
"""

import pytest
from nextract.parallel import ParallelProcessor, ParallelProcessingError
from nextract.multipass import MultiPassExtractor, MultiPassExtractionError


class TestParallelErrorHandling:
    """Test error handling in parallel processing"""
    
    def test_error_traceback_capture(self):
        """Test that full traceback is captured"""
        processor = ParallelProcessor(max_workers=2)
        
        def failing_function(x):
            if x == 2:
                # Create a nested error to test traceback depth
                def inner_function():
                    raise RuntimeError("Deep error in nested function")
                inner_function()
            return x
        
        batch_result = processor.process_batch(
            items=[1, 2, 3],
            process_fn=failing_function,
            return_errors=True
        )
        
        assert len(batch_result.errors) == 1
        error = batch_result.errors[0]
        
        # Check traceback is captured
        assert error.traceback is not None
        assert len(error.traceback) > 0
        
        # Check traceback contains function names
        assert "failing_function" in error.traceback
        assert "inner_function" in error.traceback
        assert "Deep error in nested function" in error.traceback
    
    def test_error_message_informativeness(self):
        """Test that error messages are informative"""
        processor = ParallelProcessor(max_workers=2)
        
        def failing_function(x):
            if x == 2:
                raise ValueError(f"Invalid value: {x}. Expected value < 2.")
            return x
        
        batch_result = processor.process_batch(
            items=[1, 2, 3],
            process_fn=failing_function,
            return_errors=True
        )
        
        error = batch_result.errors[0]
        
        # Error message should be preserved
        assert "Invalid value: 2" in error.error_message
        assert "Expected value < 2" in error.error_message
        
        # Error type should be captured
        assert error.error_type == "ValueError"
    
    def test_multiple_errors_summary(self):
        """Test that multiple errors are summarized properly"""
        processor = ParallelProcessor(max_workers=4)
        
        def failing_function(x):
            if x % 2 == 0:
                raise ValueError(f"Even number not allowed: {x}")
            return x
        
        batch_result = processor.process_batch(
            items=list(range(10)),
            process_fn=failing_function,
            return_errors=True
        )
        
        # Should have 5 errors (0, 2, 4, 6, 8)
        assert len(batch_result.errors) == 5
        
        # All should be ValueError
        assert all(e.error_type == "ValueError" for e in batch_result.errors)
        
        # Each should have unique item_index
        indices = [e.item_index for e in batch_result.errors]
        assert len(set(indices)) == 5
        assert all(i % 2 == 0 for i in indices)
    
    def test_error_with_fail_fast(self):
        """Test error handling with fail_fast=True"""
        processor = ParallelProcessor(max_workers=2)
        
        def failing_function(x):
            if x == 5:
                raise RuntimeError("Critical error at item 5")
            return x
        
        with pytest.raises(ParallelProcessingError) as exc_info:
            processor.process_batch(
                items=list(range(10)),
                process_fn=failing_function,
                fail_fast=True
            )
        
        # Exception should have informative message
        error_msg = str(exc_info.value)
        assert "Processing failed" in error_msg
        assert "Critical error at item 5" in error_msg
        
        # Should have error details
        assert len(exc_info.value.errors) > 0
    
    def test_error_item_representation(self):
        """Test that item representation is captured in errors"""
        processor = ParallelProcessor(max_workers=2)
        
        class CustomItem:
            def __init__(self, value):
                self.value = value
            
            def __repr__(self):
                return f"CustomItem(value={self.value})"
        
        def failing_function(item):
            if item.value == 2:
                raise ValueError("Invalid item")
            return item.value
        
        items = [CustomItem(i) for i in range(5)]
        
        batch_result = processor.process_batch(
            items=items,
            process_fn=failing_function,
            return_errors=True
        )
        
        error = batch_result.errors[0]
        
        # Item representation should be captured
        assert "CustomItem(value=2)" in error.item_repr


class TestMultiPassErrorHandling:
    """Test error handling in multi-pass extraction"""
    
    @pytest.mark.skip(reason="Implementation pending: error tracking in pass_results")
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test handling of partial failures in multi-pass"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail on passes 2 and 4
            if call_count in [2, 4]:
                raise RuntimeError(f"Extraction failed on pass {call_count}")
            
            return {"field1": f"value_{call_count}"}, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="union"
        )
        
        # Should track failures
        assert result.failed_passes == 2
        assert result.successful_passes == 3
        
        # Should have error details in pass_results
        failed_passes = [p for p in result.pass_results if p.errors]
        assert len(failed_passes) == 2
    
    @pytest.mark.skip(reason="Implementation pending: fail_threshold error handling")
    @pytest.mark.asyncio
    async def test_fail_threshold_exceeded_error(self):
        """Test informative error when fail_threshold is exceeded"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=1)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail on 2 passes (exceeds threshold of 1)
            if call_count in [2, 4]:
                raise ValueError(f"Pass {call_count} failed with specific error")
            
            return {"field1": "value"}, {
                "usage": {},
                "cost_estimate_usd": 0.0,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        with pytest.raises(MultiPassExtractionError) as exc_info:
            await extractor.extract_multipass(
                extraction_fn=mock_extraction,
                schema=schema,
                merge_strategy="union"
            )
        
        error_msg = str(exc_info.value)
        
        # Should have informative message
        assert "Too many failed passes" in error_msg
        assert "2 failed" in error_msg
        assert "threshold: 1" in error_msg
        
        # Should include error details
        assert exc_info.value.failed_count == 2
        assert len(exc_info.value.errors) == 2
    
    @pytest.mark.skip(reason="Implementation pending: error details preservation")
    @pytest.mark.asyncio
    async def test_error_details_in_pass_results(self):
        """Test that error details are preserved in pass results"""
        extractor = MultiPassExtractor(num_passes=3, fail_threshold=2)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 2:
                raise RuntimeError("Detailed error message with context")
            
            return {"field1": "value"}, {
                "usage": {},
                "cost_estimate_usd": 0.0,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="union"
        )
        
        # Find the failed pass
        failed_pass = result.pass_results[1]  # Pass 2 (index 1)
        
        assert failed_pass.errors is not None
        assert "Detailed error message with context" in failed_pass.errors


class TestErrorRecovery:
    """Test error recovery and resilience"""
    
    def test_parallel_continues_after_errors(self):
        """Test that parallel processing continues after errors"""
        processor = ParallelProcessor(max_workers=4)
        
        processed_items = []
        
        def process_with_tracking(x):
            processed_items.append(x)
            if x in [2, 5, 7]:
                raise ValueError(f"Error on {x}")
            return x * 2
        
        batch_result = processor.process_batch(
            items=list(range(10)),
            process_fn=process_with_tracking,
            fail_fast=False,
            return_errors=True
        )
        
        # All items should have been attempted
        assert len(processed_items) == 10
        
        # Should have 7 successful results
        assert batch_result.successful_count == 7
        
        # Should have 3 errors
        assert batch_result.failed_count == 3
    
    @pytest.mark.asyncio
    async def test_multipass_recovers_from_transient_errors(self):
        """Test that multi-pass can recover from transient errors"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Simulate transient error on pass 2
            if call_count == 2:
                raise ConnectionError("Transient network error")
            
            return {"field1": "value"}, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        # Should succeed despite one transient error
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="union"
        )
        
        assert result.successful_passes == 4
        assert result.failed_passes == 1
        assert "field1" in result.merged_data


class TestErrorMessageQuality:
    """Test quality and informativeness of error messages"""
    
    @pytest.mark.skip(reason="Implementation pending: parallel error context")
    def test_parallel_error_includes_context(self):
        """Test that parallel errors include helpful context"""
        processor = ParallelProcessor(max_workers=2)
        
        def process_fn(x):
            if x == 3:
                raise ValueError(f"Value {x} is invalid because it's odd and greater than 2")
            return x
        
        batch_result = processor.process_batch(
            items=[1, 2, 3, 4],
            process_fn=process_fn,
            return_errors=True
        )
        
        error = batch_result.errors[0]
        
        # Error should include:
        # 1. Item index
        assert error.item_index == 3
        
        # 2. Error type
        assert error.error_type == "ValueError"
        
        # 3. Full error message
        assert "Value 3 is invalid" in error.error_message
        assert "odd and greater than 2" in error.error_message
        
        # 4. Traceback
        assert error.traceback is not None
        
        # 5. Item representation
        assert "3" in error.item_repr
    
    @pytest.mark.skip(reason="Implementation pending: multipass error details")
    @pytest.mark.asyncio
    async def test_multipass_error_includes_pass_number(self):
        """Test that multi-pass errors include pass number"""
        extractor = MultiPassExtractor(num_passes=3, fail_threshold=2)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 2:
                raise ValueError("Error in extraction")
            
            return {"field1": "value"}, {
                "usage": {},
                "cost_estimate_usd": 0.0,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="union"
        )
        
        # Failed pass should have pass_number
        failed_pass = result.pass_results[1]
        assert failed_pass.pass_number == 2
        assert failed_pass.errors is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

