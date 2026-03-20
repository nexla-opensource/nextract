"""
Integration tests for the hybrid architecture features.

Tests:
- Multi-pass extraction with real schemas
- Combined features (parallel + multi-pass)
"""

import pytest
from nextract.multipass import MultiPassExtractor
from nextract.parallel import ParallelProcessor
from nextract.config import RuntimeConfig


class TestMultiPassIntegration:
    """Test integration of multi-pass extraction"""
    
    @pytest.mark.asyncio
    async def test_multipass_with_simple_schema(self):
        """Test multi-pass extraction with a simple schema"""
        extractor = MultiPassExtractor(num_passes=3)
        
        # Mock extraction function
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Simulate varying extraction results
            data = {
                "name": "John Doe",
                "age": 30 + call_count  # Slightly different each time
            }
            
            return data, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="union"
        )
        
        assert result.total_passes == 3
        assert result.successful_passes == 3
        assert "name" in result.merged_data
        assert "age" in result.merged_data


class TestCombinedFeatures:
    """Test combining multiple features together"""

    def test_config_with_all_features(self):
        """Test RuntimeConfig with all new features enabled"""
        config = RuntimeConfig(
            model="openai:gpt-4o",
            max_workers=10,
            enable_multipass=True,
            num_passes=3,
            multipass_merge_strategy="union",
            enable_provenance=True
        )

        assert config.max_workers == 10
        assert config.enable_multipass is True
        assert config.num_passes == 3
        assert config.multipass_merge_strategy == "union"
        assert config.enable_provenance is True

class TestErrorHandlingIntegration:
    """Test error handling across integrated components"""
    
    @pytest.mark.asyncio
    async def test_multipass_with_partial_failures(self):
        """Test multi-pass extraction handles partial failures gracefully"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail on 2 passes
            if call_count in [2, 4]:
                raise RuntimeError(f"Simulated failure on pass {call_count}")
            
            return {"field1": "value"}, {
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
        
        # Should succeed with 3 successful passes
        assert result.successful_passes == 3
        assert result.failed_passes == 2
        assert "field1" in result.merged_data
    
    def test_parallel_processor_with_mixed_results(self):
        """Test parallel processor handles mix of successes and failures"""
        processor = ParallelProcessor(max_workers=4)
        
        def process_fn(x):
            if x % 3 == 0:
                raise ValueError(f"Multiple of 3: {x}")
            return x * 2
        
        batch_result = processor.process_batch(
            items=list(range(10)),
            process_fn=process_fn,
            fail_fast=False,
            return_errors=True
        )
        
        # Should have some successes and some failures
        assert batch_result.successful_count > 0
        assert batch_result.failed_count > 0
        assert batch_result.total_count == 10
        
        # Errors should be for multiples of 3
        error_indices = [e.item_index for e in batch_result.errors]
        assert all(i % 3 == 0 for i in error_indices)


class TestPerformanceIntegration:
    """Test performance characteristics of integrated features"""
    
    def test_parallel_processing_speedup(self):
        """Test that parallel processing provides speedup"""
        import time
        
        def slow_process(x):
            time.sleep(0.05)
            return x * 2
        
        items = list(range(10))
        
        # Sequential
        processor_seq = ParallelProcessor(max_workers=1)
        start = time.time()
        processor_seq.process_batch(items=items, process_fn=slow_process)
        time_seq = time.time() - start
        
        # Parallel
        processor_par = ParallelProcessor(max_workers=5)
        start = time.time()
        processor_par.process_batch(items=items, process_fn=slow_process)
        time_par = time.time() - start
        
        # Parallel should be faster
        speedup = time_seq / time_par
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"
    
class TestConfigurationIntegration:
    """Test configuration propagation through the system"""
    
    def test_config_defaults(self):
        """Test that config defaults are sensible"""
        from nextract.config import (
            DEFAULT_MAX_WORKERS,
            DEFAULT_ENABLE_MULTIPASS,
            DEFAULT_NUM_PASSES,
            DEFAULT_MULTIPASS_MERGE_STRATEGY,
            DEFAULT_ENABLE_PROVENANCE
        )
        
        assert DEFAULT_MAX_WORKERS == 10
        assert DEFAULT_ENABLE_MULTIPASS is False
        assert DEFAULT_NUM_PASSES == 3
        assert DEFAULT_MULTIPASS_MERGE_STRATEGY == "union"
        assert DEFAULT_ENABLE_PROVENANCE is False
    
    def test_config_override(self):
        """Test that config can be overridden"""
        config = RuntimeConfig(
            max_workers=20,
            enable_multipass=True,
            num_passes=5,
            multipass_merge_strategy="intersection",
            enable_provenance=True
        )
        
        assert config.max_workers == 20
        assert config.enable_multipass is True
        assert config.num_passes == 5
        assert config.multipass_merge_strategy == "intersection"
        assert config.enable_provenance is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

