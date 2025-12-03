"""
Tests for multi-pass extraction functionality.

Tests:
- MultiPassExtractor basic functionality
- Different merge strategies
- Error handling and fail_threshold
- Usage and cost aggregation
"""

import pytest
from nextract.multipass import MultiPassExtractor, MultiPassExtractionError, PassResult, MultiPassResult
from nextract.schema import JsonSchema


class TestMultiPassExtractor:
    """Test MultiPassExtractor class"""
    
    def test_initialization(self):
        """Test extractor initialization"""
        extractor = MultiPassExtractor(num_passes=3)
        assert extractor.num_passes == 3
        assert extractor.fail_threshold == 2  # num_passes - 1
    
    def test_initialization_custom_fail_threshold(self):
        """Test initialization with custom fail_threshold"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=3)
        assert extractor.num_passes == 5
        assert extractor.fail_threshold == 3
    
    def test_initialization_invalid_num_passes(self):
        """Test initialization with invalid num_passes"""
        with pytest.raises(ValueError, match="num_passes must be >= 1"):
            MultiPassExtractor(num_passes=0)
    
    def test_initialization_invalid_fail_threshold(self):
        """Test initialization with invalid fail_threshold"""
        with pytest.raises(ValueError, match="fail_threshold must be >= 0"):
            MultiPassExtractor(num_passes=3, fail_threshold=-1)
    
    @pytest.mark.asyncio
    async def test_extract_multipass_union_strategy(self):
        """Test multi-pass extraction with union merge strategy"""
        extractor = MultiPassExtractor(num_passes=3)
        
        # Mock extraction function that returns different data each time
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                data = {"field1": "value1", "field2": "value2"}
            elif call_count == 2:
                data = {"field2": "value2_updated", "field3": "value3"}
            else:
                data = {"field1": "value1", "field4": "value4"}
            
            return data, {
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
        
        assert isinstance(result, MultiPassResult)
        assert result.total_passes == 3
        assert result.successful_passes == 3
        assert result.failed_passes == 0
        
        # Union should have all fields
        assert "field1" in result.merged_data
        assert "field2" in result.merged_data
        assert "field3" in result.merged_data
        assert "field4" in result.merged_data
    
    @pytest.mark.asyncio
    async def test_extract_multipass_intersection_strategy(self):
        """Test multi-pass extraction with intersection merge strategy"""
        extractor = MultiPassExtractor(num_passes=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # All passes have field1, only some have field2
            if call_count == 1:
                data = {"field1": "value1", "field2": "value2"}
            elif call_count == 2:
                data = {"field1": "value1", "field3": "value3"}
            else:
                data = {"field1": "value1", "field2": "value2"}
            
            return data, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="intersection"
        )
        
        # Intersection should only have field1 (present in all passes)
        assert "field1" in result.merged_data
        assert "field2" not in result.merged_data
        assert "field3" not in result.merged_data
    
    @pytest.mark.asyncio
    async def test_extract_multipass_majority_strategy(self):
        """Test multi-pass extraction with majority merge strategy"""
        extractor = MultiPassExtractor(num_passes=5)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # field1 appears in 4/5 passes (majority)
            # field2 appears in 2/5 passes (not majority)
            if call_count <= 4:
                data = {"field1": "value1"}
            else:
                data = {"field2": "value2"}
            
            if call_count in [1, 3]:
                data["field2"] = "value2"
            
            return data, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="majority"
        )
        
        # field1 should be present (4/5 = 80% > 50%)
        assert "field1" in result.merged_data
        
        # field2 should not be present (2/5 = 40% < 50%)
        assert "field2" not in result.merged_data
    
    @pytest.mark.asyncio
    async def test_extract_multipass_highest_confidence_strategy(self):
        """Test multi-pass extraction with highest_confidence merge strategy"""
        extractor = MultiPassExtractor(num_passes=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Each pass has different confidence
            data = {
                "field1": f"value_{call_count}",
                "_confidence": 0.5 + (call_count * 0.1)  # 0.6, 0.7, 0.8
            }
            
            return data, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="highest_confidence"
        )
        
        # Should use data from pass 3 (highest confidence = 0.8)
        assert result.merged_data["field1"] == "value_3"
    
    @pytest.mark.asyncio
    async def test_extract_multipass_first_non_empty_strategy(self):
        """Test multi-pass extraction with first_non_empty merge strategy"""
        extractor = MultiPassExtractor(num_passes=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # First pass has empty field1, second has value
            if call_count == 1:
                data = {"field1": "", "field2": "value2"}
            elif call_count == 2:
                data = {"field1": "value1", "field2": ""}
            else:
                data = {"field1": "value1_updated", "field2": "value2_updated"}
            
            return data, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        result = await extractor.extract_multipass(
            extraction_fn=mock_extraction,
            schema=schema,
            merge_strategy="first_non_empty"
        )
        
        # field1 should be from pass 2 (first non-empty)
        assert result.merged_data["field1"] == "value1"
        
        # field2 should be from pass 1 (first non-empty)
        assert result.merged_data["field2"] == "value2"
    
    @pytest.mark.asyncio
    async def test_extract_multipass_with_errors(self):
        """Test multi-pass extraction with some failed passes"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=3)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail on passes 2 and 4
            if call_count in [2, 4]:
                raise ValueError(f"Extraction failed on pass {call_count}")
            
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
        
        assert result.total_passes == 5
        assert result.successful_passes == 3
        assert result.failed_passes == 2
        
        # Should still have data from successful passes
        assert "field1" in result.merged_data
    
    @pytest.mark.asyncio
    async def test_extract_multipass_exceeds_fail_threshold(self):
        """Test multi-pass extraction that exceeds fail_threshold"""
        extractor = MultiPassExtractor(num_passes=5, fail_threshold=2)
        
        call_count = 0
        
        async def mock_extraction(**kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail on 3 passes (exceeds threshold of 2)
            if call_count in [1, 3, 5]:
                raise ValueError(f"Extraction failed on pass {call_count}")
            
            return {"field1": f"value_{call_count}"}, {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "cost_estimate_usd": 0.01,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        with pytest.raises(MultiPassExtractionError) as exc_info:
            await extractor.extract_multipass(
                extraction_fn=mock_extraction,
                schema=schema,
                merge_strategy="union"
            )
        
        assert "Too many failed passes" in str(exc_info.value)
        assert "3 failed" in str(exc_info.value)
        assert "threshold: 2" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_extract_multipass_usage_aggregation(self):
        """Test that usage is correctly aggregated across passes"""
        extractor = MultiPassExtractor(num_passes=3)
        
        async def mock_extraction(**kwargs):
            return {"field1": "value1"}, {
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
        
        # Should aggregate usage from all 3 passes
        assert result.total_usage["prompt_tokens"] == 300  # 100 * 3
        assert result.total_usage["completion_tokens"] == 150  # 50 * 3
        assert result.total_cost == pytest.approx(0.03)  # 0.01 * 3
    
    @pytest.mark.asyncio
    async def test_extract_multipass_invalid_strategy(self):
        """Test multi-pass extraction with invalid merge strategy"""
        extractor = MultiPassExtractor(num_passes=2)
        
        async def mock_extraction(**kwargs):
            return {"field1": "value1"}, {
                "usage": {},
                "cost_estimate_usd": 0.0,
                "warnings": []
            }
        
        schema = {"type": "object", "properties": {}}
        
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            await extractor.extract_multipass(
                extraction_fn=mock_extraction,
                schema=schema,
                merge_strategy="invalid_strategy"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

