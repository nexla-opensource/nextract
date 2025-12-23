"""
Unit tests for prompt building functions.

Tests:
- default_system_prompt
- build_examples_block
- combine_system_prompt
- build_improvement_system_prompt
"""


from nextract.prompts import (
    build_examples_block,
    build_improvement_system_prompt,
    combine_system_prompt,
    default_system_prompt,
)


class TestDefaultSystemPrompt:
    """Tests for default_system_prompt."""

    def test_without_extra(self):
        """Prompt without include_extra."""
        prompt = default_system_prompt(include_extra=False)
        
        assert "information extraction agent" in prompt
        assert "structured output" in prompt
        assert "additional, clearly relevant fields" not in prompt

    def test_with_extra(self):
        """Prompt with include_extra=True."""
        prompt = default_system_prompt(include_extra=True)
        
        assert "information extraction agent" in prompt
        assert "extra" in prompt
        assert "additional" in prompt

    def test_contains_anti_hallucination_guidance(self):
        """Prompt should contain anti-hallucination guidance."""
        prompt = default_system_prompt(include_extra=False)
        
        assert "Do not invent" in prompt
        assert "Only rely on the content" in prompt


class TestBuildExamplesBlock:
    """Tests for build_examples_block."""

    def test_none_returns_empty(self):
        """None examples returns empty string."""
        result = build_examples_block(None)
        assert result == ""

    def test_empty_list_returns_empty(self):
        """Empty list returns empty string."""
        result = build_examples_block([])
        assert result == ""

    def test_dict_examples(self):
        """Dict examples are formatted correctly."""
        examples = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
        ]
        result = build_examples_block(examples)
        
        assert "EXAMPLES" in result
        assert "OUTPUT EXAMPLE" in result
        assert "John" in result

    def test_tuple_examples(self):
        """Tuple examples include input excerpt."""
        examples = [
            ("Invoice #123", {"invoice_number": "123"}),
        ]
        result = build_examples_block(examples)
        
        assert "INPUT EXCERPT" in result
        assert "Invoice #123" in result
        assert "OUTPUT EXAMPLE" in result

    def test_mixed_examples(self):
        """Mixed dict and tuple examples work."""
        examples = [
            {"simple": "example"},
            ("input text", {"output": "example"}),
        ]
        result = build_examples_block(examples)
        
        assert "simple" in result
        assert "input text" in result


class TestCombineSystemPrompt:
    """Tests for combine_system_prompt."""

    def test_base_only(self):
        """Base prompt without additions."""
        result = combine_system_prompt(
            user_hint=None,
            include_extra=False,
            examples_block="",
        )
        
        assert "information extraction agent" in result
        assert "USER HINT" not in result

    def test_with_user_hint(self):
        """Prompt with user hint."""
        result = combine_system_prompt(
            user_hint="Extract invoice data carefully",
            include_extra=False,
            examples_block="",
        )
        
        assert "USER HINT" in result
        assert "Extract invoice data carefully" in result

    def test_with_examples(self):
        """Prompt with examples block."""
        examples = "EXAMPLES:\n- Example 1"
        result = combine_system_prompt(
            user_hint=None,
            include_extra=False,
            examples_block=examples,
        )
        
        assert "EXAMPLES" in result
        assert "Example 1" in result

    def test_with_all_options(self):
        """Prompt with all options."""
        result = combine_system_prompt(
            user_hint="Custom hint",
            include_extra=True,
            examples_block="EXAMPLES:\n- Test",
        )
        
        assert "Custom hint" in result
        assert "extra" in result
        assert "EXAMPLES" in result


class TestBuildImprovementSystemPrompt:
    """Tests for build_improvement_system_prompt."""

    def test_basic_prompt(self):
        """Basic improvement prompt."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = build_improvement_system_prompt(schema, user_hint=None)
        
        assert "improves a JSON Schema" in result
        assert "CURRENT SCHEMA" in result
        assert "name" in result

    def test_with_user_hint(self):
        """Improvement prompt with user hint."""
        schema = {"type": "object", "properties": {}}
        result = build_improvement_system_prompt(
            schema,
            user_hint="Extract vendor information",
        )
        
        assert "CURRENT USER PROMPT" in result
        assert "Extract vendor information" in result

    def test_contains_constraints(self):
        """Prompt contains improvement constraints."""
        schema = {"type": "object", "properties": {}}
        result = build_improvement_system_prompt(schema, user_hint=None)
        
        assert "Preserve field semantics" in result
        assert "Do not invent fields" in result
