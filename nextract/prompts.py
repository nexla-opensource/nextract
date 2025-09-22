from __future__ import annotations

import json
from typing import Any, Iterable, Optional

def default_system_prompt(include_extra: bool) -> str:
    extra_hint = (
        "\nAlso include any additional, clearly relevant fields you find under the top-level key `extra`."
        if include_extra else ""
    )
    return (
        "You are an information extraction agent.\n"
        "Read the attached files exactly as provided (no assumptions), and produce a structured output that STRICTLY matches the schema.\n"
        "If a required field is not supported by the content, infer conservatively or leave it blank/null if the schema allows. "
        "Do not invent facts. Only rely on the content provided.\n"
        f"Return only the structured object—no prose, no explanations.{extra_hint}"
    )

def build_examples_block(examples: Optional[Iterable[dict[str, Any] | tuple[Optional[str], dict[str, Any]]]]) -> str:
    """Examples can be either:
      - dicts (treated as output-only examples), or
      - tuples (input_excerpt, output_example_dict)
    We'll serialize into a compact instruction block.
    """
    if not examples:
        return ""
    lines: list[str] = ["\nEXAMPLES (for the model, not to echo):"]
    for ex in examples:
        if isinstance(ex, dict):
            lines.append("  - OUTPUT EXAMPLE:\n" + json.dumps(ex, ensure_ascii=False))
        else:
            inp, out = ex  # type: ignore[misc]
            if inp:
                lines.append("  - INPUT EXCERPT:\n" + str(inp))
            lines.append("    OUTPUT EXAMPLE:\n" + json.dumps(out, ensure_ascii=False))
    return "\n".join(lines)

def combine_system_prompt(user_hint: Optional[str], include_extra: bool, examples_block: str) -> str:
    base = default_system_prompt(include_extra)
    if user_hint:
        base = base + f"\n\nUSER HINT:\n{user_hint.strip()}"
    if examples_block:
        base = base + f"\n\n{examples_block}"
    return base
