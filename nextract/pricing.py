from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic_ai.usage import RunUsage

@dataclass
class ModelPricing:
    input_per_1k: float
    output_per_1k: float

def parse_pricing_json(json_str: str) -> dict[str, ModelPricing]:
    if not json_str:
        return {}
    raw = json.loads(json_str)
    out: dict[str, ModelPricing] = {}
    for model, entry in raw.items():
        try:
            out[model] = ModelPricing(
                input_per_1k=float(entry["input_per_1k"]),
                output_per_1k=float(entry["output_per_1k"]),
            )
        except Exception:
            continue
    return out

def estimate_cost_usd(usage: RunUsage, model_name: str, pricing_map: dict[str, ModelPricing]) -> Optional[float]:
    mp = pricing_map.get(model_name)
    if not mp:
        return None
    cost = (usage.input_tokens / 1000.0) * mp.input_per_1k + (usage.output_tokens / 1000.0) * mp.output_per_1k
    return float(cost)
