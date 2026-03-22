from __future__ import annotations

from typing import Any

from nextract.core import ExtractionPlan


class PipelineRouter:
    """Placeholder router for future extractor/provider routing."""

    def route(self, plan: ExtractionPlan, **kwargs: Any) -> ExtractionPlan:
        return plan
