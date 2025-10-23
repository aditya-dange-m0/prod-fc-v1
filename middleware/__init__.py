"""
Middleware for full-stack agent.

Import all middleware classes for easy access.
"""

from .metadata_extractor import MetadataExtractorMiddleware
from .dynamic_prompts import phase_aware_prompt
from .token_tracker import TokenTrackerMiddleware
from .iteration_tracker import IterationTrackerMiddleware
from langchain_core.callbacks import UsageMetadataCallbackHandler

__all__ = [
    "MetadataExtractorMiddleware",
    "phase_aware_prompt",
    "TokenTrackerMiddleware",
    "IterationTrackerMiddleware",
    "UsageMetadataCallbackHandler"
]



# callback = UsageMetadataCallbackHandler()

# result = await agent.ainvoke(
#     state,
#     context=context,
#     config={"callbacks": [callback]}
# )

# print(callback.usage_metadata)
