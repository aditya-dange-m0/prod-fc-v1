"""
Iteration Tracker Middleware - Track agent iterations.

Runs: before_model hook
Purpose: Increment iteration counter for debugging and triggers
"""

import logging
from typing import Optional, Dict, Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from agent_state import FullStackAgentState

logger = logging.getLogger(__name__)


class IterationTrackerMiddleware(AgentMiddleware[FullStackAgentState]):
    """Track iteration count for debugging and summarization triggers"""

    state_schema = FullStackAgentState

    def before_model(
        self, state: FullStackAgentState, runtime: Runtime
    ) -> Optional[Dict[str, Any]]:
        """Increment iteration count before each model call"""

        current_count = state.get("iteration_count", 0)
        new_count = current_count + 1

        logger.debug(f"🔢 Iteration: {new_count}")

        return {"iteration_count": new_count}
