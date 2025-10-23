"""
Production Singleton Agent Graph
===============================

Creates agent graph ONCE on startup, reuses for all requests.
This is the PRODUCTION approach for FastAPI/web servers.
"""

import asyncio
import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from checkpoint.postgres_checkpointer import checkpointer_service
from agent_state import FullStackAgentState
from context.runtime_context import RuntimeContext
from agent.prompts import BASE_SYSTEM_PROMPT
from agent.tools_loader import load_all_tools
from middleware.metadata_extractor import MetadataExtractorMiddleware

logger = logging.getLogger(__name__)

# Global compiled graph (singleton)
_compiled_graph: Optional = None
_graph_lock = asyncio.Lock()


async def get_agent_graph():
    """
    Get singleton agent graph instance.
    
    Creates graph ONCE on first call, reuses for all subsequent requests.
    This is the PRODUCTION approach for FastAPI/web servers.
    """
    global _compiled_graph
    
    if _compiled_graph is not None:
        return _compiled_graph
    
    async with _graph_lock:
        # Double-check after acquiring lock
        if _compiled_graph is not None:
            return _compiled_graph
        
        logger.info("[AGENT] Initializing singleton agent graph...")
        
        # Ensure checkpointer initialized
        if not checkpointer_service._initialized:
            await checkpointer_service.initialize()
        
        # Get checkpointer instance (shared across all requests)
        checkpointer = checkpointer_service.get_checkpointer()
        
        # Load tools
        tools = load_all_tools()
        logger.info(f"📦 Loaded {len(tools)} tools")
        
        # Create model
        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            temperature=0.1,
            timeout=120,
        )
        
        
        # Compile graph ONCE
        _compiled_graph = create_agent(
            model=model,
            tools=tools,
            system_prompt=BASE_SYSTEM_PROMPT,
            state_schema=FullStackAgentState,
            context_schema=RuntimeContext,
            checkpointer=checkpointer,  # ← Shared checkpointer!
            debug=True,
            name="production-fullstack-agent",
            middleware=[MetadataExtractorMiddleware()],
        )
        
        logger.info("✅ Agent graph initialized (singleton)")
        
        return _compiled_graph


# For backward compatibility
async def create_agent_graph(project_id: str = None):
    """Legacy function - redirects to singleton"""
    return await get_agent_graph()


async def create_streaming_agent(project_id: str = None, **kwargs):
    """Legacy function - redirects to singleton"""
    return await get_agent_graph()