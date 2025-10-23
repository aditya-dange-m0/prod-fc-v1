"""
Full-Stack Development Agent Factory - Production Ready
"""

import os
import platform
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# # Windows fix
# if platform.system() == "Windows":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from agent_state import FullStackAgentState
from context.runtime_context import RuntimeContext
from agent.prompts import BASE_SYSTEM_PROMPT
from agent.tools_loader import load_all_tools
from checkpoint.postgres_checkpointer import checkpointer_service

from .middleware_stack import create_middleware_stack

from langchain.embeddings import init_embeddings
from langgraph.store.postgres import PostgresStore  # ← Built-in!
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Configure logging
os.makedirs("log", exist_ok=True)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "log/log_agent_main_01.log", mode="w", encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


async def create_agent_graph(
    project_id: str,
    model_name: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.1,
    max_tokens: int = 8192,
    use_postgres_checkpointer: bool = True,
    debug: bool = True,
):
    """
    Create a LangGraph agent ready for streaming with PostgreSQL persistence.

    Args:
        project_id: Project identifier (used as thread_id for message history)
        model_name: Anthropic model to use
        temperature: Model temperature
        max_tokens: Maximum output tokens
        use_postgres_checkpointer: Enable PostgreSQL message persistence
        debug: Enable debug mode

    Returns:
        LangGraph agent ready for streaming with astream() and persistent history
    """
    logger.info(f"🤖 Creating streaming agent for project: {project_id}")

    # Load tools
    logger.info("\n📦 Loading tools...")
    tools = load_all_tools()
    if len(tools) == 0:
        raise RuntimeError("No tools loaded!")
    logger.info(f"✅ Loaded {len(tools)} tools")
    # Create model
    model = ChatAnthropic(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=120,
        max_retries=2,
    )
    logger.info("\n🔧 Creating middleware stack...")
    middleware = create_middleware_stack()
    logger.info(f"✅ Created {len(middleware)} middleware components")

    # Create checkpointer for persistent memory
    checkpointer = None
    if use_postgres_checkpointer:
        # Use the global checkpointer service (should already be initialized by FastAPI lifespan)
        if checkpointer_service._initialized:
            checkpointer = checkpointer_service.get_checkpointer()
            logger.info(
                "✅ Using PostgreSQL checkpointer for persistent message history"
            )
        else:
            logger.warning(
                "⚠️ Checkpointer service not initialized, falling back to in-memory"
            )
            checkpointer = InMemorySaver()
    else:
        checkpointer = InMemorySaver()
        logger.info("⚠️ Using in-memory checkpointer (messages will not persist)")

    # Create agent with persistent checkpointer
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=BASE_SYSTEM_PROMPT,
        state_schema=FullStackAgentState,
        context_schema=RuntimeContext,
        checkpointer=checkpointer,  # ← This enables message persistence!
        debug=debug,
        name="fullstack-agent",
        middleware=middleware,
    )

    logger.info("✅ Streaming agent created successfully with persistent history")
    return agent


# Convenience function for the streaming API
async def create_streaming_agent(project_id: str, **kwargs):
    """Create agent for streaming API usage with PostgreSQL persistence"""
    return await create_agent_graph(project_id=project_id, **kwargs)
