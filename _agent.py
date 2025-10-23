# # pip install -qU "langchain[anthropic]" to call the model
# import os
# import dotenv
# import asyncio

# dotenv.load_dotenv()

# from langchain.agents import create_agent, AgentState
# from langchain_anthropic import ChatAnthropic
# from langchain.tools import tool, ToolRuntime
# from dataclasses import dataclass
# from langchain_core.tools import BaseTool
# from langgraph.checkpoint.memory import InMemorySaver
# from db.service import DatabaseService, db_service
# from langchain_openai import ChatOpenAI
# from tools.file_tools_e2b import (
#     read_file,
#     write_file,
#     file_exists,
#     list_directory,
#     create_directory,
#     delete_file,
#     batch_read_files,
#     batch_write_files,
# )
# from langchain.agents.middleware import (
#     SummarizationMiddleware,
#     ContextEditingMiddleware,
#     ModelCallLimitMiddleware,
# )
# from langchain.agents.middleware.context_editing import ClearToolUsesEdit


# """
# Agent state schema for full-stack development agent.

# """

# from typing import TypedDict, Optional, Annotated, List
# from typing_extensions import NotRequired
# from langchain.agents import AgentState
# import logging

# logger = logging.getLogger(__name__)


# class FullStackAgentState(AgentState):
#     user_id: NotRequired[str]
#     project_id: NotRequired[str]
#     current_phase: NotRequired[str]
#     """
#     Current development phase.
#     Values: 'planning' | 'backend_dev' | 'frontend_dev' | 'testing' | 'integration'
#     Updated from: <phase> XML tag in agent responses
#     Used for: Dynamic prompts, context injection, progress tracking
#     """

#     next_steps: NotRequired[list[str]]
#     """
#     List of concrete next actions planned by agent.
#     Updated from: <next_steps> XML tag in agent responses
#     Used for: UI progress display, recovery after interruption
#     """

#     recent_thinking: NotRequired[list[dict]]
#     """
#     Last 5 thoughts from agent (lightweight memory).
#     Structure:
#         [{
#             'thinking': str,      # 2-3 sentence reasoning
#             'phase': str,         # Phase when thought occurred
#             'iteration': int,     # Iteration number
#             'timestamp': str      # ISO timestamp
#         }, ...]
#     Updated from: <thinking> XML tag in agent responses
#     Used for: Quick context retrieval, debugging, continuity
#     Size: Capped at 5 entries (oldest removed)
#     """

#     error_count: NotRequired[int]
#     """
#     Consecutive error count (circuit breaker).
#     Incremented: When <error> tag present in response
#     Reset to 0: When no error in response
#     Used for: Stop agent after N consecutive failures
#     """

#     last_error: NotRequired[Optional[dict]]
#     """
#     Most recent error details.
#     Structure:
#         {
#             'description': str,   # Error description
#             'severity': str,      # 'low' | 'medium' | 'high' | 'critical'
#             'timestamp': str      # ISO timestamp
#         }
#     Updated from: <error> XML tag in agent responses
#     Used for: Error recovery, debugging, logging
#     """

#     iteration_count: NotRequired[int]
#     """
#     Total LLM calls in this session.

#     Updated by: IterationTrackerMiddleware (before each model call)
#     Used for: Debugging, cost tracking, trigger periodic summarization
#     """

#     last_summarized_at: NotRequired[int]
#     """
#     Iteration number when last summarization occurred.

#     Updated by: SummarizationMiddleware (after summarization)
#     Used for: Determine when next summarization needed
#     """

#     working_directory: NotRequired[str]
#     """
#     Current working directory in E2B sandbox.

#     Default: '/workspace'
#     Updated by: File tools (change_directory, etc.)
#     Used for: Resolve relative paths, context in prompts
#     """

#     active_files: NotRequired[list[str]]
#     """
#     Recently accessed files (last 10).
#     Updated by: File/edit tools (read_file, write_file, edit_file)
#     Used for: Context injection, "what was I working on?"
#     Size: Capped at 10 entries (oldest removed)
#     """

#     service_pids: NotRequired[dict[str, int]]
#     """
#     Running background services (dev servers, etc.).
#     Structure:
#         {
#             'fastapi_dev': 1234,
#             'react_dev': 5678
#         }
#     Updated by: Command tools (run_service, stop_service)
#     Used for: Track running processes, stop services cleanly
#     """

#     tokens_used: NotRequired[dict]
#     """
#     Token usage tracking across the session.

#     Structure:
#         {
#             'total_input': int,       # Total input tokens
#             'total_output': int,      # Total output tokens
#             'total_cost': float,      # Estimated cost in USD
#             'by_model': {             # Per-model breakdown
#                 'claude-sonnet-4': {
#                     'input': int,
#                     'output': int,
#                     'calls': int
#                 }
#             }
#         }

#     Updated by: TokenTrackerMiddleware (post-model hook)
#     Used for: Cost monitoring, optimization, user billing
#     """


# def load_file_tools() -> List[BaseTool]:
#     """
#     Load all available tools for the agent.

#     Returns:
#         List of LangChain tools ready for agent use
#     """

#     tools = []

#     # =========================================================================
#     # FILE TOOLS (E2B Sandbox) - Import individual tools
#     # =========================================================================
#     try:
#         from tools.file_tools_e2b import (
#             read_file,
#             write_file,
#             file_exists,
#             list_directory,
#             create_directory,
#             delete_file,
#             batch_read_files,
#             batch_write_files,
#         )

#         file_tools = [
#             read_file,
#             write_file,
#             file_exists,
#             list_directory,
#             create_directory,
#             delete_file,
#             batch_read_files,
#             batch_write_files,
#         ]

#         tools.extend(file_tools)
#         logger.info(f"✅ Loaded {len(file_tools)} file tools")
#     except Exception as e:
#         logger.error(f"❌ Failed to load file tools: {e}", exc_info=True)

#     return tools


# @dataclass
# class RuntimeContext:

#     user_id: str
#     project_id: str

#     @property
#     def session_id(self) -> str:
#         """Alias for project_id (compatibility)"""
#         return self.project_id

#     @property
#     def thread_id(self) -> str:
#         """LangChain thread_id (same as project_id)"""
#         return self.project_id

#     sandbox_id: Optional[str] = None
#     """E2B sandbox ID (from Project.active_sandbox_id)"""

#     sandbox_state: str = "none"
#     """
#     Sandbox state (from Project.sandbox_state)
#     Values: 'running' | 'paused' | 'killed' | 'none'
#     """

#     @property
#     def sandbox_active(self) -> bool:
#         """Whether sandbox is currently running"""
#         return self.sandbox_state == "running"

#     # =========================================================================
#     # SESSION STATUS (From Database)
#     # =========================================================================

#     session_status: str = "active"
#     """
#     Session status (from Project.status)
#     Values: 'active' | 'paused' | 'ended'
#     """

#     # =========================================================================
#     # CONFIGURATION
#     # =========================================================================

#     environment: str = "development"
#     """Environment: 'development' | 'staging' | 'production'"""

#     debug_mode: bool = False
#     """Enable debug logging"""

#     max_iterations: int = 25
#     """Maximum agent iterations"""


# system_prompt = """You are a full stack code generation agent for now only work with fastapi backend"""

# model = ChatAnthropic(
#     model="claude-sonnet-4-5-20250929",
#     max_tokens=10000,
#     temperature=0.1,
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     timeout=120,
# )

# tools = load_file_tools()

# # context = RuntimeContext(user_id="test_adi_boy_01", project_id="test_adi_boy_prod_01")
# checkpointer = InMemorySaver()
# graph = create_agent(
#     model=model,
#     system_prompt=system_prompt,
#     middleware=[
#         SummarizationMiddleware(
#             model=ChatOpenAI(
#                 model="gpt-4o-mini", temperature=0
#             ),  # Use ChatOpenAI instance
#             max_tokens_before_summary=5000,
#             messages_to_keep=2,
#             summary_prompt="""Summarize ONLY the key technical facts from the conversation:

#             **Files Created/Modified:**
#             - List file paths only

#             **Commands Executed:**
#             - List commands only

#             **Current Task:**
#             - One sentence describing what's being worked on

#             **Errors Fixed:**
#             - List any resolved errors

#             Be EXTREMELY concise. Do NOT provide explanations or recommendations.""",
#         ),
#         # ContextEditingMiddleware(
#         #     edits=[
#         #         ClearToolUsesEdit(
#         #             trigger=500,
#         #             clear_at_least=200,
#         #             keep=5,
#         #             exclude_tools=["ask_user"],
#         #             placeholder="[Tool output cleared to save context. Use the tool again if needed.$$$$$$$$$]",
#         #         )
#         #     ]
#         # ),
#     ],
#     tools=[
#         read_file,
#         write_file,
#         file_exists,
#         list_directory,
#         create_directory,
#         delete_file,
#         batch_read_files,
#         batch_write_files,
#     ],
#     context_schema=RuntimeContext,
#     state_schema=FullStackAgentState,
#     debug=True,
# )


# async def main():
#     user_id = "test_adi_01"
#     project_id = "test_project_adi_01"
#     config = {"configurable": {"thread_id": "test-123"}}
#     context = RuntimeContext(
#         user_id=user_id,
#         project_id=project_id,
#     )
#     await graph.ainvoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "can u read created files using read tool",
#                 }
#             ],
#             "user_id": "1",
#             "project_id": "2",
#         },
#         config=config,
#         context=context,
#         stream_mode="debug",
#         print_mode="debug",
#     )


# if __name__ == "__main__":
#     asyncio.run(main())

# # config = {"configurable": {"thread_id": "1"}}
# # graph.invoke({"foo": ""}, config, stream_mode="updates", print_mode="updates")
# # config = {"configurable": {"thread_id": "1"}}
# # # result = graph.get_state(config)
# # config = {
# #     "configurable": {
# #         "thread_id": "1",
# #     }
# # }
# # result = list(graph.get_state_history(config))
# # print(result)


# pip install -qU "langchain[anthropic]" to call the model
import os
import dotenv
import asyncio
import logging
from pathlib import Path

dotenv.load_dotenv()

from langchain.agents import create_agent, AgentState
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from db.service import DatabaseService, db_service
from langchain_openai import ChatOpenAI
from tools.file_tools_e2b import (
    read_file,
    write_file,
    file_exists,
    list_directory,
    create_directory,
    delete_file,
    batch_read_files,
    batch_write_files,
)
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit


# ============================================================================
# LOGGING CONFIGURATION - Captures ALL logs including LangChain internals
# ============================================================================

# Create log directory
os.makedirs("log", exist_ok=True)

# Get root logger (captures everything)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear existing handlers to avoid duplicates
root_logger.handlers.clear()

# File Handler - Saves ALL logs to file (DEBUG level)
file_handler = logging.FileHandler("log/log_latest.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Console Handler - Shows only important logs (INFO level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Optional: Suppress extremely noisy libraries (uncomment if needed)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("anthropic").setLevel(logging.WARNING)

# Get logger for this module
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Logging initialized - All logs saved to log/log_latest.log")
logger.info("=" * 80)


# ============================================================================
# AGENT STATE & CONFIGURATION
# ============================================================================

from typing import TypedDict, Optional, Annotated, List
from typing_extensions import NotRequired


class FullStackAgentState(AgentState):
    user_id: NotRequired[str]
    project_id: NotRequired[str]
    current_phase: NotRequired[str]
    """
    Current development phase.
    Values: 'planning' | 'backend_dev' | 'frontend_dev' | 'testing' | 'integration'
    Updated from: <phase> XML tag in agent responses
    Used for: Dynamic prompts, context injection, progress tracking
    """

    next_steps: NotRequired[list[str]]
    """
    List of concrete next actions planned by agent.
    Updated from: <next_steps> XML tag in agent responses
    Used for: UI progress display, recovery after interruption
    """

    recent_thinking: NotRequired[list[dict]]
    """
    Last 5 thoughts from agent (lightweight memory).
    Structure:
        [{
            'thinking': str,      # 2-3 sentence reasoning
            'phase': str,         # Phase when thought occurred
            'iteration': int,     # Iteration number
            'timestamp': str      # ISO timestamp
        }, ...]
    Updated from: <thinking> XML tag in agent responses
    Used for: Quick context retrieval, debugging, continuity
    Size: Capped at 5 entries (oldest removed)
    """

    error_count: NotRequired[int]
    """
    Consecutive error count (circuit breaker).
    Incremented: When <error> tag present in response
    Reset to 0: When no error in response
    Used for: Stop agent after N consecutive failures
    """

    last_error: NotRequired[Optional[dict]]
    """
    Most recent error details.
    Structure:
        {
            'description': str,   # Error description
            'severity': str,      # 'low' | 'medium' | 'high' | 'critical'
            'timestamp': str      # ISO timestamp
        }
    Updated from: <error> XML tag in agent responses
    Used for: Error recovery, debugging, logging
    """

    iteration_count: NotRequired[int]
    """
    Total LLM calls in this session.
    
    Updated by: IterationTrackerMiddleware (before each model call)
    Used for: Debugging, cost tracking, trigger periodic summarization
    """

    last_summarized_at: NotRequired[int]
    """
    Iteration number when last summarization occurred.
    
    Updated by: SummarizationMiddleware (after summarization)
    Used for: Determine when next summarization needed
    """

    working_directory: NotRequired[str]
    """
    Current working directory in E2B sandbox.
    
    Default: '/workspace'
    Updated by: File tools (change_directory, etc.)
    Used for: Resolve relative paths, context in prompts
    """

    active_files: NotRequired[list[str]]
    """
    Recently accessed files (last 10).
    Updated by: File/edit tools (read_file, write_file, edit_file)
    Used for: Context injection, "what was I working on?"
    Size: Capped at 10 entries (oldest removed)
    """

    service_pids: NotRequired[dict[str, int]]
    """
    Running background services (dev servers, etc.).
    Structure:
        {
            'fastapi_dev': 1234,
            'react_dev': 5678
        }
    Updated by: Command tools (run_service, stop_service)
    Used for: Track running processes, stop services cleanly
    """

    tokens_used: NotRequired[dict]
    """
    Token usage tracking across the session.
    
    Structure:
        {
            'total_input': int,       # Total input tokens
            'total_output': int,      # Total output tokens
            'total_cost': float,      # Estimated cost in USD
            'by_model': {             # Per-model breakdown
                'claude-sonnet-4': {
                    'input': int,
                    'output': int,
                    'calls': int
                }
            }
        }
    
    Updated by: TokenTrackerMiddleware (post-model hook)
    Used for: Cost monitoring, optimization, user billing
    """


def load_file_tools() -> List[BaseTool]:
    """
    Load all available tools for the agent.

    Returns:
        List of LangChain tools ready for agent use
    """

    tools = []

    # =========================================================================
    # FILE TOOLS (E2B Sandbox) - Import individual tools
    # =========================================================================
    try:
        from tools.file_tools_e2b import (
            read_file,
            write_file,
            file_exists,
            list_directory,
            create_directory,
            delete_file,
            batch_read_files,
            batch_write_files,
        )

        file_tools = [
            read_file,
            write_file,
            file_exists,
            list_directory,
            create_directory,
            delete_file,
            batch_read_files,
            batch_write_files,
        ]

        tools.extend(file_tools)
        logger.info(f"✅ Loaded {len(file_tools)} file tools")
    except Exception as e:
        logger.error(f"❌ Failed to load file tools: {e}", exc_info=True)

    return tools


@dataclass
class RuntimeContext:

    user_id: str
    project_id: str

    @property
    def session_id(self) -> str:
        """Alias for project_id (compatibility)"""
        return self.project_id

    @property
    def thread_id(self) -> str:
        """LangChain thread_id (same as project_id)"""
        return self.project_id

    sandbox_id: Optional[str] = None
    """E2B sandbox ID (from Project.active_sandbox_id)"""

    sandbox_state: str = "none"
    """
    Sandbox state (from Project.sandbox_state)
    Values: 'running' | 'paused' | 'killed' | 'none'
    """

    @property
    def sandbox_active(self) -> bool:
        """Whether sandbox is currently running"""
        return self.sandbox_state == "running"

    # =========================================================================
    # SESSION STATUS (From Database)
    # =========================================================================

    session_status: str = "active"
    """
    Session status (from Project.status)
    Values: 'active' | 'paused' | 'ended'
    """

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    environment: str = "development"
    """Environment: 'development' | 'staging' | 'production'"""

    debug_mode: bool = False
    """Enable debug logging"""

    max_iterations: int = 25
    """Maximum agent iterations"""


system_prompt = """You are a full stack code generation agent for now only work with fastapi backend"""

model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    max_tokens=10000,
    temperature=0.1,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=120,
)

tools = load_file_tools()

# context = RuntimeContext(user_id="test_adi_boy_01", project_id="test_adi_boy_prod_01")
checkpointer = InMemorySaver()
graph = create_agent(
    model=model,
    system_prompt=system_prompt,
    middleware=[
        SummarizationMiddleware(
            model=ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                max_tokens=10000,
                temperature=0.1,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                timeout=120,
            ),
            max_tokens_before_summary=2000,
            messages_to_keep=5,
        ),
        # ContextEditingMiddleware(
        #     edits=[
        #         ClearToolUsesEdit(
        #             trigger=500,
        #             clear_at_least=200,
        #             keep=5,
        #             exclude_tools=["ask_user"],
        #             placeholder="[Tool output cleared to save context. Use the tool again if needed.$$$$$$$$$]",
        #         )
        #     ]
        # ),
    ],
    tools=[
        read_file,
        write_file,
        file_exists,
        list_directory,
        create_directory,
        delete_file,
        batch_read_files,
        batch_write_files,
    ],
    context_schema=RuntimeContext,
    state_schema=FullStackAgentState,
    debug=True,
)


async def main():
    user_id = "test_adi_06"
    project_id = "test_project_adi_06"
    config = {"configurable": {"thread_id": project_id}}
    context = RuntimeContext(
        user_id=user_id,
        project_id=project_id,
    )

    logger.info("🚀 Starting agent execution")
    logger.info(f"User ID: {user_id}")
    logger.info(f"Project ID: {project_id}")

    try:
        await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a fastapi todo app backend also next frontend for it use mongodb as db with pymongo and motor such that use next.js frontend which is already set up using create next app the working directory for this is home/user/code/frontend which has nextjs setup and create backend folder in code/ for fastapi backend u have 8k token budget do read files in frontend before modifying Do not stop until build successfully",
                    }
                ],
                "user_id": "1",
                "project_id": "2",
            },
            config=config,
            context=context,
            stream_mode="debug",
            print_mode="debug",
        )

        logger.info("✅ Agent execution completed successfully")

    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}", exc_info=True)
        raise

    logger.info("=" * 80)
    logger.info("All logs saved to: log/log_latest.log")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())


# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"foo": ""}, config, stream_mode="updates", print_mode="updates")
# config = {"configurable": {"thread_id": "1"}}
# # result = graph.get_state(config)
# config = {
#     "configurable": {
#         "thread_id": "1",
#     }
# }
# result = list(graph.get_state_history(config))
# print(result)






# agent.py

"""
Simple Agent Implementation
Basic agent setup for testing
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from agent.state import FullStackAgentState
from context.runtime_context import RuntimeContext
from agent.prompts import BASE_SYSTEM_PROMPT
from agent.tools_loader import load_all_tools

load_dotenv()

logger = logging.getLogger(__name__)

def create_simple_agent():
    """Create a simple agent for testing"""
    
    # Load tools
    tools = load_all_tools()
    
    # Create model
    model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        max_tokens=8192,
        temperature=0.1,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=120,
    )
    
    # Create checkpointer
    checkpointer = InMemorySaver()
    
    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=BASE_SYSTEM_PROMPT,
        state_schema=FullStackAgentState,
        context_schema=RuntimeContext,
        checkpointer=checkpointer,
        debug=True,
        name="simple-agent",
    )
    
    return agent

async def test_agent():
    """Test the agent"""
    agent = create_simple_agent()
    
    context = RuntimeContext(
        user_id="test_user",
        project_id="test_project",
        email_id="test@example.com"
    )
    
    config = {
        "configurable": {
            "thread_id": "test_project",
        }
    }
    
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "List the current directory"}
            ],
            "user_id": "test_user",
            "project_id": "test_project",
        },
        config=config,
        context=context,
    )
    
    print("Agent result:", result)

if __name__ == "__main__":
    asyncio.run(test_agent())