"""
Tool loading for the full-stack agent.

Loads all available tools from different modules.
"""

import logging
from typing import List
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def load_all_tools() -> List[BaseTool]:
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

    # =========================================================================
    # EDIT TOOLS (E2B Sandbox) - Import individual tools
    # =========================================================================
    try:
        from tools.edit_tools_e2b import (
            edit_file,
            smart_edit_file,
        )

        edit_tools = [
            edit_file,
            smart_edit_file,
        ]

        tools.extend(edit_tools)
        logger.info(f"✅ Loaded {len(edit_tools)} edit tools")
    except Exception as e:
        logger.error(f"❌ Failed to load edit tools: {e}", exc_info=True)

    # =========================================================================
    # COMMAND TOOLS (E2B Sandbox) - Import individual tools
    # =========================================================================
    try:
        from tools.command_tools_e2b import (
            run_command,
            run_service,
            list_processes,
            kill_process,
            get_service_url,
        )

        # Use CORE tools only (most commonly used)
        command_tools = [
            run_command,
            run_service,
            list_processes,
            kill_process,
            get_service_url,
        ]

        tools.extend(command_tools)
        logger.info(f"✅ Loaded {len(command_tools)} command tools")
    except Exception as e:
        logger.error(f"❌ Failed to load command tools: {e}", exc_info=True)

    # =========================================================================
    # WEB SEARCH TOOL - Import individual tool
    # =========================================================================
    try:
        from tools.web_search_tool import search_web

        tools.append(search_web)
        logger.info("✅ Loaded search_web tool")
    except Exception as e:
        logger.error(f"❌ Failed to load web search tool: {e}", exc_info=True)

    logger.info(f"📦 Total tools loaded: {len(tools)}")

    if len(tools) == 0:
        logger.warning("⚠️ No tools loaded! Agent will have limited functionality.")

    return tools
