"""Convenience package exports for tools.

This module re-exports the most commonly used tools and factory
functions so other modules can simply `from tools import ...`.

Imports are guarded so importing the package won't fail if a single
tool has missing optional dependencies.
"""
from typing import List

# File tools
try:
    from .file_tools_e2b import (
        FILE_TOOLS as FILE_TOOLS,
        create_file_tools as create_file_tools,
        configure_file_tools as configure_file_tools,
    )
except Exception:  # pragma: no cover - best-effort import
    FILE_TOOLS = []
    create_file_tools = lambda **kwargs: []
    configure_file_tools = lambda *args, **kwargs: None

# Command tools
try:
    from .command_tools_e2b import (
        COMMAND_TOOLS as COMMAND_TOOLS,
        CORE_COMMAND_TOOLS as CORE_COMMAND_TOOLS,
        configure_command_tools as configure_command_tools,
    )
except Exception:  # pragma: no cover
    COMMAND_TOOLS = []
    CORE_COMMAND_TOOLS = []
    configure_command_tools = lambda *a, **k: None

# Edit tools
try:
    from .edit_tools_e2b import EDIT_TOOLS as EDIT_TOOLS, configure_edit_tools as configure_edit_tools
except Exception:  # pragma: no cover
    EDIT_TOOLS = []
    configure_edit_tools = lambda *a, **k: None

# Ask user tool
try:
    from .ask_user_tool import ask_user as ask_user
except Exception:  # pragma: no cover
    def ask_user(prompt: str) -> str:  # fallback stub
        raise RuntimeError("ask_user tool not available")

# Web search tool
try:
    from .web_search_tool import SEARCH_TOOL as search_web
except Exception:  # pragma: no cover
    def search_web(query: str) -> str:
        raise RuntimeError("search_web tool not available")

__all__ = [
    "FILE_TOOLS",
    "create_file_tools",
    "configure_file_tools",
    "COMMAND_TOOLS",
    "CORE_COMMAND_TOOLS",
    "configure_command_tools",
    "EDIT_TOOLS",
    "configure_edit_tools",
    "ask_user",
    "search_web",
]
