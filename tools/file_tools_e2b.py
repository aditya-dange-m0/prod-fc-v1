"""
FileTools for LangGraph - E2B Sandbox File Operations

Features:
- LangChain @tool decorator
- Consistent use of global_context for ALL tools
- Batch operations use RunnableConfig
- Non-batch operations support both config and fallback
- Same E2B sandbox integration
- Same database tracking with db_service
- Same multi-tenant support (userid/projectid)
- Production-ready with all error handling preserved
"""

import asyncio
import mimetypes
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any, Annotated
from dataclasses import dataclass
from enum import Enum

from langchain.tools import ToolRuntime, tool
from context.runtime_context import RuntimeContext
from context.runtime_context import RuntimeContext
from agent_state import FullStackAgentState


from db.service import db_service
from sandbox_manager import get_user_sandbox

from langgraph.config import get_stream_writer


def _resolve_ids_from_runtime(
    runtime: "ToolRuntime[RuntimeContext, FullStackAgentState]",
) -> tuple[str, str]:
    """Return (user_id, project_id) using runtime.context when present, otherwise fall back to runtime.state values.

    This centralizes the fallback behavior and avoids repeated checks.
    """
    # Prefer runtime.context attributes if present and truthy
    user_id = None
    project_id = None

    try:
        if hasattr(runtime, "context") and runtime.context is not None:
            # Some RuntimeContext implementations use attributes user_id/project_id
            user_id = getattr(runtime.context, "user_id", None)
            project_id = getattr(runtime.context, "project_id", None)
    except Exception:
        # defensive: ignore and continue to other fallbacks
        user_id = None
        project_id = None

    # Fallback to runtime.state (agent state) if context did not provide values
    try:
        if not user_id and hasattr(runtime, "state") and runtime.state is not None:
            user_id = (
                runtime.state.get("user_id")
                if isinstance(runtime.state, dict)
                else getattr(runtime.state, "user_id", None)
            )
        if not project_id and hasattr(runtime, "state") and runtime.state is not None:
            project_id = (
                runtime.state.get("project_id")
                if isinstance(runtime.state, dict)
                else getattr(runtime.state, "project_id", None)
            )
    except Exception:
        # defensive: ignore and allow final fallback below
        pass

    # Final safe defaults to avoid passing None to sandbox manager
    if not user_id:
        user_id = "unknown_user"
    if not project_id:
        project_id = "unknown_project"

    return user_id, project_id


# =============================================================================
# TYPE DEFINITIONS (Same as Agno version)
# =============================================================================


class FileType(Enum):
    """Supported file types for code operations"""

    TEXT = "text"


class ErrorType(Enum):
    """Error types for structured error handling"""

    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    PATH_TRAVERSAL = "path_traversal"
    FILE_TOO_LARGE = "file_too_large"
    INVALID_PATH = "invalid_path"
    TARGET_IS_DIRECTORY = "target_is_directory"
    READ_FAILURE = "read_failure"
    WRITE_FAILURE = "write_failure"
    OPERATION_FAILED = "operation_failed"


@dataclass
class FileInfo:
    """Information about a file or directory"""

    name: str
    path: str
    type: Literal["file", "directory"]
    size: int
    modified: Optional[str] = None
    permissions: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class FileOperationResult:
    """Result of a file operation"""

    success: bool
    path: str
    operation: str
    message: str
    size_bytes: int = 0
    error_type: Optional[ErrorType] = None
    error_details: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


# =============================================================================
# UTILITY FUNCTIONS (Same as Agno version)
# =============================================================================


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger for file operations"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def validate_sandbox_path(path: str) -> tuple[str, Optional[str]]:
    """
    Validate and normalize paths for E2B sandbox operations.

    Args:
        path: Path to validate

    Returns:
        Tuple of (normalized_path, error_message). If error_message is not None,
        the path is invalid and should not be used.
    """
    if not path or not isinstance(path, str):
        return "", "Path must be a non-empty string"

    path = path.strip()
    if not path:
        return "", "Path cannot be empty"

    # Prevent directory traversal attacks
    if ".." in path:
        return "", "Path traversal '..' not allowed for security reasons"

    # Use posixpath for Linux sandbox compatibility
    import posixpath

    normalized = posixpath.normpath(path)

    # Double-check after normalization
    if ".." in normalized:
        return "", "Path traversal detected after normalization"

    return normalized, None


def get_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type for a file path."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type if isinstance(mime_type, str) else None


# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

_logger = setup_logger("file_tools_langgraph")

# Default configuration for backward compatibility
_default_config = {
    "max_file_size": 50 * 1024 * 1024,
    "default_encoding": "utf-8",
    "enable_db_tracking": True,
}


def configure_file_tools(
    max_file_size: int = 50 * 1024 * 1024,
    enable_db_tracking: bool = True,
):
    """
    Configure global file tools settings.

    Args:
        max_file_size: Maximum file size in bytes
        enable_db_tracking: Enable database tracking
    """
    _default_config["max_file_size"] = max_file_size
    _default_config["enable_db_tracking"] = enable_db_tracking


# =============================================================================
# HELPER FUNCTIONS (Internal, shared by tools)
# =============================================================================


async def _persist_file_to_db(
    user_id: str, project_id: str, path: str, content: str, tool_name: str
):
    """Persist file changes to database"""
    try:
        # ✅ NEW: Just save the file, assume project exists
        # NestJS validated ownership before proxying request
        # # Create user/project with better error handling
        # try:
        #     await db_service.create_user(user_id, email)
        # except Exception as user_err:
        #     # User creation might fail due to IntegrityError (race condition)
        #     # This is OK - user likely exists
        #     _logger.debug(
        #         f"User {user_id} already exists or creation failed: {user_err}"
        #     )

        # try:
        #     await db_service.create_project(
        #         user_id, project_id, f"Project {project_id}"
        #     )
        # except Exception as proj_err:
        #     # Project creation might fail if already exists
        #     _logger.debug(
        #         f"Project {project_id} already exists or creation failed: {proj_err}"
        #     )

        # Now save the file
        success = await db_service.save_file(
            project_id=project_id,
            file_path=path,
            content=content,
            created_by_tool=tool_name,
        )

        if success:
            _logger.info(f"✅ DB: Persisted {path}")  # ← Changed to INFO
        else:
            _logger.error(f"❌ DB: Failed to persist {path}")  # ← Changed to ERROR

    except Exception as e:
        _logger.error(f"❌ DB persist exception for {path}: {e}", exc_info=True)


async def _track_file_deletion(project_id: str, path: str, tool_name: str):
    """Track file deletion in database"""
    if not _default_config["enable_db_tracking"]:
        return

    try:
        success = await db_service.delete_file(project_id=project_id, file_path=path)

        if success:
            _logger.debug(f"✅ Tracked deletion of {path}")
    except Exception as e:
        _logger.error(f"❌ Failed to track deletion: {e}")


# =============================================================================
# LANGCHAIN FILE TOOLS
# =============================================================================


@tool
async def read_file(
    path: str,
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Read file content from E2B sandbox.

    Args:
        path: File path in sandbox (absolute or relative)
        config: LangGraph config (optional, uses fallback if not provided)
        offset: Starting line number (0-based, optional)
        limit: Maximum lines to read (optional)

    Returns:
        File content as string
    """
    # # Resolve ids from injected runtime (prefer context, fallback to state)
    # user_id, project_id = _resolve_ids_from_runtime(runtime)
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        sandbox = await get_user_sandbox(user_id, project_id)
        path, path_error = validate_sandbox_path(path)
        if path_error:
            return f"❌ ERROR: Invalid path '{path}': {path_error}"

        # Read from E2B
        content = await sandbox.files.read(path, format="text")

        # Handle line-based reading
        if offset is not None or limit is not None:
            lines = content.splitlines()
            start = offset or 0
            end = (start + limit) if limit else len(lines)
            if start >= len(lines):
                return ""
            selected_lines = lines[start : min(end, len(lines))]
            content = "\n".join(selected_lines)

        _logger.info(f"Read file: {path} ({len(content)} chars)")
        return content

    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "no such file" in error_msg:
            return f"❌ ERROR: File '{path}' does not exist. Please verify the path is correct or create the file first. You can use 'file_exists' to check if a file exists before reading it."
        elif "permission" in error_msg:
            return f"❌ ERROR: Permission denied accessing '{path}'. The file may be protected or you may not have read permissions."
        else:
            return f"❌ ERROR: Unable to read file '{path}': {e}. Please check if the path is correct and the file is accessible."


@tool
async def write_file(
    path: str,
    content: str,
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
    overwrite: bool = True,
) -> str:
    """
    Write content to file in E2B sandbox with automatic directory creation and streaming progress.

    Args:
        path: Target file path (must be valid sandbox path)
        content: Text content to write (strings only)
        runtime: Tool runtime with context
        overwrite: Whether to overwrite existing files (default: True)

    Returns:
        Success message with file details
    """

    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        # Get stream writer for progress updates
        writer = get_stream_writer()

        sandbox = await get_user_sandbox(user_id, project_id)

        path, path_error = validate_sandbox_path(path)
        if path_error:
            return f"❌ ERROR: Invalid path '{path}': {path_error}"

        if not isinstance(content, str):
            return f"❌ ERROR: Content must be a string, got {type(content).__name__}. Please provide text content as a string."

        # Validate content size
        content_size = len(content.encode(_default_config["default_encoding"]))
        if content_size > _default_config["max_file_size"]:
            return f"❌ ERROR: Content too large: {content_size:,} bytes exceeds {_default_config['max_file_size']:,} bytes limit. Please reduce the content size."

        # Check if file exists
        file_exists = await sandbox.files.exists(path)
        if file_exists and not overwrite:
            return f"❌ ERROR: File '{path}' already exists. Set overwrite=True to replace it, or choose a different filename."

        # Stream progress update
        operation = "Updating" if file_exists else "Creating"
        writer(f"{operation} file: {path}")

        await sandbox.files.write(path=path, data=content)

        await _persist_file_to_db(user_id, project_id, path, content, "write_file")

        line_count = len(content.splitlines())
        operation = "created" if not file_exists else "updated"

        message = (
            f"Successfully {operation} file '{Path(path).name}' "
            f"({content_size} bytes, {line_count} lines)"
        )

        # Stream completion update
        writer(f"✅ {message}")

        _logger.info(message)
        return message

    except ValueError as e:
        return f"❌ ERROR: Invalid input for write_file: {e}. Please check your content and path parameters."
    except FileExistsError as e:
        return f"❌ ERROR: {e}. The file already exists and overwrite is disabled."
    except PermissionError as e:
        return f"❌ ERROR: Permission denied: {e}. You may not have write permissions to this location."
    except OSError as e:
        return f"❌ ERROR: File system error: {e}. Please check the path and try again."
    except Exception as e:
        error_msg = str(e)
        if "permission" in error_msg.lower():
            return f"❌ ERROR: Access denied writing to '{path}'. You may not have write permissions to this location."
        else:
            return f"❌ ERROR: Failed to write file '{path}': {e}. Please check the path and content are valid."


@tool
async def file_exists(
    path: str, runtime: ToolRuntime[RuntimeContext, FullStackAgentState]
) -> bool:
    """
    Check if a file or directory exists in the sandbox.

    Args:
        path: Path to check
        config: LangGraph config (optional)

    Returns:
        True if exists, False otherwise
    """
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        sandbox = await get_user_sandbox(user_id, project_id)
        path, path_error = validate_sandbox_path(path)
        if path_error:
            # For file_exists, return False if path is invalid
            return False
        return await sandbox.files.exists(path)
    except Exception as e:
        _logger.debug(f"Error checking existence: {e}")
        # Return False for file_exists is appropriate - it means "file does not exist"
        return False


@tool
async def list_directory(
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
    path: str = ".",
    depth: int = 1,
    include_hidden: bool = False,
) -> str:
    """
    List directory contents in E2B sandbox.

    Args:
        path: Directory path to list (default: current directory)
        config: LangGraph config (optional)
        depth: Depth of recursion (1 = immediate children)
        include_hidden: Whether to include hidden files (starting with .)

    Returns:
        Formatted string with directory contents
    """
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        sandbox = await get_user_sandbox(user_id, project_id)
        path, path_error = validate_sandbox_path(path)
        if path_error:
            return f"❌ ERROR: Invalid path '{path}': {path_error}"

        # List using E2B
        entries_info = await sandbox.files.list(path, depth=depth)

        file_entries = []
        for entry_info in entries_info:
            # Skip hidden files if not requested
            if not include_hidden and entry_info.name.startswith("."):
                continue

            file_type = "directory" if entry_info.type.value == "dir" else "file"
            file_entries.append(
                {
                    "name": entry_info.name,
                    "path": entry_info.path,
                    "type": file_type,
                    "size": entry_info.size,
                }
            )

        # Format output for LLM
        if not file_entries:
            return f"Directory '{path}' is empty"

        output = f"Directory listing for '{path}' ({len(file_entries)} items):\n"
        for entry in file_entries:
            icon = "📁" if entry["type"] == "directory" else "📄"
            size = f"{entry['size']:,} bytes" if entry["type"] == "file" else ""
            output += f"{icon} {entry['name']} {size}\n"

        return output

    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            return f"❌ ERROR: Directory '{path}' not found. Please check the path is correct or create the directory first using 'create_directory'."
        elif "not a directory" in error_msg:
            return f"❌ ERROR: '{path}' is not a directory, it's a file. Please provide a directory path to list its contents."
        else:
            return f"❌ ERROR: Failed to list directory '{path}': {e}. Please check if the path exists and is accessible."


@tool
async def create_directory(
    path: str, runtime: ToolRuntime[RuntimeContext, FullStackAgentState]
) -> str:
    """
    Create a directory in the sandbox (automatically recursive).

    Args:
        path: Directory path to create
        config: LangGraph config (optional)

    Returns:
        Success message
    """
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        sandbox = await get_user_sandbox(user_id, project_id)
        path, path_error = validate_sandbox_path(path)
        if path_error:
            return f"❌ ERROR: Invalid path '{path}': {path_error}"

        created = await sandbox.files.make_dir(path)

        if created:
            message = f"Successfully created directory '{path}'"
        else:
            message = f"Directory '{path}' already exists"

        _logger.info(message)
        return message

    except Exception as e:
        return f"❌ ERROR: Failed to create directory '{path}': {e}. Please check if the path is valid and you have write permissions."


@tool
async def delete_file(
    path: str,
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
    force: bool = False,
) -> str:
    """
    Delete a file or directory from the sandbox.

    Args:
        path: Path to file or directory to delete
        config: LangGraph config (optional)
        force: Whether to ignore if path doesn't exist (default: False)

    Returns:
        Success message
    """
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        sandbox = await get_user_sandbox(user_id, project_id)
        path, path_error = validate_sandbox_path(path)
        if path_error:
            return f"❌ ERROR: Invalid path '{path}': {path_error}"

        # Check if exists
        if not force and not await sandbox.files.exists(path):
            return f"❌ ERROR: Path '{path}' does not exist. Use force=True to ignore missing files, or check if the path is correct."

        # Delete using E2B
        await sandbox.files.remove(path)

        # Track deletion in database
        await _track_file_deletion(project_id, path, "delete_file")

        message = f"Successfully deleted '{path}'"
        _logger.info(message)
        return message

    except FileNotFoundError as e:
        return f"❌ ERROR: {e}. The file or directory does not exist."
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg and force:
            return f"✅ Path '{path}' already deleted or doesn't exist (force=True, so this is OK)"
        else:
            return f"❌ ERROR: Failed to delete '{path}': {e}. Please check if the path exists and you have delete permissions."


# =============================================================================
# BATCH OPERATIONS (Require config for consistency)
# =============================================================================


@tool
async def batch_read_files(
    file_paths: List[str], runtime: ToolRuntime[RuntimeContext, FullStackAgentState]
) -> str:
    """
    Read multiple files from E2B sandbox in parallel.

    Efficient for reading project files, configuration files, or multiple
    source files at once.

    Args:
        file_paths: List of file paths to read
        config: LangGraph config with user/project context (required)

    Returns:
        Formatted string with all file contents

    Example:
        files = ["src/app.py", "src/utils.py", "README.md"]
        content = await batch_read_files(files, config)
    """
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    if not file_paths:
        return "No files specified to read"

    try:
        sandbox = await get_user_sandbox(user_id, project_id)

        # Validate all paths
        valid_paths = []
        for p in file_paths:
            validated_path, path_error = validate_sandbox_path(p)
            if path_error:
                return f"❌ ERROR: Invalid path '{p}': {path_error}"
            valid_paths.append(validated_path)

        # Read all files in parallel
        async def read_one(path: str) -> tuple[str, str, Optional[str]]:
            try:
                content = await sandbox.files.read(path, format="text")
                return (path, content, None)
            except Exception as e:
                return (path, "", str(e))

        results = await asyncio.gather(*[read_one(p) for p in valid_paths])

        # Format results
        output = f"Read {len(file_paths)} files:\n\n"

        success_count = 0
        for path, content, error in results:
            if error:
                output += f"❌ {path}: ERROR - {error}\n\n"
            else:
                success_count += 1
                output += f"✅ {path} ({len(content)} chars):\n"
                output += "```"
                output += content
                output += "\n```\n\n"

        summary = f"Successfully read {success_count}/{len(file_paths)} files"
        return summary + "\n\n" + output

    except Exception as e:
        return f"❌ ERROR: Batch read operation failed: {e}. Please check if all file paths are valid and accessible."


@tool
async def batch_write_files(
    files: List[Dict[str, str]],
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
    overwrite: bool = True,
) -> str:
    """
    Write multiple files to E2B sandbox with streaming progress updates.

    Efficient for creating project scaffolding, multiple components,
    or updating several files at once.

    Args:
        files: List of dicts with 'path' and 'content' keys
            Example: [{"path": "src/app.py", "content": "..."}, ...]
        runtime: Tool runtime with context
        overwrite: Whether to overwrite existing files

    Returns:
        Summary of batch write operation

    Example:
        files = [
            {"path": "src/app.py", "content": "print('hello')"},
            {"path": "src/__init__.py", "content": ""},
            {"path": "README.md", "content": "# My Project"}
        ]
        result = await batch_write_files(files, runtime)
    """
    from langgraph.config import get_stream_writer

    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    if not files:
        return "No files specified to write"

    # Validate input format
    for i, f in enumerate(files):
        if "path" not in f or "content" not in f:
            return f"❌ ERROR: File {i+1} is missing required keys. Each file dict must have 'path' and 'content' keys. Got: {list(f.keys())}. Example: {{'path': 'file.py', 'content': 'print(\"hello\")'}}"

    try:
        # Get stream writer for custom updates
        writer = get_stream_writer()

        sandbox = await get_user_sandbox(user_id, project_id)

        success_count = 0
        total = len(files)

        # Write files with progress updates
        for i, file_data in enumerate(files, 1):
            path = file_data["path"]
            content = file_data["content"]

            # Stream progress to frontend
            writer(f"Creating file {i}/{total}: {path}")

            try:
                path, path_error = validate_sandbox_path(path)
                if path_error:
                    writer(f"❌ Failed: {path} - Invalid path: {path_error}")
                    continue

                # Check if exists
                if not overwrite:
                    exists = await sandbox.files.exists(path)
                    if exists:
                        writer(f"⚠️ Skipped: {path} - File exists (overwrite=False)")
                        continue

                # Write file (auto-creates directories)
                await sandbox.files.write(path=path, data=content)

                # Persist to database
                await _persist_file_to_db(
                    user_id, project_id, path, content, "batch_write_files"
                )

                success_count += 1
                writer(f"✅ Created: {path}")

            except Exception as e:
                writer(f"❌ Failed: {path} - {str(e)}")

        return f"Created {success_count}/{total} files"

    except Exception as e:
        return f"❌ ERROR: Batch write operation failed: {e}. Please check if all file paths are valid and you have write permissions."


# =============================================================================
# TOOL LIST & FACTORY
# =============================================================================

# All tools ready for LangGraph agent
FILE_TOOLS = [
    read_file,
    write_file,
    file_exists,
    list_directory,
    create_directory,
    delete_file,
    batch_read_files,
    batch_write_files,
]


def create_file_tools(**kwargs):
    """
    Configure file tools settings.

    Note: user_id/project_id come from config in each tool call, not here.

    Args:
        **kwargs: Configuration (max_file_size, enable_db_tracking)

    Returns:
        List of configured tools ready for agent

    Usage:
        ```
        from tools.file_tools_e2b import FILE_TOOLS, configure_file_tools
        from context.global_context import create_config

        # Configure settings (optional)
        configure_file_tools(max_file_size=100*1024*1024)

        # Create agent with tools
        agent = create_react_agent(model, tools=FILE_TOOLS)

        # Create config with user/project context
        config = create_config(
            user_id="user_123",
            project_id="project_456",
            session_id="session_789"
        )

        # Invoke with config
        agent.invoke({"messages": [...]}, config)
        ```
    """
    if kwargs:
        configure_file_tools(**kwargs)
    return FILE_TOOLS


if __name__ == "__main__":
    print("FileTools for LangGraph - E2B Sandbox File Operations")
    print("Available tools:", [t.name for t in FILE_TOOLS])
    print("\nFeatures:")
    print("  ✅ Consistent global_context usage")
    print("  ✅ Batch operations with parallel execution")
    print("  ✅ Database tracking and E2B integration")
    print("\nReady for LangGraph agents!")
