"""
EditTools for LangGraph - E2B Sandbox Code Editing Operations
Migrated from Agno to LangChain tools

Features:
- ALL tools require RunnableConfig with user/project context
- Intelligent code editing with exact, flexible, and fuzzy matching strategies
- Diff generation for change visibility
- Line ending preservation
- Indentation-aware matching
- Automatic database tracking
"""

import re
import logging
from typing import Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import difflib

from langchain.tools import tool, ToolRuntime

from sandbox_manager import get_user_sandbox
from db.service import db_service
from agent_state import FullStackAgentState
from context.runtime_context import RuntimeContext

# =============================================================================
# TYPE DEFINITIONS AND ENUMS
# =============================================================================


class EditErrorType(Enum):
    """Error types for edit operations"""

    EDIT_NO_OCCURRENCE_FOUND = "edit_no_occurrence_found"
    EDIT_MULTIPLE_OCCURRENCES = "edit_multiple_occurrences"
    EDIT_FILE_NOT_FOUND = "edit_file_not_found"
    EDIT_INVALID_PATH = "edit_invalid_path"
    EDIT_READ_ERROR = "edit_read_error"
    EDIT_WRITE_ERROR = "edit_write_error"
    EDIT_NO_CHANGES = "edit_no_changes"


@dataclass
class EditOperationResult:
    """Result of an edit operation"""

    success: bool
    path: str
    operation: str
    message: str
    replacements: int = 0
    strategy_used: str = "none"
    diff: Optional[str] = None
    error_type: Optional[EditErrorType] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger for edit operations"""
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


def validate_sandbox_path(path: str) -> str:
    """
    Validate and normalize paths for E2B sandbox operations.

    Args:
        path: Path to validate

    Returns:
        Normalized path safe for sandbox operations

    Raises:
        ValueError: If path is invalid or contains security risks
    """
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    path = path.strip()
    if not path:
        raise ValueError("Path cannot be empty")

    # Prevent directory traversal attacks
    if ".." in path:
        raise ValueError("Path traversal '..' not allowed for security")

    # Use posixpath for Linux sandbox compatibility
    import posixpath

    normalized = posixpath.normpath(path)

    # Double-check after normalization
    if ".." in normalized:
        raise ValueError("Path traversal detected after normalization")

    return normalized


def _create_structured_error(display: str, raw: str, error_type: EditErrorType) -> str:
    """Create a structured error response"""
    return f"""❌ {display}
🔍 Details: {raw}
📋 Error Type: {error_type.value}"""


def _safe_literal_replace(text: str, old: str, new: str) -> str:
    """Safe literal replacement handling special regex characters"""
    if not old:
        return text
    escaped_old = re.escape(old)
    safe_new = new.replace("\\", "\\\\").replace("$", "\\$")
    return re.sub(escaped_old, safe_new, text)


def _detect_line_ending(content: str) -> str:
    """Detect line ending style of content"""
    return "\r\n" if "\r\n" in content else "\n"


def _restore_trailing_newline(original: str, modified: str) -> str:
    """Restore original trailing newline behavior"""
    had_trailing = original.endswith("\n")
    if had_trailing and not modified.endswith("\n"):
        return modified + "\n"
    elif not had_trailing and modified.endswith("\n"):
        return modified.rstrip("\n")
    return modified


def _calculate_exact_replacement(
    content: str, old_string: str, new_string: str
) -> Tuple[str, int]:
    """Calculate exact string replacement"""
    normalized_content = content.replace("\r\n", "\n")
    normalized_old = old_string.replace("\r\n", "\n")
    normalized_new = new_string.replace("\r\n", "\n")

    occurrences = normalized_content.count(normalized_old)

    if occurrences > 0:
        result = _safe_literal_replace(
            normalized_content, normalized_old, normalized_new
        )
        result = _restore_trailing_newline(content, result)
        return result, occurrences

    return content, 0


def _calculate_flexible_replacement(
    content: str, old_string: str, new_string: str
) -> Tuple[str, int]:
    """Calculate flexible replacement handling indentation differences"""
    normalized_content = content.replace("\r\n", "\n")
    normalized_old = old_string.replace("\r\n", "\n")
    normalized_new = new_string.replace("\r\n", "\n")

    content_lines = normalized_content.splitlines(keepends=True)
    old_lines = normalized_old.splitlines()
    new_lines = normalized_new.splitlines()
    old_lines_stripped = [line.strip() for line in old_lines]

    occurrences = 0
    i = 0

    while i <= len(content_lines) - len(old_lines_stripped):
        window = content_lines[i : i + len(old_lines_stripped)]
        window_stripped = [line.strip() for line in window]

        if window_stripped == old_lines_stripped:
            occurrences += 1

            # Preserve indentation from first line
            first_line = window[0] if window else ""
            indent_match = re.match(r"^(\s*)", first_line)
            indentation = indent_match.group(1) if indent_match else ""

            # Apply indentation to new lines
            indented_new_lines = []
            for line in new_lines:
                if line.strip():
                    indented_new_lines.append(indentation + line)
                else:
                    indented_new_lines.append(line)

            replacement_text = "\n".join(indented_new_lines)
            if window and window[-1].endswith("\n"):
                replacement_text += "\n"

            content_lines[i : i + len(old_lines_stripped)] = [replacement_text]
            i += 1
        else:
            i += 1

    if occurrences > 0:
        result = "".join(content_lines)
        result = _restore_trailing_newline(content, result)
        return result, occurrences

    return content, 0


def _calculate_fuzzy_replacement(
    content: str, old_string: str, new_string: str
) -> Tuple[str, int]:
    """Smart fuzzy replacement with whitespace normalization"""

    def normalize_for_matching(text: str) -> str:
        lines = text.split("\n")
        normalized_lines = []
        for line in lines:
            stripped = line.rstrip()
            if stripped:
                normalized_lines.append(stripped)
            else:
                normalized_lines.append("")
        return "\n".join(normalized_lines)

    normalized_content = normalize_for_matching(content)
    normalized_old = normalize_for_matching(old_string)

    if normalized_old in normalized_content:
        result = _safe_literal_replace(normalized_content, normalized_old, new_string)
        return result, 1

    return content, 0


def _generate_diff(original: str, new: str, filename: str) -> str:
    """Generate unified diff for display"""
    original_lines = original.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )

    return "".join(diff)


# =============================================================================
# CONFIGURATION
# =============================================================================

_logger = setup_logger("edit_tools_langgraph")

_settings = {
    "default_encoding": "utf-8",
    "enable_db_tracking": True,
}


def configure_edit_tools(enable_db_tracking: bool = True):
    """Configure edit tools settings"""
    _settings["enable_db_tracking"] = enable_db_tracking


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _persist_file_to_db(
    user_id: str, project_id: str, path: str, content: str, tool_name: str
):
    """Persist file changes to database"""
    if not _settings["enable_db_tracking"]:
        _logger.info("📝 Database tracking disabled - skipping file persistence")
        return

    try:
        # ✅ NEW: Just save the file, assume project exists
        # NestJS validated ownership before proxying request
        # Create user/project with better error handling
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
            _logger.info(f"✅ DB: Persisted edit of {path}")
        else:
            _logger.error(f"❌ DB: Failed to persist edit of {path}")

    except Exception as e:
        _logger.error(f"❌ DB persist exception for edit of {path}: {e}", exc_info=True)


# =============================================================================
# CORE EDIT TOOLS
# =============================================================================


@tool
async def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
    expected_replacements: int = 1,
) -> str:
    """
    Edit a file by replacing exact text with validation.

    Supports creating new files (old_string='') and precise text replacement
    with occurrence counting. Uses exact matching by default.

    Args:
        file_path: Path to file in sandbox (relative or absolute)
        old_string: Exact text to replace (empty string creates new file)
        new_string: Text to replace with
        config: LangGraph config with user/project context (REQUIRED)
        expected_replacements: Number of replacements expected (default: 1)

    Returns:
        Success message with diff or structured error

    Examples:
        - edit_file("backend/app.py", "old_code", "new_code", config)
        - edit_file("backend/new.py", "", "print('hello')", config)  # Create file
        - edit_file("backend/utils.py", "def old()", "def new()", config, expected_replacements=2)
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
        path = validate_sandbox_path(file_path)

        _logger.debug(f"Editing file: {path}")

        # Check if file exists
        file_exists = await sandbox.files.exists(path)
        is_new_file = old_string == "" and not file_exists

        # Handle new file creation
        if is_new_file:
            try:
                await sandbox.files.write(path, new_string)

                # Persist to database
                email_id = getattr(
                    runtime.context, "email_id", f"{user_id}@system.local"
                )
                await _persist_file_to_db(
                    user_id, project_id, path, new_string, "edit_file"
                )

                return f"""✅ Created new file: {path}
📄 Content: {len(new_string)} characters, {len(new_string.splitlines())} lines"""
            except Exception as e:
                return _create_structured_error(
                    display="Error creating file.",
                    raw=f"Failed to create file {path}: {str(e)}",
                    error_type=EditErrorType.EDIT_WRITE_ERROR,
                )

        # Handle existing file edits
        if not file_exists:
            return _create_structured_error(
                display="File not found.",
                raw=f"File '{path}' does not exist in sandbox. Use empty old_string to create new file.",
                error_type=EditErrorType.EDIT_FILE_NOT_FOUND,
            )

        if old_string == "":
            return _create_structured_error(
                display="Cannot create file that already exists.",
                raw=f"File '{path}' already exists. Provide old_string to edit it.",
                error_type=EditErrorType.EDIT_INVALID_PATH,
            )

        # Read current content
        try:
            current_content = await sandbox.files.read(path, format="text")
        except Exception as e:
            return _create_structured_error(
                display="Error reading file.",
                raw=f"Failed to read {path}: {str(e)}",
                error_type=EditErrorType.EDIT_READ_ERROR,
            )

        # Normalize and detect line endings
        normalized_content = current_content.replace("\r\n", "\n")
        original_line_ending = _detect_line_ending(current_content)

        # Try exact replacement
        new_content, occurrences = _calculate_exact_replacement(
            normalized_content, old_string, new_string
        )

        # If exact failed, try flexible matching
        if occurrences == 0:
            new_content, occurrences = _calculate_flexible_replacement(
                normalized_content, old_string, new_string
            )

        # Validate replacement count
        if occurrences == 0:
            searched_preview = old_string[:100] + (
                "..." if len(old_string) > 100 else ""
            )
            return _create_structured_error(
                display="Failed to edit, could not find the string to replace.",
                raw=f"0 occurrences found for old_string in {path}. Searched for: {searched_preview}. Use search_file_content to verify exact text.",
                error_type=EditErrorType.EDIT_NO_OCCURRENCE_FOUND,
            )

        if occurrences != expected_replacements:
            term = "occurrence" if expected_replacements == 1 else "occurrences"
            return _create_structured_error(
                display=f"Expected {expected_replacements} {term} but found {occurrences}.",
                raw=f"Found {occurrences} occurrences instead of expected {expected_replacements} in {path}. Verify expected_replacements parameter.",
                error_type=EditErrorType.EDIT_MULTIPLE_OCCURRENCES,
            )

        if old_string == new_string:
            return _create_structured_error(
                display="No changes needed.",
                raw=f"old_string and new_string are identical in {path}.",
                error_type=EditErrorType.EDIT_NO_CHANGES,
            )

        if normalized_content == new_content:
            return _create_structured_error(
                display="No changes applied.",
                raw=f"Content would remain unchanged in {path}.",
                error_type=EditErrorType.EDIT_NO_CHANGES,
            )

        # Restore line endings and write
        if original_line_ending == "\r\n":
            final_content = new_content.replace("\n", "\r\n")
        else:
            final_content = new_content

        try:
            await sandbox.files.write(path, final_content)

            # Persist to database
            email_id = getattr(runtime.context, "email_id", f"{user_id}@system.local")
            await _persist_file_to_db(
                user_id, project_id, path, final_content, "edit_file"
            )

            # Generate diff
            import os

            filename = os.path.basename(path)
            diff = _generate_diff(current_content, final_content, filename)

            _logger.info(f"Successfully edited {path} ({occurrences} replacements)")

            return f"""✅ Successfully edited {path} ({occurrences} replacements)
📊 Changes:
{diff}
📈 Stats: {len(final_content)} characters, {len(final_content.splitlines())} lines"""

        except Exception as e:
            return _create_structured_error(
                display="Error writing file.",
                raw=f"Failed to write {path}: {str(e)}",
                error_type=EditErrorType.EDIT_WRITE_ERROR,
            )

    except ValueError as e:
        return _create_structured_error(
            display="Invalid path.",
            raw=str(e),
            error_type=EditErrorType.EDIT_INVALID_PATH,
        )

    except Exception as e:
        _logger.error(f"Unexpected error in edit_file: {e}")
        return f"❌ Error in edit_file: {str(e)}"


@tool
async def smart_edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    instruction: str,
    runtime: ToolRuntime[RuntimeContext, FullStackAgentState],
) -> str:
    """
    Intelligently edit a file with semantic understanding and auto-correction.

    Uses multiple matching strategies (exact → flexible → fuzzy) to find and
    replace code even with minor formatting differences. Preserves indentation
    and provides clear feedback about the strategy used.

    Args:
        file_path: Path to file in sandbox (relative or absolute)
        old_string: Text to replace (with context for better matching)
        new_string: Replacement text
        instruction: Clear description of what change is being made
        config: LangGraph config with user/project context (REQUIRED)

    Returns:
        Success message with diff or structured error

    Examples:
        - smart_edit_file("app.py", "old_code", "new_code", "Fix login bug", config)
        - smart_edit_file("utils.py", "def old()", "def new()", "Rename function", config)
    """
    user_id = runtime.context.user_id
    project_id = runtime.context.project_id
    if not user_id or project_id:
        user_id = runtime.state["user_id"]
        project_id = runtime.state["project_id"]
    project_id = runtime.context.project_id

    # Final safety check
    if not user_id or not project_id:
        return f"ERROR: Missing user_id or project_id in runtime context and state. Cannot access sandbox."

    try:
        sandbox = await get_user_sandbox(user_id, project_id)
        path = validate_sandbox_path(file_path)

        if not instruction:
            return _create_structured_error(
                display="Instruction required.",
                raw="instruction parameter is required for smart_edit_file.",
                error_type=EditErrorType.EDIT_INVALID_PATH,
            )

        _logger.debug(f"Smart editing file: {path} - {instruction}")

        # Check if file exists
        file_exists = await sandbox.files.exists(path)
        is_new_file = old_string == "" and not file_exists

        # Handle new file creation
        if is_new_file:
            try:
                await sandbox.files.write(path, new_string)

                # Persist to database
                email_id = getattr(
                    runtime.context, "email_id", f"{user_id}@system.local"
                )
                await _persist_file_to_db(
                    user_id, project_id, path, new_string, "smart_edit_file"
                )

                return f"""✅ Created new file: {path}
📝 Instruction: {instruction}
📄 Content: {len(new_string)} characters, {len(new_string.splitlines())} lines"""
            except Exception as e:
                return _create_structured_error(
                    display="Error creating file.",
                    raw=f"Failed to create file {path}: {str(e)}",
                    error_type=EditErrorType.EDIT_WRITE_ERROR,
                )

        # Handle existing file edits
        if not file_exists:
            return _create_structured_error(
                display="File not found.",
                raw=f"File '{path}' does not exist. Use empty old_string to create new file.",
                error_type=EditErrorType.EDIT_FILE_NOT_FOUND,
            )

        if old_string == "":
            return _create_structured_error(
                display="Cannot create file that already exists.",
                raw=f"File '{path}' already exists.",
                error_type=EditErrorType.EDIT_INVALID_PATH,
            )

        # Read current content
        try:
            current_content = await sandbox.files.read(path, format="text")
        except Exception as e:
            return _create_structured_error(
                display="Error reading file.",
                raw=f"Failed to read {path}: {str(e)}",
                error_type=EditErrorType.EDIT_READ_ERROR,
            )

        # Normalize and detect line endings
        normalized_content = current_content.replace("\r\n", "\n")
        original_line_ending = _detect_line_ending(current_content)

        # Try multiple strategies
        strategies = [
            (
                "exact",
                lambda: _calculate_exact_replacement(
                    normalized_content, old_string, new_string
                ),
            ),
            (
                "flexible",
                lambda: _calculate_flexible_replacement(
                    normalized_content, old_string, new_string
                ),
            ),
            (
                "fuzzy",
                lambda: _calculate_fuzzy_replacement(
                    normalized_content, old_string, new_string
                ),
            ),
        ]

        new_content = normalized_content
        occurrences = 0
        strategy_used = "none"

        for strategy_name, strategy_func in strategies:
            try:
                new_content, occurrences = strategy_func()
                if occurrences > 0:
                    strategy_used = strategy_name
                    break
            except Exception:
                continue

        # Validate results
        if occurrences == 0:
            searched_preview = old_string[:100] + (
                "..." if len(old_string) > 100 else ""
            )
            return _create_structured_error(
                display="Smart edit failed, could not find the string to replace.",
                raw=f"0 occurrences found. Instruction: {instruction}. Searched for: {searched_preview}",
                error_type=EditErrorType.EDIT_NO_OCCURRENCE_FOUND,
            )

        if old_string == new_string:
            return _create_structured_error(
                display="No changes needed.",
                raw=f"old_string and new_string are identical. Instruction: {instruction}",
                error_type=EditErrorType.EDIT_NO_CHANGES,
            )

        if normalized_content == new_content:
            return _create_structured_error(
                display="No changes applied.",
                raw=f"Content would remain unchanged. Instruction: {instruction}",
                error_type=EditErrorType.EDIT_NO_CHANGES,
            )

        # Restore line endings and write
        if original_line_ending == "\r\n":
            final_content = new_content.replace("\n", "\r\n")
        else:
            final_content = new_content

        try:
            await sandbox.files.write(path, final_content)

            # Persist to database
            email_id = getattr(runtime.context, "email_id", f"{user_id}@system.local")
            await _persist_file_to_db(
                user_id, project_id, path, final_content, "smart_edit_file"
            )

            # Generate diff
            import os

            filename = os.path.basename(path)
            diff = _generate_diff(current_content, final_content, filename)

            _logger.info(
                f"Smart edit successful: {path} ({strategy_used} strategy, {occurrences} replacements)"
            )

            return f"""✅ Smart edit successful: {path}
📝 Instruction: {instruction}
🔧 Strategy: {strategy_used} matching
🔄 Replacements: {occurrences}
📊 Changes:
{diff}
📈 Stats: {len(final_content)} characters, {len(final_content.splitlines())} lines"""

        except Exception as e:
            return _create_structured_error(
                display="Error writing file.",
                raw=f"Failed to write {path}: {str(e)}",
                error_type=EditErrorType.EDIT_WRITE_ERROR,
            )

    except ValueError as e:
        return _create_structured_error(
            display="Invalid path.",
            raw=str(e),
            error_type=EditErrorType.EDIT_INVALID_PATH,
        )

    except Exception as e:
        _logger.error(f"Unexpected error in smart_edit_file: {e}")
        return f"❌ Error in smart_edit_file: {str(e)}"


# =============================================================================
# TOOL EXPORT
# =============================================================================

EDIT_TOOLS = [
    edit_file,
    smart_edit_file,
]

if __name__ == "__main__":
    print("EditTools for LangGraph - E2B Sandbox Code Editing Operations")
    print("Available tools:", [t.name for t in EDIT_TOOLS])
    print("\n✅ Intelligent code editing with multiple matching strategies!")
    print("✅ Automatic diff generation!")
    print("✅ Database tracking enabled!")
    print("✅ Ready for LangGraph agents!")
