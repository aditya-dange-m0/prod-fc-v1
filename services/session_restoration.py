# """
# Session Restoration for LangGraph - Simplified Architecture
# ===========================================================

# Restores complete project state for LangGraph agents with PostgreSQL checkpointer.

# KEY CHANGES FOR LANGGRAPH:
# - Uses LangGraph's built-in checkpointer for session state
# - File restoration remains the same (load from DB → write to E2B)
# - No manual session management (LangGraph handles this via thread_id)
# - Compatible with PostgresSaver checkpointer
# - Enhanced error handling with structured responses

# Architecture Flow:
# 1. Get project files from database using project_id
# 2. Get or create E2B sandbox using multi-tenant manager
# 3. Write all files to sandbox (triggers dependency watcher)
# 4. Auto-install dependencies via file system watchers
# 5. Session ready for LangGraph agent!

# LangGraph Integration:
# - Agent state managed by checkpointer (PostgresSaver)
# - thread_id = project_id for session continuity
# - File restoration is separate from conversation state
# - Sandbox is persistent across agent invocations
# """

# import logging
# from datetime import datetime, UTC
# from typing import Dict, Any, List, Optional
# import asyncio

# # Import simplified database adapter
# from db.service import db_service

# # Import sandbox manager - using multi-tenant manager
# from sandbox_manager import (
#     get_user_sandbox,
#     get_multi_tenant_manager,
#     MultiTenantSandboxManager,
# )

# logger = logging.getLogger(__name__)

# # =============================================================================
# # ERROR CLASSES (Same as before - unchanged)
# # =============================================================================


# class SessionRestorationError(Exception):
#     """
#     Enhanced session restoration error with structured information for AI systems.
#     """

#     def __init__(
#         self,
#         message: str,
#         error_type: str = "unknown",
#         ai_context: str = None,
#         retry_possible: bool = True,
#         user_action_required: str = None,
#         system_action_required: str = None,
#         severity: str = "medium",
#         metadata: Dict[str, Any] = None,
#     ):
#         super().__init__(message)
#         self.error_type = error_type
#         self.message = message
#         self.ai_context = ai_context or "No additional context available"
#         self.retry_possible = retry_possible
#         self.user_action_required = user_action_required
#         self.system_action_required = system_action_required
#         self.severity = severity
#         self.metadata = metadata or {}
#         self.timestamp = datetime.now(UTC).isoformat()

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert error to dictionary for API responses and logging"""
#         return {
#             "error_type": self.error_type,
#             "message": self.message,
#             "ai_context": self.ai_context,
#             "retry_possible": self.retry_possible,
#             "user_action_required": self.user_action_required,
#             "system_action_required": self.system_action_required,
#             "severity": self.severity,
#             "metadata": self.metadata,
#             "timestamp": self.timestamp,
#         }

#     def for_ai_debug(self) -> str:
#         """Generate AI-friendly debug information"""
#         return f"""
# ERROR ANALYSIS FOR AI SYSTEM:
# - Error Type: {self.error_type}
# - Severity: {self.severity}
# - Context: {self.ai_context}
# - Retry Possible: {self.retry_possible}
# - User Action: {self.user_action_required or 'None required'}
# - System Action: {self.system_action_required or 'None required'}
# - Message: {self.message}
# - Metadata: {self.metadata}
# - Timestamp: {self.timestamp}
# """


# class DatabaseRestorationError(SessionRestorationError):
#     """Database-related restoration errors"""

#     def __init__(self, message: str, **kwargs):
#         kwargs.setdefault("error_type", "database_error")
#         kwargs.setdefault("ai_context", "Database operation failed during restoration")
#         super().__init__(message, **kwargs)


# class SandboxRestorationError(SessionRestorationError):
#     """E2B sandbox-related restoration errors"""

#     def __init__(self, message: str, **kwargs):
#         kwargs.setdefault("error_type", "sandbox_error")
#         kwargs.setdefault(
#             "ai_context", "E2B sandbox operation failed during restoration"
#         )
#         super().__init__(message, **kwargs)


# class FileRestorationError(SessionRestorationError):
#     """File system-related restoration errors"""

#     def __init__(self, message: str, **kwargs):
#         kwargs.setdefault("error_type", "file_error")
#         kwargs.setdefault("ai_context", "File operation failed during restoration")
#         super().__init__(message, **kwargs)


# # =============================================================================
# # MAIN RESTORATION CLASS (Updated for LangGraph)
# # =============================================================================


# class SessionRestorationLangGraph:
#     """
#     Service for restoring project files for LangGraph agents.

#     LANGGRAPH CHANGES:
#     - Session state managed by LangGraph checkpointer (not here)
#     - Only handles file restoration to E2B sandbox
#     - Agent conversation history managed by PostgresSaver
#     - thread_id = project_id for continuity
#     """

#     def __init__(self, user_id: str, project_id: str):
#         """
#         Initialize restoration service.

#         Args:
#             user_id: User identifier
#             project_id: Project identifier (also used as thread_id in LangGraph)
#         """
#         self.user_id = user_id
#         self.project_id = project_id
#         self.sandbox = None
#         self.restoration_log = []

#     async def restore_project_files(self) -> Dict[str, Any]:
#         """
#         Restore project files to E2B sandbox for LangGraph agent.

#         Note: This does NOT restore conversation history - that's handled
#         by LangGraph's checkpointer when you invoke the agent with the
#         same thread_id.

#         Returns:
#             Restoration status with file counts and sandbox instance
#         """
#         try:
#             logger.info(
#                 f"[RESTORE-LANGGRAPH] Starting file restoration for {self.user_id}/{self.project_id}"
#             )

#             result = {
#                 "status": "restoring",
#                 "user_id": self.user_id,
#                 "project_id": self.project_id,
#                 "thread_id": self.project_id,  # For LangGraph continuity
#                 "framework": "langgraph",
#                 "started_at": datetime.now(UTC).isoformat(),
#                 "sandbox_created": False,
#                 "files_restored": 0,
#                 "dependency_files_detected": [],
#                 "errors": [],
#                 "file_errors": [],
#             }

#             # Store reference for file error tracking
#             self._current_result = result

#             # Step 1: Get project info
#             project_info = await self._get_project_info()
#             if project_info:
#                 result["project_info"] = {
#                     "name": project_info.get("name", "Unnamed Project"),
#                     "status": project_info.get("status", "unknown"),
#                     "last_active": project_info.get("last_active"),
#                 }

#             # Step 2: Get or create sandbox
#             self.sandbox = await self._get_or_create_sandbox()
#             if not self.sandbox:
#                 raise SessionRestorationError("Failed to get/create E2B sandbox")

#             result["sandbox_created"] = True
#             result["sandbox_id"] = getattr(self.sandbox, "id", "unknown")
#             logger.info(f"[RESTORE-LANGGRAPH] Sandbox ready: {result['sandbox_id']}")

#             # Step 3: Load files from database
#             files = await self._load_project_files()

#             if not files:
#                 logger.warning(
#                     f"[RESTORE-LANGGRAPH] No files found for project {self.project_id}"
#                 )
#                 result["status"] = "completed"
#                 result["message"] = "No files to restore (new project)"
#                 result["sandbox"] = self.sandbox
#                 return result

#             logger.info(f"[RESTORE-LANGGRAPH] Loaded {len(files)} files from database")

#             # Step 4: Restore files to sandbox
#             restored_files = await self._restore_files_to_sandbox(files)
#             result["files_restored"] = len(restored_files)
#             result["restored_files"] = restored_files

#             # Step 5: Detect dependency files
#             dependency_files = self._detect_dependency_files(restored_files)
#             result["dependency_files_detected"] = dependency_files

#             # Step 6: Update project status
#             await self._update_project_status()

#             # Step 7: Handle dependencies
#             if dependency_files:
#                 logger.info(f"[RESTORE-LANGGRAPH] Dependency files: {dependency_files}")
#                 await asyncio.sleep(2)  # Wait for watcher
#                 result["dependency_status"] = "installation_triggered"
#                 result["message"] = (
#                     f"Restored {len(restored_files)} files. Dependencies installing."
#                 )
#             else:
#                 result["dependency_status"] = "none"
#                 result["message"] = (
#                     f"Restored {len(restored_files)} files. No dependencies detected."
#                 )

#             # Final status
#             result["status"] = "completed"
#             result["completed_at"] = datetime.now(UTC).isoformat()
#             result["sandbox"] = self.sandbox

#             logger.info(
#                 f"[RESTORE-LANGGRAPH] ✅ File restoration completed for {self.project_id}"
#             )

#             return result

#         except SessionRestorationError as e:
#             logger.error(f"[RESTORE-LANGGRAPH] ❌ Restoration failed: {e.message}")
#             return {
#                 "status": "error",
#                 "framework": "langgraph",
#                 "error_details": e.to_dict(),
#                 "ai_debug_info": e.for_ai_debug(),
#                 "recovery_suggestions": self._get_recovery_suggestions(e),
#                 "restoration_log": self.restoration_log,
#             }

#         except Exception as e:
#             logger.error(f"[RESTORE-LANGGRAPH] ❌ Unexpected error: {e}")
#             unexpected_error = SessionRestorationError(
#                 message=f"Unexpected error during restoration: {str(e)}",
#                 error_type="unexpected_error",
#                 ai_context="Unhandled exception in LangGraph file restoration",
#                 retry_possible=False,
#                 severity="critical",
#             )

#             return {
#                 "status": "error",
#                 "framework": "langgraph",
#                 "error_details": unexpected_error.to_dict(),
#                 "restoration_log": self.restoration_log,
#             }

#     # =============================================================================
#     # HELPER METHODS (Same as before, just renamed _get_or_create_sandbox)
#     # =============================================================================

#     async def _get_project_info(self) -> Optional[Dict[str, Any]]:
#         """Get project information"""
#         try:
#             project_data = await db_service.get_project(self.project_id)
#             if project_data:
#                 self.restoration_log.append(
#                     f"Found project: {project_data.get('name', 'Unnamed')}"
#                 )
#                 return project_data
#             else:
#                 self.restoration_log.append(f"Project {self.project_id} not found")
#                 return None
#         except Exception as e:
#             logger.warning(f"[RESTORE-LANGGRAPH] Could not get project info: {e}")
#             self.restoration_log.append(f"Warning: Could not get project info - {e}")
#             return None

#     async def _get_or_create_sandbox(self):
#         """Get existing or create new E2B sandbox"""
#         try:
#             # Use multi-tenant sandbox manager (handles persistence)
#             sandbox = await get_user_sandbox(self.user_id, self.project_id)
#             sandbox_id = getattr(
#                 sandbox, "sandbox_id", getattr(sandbox, "id", "unknown")
#             )

#             self.restoration_log.append(
#                 f"Got sandbox via MultiTenantSandboxManager: {sandbox_id}"
#             )

#             logger.info(f"[RESTORE-LANGGRAPH] Sandbox ready: {sandbox_id}")
#             return sandbox

#         except Exception as e:
#             logger.error(f"[RESTORE-LANGGRAPH] Failed to get/create sandbox: {e}")
#             raise SandboxRestorationError(
#                 message=f"E2B sandbox setup failed: {e}",
#                 error_type="sandbox_creation_error",
#                 retry_possible=True,
#                 severity="high",
#             )

#     async def _load_project_files(self) -> Dict[str, Dict[str, Any]]:
#         """Load all project files from database"""
#         try:
#             files_data = await db_service.get_project_files(self.project_id)

#             if not files_data or not files_data.get("files"):
#                 return {}

#             files = files_data["files"]
#             self.restoration_log.append(f"Loaded {len(files)} files from database")

#             return files

#         except Exception as e:
#             logger.error(f"[RESTORE-LANGGRAPH] Failed to load files: {e}")
#             raise DatabaseRestorationError(
#                 message=f"Database query failed: {e}",
#                 error_type="database_query_error",
#                 retry_possible=True,
#                 severity="high",
#             )

#     async def _restore_files_to_sandbox(
#         self, files: Dict[str, Dict[str, Any]]
#     ) -> List[str]:
#         """Write all files to E2B sandbox"""
#         restored_files = []

#         for file_path, file_data in files.items():
#             try:
#                 content = file_data.get("content", "")
#                 normalized_path = file_path.lstrip("/")

#                 if not normalized_path.startswith("home/user/code/"):
#                     full_path = f"/home/user/code/{normalized_path}"
#                 else:
#                     full_path = f"/{normalized_path}"

#                 # Create directory
#                 directory = full_path.rsplit("/", 1)[0]
#                 try:
#                     await self.sandbox.commands.run(f"mkdir -p {directory}")
#                 except Exception:
#                     pass

#                 # Write file
#                 await self.sandbox.files.write(full_path, content)
#                 restored_files.append(full_path)
#                 self.restoration_log.append(f"Restored: {full_path}")

#                 logger.debug(
#                     f"[RESTORE-LANGGRAPH] ✅ Wrote {full_path} ({len(content)} bytes)"
#                 )

#             except Exception as e:
#                 error_msg = f"Failed to restore {file_path}: {e}"
#                 self.restoration_log.append(f"ERROR: {error_msg}")
#                 logger.error(f"[RESTORE-LANGGRAPH] {error_msg}")

#                 if hasattr(self, "_current_result"):
#                     if "file_errors" not in self._current_result:
#                         self._current_result["file_errors"] = []
#                     self._current_result["file_errors"].append(
#                         {
#                             "file_path": file_path,
#                             "error_message": str(e),
#                         }
#                     )

#         logger.info(
#             f"[RESTORE-LANGGRAPH] Restored {len(restored_files)}/{len(files)} files"
#         )
#         return restored_files

#     async def _update_project_status(self) -> None:
#         """Update project status to active"""
#         try:
#             await db_service.update_project_status(
#                 project_id=self.project_id, status="active"
#             )
#             self.restoration_log.append("Updated project status to active")
#         except Exception as e:
#             logger.warning(f"[RESTORE-LANGGRAPH] Could not update status: {e}")

#     def _detect_dependency_files(self, restored_files: List[str]) -> List[str]:
#         """Detect dependency files"""
#         dependency_files = []
#         for file_path in restored_files:
#             if file_path.endswith("package.json"):
#                 dependency_files.append(f"npm (package.json at {file_path})")
#             elif file_path.endswith("requirements.txt"):
#                 dependency_files.append(f"pip (requirements.txt at {file_path})")
#             elif file_path.endswith("Pipfile"):
#                 dependency_files.append(f"pipenv (Pipfile at {file_path})")
#             elif file_path.endswith(("poetry.lock", "pyproject.toml")):
#                 dependency_files.append(f"poetry (at {file_path})")
#             elif file_path.endswith("yarn.lock"):
#                 dependency_files.append(f"yarn (yarn.lock at {file_path})")
#             elif file_path.endswith("pnpm-lock.yaml"):
#                 dependency_files.append(f"pnpm (pnpm-lock.yaml at {file_path})")
#         return dependency_files

#     def _get_recovery_suggestions(self, error: SessionRestorationError) -> List[str]:
#         """Get recovery suggestions based on error type"""
#         suggestions = {
#             "database_connection_failure": [
#                 "Check database server status",
#                 "Verify connection string",
#                 "Retry in 30 seconds",
#             ],
#             "sandbox_creation_error": [
#                 "Check E2B service status",
#                 "Verify E2B API key",
#                 "Wait and retry",
#             ],
#             "file_error": [
#                 "Check sandbox permissions",
#                 "Verify file paths",
#                 "Retry operation",
#             ],
#         }

#         return suggestions.get(error.error_type, ["Check logs", "Retry operation"])


# # =============================================================================
# # PUBLIC API FOR LANGGRAPH
# # =============================================================================


# async def restore_project_for_langgraph(
#     user_id: str, project_id: str
# ) -> Dict[str, Any]:
#     """
#     Restore project files for LangGraph agent.

#     This only restores FILES from database to E2B sandbox.
#     Conversation history is automatically handled by LangGraph's
#     checkpointer when you use the same thread_id.

#     Args:
#         user_id: User identifier
#         project_id: Project identifier (use as thread_id in agent.invoke())

#     Returns:
#         Restoration result with sandbox instance

#     Example:
#         ```
#         # In your FastAPI endpoint
#         from langgraph.checkpoint.postgres import PostgresSaver
#         from context.global_context import create_config

#         # Restore files to sandbox
#         result = await restore_project_for_langgraph("user_123", "project_456")

#         if result["status"] == "completed":
#             sandbox = result["sandbox"]

#             # Create config for agent (thread_id = project_id)
#             config = create_config(
#                 user_id="user_123",
#                 project_id="project_456",
#                 session_id="project_456"  # Same as project_id
#             )

#             # Invoke agent (conversation history auto-loaded by checkpointer)
#             response = agent.invoke(
#                 {"messages": [("user", "Continue where we left off")]},
#                 config  # LangGraph loads conversation history automatically!
#             )
#         ```
#     """
#     try:
#         restoration = SessionRestorationLangGraph(user_id, project_id)
#         result = await restoration.restore_project_files()
#         return result

#     except Exception as e:
#         logger.error(f"[RESTORE-API-LANGGRAPH] File restoration failed: {e}")
#         return {
#             "status": "error",
#             "framework": "langgraph",
#             "error": str(e),
#             "message": "Failed to restore project files for LangGraph",
#         }


# # Export main function
# __all__ = [
#     "restore_project_for_langgraph",
#     "SessionRestorationLangGraph",
# ]


# services/session_restoration.py - PRODUCTION ENHANCED

"""
Session Restoration with Auto-Activation
- Restores files from database
- Installs dependencies (npm/pip/etc.)
- Starts backend and frontend services
- Returns service URLs
- Handles incomplete code gracefully
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from db.service import db_service
from sandbox_manager import get_user_sandbox

logger = logging.getLogger(__name__)


# =============================================================================
# DEPENDENCY DETECTION & INSTALLATION
# =============================================================================

async def detect_and_install_dependencies(
    sandbox,
    project_files: Dict[str, str]
) -> Dict[str, Any]:
    """
    Detect project type and install dependencies.
    
    Returns:
        {
            "installed": {
                "npm": bool,
                "pip": bool,
                "other": []
            },
            "errors": []
        }
    """
    
    result = {
        "installed": {
            "npm": False,
            "pip": False,
            "yarn": False,
            "pnpm": False,
        },
        "errors": [],
    }
    
    # Check for Node.js projects
    if "package.json" in project_files:
        logger.info("📦 Detected Node.js project (package.json)")
        
        # Detect package manager
        has_yarn_lock = "yarn.lock" in project_files
        has_pnpm_lock = "pnpm-lock.yaml" in project_files
        
        try:
            if has_pnpm_lock:
                logger.info("   Using pnpm...")
                cmd_result = await sandbox.commands.run(
                    "cd /home/user/code && pnpm install",
                    timeout=120,
                )
                result["installed"]["pnpm"] = cmd_result.exit_code == 0
                
            elif has_yarn_lock:
                logger.info("   Using yarn...")
                cmd_result = await sandbox.commands.run(
                    "cd /home/user/code && yarn install",
                    timeout=120,
                )
                result["installed"]["yarn"] = cmd_result.exit_code == 0
            
            else:
                logger.info("   Using npm...")
                cmd_result = await sandbox.commands.run(
                    "cd /home/user/code && npm install",
                    timeout=120,
                )
                result["installed"]["npm"] = cmd_result.exit_code == 0
            
            if cmd_result.exit_code != 0:
                result["errors"].append({
                    "type": "npm_install",
                    "error": cmd_result.stderr or "Installation failed"
                })
                logger.error(f"❌ npm install failed: {cmd_result.stderr}")
            else:
                logger.info("✅ Node.js dependencies installed")
                
        except Exception as e:
            result["errors"].append({
                "type": "npm_install",
                "error": str(e)
            })
            logger.error(f"❌ npm install error: {e}")
    
    # Check for Python projects
    if "requirements.txt" in project_files:
        logger.info("🐍 Detected Python project (requirements.txt)")
        
        try:
            cmd_result = await sandbox.commands.run(
                "cd /home/user/code && pip install -r requirements.txt",
                timeout=180,
            )
            result["installed"]["pip"] = cmd_result.exit_code == 0
            
            if cmd_result.exit_code != 0:
                result["errors"].append({
                    "type": "pip_install",
                    "error": cmd_result.stderr or "Installation failed"
                })
                logger.error(f"❌ pip install failed: {cmd_result.stderr}")
            else:
                logger.info("✅ Python dependencies installed")
                
        except Exception as e:
            result["errors"].append({
                "type": "pip_install",
                "error": str(e)
            })
            logger.error(f"❌ pip install error: {e}")
    
    return result


# =============================================================================
# SERVICE DETECTION & STARTUP
# =============================================================================

async def detect_and_start_services(
    sandbox,
    project_files: Dict[str, str]
) -> Dict[str, Any]:
    """
    Detect and start backend/frontend services.
    
    Returns:
        {
            "services": {
                "backend": {
                    "started": bool,
                    "url": str,
                    "port": int,
                    "pid": int,
                    "type": "fastapi|flask|express|..."
                },
                "frontend": {
                    "started": bool,
                    "url": str,
                    "port": int,
                    "pid": int,
                    "type": "react|next|vue|..."
                }
            },
            "incomplete_code": bool,
            "errors": []
        }
    """
    
    result = {
        "services": {},
        "incomplete_code": False,
        "errors": [],
    }
    
    # =========================================================================
    # BACKEND DETECTION & START
    # =========================================================================
    
    backend_started = False
    backend_port = None
    backend_type = None
    
    # Check for FastAPI/Flask (Python)
    if "main.py" in project_files or "app.py" in project_files:
        backend_type = "python"
        
        # Try FastAPI first
        for filename in ["main.py", "app.py"]:
            if filename in project_files:
                logger.info(f"🚀 Attempting to start Python backend ({filename})...")
                
                try:
                    # Start with uvicorn (FastAPI) or gunicorn (Flask)
                    start_cmd = f"cd /home/user/code && uvicorn {filename.replace('.py', '')}:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &"
                    
                    cmd_result = await sandbox.commands.run(start_cmd)
                    
                    # Wait for service to start
                    await asyncio.sleep(3)
                    
                    # Check if service is running
                    check_result = await sandbox.commands.run(
                        "curl -s http://localhost:8000/health || curl -s http://localhost:8000"
                    )
                    
                    if check_result.exit_code == 0:
                        # Get PID
                        pid_result = await sandbox.commands.run(
                            f"ps aux | grep uvicorn | grep -v grep | awk '{{print $2}}' | head -1"
                        )
                        pid = int(pid_result.stdout.strip()) if pid_result.stdout.strip() else None
                        
                        backend_started = True
                        backend_port = 8000
                        
                        result["services"]["backend"] = {
                            "started": True,
                            "url": f"http://8000-{sandbox.get_hostname()}",
                            "port": 8000,
                            "pid": pid,
                            "type": "fastapi",
                            "file": filename,
                        }
                        
                        logger.info(f"✅ Backend started on port 8000 (PID: {pid})")
                        break
                    
                except Exception as e:
                    result["errors"].append({
                        "type": "backend_start",
                        "file": filename,
                        "error": str(e)
                    })
                    logger.warning(f"⚠️ Failed to start {filename}: {e}")
    
    # Check for Express.js (Node)
    elif "server.js" in project_files or "index.js" in project_files:
        backend_type = "nodejs"
        
        for filename in ["server.js", "index.js"]:
            if filename in project_files:
                logger.info(f"🚀 Attempting to start Node.js backend ({filename})...")
                
                try:
                    start_cmd = f"cd /home/user/code && node {filename} > backend.log 2>&1 &"
                    await sandbox.commands.run(start_cmd)
                    await asyncio.sleep(3)
                    
                    # Check if running
                    check_result = await sandbox.commands.run(
                        "curl -s http://localhost:3000 || curl -s http://localhost:8000"
                    )
                    
                    if check_result.exit_code == 0:
                        pid_result = await sandbox.commands.run(
                            "ps aux | grep node | grep -v grep | awk '{print $2}' | head -1"
                        )
                        pid = int(pid_result.stdout.strip()) if pid_result.stdout.strip() else None
                        
                        backend_started = True
                        backend_port = 3000
                        
                        result["services"]["backend"] = {
                            "started": True,
                            "url": f"http://3000-{sandbox.get_hostname()}",
                            "port": 3000,
                            "pid": pid,
                            "type": "express",
                            "file": filename,
                        }
                        
                        logger.info(f"✅ Backend started on port 3000 (PID: {pid})")
                        break
                        
                except Exception as e:
                    result["errors"].append({
                        "type": "backend_start",
                        "file": filename,
                        "error": str(e)
                    })
    
    # =========================================================================
    # FRONTEND DETECTION & START
    # =========================================================================
    
    frontend_started = False
    frontend_port = None
    
    # Check for React/Next.js
    if "package.json" in project_files:
        package_json_data = project_files["package.json"]
        
        # Convert to string if needed (FIXED!)
        if isinstance(package_json_data, dict):
            package_json_content = package_json_data.get("content", "")
        else:
            package_json_content = str(package_json_data)
        
        # Detect frontend framework (FIXED!)
        if "next" in package_json_content.lower():
            logger.info("🚀 Attempting to start Next.js frontend...")
            frontend_type = "nextjs"
            start_cmd = "cd /home/user/code && npm run dev > frontend.log 2>&1 &"
            frontend_port = 3000
            
        elif "react" in package_json_content.lower() or "vite" in package_json_content.lower():
            logger.info("🚀 Attempting to start React/Vite frontend...")
            frontend_type = "react"
            start_cmd = "cd /home/user/code && npm run dev > frontend.log 2>&1 &"
            frontend_port = 5173  # Vite default
        
        else:
            frontend_type = "nodejs"
            start_cmd = "cd /home/user/code && npm start > frontend.log 2>&1 &"
            frontend_port = 3000
        
        try:
            await sandbox.commands.run(start_cmd)
            await asyncio.sleep(5)  # Wait for frontend to compile
            
            # Check if running
            check_result = await sandbox.commands.run(
                f"curl -s http://localhost:{frontend_port}"
            )
            
            if check_result.exit_code == 0:
                pid_result = await sandbox.commands.run(
                    f"ps aux | grep 'npm\\|node' | grep -v grep | awk '{{print $2}}' | head -1"
                )
                pid = int(pid_result.stdout.strip()) if pid_result.stdout.strip() else None
                
                frontend_started = True
                
                result["services"]["frontend"] = {
                    "started": True,
                    "url": f"http://{frontend_port}-{sandbox.get_hostname()}",
                    "port": frontend_port,
                    "pid": pid,
                    "type": frontend_type,
                }
                
                logger.info(f"✅ Frontend started on port {frontend_port} (PID: {pid})")
                
        except Exception as e:
            result["errors"].append({
                "type": "frontend_start",
                "error": str(e)
            })
            logger.warning(f"⚠️ Failed to start frontend: {e}")
    
    # =========================================================================
    # INCOMPLETE CODE DETECTION
    # =========================================================================
    
    # If dependencies installed but services didn't start
    if not backend_started and not frontend_started:
        result["incomplete_code"] = True
        logger.info("ℹ️ Code appears incomplete - dependencies installed but services won't start")
    
    return result


# =============================================================================
# MAIN RESTORATION FUNCTION (ENHANCED)
# =============================================================================

async def restore_and_activate_project(
    user_id: str,
    project_id: str,
) -> Dict[str, Any]:
    """
    Complete session restoration with auto-activation.
    
    Steps:
    1. Load files from database
    2. Write files to E2B sandbox
    3. Detect and install dependencies
    4. Start backend service (if applicable)
    5. Start frontend service (if applicable)
    6. Return service URLs
    
    Returns:
        {
            "status": "completed" | "partial" | "error",
            "files_restored": int,
            "dependencies": {
                "installed": {...},
                "errors": []
            },
            "services": {
                "backend": {
                    "started": bool,
                    "url": str,
                    "port": int,
                    "pid": int
                },
                "frontend": {...}
            },
            "incomplete_code": bool,
            "message": str
        }
    """
    
    logger.info(f"🔄 Starting restoration and activation for {user_id}/{project_id}")
    
    result = {
        "status": "in_progress",
        "user_id": user_id,
        "project_id": project_id,
        "timestamp": datetime.utcnow().isoformat(),
        "files_restored": 0,
        "dependencies": {},
        "services": {},
        "incomplete_code": False,
        "errors": [],
    }
    
    try:
        # Step 1: Load files
        logger.info("📂 Step 1: Loading files from database...")
        files_data = await db_service.get_project_files(project_id)
        project_files = files_data.get("files", {})
        
        if not project_files:
            result["status"] = "error"
            result["message"] = "No files found for this project"
            return result
        
        logger.info(f"✅ Loaded {len(project_files)} files")
        
        # Step 2: Get/Create sandbox
        logger.info("🏗️ Step 2: Getting E2B sandbox...")
        sandbox = await get_user_sandbox(user_id, project_id)
        result["sandbox_id"] = sandbox.sandbox_id
        
        # Step 3: Write files (FIXED!)
        logger.info("📝 Step 3: Writing files to sandbox...")
        
        # Convert files to string format for processing
        project_files_processed = {}
        
        for file_path, file_data in project_files.items():
            try:
                # Handle different data types from database
                if isinstance(file_data, dict):
                    content = file_data.get("content", "")
                elif isinstance(file_data, bytes):
                    content = file_data.decode('utf-8')
                else:
                    content = str(file_data)
                
                # Store in processed format
                project_files_processed[file_path] = content
                
                # Write to sandbox
                await sandbox.files.write(f"/home/user/code/{file_path}", content)
                result["files_restored"] += 1
                
            except Exception as e:
                result["errors"].append({
                    "type": "file_write",
                    "file": file_path,
                    "error": str(e)
                })
                logger.error(f"❌ Failed to write {file_path}: {e}")
        
        # Use processed files for subsequent operations
        project_files = project_files_processed
        
        logger.info(f"✅ Wrote {result['files_restored']} files")
        
        # Step 4: Install dependencies
        logger.info("📦 Step 4: Installing dependencies...")
        dep_result = await detect_and_install_dependencies(sandbox, project_files)
        result["dependencies"] = dep_result
        
        # Step 5: Start services
        logger.info("🚀 Step 5: Starting services...")
        service_result = await detect_and_start_services(sandbox, project_files)
        result["services"] = service_result["services"]
        result["incomplete_code"] = service_result["incomplete_code"]
        result["errors"].extend(service_result["errors"])
        
        # Determine final status
        if result["incomplete_code"]:
            result["status"] = "partial"
            result["message"] = "Files and dependencies restored. Code appears incomplete - services could not start."
        elif result["services"]:
            result["status"] = "completed"
            result["message"] = "Project fully restored and activated!"
        else:
            result["status"] = "partial"
            result["message"] = "Files restored, but no services detected or started."
        
        logger.info(f"✅ Restoration complete: {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Restoration failed: {e}", exc_info=True)
        result["status"] = "error"
        result["message"] = str(e)
        result["errors"].append({
            "type": "restoration_error",
            "error": str(e)
        })
        return result

