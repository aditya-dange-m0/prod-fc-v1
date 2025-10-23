"""
Runtime Context for LangChain v1 create_agent.

Integrates with existing database infrastructure:
- db.models (User, Project, ProjectFile, etc.)
- db.service (DatabaseService)
- db.data_access (Repositories)

Architecture:
- Project ID = Session ID = Thread ID
- Coordinates with DatabaseService
- Used for LangChain v1 context= parameter
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, UTC
import logging

if TYPE_CHECKING:
    from db.models import Project, User
    from db.service import DatabaseService

logger = logging.getLogger(__name__)

@dataclass
class RuntimeContext:

    user_id: str
    project_id: str
    email_id: str

    @property
    def thread_id(self) -> str:
        """LangChain thread_id (same as project_id)"""
        return self.project_id

    sandbox_id: Optional[str] = None

    sandbox_state: str = "none" #Values: 'running' | 'paused' | 'killed' | 'none'

    max_iterations: int = 25
    """Maximum agent iterations"""

def get_runtime_context() -> RuntimeContext:
    """
    Get RuntimeContext inside tools or middleware.

    Usage in tools:
        from langgraph.runtime import get_runtime
        from context.runtime_context import RuntimeContext

        @tool
        def my_tool():
            runtime = get_runtime(RuntimeContext)
            project_id = runtime.context.project_id
            user_id = runtime.context.user_id

    Returns:
        RuntimeContext from current runtime
    """
    from langgraph.runtime import get_runtime

    runtime = get_runtime(RuntimeContext)
    return runtime.context
