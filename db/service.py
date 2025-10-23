# db/service.py - Clean Business Logic Layer
"""
Database service layer - Business logic for database operations.
Renamed from integration.py for clarity.
"""

import json
import logging
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
from sqlalchemy import select, update, delete, desc, func

from db import get_db_session, init_db
from db.models import (
    User,
    Project,
    ProjectFile,
    SessionStatus,
    SandboxState,
)
from db.data_access import (
    UserRepository,
    ProjectRepository,
    FileRepository,
    ThoughtRepository,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Clean database service - Project = Session architecture.
    No redundant methods, no nested sessions, clear API.
    """

    def __init__(self):
        self.initialized = False

    async def initialize(self, db_url: Optional[str] = None) -> bool:
        """Initialize database connection"""
        try:
            await init_db(use_direct=False)
            self.initialized = True
            logger.info("✅ Database service initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False

    # ========================================================================
    # USER OPERATIONS
    # ========================================================================

    # db/service.py

    async def create_user(
        self, user_id: str, email: str = None, username: str = None
    ) -> str:
        """Create or get user with unique email"""
        async with get_db_session() as session:
            repo = UserRepository(session)

            # Generate unique email if not provided
            if not email:
                email = f"{user_id}@generated.local"

            # Use user_id as username if not provided
            if not username:
                username = user_id

            user = await repo.get_or_create_user(
                user_id=user_id,
                email=email,
                username=username,
            )

            return user.id

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        async with get_db_session() as session:
            repo = UserRepository(session)
            user = await repo.get_or_create_user(user_id)

            if user:
                return {
                    "user_id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "created_at": user.created_at,
                }
            return None

    async def delete_user(self, user_id: str) -> bool:
        """Delete user (cascades to projects)"""
        try:
            async with get_db_session() as session:
                stmt = delete(User).where(User.id == user_id)
                result = await session.execute(stmt)
                return result.rowcount > 0
        except Exception:
            return False

    # ========================================================================
    # PROJECT OPERATIONS (Project = Session)
    # ========================================================================

    async def create_project(
        self, user_id: str, project_id: str, project_name: str = None
    ) -> str:
        """
        Create or get project.
        This is the MAIN method - project_id IS your session_id.
        Use this ID everywhere: Agno, E2B, database queries.
        """
        async with get_db_session() as session:
            # Create user first (within same session!)
            user_repo = UserRepository(session)
            await user_repo.get_or_create_user(
                user_id=user_id, email=f"{user_id}@system.local", username=user_id
            )

            # Create/get project
            project_repo = ProjectRepository(session)
            project = await project_repo.get_or_create_project(
                user_id=user_id,
                project_id=project_id,
                project_name=project_name or f"Project {project_id}",
            )

            # Update activity timestamp
            project.last_active = datetime.now()  # Use naive datetime
            project.status = SessionStatus.ACTIVE
            # Context manager commits automatically

            logger.info(f"✅ Project ready: {project_id}")
            return project.id

    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID"""
        async with get_db_session() as session:
            repo = ProjectRepository(session)
            project = await repo.get_project(project_id)

            if project:
                return {
                    "project_id": project.id,
                    "user_id": project.user_id,
                    "name": project.name,
                    "description": project.description,
                    "active_sandbox_id": project.active_sandbox_id,
                    "sandbox_state": (
                        project.sandbox_state.value if project.sandbox_state else "none"
                    ),
                    "status": project.status.value if project.status else "active",
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                    "last_active": project.last_active,
                    "ended_at": project.ended_at,
                }
            return None

    async def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all projects for user"""
        async with get_db_session() as session:
            repo = ProjectRepository(session)
            projects = await repo.get_user_projects(user_id)

            return [
                {
                    "project_id": p.id,
                    "name": p.name,
                    "status": p.status.value,
                    "created_at": p.created_at,
                    "last_active": p.last_active,
                }
                for p in projects
            ]

    async def update_project_status(
        self, project_id: str, status: str, ended_at: datetime = None
    ):
        """Update project status"""
        async with get_db_session() as session:
            repo = ProjectRepository(session)
            status_enum = (
                SessionStatus(status)
                if status in [s.value for s in SessionStatus]
                else SessionStatus.ACTIVE
            )
            await repo.update_project_status(project_id, status_enum)

            # Update ended_at if ending
            if status == "ended" and ended_at:
                stmt = (
                    update(Project)
                    .where(Project.id == project_id)
                    .values(ended_at=ended_at)
                )
                await session.execute(stmt)

    async def update_project_sandbox(
        self, project_id: str, sandbox_id: Optional[str], state: str
    ):
        """Update project sandbox state"""
        async with get_db_session() as session:
            repo = ProjectRepository(session)
            sandbox_state = (
                SandboxState(state)
                if state in [s.value for s in SandboxState]
                else SandboxState.NONE
            )
            await repo.update_sandbox_state(project_id, sandbox_id, sandbox_state)

    async def delete_project(self, project_id: str) -> bool:
        """Delete project (cascades to files/snapshots)"""
        try:
            async with get_db_session() as session:
                stmt = delete(Project).where(Project.id == project_id)
                result = await session.execute(stmt)
                return result.rowcount > 0
        except Exception:
            return False

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    async def save_file(
        self,
        project_id: str,
        file_path: str,
        content: str,
        created_by_tool: str = "write_file",
    ) -> bool:
        """Save file to project (upsert)"""
        try:
            # SEPARATE TRANSACTION: Ensure project exists
            # This prevents rollback of file save if user creation fails
            async with get_db_session() as session_prep:
                user_repo = UserRepository(session_prep)
                project_repo = ProjectRepository(session_prep)

                # Get project to extract user_id
                project = await project_repo.get_project(project_id)

                if project:
                    # Ensure user exists (idempotent)
                    await user_repo.get_or_create_user(
                        user_id=project.user_id,
                        email=f"{project.user_id}@system.local",
                        username=project.user_id,
                    )

            # MAIN TRANSACTION: Save file
            async with get_db_session() as session:
                repo = FileRepository(session)
                await repo.save_project_file(
                    project_id=project_id,
                    file_path=file_path,
                    content=content,
                    created_by_tool=created_by_tool,
                )

            logger.info(f"✅ DB: Saved {file_path}")  # ← Changed to INFO
            return True

        except Exception as e:
            logger.error(f"❌ DB: Save failed for {file_path}: {e}", exc_info=True)
            return False

    async def get_file(
        self, project_id: str, file_path: str
    ) -> Optional[Dict[str, Any]]:
        """Get single file"""
        async with get_db_session() as session:
            repo = FileRepository(session)
            file = await repo.get_project_file(project_id, file_path)

            if file:
                return {
                    "file_path": file.file_path,
                    "content": file.content,
                    "created_at": file.created_at,
                    "updated_at": file.updated_at,
                    "created_by": file.created_by_tool,
                }
            return None

    async def get_project_files(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get all project files"""
        async with get_db_session() as session:
            repo = FileRepository(session)
            project_files = await repo.get_all_project_files(project_id)

            if not project_files:
                return None

            files = {}
            for pf in project_files:
                files[pf.file_path] = {
                    "path": pf.file_path,
                    "content": pf.content,
                    "created_at": pf.created_at,
                    "updated_at": pf.updated_at,
                    "created_by": pf.created_by_tool,
                }

            return {
                "project_id": project_id,
                "files": files,
                "file_count": len(files),
                "updated_at": (
                    max(f["updated_at"] for f in files.values())
                    if files
                    else datetime.now()  # Use naive datetime
                ),
            }

    async def save_multiple_files(
        self,
        project_id: str,
        files: Dict[str, str],
        created_by_tool: str = "batch_write",
    ) -> bool:
        """Save multiple files efficiently"""
        try:
            async with get_db_session() as session:
                repo = FileRepository(session)
                await repo.save_multiple_files(project_id, files, created_by_tool)
                logger.info(f"📦 Saved {len(files)} files")
                return True
        except Exception as e:
            logger.error(f"❌ Batch save failed: {e}")
            return False

    async def delete_file(self, project_id: str, file_path: str) -> bool:
        """Delete file from project"""
        try:
            async with get_db_session() as session:
                repo = FileRepository(session)
                await repo.delete_project_file(project_id, file_path, soft_delete=False)
                logger.debug(f"🗑️ Deleted: {file_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            return False

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            async with get_db_session() as session:
                user_count = await session.scalar(select(func.count(User.id)))
                project_count = await session.scalar(select(func.count(Project.id)))
                file_count = await session.scalar(select(func.count(ProjectFile.id)))

                return {
                    "status": "healthy",
                    "database": "postgresql",
                    "architecture": "project_equals_session",
                    "user_count": user_count,
                    "project_count": project_count,
                    "file_count": file_count,
                    "timestamp": datetime.now(),  # Use naive datetime
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(),  # Use naive datetime
            }

    # ========================================================================
    # THOUGHT OPERATIONS (For context engineering)
    # ========================================================================

    async def save_thought(self, project_id: str, record: Dict[str, Any]) -> bool:
        """Save think() tool output"""
        try:
            async with get_db_session() as session:
                repo = ThoughtRepository(session)
                await repo.save_thought(
                    project_id=project_id,
                    thought=record["thought"],
                    thought_type=record.get("thought_type", "planning"),
                    phase=record.get("phase"),
                    milestone=record.get("milestone"),
                    priority=record.get("priority", "normal"),
                )
                logger.debug(f"💭 Saved thought for project {project_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save thought: {e}")
            return False

    async def get_thoughts(
        self, project_id: str, filters: Optional[Dict[str, str]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve thoughts with optional filters"""
        try:
            async with get_db_session() as session:
                repo = ThoughtRepository(session)

                thoughts = await repo.get_thoughts(
                    project_id=project_id,
                    thought_type=filters.get("thought_type") if filters else None,
                    phase=filters.get("phase") if filters else None,
                    limit=limit,
                )

                return [
                    {
                        "thought": t.thought,
                        "thought_type": t.thought_type,
                        "phase": t.phase,
                        "milestone": t.milestone,
                        "priority": t.priority,
                        "timestamp": t.timestamp.isoformat(),
                    }
                    for t in thoughts
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get thoughts: {e}")
            return []

    async def get_milestones(
        self, project_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get milestone thoughts"""
        try:
            async with get_db_session() as session:
                repo = ThoughtRepository(session)
                milestones = await repo.get_milestones(project_id, limit)

                return [
                    {
                        "thought": m.thought,
                        "milestone": m.milestone,
                        "phase": m.phase,
                        "timestamp": m.timestamp.isoformat(),
                    }
                    for m in milestones
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get milestones: {e}")
            return []

    async def count_thoughts(self, project_id: str) -> int:
        """Count total thoughts"""
        try:
            async with get_db_session() as session:
                repo = ThoughtRepository(session)
                return await repo.count_thoughts(project_id)
        except Exception as e:
            logger.error(f"❌ Failed to count thoughts: {e}")
            return 0

    async def delete_old_thoughts(self, project_id: str, days: int = 30) -> int:
        """Delete thoughts older than N days"""
        try:
            from datetime import timedelta

            cutoff = datetime.now() - timedelta(days=days)  # Use naive datetime

            async with get_db_session() as session:
                repo = ThoughtRepository(session)
                deleted = await repo.delete_old_thoughts(project_id, cutoff)
                logger.info(f"🗑️ Deleted {deleted} old thoughts")
                return deleted
        except Exception as e:
            logger.error(f"❌ Failed to delete old thoughts: {e}")
            return 0


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

db_service = DatabaseService()


# Convenience functions
async def init_database(db_url: Optional[str] = None) -> bool:
    """Initialize database"""
    return await db_service.initialize(db_url)


async def health_check() -> Dict[str, Any]:
    """Database health check"""
    return await db_service.health_check()


async def close_database():
    """Close database connections"""
    from db import close_db

    await close_db()
