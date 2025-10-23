# db/data_access.py - Clean Data Access Layer
"""
Data access layer for database operations.
Pure CRUD operations without business logic.
"""

from sqlalchemy import select, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone  # Use explicit timezone
import asyncio
from sqlalchemy.exc import OperationalError, DatabaseError
from .models import (
    User,
    Project,
    ProjectFile,
    SandboxState,
    SessionStatus,
    ProjectThought,
)
import logging

# Simple module-level logger for this module
logger = logging.getLogger(__name__)

# Configure basic logging only if the root logger has no handlers to avoid
# interfering with application-level logging configuration.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# db/data_access.py

from sqlalchemy.exc import IntegrityError
from sqlalchemy import or_
import logging

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_user(self, user_id: str, email: str, username: str) -> User:
        """
        Get existing user or create new one (handles user_id AND email conflicts).

        Race-condition safe for concurrent calls.
        """

        # Step 1: Check if user_id OR email already exist
        stmt = select(User).where(or_(User.id == user_id, User.email == email))
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            # User exists
            if user.id == user_id:
                # Exact match - return existing user
                logger.debug(f"User {user_id} already exists")
                return user
            else:
                # Email collision - return existing user to avoid conflict
                logger.warning(
                    f"Email {email} already used by user {user.id}, returning existing user"
                )
                return user

        # Step 2: User doesn't exist - try to create
        try:
            user = User(id=user_id, email=email, username=username)
            self.session.add(user)
            await self.session.flush()

            logger.info(f"✅ Created new user: {user_id}")
            return user

        except IntegrityError as e:
            # Race condition: Another process created user between SELECT and INSERT
            await self.session.rollback()

            error_msg = str(e.orig).lower()

            # Determine which constraint was violated
            if "ix_users_email" in error_msg or "email" in error_msg:
                # Email conflict - fetch by email
                logger.info(
                    f"✅ User with email {email} already created by another transaction"
                )

                stmt = select(User).where(User.email == email)
                result = await self.session.execute(stmt)
                user = result.scalar_one_or_none()

                if user:
                    return user

            else:
                # user_id conflict - fetch by user_id
                logger.info(f"✅ User {user_id} already created by another transaction")

                stmt = select(User).where(User.id == user_id)
                result = await self.session.execute(stmt)
                user = result.scalar_one_or_none()

                if user:
                    return user

            # If we still don't have a user, something is very wrong
            logger.error(f"❌ IntegrityError but couldn't fetch user: {e}")
            raise RuntimeError(f"Failed to create or fetch user {user_id}: {e}")


class FileRepository:
    """Repository for file operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_project_file(
        self,
        project_id: str,
        file_path: str,
        content: str,
        created_by_tool: str = "write_file",
    ) -> ProjectFile:
        """Save project file - UPSERT operation"""
        now = (
            datetime.now()
        )  # Use naive datetime for PostgreSQL TIMESTAMP WITHOUT TIME ZONE

        # use byte-length for accurate size accounting
        byte_size = len(content.encode("utf-8"))

        stmt = insert(ProjectFile).values(
            project_id=project_id,
            file_path=file_path,
            content=content,
            created_by_tool=created_by_tool,
            size_bytes=byte_size,
            created_at=now,
            updated_at=now,
        )

        # On conflict, update
        stmt = stmt.on_conflict_do_update(
            index_elements=["project_id", "file_path"],
            set_=dict(
                content=stmt.excluded.content,
                size_bytes=stmt.excluded.size_bytes,
                created_by_tool=stmt.excluded.created_by_tool,
                updated_at=now,
                is_deleted=False,
                deleted_at=None,
            ),
        ).returning(ProjectFile)

        result = await self.session.execute(stmt)
        project_file = result.scalar_one()
        await self.session.flush()
        return project_file

    async def get_project_file(
        self, project_id: str, file_path: str
    ) -> Optional[ProjectFile]:
        """Get a specific file from project"""
        stmt = select(ProjectFile).where(
            ProjectFile.project_id == project_id,
            ProjectFile.file_path == file_path,
            ProjectFile.is_deleted == False,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_project_files(self, project_id: str) -> List[ProjectFile]:
        """Get all files for a project (excluding deleted)"""
        stmt = (
            select(ProjectFile)
            .where(
                ProjectFile.project_id == project_id, ProjectFile.is_deleted == False
            )
            .order_by(ProjectFile.file_path)
        )

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def delete_project_file(
        self, project_id: str, file_path: str, soft_delete: bool = True
    ) -> bool:
        """Delete a project file"""
        if soft_delete:
            stmt = select(ProjectFile).where(
                ProjectFile.project_id == project_id, ProjectFile.file_path == file_path
            )
            result = await self.session.execute(stmt)
            project_file = result.scalar_one_or_none()

            if project_file:
                project_file.is_deleted = True
                project_file.deleted_at = datetime.now()  # Use naive datetime
                await self.session.flush()
                return True
        else:
            stmt = delete(ProjectFile).where(
                ProjectFile.project_id == project_id, ProjectFile.file_path == file_path
            )
            result = await self.session.execute(stmt)
            return result.rowcount > 0

        return False

    async def get_project_file_count(self, project_id: str) -> int:
        """Get count of files in project"""
        stmt = select(func.count(ProjectFile.id)).where(
            ProjectFile.project_id == project_id, ProjectFile.is_deleted == False
        )
        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)

    async def save_multiple_files(
        self,
        project_id: str,
        files: Dict[str, str],
        created_by_tool: str = "batch_write",
    ) -> List[ProjectFile]:
        """Save multiple files efficiently"""
        saved_files = []
        for file_path, content in files.items():
            project_file = await self.save_project_file(
                project_id=project_id,
                file_path=file_path,
                content=content,
                created_by_tool=created_by_tool,
            )
            saved_files.append(project_file)
        return saved_files


class ProjectRepository:
    """Repository for project operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_project(
        self, user_id: str, project_id: str, project_name: str = "Default Project"
    ) -> Project:
        """Get existing project or create new one"""
        stmt = select(Project).where(Project.id == project_id)
        result = await self.session.execute(stmt)
        project = result.scalar_one_or_none()

        if not project:
            project = Project(id=project_id, user_id=user_id, name=project_name)
            self.session.add(project)
            await self.session.flush()

        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        stmt = select(Project).where(Project.id == project_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_sandbox_state(
        self, project_id: str, sandbox_id: Optional[str], state: SandboxState
    ):
        """Update project's sandbox information"""
        stmt = select(Project).where(Project.id == project_id)
        result = await self.session.execute(stmt)
        project = result.scalar_one_or_none()

        if project:
            project.active_sandbox_id = sandbox_id
            project.sandbox_state = state
            project.updated_at = datetime.now()  # Use naive datetime
            await self.session.flush()

    async def get_user_projects(self, user_id: str) -> List[Project]:
        """Get all projects for a user"""
        stmt = (
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(desc(Project.updated_at))
        )

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_project_status(self, project_id: str, status: SessionStatus):
        """Update project status"""
        stmt = select(Project).where(Project.id == project_id)
        result = await self.session.execute(stmt)
        project = result.scalar_one_or_none()

        if project:
            project.status = status
            project.updated_at = datetime.now()  # Use naive datetime
            if status == SessionStatus.ENDED:
                project.ended_at = datetime.now()  # Use naive datetime
            await self.session.flush()


class ThoughtRepository:
    """Repository for thought operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_thought(
        self,
        project_id: str,
        thought: str,
        thought_type: str = "planning",
        phase: Optional[str] = None,
        milestone: Optional[str] = None,
        priority: str = "normal",
    ) -> ProjectThought:
        """Save a new thought"""
        thought_obj = ProjectThought(
            project_id=project_id,
            thought=thought,
            thought_type=thought_type,
            phase=phase,
            milestone=milestone,
            priority=priority,
        )
        self.session.add(thought_obj)
        await self.session.commit()
        return thought_obj

    async def get_thoughts(
        self,
        project_id: str,
        thought_type: Optional[str] = None,
        phase: Optional[str] = None,
        limit: int = 10,
    ) -> List[ProjectThought]:
        """Get thoughts with optional filters"""
        stmt = select(ProjectThought).where(ProjectThought.project_id == project_id)

        if thought_type:
            stmt = stmt.where(ProjectThought.thought_type == thought_type)

        if phase:
            stmt = stmt.where(ProjectThought.phase == phase)

        stmt = stmt.order_by(desc(ProjectThought.timestamp)).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_milestones(
        self, project_id: str, limit: int = 20
    ) -> List[ProjectThought]:
        """Get milestone thoughts"""
        stmt = (
            select(ProjectThought)
            .where(
                ProjectThought.project_id == project_id,
                ProjectThought.milestone.isnot(None),
            )
            .order_by(desc(ProjectThought.timestamp))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_thoughts(self, project_id: str) -> int:
        """Count total thoughts"""
        stmt = select(func.count(ProjectThought.id)).where(
            ProjectThought.project_id == project_id
        )
        return await self.session.scalar(stmt)

    async def delete_old_thoughts(self, project_id: str, cutoff_date: datetime) -> int:
        """Delete thoughts older than cutoff date"""
        stmt = delete(ProjectThought).where(
            ProjectThought.project_id == project_id,
            ProjectThought.timestamp < cutoff_date,
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
