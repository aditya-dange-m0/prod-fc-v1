# db/models.py - SIMPLIFIED ARCHITECTURE
"""
Database Models - Simplified Session Management
==============================================

SIMPLIFIED ARCHITECTURE:
- Project = Session (One-to-One mapping)
- project.id is used as session_id everywhere
- Direct relationship: User -> Projects -> Files/Snapshots

USAGE:
- Agno session_id = project.id
- E2B sandbox_id = project.id
- Database queries use project.id
- Session restoration uses project.id
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String,
    Integer,
    Text,
    DateTime,
    Boolean,
    Enum,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import uuid
import enum


class Base(AsyncAttrs, DeclarativeBase):
    """Base model for all tables"""

    pass


def generate_uuid():
    return str(uuid.uuid4())


# =============================================================================
# ENUMS
# =============================================================================


class SandboxState(str, enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    KILLED = "killed"
    NONE = "none"


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


# =============================================================================
# USER MODEL
# =============================================================================


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    username: Mapped[str] = mapped_column(String, unique=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    projects: Mapped[list["Project"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Project(Base):
    """
    Project = Session (Simplified Architecture)

    Each project represents a continuous working session.
    Project ID is used as:
    - Database identifier
    - Agno session_id
    - E2B sandbox identifier
    - Session restoration identifier
    """

    __tablename__ = "projects"

    # PRIMARY KEY - Use this as session_id everywhere!
    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)

    # User relationship
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # Project metadata
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # E2B Sandbox tracking
    active_sandbox_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    sandbox_state: Mapped[SandboxState] = mapped_column(
        Enum(SandboxState), default=SandboxState.NONE
    )

    # Session status (merged from Session table)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.ACTIVE
    )

    # Timestamps (merged from Session table)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships (removed sessions relationship - Project IS the session)
    user: Mapped["User"] = relationship(back_populates="projects")
    project_files: Mapped[list["ProjectFile"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    thoughts: Mapped[list["ProjectThought"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProjectFile(Base):
    __tablename__ = "project_files"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )

    # File metadata
    file_path: Mapped[str] = mapped_column(String(500), index=True)
    content: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Deletion tracking (soft delete)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Tracking
    created_by_tool: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="project_files")

    __table_args__ = (
        Index(
            "ix_project_files_project_path",
            "project_id",
            "file_path",
        ),
        Index("ix_project_files_deleted", "project_id", "is_deleted", "deleted_at"),
        UniqueConstraint("project_id", "file_path", name="uq_project_file_path"),
    )


class ProjectThought(Base):
    """
    Agent thoughts for context management.
    Stores agent's internal reasoning and planning.
    """

    __tablename__ = "project_thoughts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )

    # Thought content
    thought: Mapped[str] = mapped_column(Text)
    thought_type: Mapped[str] = mapped_column(String(50), default="planning")

    # Organization
    phase: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    milestone: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    priority: Mapped[str] = mapped_column(String(20), default="normal", index=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="thoughts")

    __table_args__ = (
        Index("ix_project_thoughts_project_type", "project_id", "thought_type"),
        Index("ix_project_thoughts_project_phase", "project_id", "phase"),
        Index("ix_project_thoughts_milestone", "milestone"),
    )


# =============================================================================
# EXPORTS - Ensure all models are registered for Alembic autogenerate
# =============================================================================

__all__ = [
    "Base",
    "User",
    "Project",  # Project IS the session now
    "ProjectFile",  # Simplified file storage (no versioning)
    "ProjectThought",
    "SandboxState",
    "SessionStatus",
]
