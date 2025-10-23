# api/streaming_agent.py - COMPLETE PRODUCTION VERSION

"""
Production Streaming API for Full-Stack Code Gen Agent
Implements Server-Sent Events (SSE) with LangGraph streaming
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Annotated, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from agent.singleton_agent import get_agent_graph
from context.runtime_context import RuntimeContext
from db.service import db_service
from checkpoint.postgres_checkpointer import checkpointer_service
from services.session_restoration import restore_and_activate_project
from agent_state.state import get_state_summary



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ChatRequest(BaseModel):
    message: str
    user_id: str
    project_id: str
    email_id: str = "user@system.local"


class MessageResponse(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_name: Optional[str] = None


class ProjectHistoryResponse(BaseModel):
    project_id: str
    messages: list[MessageResponse]
    message_count: int
    current_phase: Optional[str] = None
    next_steps: Optional[list[str]] = None


# =============================================================================
# FASTAPI APP SETUP
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Production lifespan handler.
    Initializes ALL services ONCE on startup.
    """

    # ==================== STARTUP ====================
    logger.info("[APP] 🚀 Starting application...")

    try:
        # 1. Initialize database
        await db_service.initialize()
        logger.info("✅ Database initialized")

        # 2. Initialize checkpointer (creates connection pool)
        await checkpointer_service.initialize()
        logger.info("✅ Checkpointer initialized")

        # 3. Initialize agent graph (singleton, uses checkpointer)
        graph = await get_agent_graph()
        logger.info("✅ Agent graph compiled")

        logger.info("[APP] ✅ All services ready for production!")

    except Exception as e:
        logger.error(f"[APP] ❌ Startup failed: {e}", exc_info=True)
        raise

    yield

    # ==================== SHUTDOWN ====================
    logger.info("[APP] 🛑 Shutting down...")

    await checkpointer_service.close()
    logger.info("✅ Checkpointer connections closed")


app = FastAPI(
    title="Full-Stack Agent Streaming API",
    description="Real-time streaming API for code generation agent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# STREAMING UTILITIES
# =============================================================================


def format_sse_event(event_type: str, data: dict) -> str:
    """Format Server-Sent Event"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_agent_response(
    message: str, user_id: str, project_id: str, email_id: str = "user@system.local"
) -> AsyncGenerator[str, None]:
    """Stream agent execution with multiple event types"""

    try:
        # Initialize database and create/get project
        await db_service.create_user(user_id, email_id)
        await db_service.create_project(user_id, project_id, f"Project {project_id}")

        # Configuration for LangGraph
        config = {
            "configurable": {
                "thread_id": project_id,
                "user_id": user_id,
                "project_id": project_id,
                "email_id": email_id,
            }
        }
        context = RuntimeContext(
            user_id=user_id,
            project_id=project_id,
            email_id=email_id,
        )

        # Send start event
        yield format_sse_event(
            "agent_start",
            {
                "timestamp": datetime.utcnow().isoformat(),
                "project_id": project_id,
            },
        )

        # Get singleton agent graph
        agent = await get_agent_graph()

        # Stream using multiple modes: updates (state) + messages (LLM tokens)
        # Based on official docs: https://docs.langchain.com/oss/python/langgraph/streaming
        async for stream_mode, chunk in agent.astream(
            {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "project_id": project_id,
                "email_id": email_id,
            },
            config=config,
            stream_mode=["updates", "messages"],  # Multiple modes as per docs
            context=context,
        ):

            # Handle state updates (node transitions)
            if stream_mode == "updates":
                # chunk is a dict: {node_name: {state_updates}}
                for node_name, node_data in chunk.items():
                    # Skip if node_data is None (can happen with middleware)
                    if node_data is None:
                        continue
                        
                    # Extract messages from node data
                    messages = node_data.get("messages", [])
                    if messages:
                        last_message = messages[-1]

                        # Check for tool calls
                        if (
                            hasattr(last_message, "tool_calls")
                            and last_message.tool_calls
                        ):
                            for tool_call in last_message.tool_calls:
                                yield format_sse_event(
                                    "tool_start",
                                    {
                                        "tool_name": tool_call.get("name"),
                                        "tool_id": tool_call.get("id"),
                                        "tool_args": tool_call.get("args", {}),
                                        "node": node_name,
                                    },
                                )

                        # Check for tool results
                        if hasattr(last_message, "name") and last_message.name:
                            # This is a tool result message
                            output_preview = (
                                str(last_message.content)[:200] + "..."
                                if len(str(last_message.content)) > 200
                                else str(last_message.content)
                            )
                            yield format_sse_event(
                                "tool_complete",
                                {
                                    "tool_name": last_message.name,
                                    "output_preview": output_preview,
                                    "node": node_name,
                                },
                            )

            # Handle LLM token streaming
            elif stream_mode == "messages":
                # chunk is a tuple: (message_chunk, metadata)
                message_chunk, metadata = chunk

                # Skip if metadata is None
                if metadata is None:
                    continue

                # Get node name from metadata
                node_name = metadata.get("langgraph_node", "unknown")

                # Extract content from message_chunk
                if hasattr(message_chunk, "content"):
                    content = message_chunk.content

                    # Handle string content (simple text tokens)
                    if isinstance(content, str) and content:
                        yield format_sse_event(
                            "agent_thinking",
                            {
                                "token": content,
                                "node": node_name,
                            },
                        )

                    # Handle content blocks (structured content)
                    elif isinstance(content, list) and content:
                        for block in content:
                            if isinstance(block, dict):
                                block_type = block.get("type")

                                # Stream text blocks
                                if block_type == "text":
                                    text = block.get("text", "")
                                    if text:
                                        yield format_sse_event(
                                            "agent_thinking",
                                            {
                                                "token": text,
                                                "node": node_name,
                                            },
                                        )

                                # Tool call chunks
                                elif block_type == "tool_use":
                                    yield format_sse_event(
                                        "tool_calling",
                                        {
                                            "tool_name": block.get("name"),
                                            "tool_id": block.get("id"),
                                            "node": node_name,
                                        },
                                    )

        # Send completion event
        yield format_sse_event(
            "agent_complete",
            {
                "timestamp": datetime.utcnow().isoformat(),
                "project_id": project_id,
            },
        )

    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        yield format_sse_event(
            "error",
            {
                "message": str(e),
                "type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/api/projects/{project_id}/chat")
async def chat_endpoint(
    project_id: str,
    request: Annotated[ChatRequest, Body()],
):
    """Main streaming chat endpoint"""

    if request.project_id != project_id:
        raise HTTPException(status_code=400, detail="project_id mismatch")

    return StreamingResponse(
        stream_agent_response(
            message=request.message,
            user_id=request.user_id,
            project_id=request.project_id,
            email_id=request.email_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/projects/{project_id}/history")
async def get_history(project_id: str) -> ProjectHistoryResponse:
    """
    Get conversation history with proper message serialization.

    FIXED: Properly handles BaseMessage objects from LangGraph.
    """
    try:
        config = {"configurable": {"thread_id": project_id}}
        agent = await get_agent_graph()

        # Get current state from LangGraph
        state = await agent.aget_state(config)

        if not state or not state.values:
            logger.info(f"No history found for project {project_id}")
            return ProjectHistoryResponse(
                project_id=project_id,
                messages=[],
                message_count=0,
            )

        # Extract and serialize messages
        messages = state.values.get("messages", [])
        serialized_messages = []

        for msg in messages:
            try:
                # Base message data
                message_data = {
                    "role": msg.type,
                    "timestamp": str(getattr(msg, "created_at", None)),
                }

                # Handle content (could be string or list)
                if isinstance(msg.content, str):
                    message_data["content"] = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    message_data["content"] = (
                        " ".join(text_parts) if text_parts else str(msg.content)
                    )
                else:
                    message_data["content"] = str(msg.content)

                # Add tool-specific data
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    message_data["tool_calls"] = [
                        {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                        }
                        for tc in msg.tool_calls
                    ]

                if isinstance(msg, ToolMessage):
                    message_data["tool_name"] = msg.name

                serialized_messages.append(MessageResponse(**message_data))

            except Exception as msg_error:
                logger.warning(f"Failed to serialize message: {msg_error}")
                # Add minimal message data
                serialized_messages.append(
                    MessageResponse(
                        role="unknown",
                        content=str(msg),
                    )
                )

        return ProjectHistoryResponse(
            project_id=project_id,
            messages=serialized_messages,
            message_count=len(serialized_messages),
            current_phase=state.values.get("current_phase"),
            next_steps=state.values.get("next_steps", []),
        )

    except Exception as e:
        logger.error(f"Error getting history for {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve conversation history: {str(e)}"
        )

@app.get("/api/projects/{project_id}/files")
async def get_project_files(project_id: str):
    """Get all files for a project"""
    try:
        files = await db_service.get_project_files(project_id)
        return files or {"files": {}, "file_count": 0}
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/state")
async def get_agent_state_summary(project_id: str):
    """
    Get current agent state summary for a project.

    Returns:
        - Current phase
        - Iteration count
        - Error count
        - Next steps count
        - Recent thoughts count
        - Active files count
        - Running services count
        - Token usage and cost
    """
    try:
        config = {"configurable": {"thread_id": project_id}}
        agent = await get_agent_graph()

        # Get current state from LangGraph
        state = await agent.aget_state(config)

        if not state or not state.values:
            return {
                "project_id": project_id,
                "status": "no_state",
                "message": "No agent state found for this project",
                "summary": {
                    "phase": "unknown",
                    "iteration": 0,
                    "errors": 0,
                    "next_steps": 0,
                    "recent_thoughts": 0,
                    "active_files": 0,
                    "running_services": 0,
                    "total_tokens": 0,
                    "estimated_cost": "$0.0000",
                },
            }

        # Get formatted summary
        summary = get_state_summary(state.values)

        # Add detailed state information
        detailed_state = {
            "project_id": project_id,
            "status": "active",
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary,
            "details": {
                "current_phase": state.values.get("current_phase"),
                "next_phase": state.values.get("next_phase"),
                "next_steps": state.values.get("next_steps", []),
                "recent_thinking": state.values.get("recent_thinking", []),
                "last_error": state.values.get("last_error"),
                "active_files": state.values.get("active_files", []),
                "service_pids": state.values.get("service_pids", {}),
                "working_directory": state.values.get(
                    "working_directory", "/workspace"
                ),
                "tokens_used": state.values.get("tokens_used", {}),
                "iteration_count": state.values.get("iteration_count", 0),
                "last_summarized_at": state.values.get("last_summarized_at"),
            },
            "checkpoint_info": {
                "checkpoint_id": state.config.get("configurable", {}).get(
                    "checkpoint_id"
                ),
                "thread_id": state.config.get("configurable", {}).get("thread_id"),
            },
        }

        return detailed_state

    except Exception as e:
        logger.error(f"Error getting agent state for {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve agent state: {str(e)}"
        )


@app.get("/api/projects/{project_id}/state/summary")
async def get_agent_state_quick_summary(project_id: str):
    """
    Get a quick summary of agent state (lightweight version).

    Returns just the essential metrics without detailed data.
    """
    try:
        config = {"configurable": {"thread_id": project_id}}
        agent = await get_agent_graph()

        # Get current state from LangGraph
        state = await agent.aget_state(config)

        if not state or not state.values:
            return {
                "project_id": project_id,
                "exists": False,
                "phase": "unknown",
                "iteration": 0,
                "errors": 0,
            }

        # Get formatted summary
        summary = get_state_summary(state.values)

        return {
            "project_id": project_id,
            "exists": True,
            "timestamp": datetime.utcnow().isoformat(),
            **summary,  # Spread the summary fields
        }

    except Exception as e:
        logger.error(
            f"Error getting quick state summary for {project_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve state summary: {str(e)}"
        )


@app.post("/api/projects/{project_id}/restore")
async def restore_project_endpoint(
    project_id: str,
    user_id: str = Body(...),
):
    """
    Restore project session with auto-activation.

    Returns:
        - Files restored
        - Dependencies installed
        - Service URLs (frontend/backend)
        - Incomplete code warning if applicable
    """

    try:
        result = await restore_and_activate_project(user_id, project_id)

        return {
            "success": result["status"] in ["completed", "partial"],
            "status": result["status"],
            "message": result["message"],
            "data": {
                "files_restored": result["files_restored"],
                "sandbox_id": result.get("sandbox_id"),
                "dependencies": result.get("dependencies", {}),
                "services": result.get("services", {}),
                "incomplete_code": result.get("incomplete_code", False),
            },
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Restoration endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/users/{user_id}/projects")
    async def get_user_projects_endpoint(
        user_id: str,
        status: Optional[str] = None,  # Filter by status: 'active', 'ended'
        limit: int = 50,
        offset: int = 0,
    ):
        """
        Get all projects for a specific user.
        Returns project list with metadata for UI display.

        Query params:
        - status: Filter by project status ('active', 'ended')
        - limit: Max projects to return (default 50)
        - offset: Pagination offset (default 0)
        """
        try:
            # Get projects from database service
            projects = await db_service.get_user_projects(user_id)

            # Filter by status if specified
            if status:
                projects = [p for p in projects if p.get("status") == status]

            # Sort by last_active (most recent first)
            projects.sort(
                key=lambda x: x.get("last_active") or x.get("created_at"), reverse=True
            )

            # Paginate
            total = len(projects)
            paginated_projects = projects[offset : offset + limit]

            # Format response
            formatted_projects = []
            for p in paginated_projects:
                formatted_projects.append(
                    {
                        "project_id": p["project_id"],
                        "name": p["name"],
                        "status": p["status"],
                        "created_at": (
                            p["created_at"].isoformat() if p.get("created_at") else None
                        ),
                        "last_active": (
                            p["last_active"].isoformat()
                            if p.get("last_active")
                            else None
                        ),
                    }
                )

            return {
                "user_id": user_id,
                "projects": formatted_projects,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total,
            }

        except Exception as e:
            logger.error(f"Error getting projects for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Add this to api/streaming_agent.py

@app.get("/api/users/{user_id}/projects")
async def get_user_projects_endpoint(
    user_id: str,
    status: Optional[str] = None,  # Filter by status: 'active', 'ended'
    limit: int = 50,
    offset: int = 0
):
    """
    Get all projects for a specific user.
    Returns project list with metadata for UI display.
    
    Query params:
    - status: Filter by project status ('active', 'ended')
    - limit: Max projects to return (default 50)
    - offset: Pagination offset (default 0)
    """
    try:
        # Get projects from database service
        projects = await db_service.get_user_projects(user_id)
        
        # Filter by status if specified
        if status:
            projects = [p for p in projects if p.get('status') == status]
        
        # Sort by last_active (most recent first)
        projects.sort(
            key=lambda x: x.get('last_active') or x.get('created_at'),
            reverse=True
        )
        
        # Paginate
        total = len(projects)
        paginated_projects = projects[offset:offset + limit]
        
        # Format response
        formatted_projects = []
        for p in paginated_projects:
            formatted_projects.append({
                "project_id": p["project_id"],
                "name": p["name"],
                "status": p["status"],
                "created_at": p["created_at"].isoformat() if p.get("created_at") else None,
                "last_active": p["last_active"].isoformat() if p.get("last_active") else None,
            })
        
        return {
            "user_id": user_id,
            "projects": formatted_projects,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error getting projects for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Production health check"""

    checkpointer_health = await checkpointer_service.health_check()

    return {
        "status": (
            "healthy" if checkpointer_health["status"] == "healthy" else "degraded"
        ),
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "checkpointer": checkpointer_health,
            "database": {"status": "healthy", "initialized": True},
        },
    }

    # api/streaming_agent.py - ADD RESTORATION ENDPOINT

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "streaming_agent:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
