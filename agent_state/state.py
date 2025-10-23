"""
Agent state schema for full-stack development agent.

"""

from typing import TypedDict, Optional, Annotated
from typing_extensions import NotRequired
from langchain.agents import AgentState


class FullStackAgentState(AgentState):
    user_id: NotRequired[str]
    project_id: NotRequired[str]
    email_id: NotRequired[str]
    current_phase: NotRequired[str]
    next_phase: Optional[
        str
    ]  # Values: 'planning' | 'backend_dev' | 'frontend_dev' | 'testing' | 'integration'
    next_steps: NotRequired[list[str]]
    recent_thinking: NotRequired[list[dict]]
    error_count: NotRequired[int]
    last_error: NotRequired[Optional[dict]]
    working_directory: NotRequired[str]
    active_files: NotRequired[list[str]]
    service_pids: NotRequired[dict[str, int]]
    tokens_used: NotRequired[dict]


def get_state_summary(state: FullStackAgentState) -> dict:
    return {
        "user_id": state.get("user_id", "unknown"),
        "project_id": state.get("project_id", "unknown"),
        "email_id": state.get("email_id", "unknown"),
        "phase": state.get("current_phase", "unknown"),
        "next_phase": state.get("next_phase", "unknown"),
        "errors": state.get("error_count", 0),
        "next_steps": len(state.get("next_steps", [])),
        "recent_thoughts": len(state.get("recent_thinking", [])),
        "active_files": len(state.get("active_files", [])),
        "running_services": len(state.get("service_pids", {})),
        "total_tokens": (
            state.get("tokens_used", {}).get("total_input", 0)
            + state.get("tokens_used", {}).get("total_output", 0)
        ),
    }
