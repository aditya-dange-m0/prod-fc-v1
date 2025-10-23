# middleware/dynamic_prompts.py - WITH NEXT_PHASE AWARENESS

"""
Dynamic Prompt Middleware - Phase-aware system prompts.

Runs: Automatically via @dynamic_prompt decorator
Purpose: Inject phase-specific instructions into system prompt

NEW: Uses next_phase to provide transition guidance
"""

import logging
from langchain.agents.middleware import dynamic_prompt, ModelRequest

logger = logging.getLogger(__name__)


# Phase-specific instruction templates
PHASE_INSTRUCTIONS = {
    "planning": """
**PLANNING PHASE**

Focus:
- Ask clarifying questions about requirements
- Propose tech stack and architecture
- Break down project into clear phases
- Identify potential challenges

Actions:
- Use ask_user tool for clarification
- Use web_search to find best practices
- Think through architectural decisions
""",
    "backend_dev": """
**BACKEND DEVELOPMENT PHASE**

Focus:
- Build APIs with proper error handling
- Design database models with relationships
- Implement authentication/authorization
- Write clean, production-ready code

Best Practices:
- Use absolute imports
- Add type hints
- Handle edge cases
- Log important events
- Test endpoints as you build
""",
    "frontend_dev": """
**FRONTEND DEVELOPMENT PHASE**

Focus:
- Build responsive UI components
- Implement state management
- Integrate with backend APIs
- Handle loading/error states

Best Practices:
- Component composition
- Proper error boundaries
- Accessibility (a11y)
- Mobile-friendly design
""",
    "testing": """
**TESTING PHASE**

Focus:
- Write comprehensive tests
- Test happy paths and edge cases
- Debug failing tests systematically
- Fix bugs without breaking other features

Approach:
- Unit tests for business logic
- Integration tests for APIs
- E2E tests for critical flows
- Use debugging tools effectively
""",
    "integration": """
**INTEGRATION/DEPLOYMENT PHASE**

Focus:
- Connect frontend and backend
- Configure environment variables
- Setup Docker/deployment
- Ensure production readiness

Checklist:
- CORS configuration
- Environment variables
- Database migrations
- Health check endpoints
- Error monitoring
""",
}


# NEW: Phase transition guidance
PHASE_TRANSITIONS = {
    "planning->backend_dev": """
🔄 **TRANSITIONING TO BACKEND DEVELOPMENT**

Before starting:
- Review the architecture plan
- Set up project structure
- Initialize dependencies
- Create base configuration files
""",
    "backend_dev->frontend_dev": """
🔄 **TRANSITIONING TO FRONTEND DEVELOPMENT**

Before starting:
- Ensure backend APIs are working
- Document API endpoints
- Test authentication flow
- Prepare API integration guide
""",
    "frontend_dev->testing": """
🔄 **TRANSITIONING TO TESTING**

Before starting:
- Complete core UI components
- Ensure frontend-backend integration works
- Review user flows
- Prepare test scenarios
""",
    "testing->integration": """
🔄 **TRANSITIONING TO DEPLOYMENT**

Before starting:
- All critical tests passing
- Fix remaining bugs
- Review security concerns
- Prepare deployment checklist
""",
}


@dynamic_prompt
def phase_aware_prompt(request: ModelRequest) -> str:
    """
    Generate phase-aware system prompt.

    NEW: Includes transition guidance when next_phase is different from current
    """

    # Get current state
    state = request.state
    current_phase = state.get("current_phase", "planning")
    next_phase = state.get("next_phase")  # NEW!

    # Base prompt
    base_prompt = """You are an expert Full-Stack Development Agent.

IMPORTANT: Always include metadata block with <agent_metadata> tags.
Include <next_phase> to indicate what phase comes after current work.

"""

    # Add current phase instructions
    phase_instructions = PHASE_INSTRUCTIONS.get(current_phase, "")
    full_prompt = base_prompt + phase_instructions

    # NEW: Add transition guidance if next_phase is different
    if next_phase and next_phase != current_phase:
        transition_key = f"{current_phase}->{next_phase}"
        transition_guidance = PHASE_TRANSITIONS.get(transition_key, "")

        if transition_guidance:
            full_prompt += f"\n\n{transition_guidance}"
            logger.debug(
                f"🔄 Added transition guidance: {current_phase} → {next_phase}"
            )

    # Add recent thinking for continuity
    recent_thinking = state.get("recent_thinking", [])
    if recent_thinking:
        last_thought = recent_thinking[-1]
        context_reminder = f"""

**CONTINUITY**: Last thought: "{last_thought['thinking']}"
Remember what you were working on and continue from there.
"""
        full_prompt += context_reminder

    # Add error context if present
    error_count = state.get("error_count", 0)
    if error_count > 0:
        last_error = state.get("last_error")
        if last_error:
            error_reminder = f"""

⚠️ **RECENT ERROR** (Count: {error_count}): {last_error['description']}
Be careful to avoid similar errors.
"""
            full_prompt += error_reminder

    # Add next steps reminder
    next_steps = state.get("next_steps", [])
    if next_steps:
        steps_reminder = f"""

📋 **YOUR PLANNED NEXT STEPS**:
"""
        for i, step in enumerate(next_steps[:3], 1):  # Show top 3
            steps_reminder += f"\n{i}. {step}"

        full_prompt += steps_reminder

    logger.debug(f"📝 Generated prompt for phase: {current_phase}")
    if next_phase:
        logger.debug(f"🔮 Preparing for next phase: {next_phase}")

    return full_prompt
