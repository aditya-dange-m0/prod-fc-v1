"""
Think Tool - Agent's External Memory System
==========================================

The agent's internal reasoning and context management tool.

Features:
- Persistent thought storage in PostgresStore
- Semantic search for thought retrieval
- Automatic thought summarization (when token limit reached)
- Milestone tracking
- Phase-based organization
"""

import logging
from datetime import datetime, UTC, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_anthropic import ChatAnthropic

from context.runtime_context import get_runtime_context
from db.service import db_service

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Token limits for thought summarization
THOUGHT_SUMMARIZATION_THRESHOLD = 50000  # Total tokens in all thoughts
THOUGHTS_PER_SUMMARY_CHECK = 50  # Check summarization every N thoughts

# =============================================================================
# THOUGHT CATEGORIZATION
# =============================================================================

class ThoughtType(Enum):
    """Categories of thoughts for organization"""
    PLANNING = "planning"           # Architecture, requirements breakdown
    DECISION = "decision"            # Key technical decisions
    PROGRESS = "progress"            # What was accomplished
    DEBUG = "debug"                  # Problem analysis and solutions
    NEXT_STEPS = "next_steps"       # What to do next
    REMEMBER = "remember"            # Important context to preserve
    ANALYSIS = "analysis"            # Test results, evaluations
    REFLECTION = "reflection"        # Learning from what happened

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _get_store_from_config(config: RunnableConfig):
    """Get PostgresStore from config"""
    # Store is passed in agent creation
    # Access it from config if available
    return getattr(config, 'store', None)


async def _count_thought_tokens(store, namespace) -> int:
    """Count total tokens in all thoughts for a project"""
    try:
        # Get all thoughts
        all_thoughts = await store.search(
            namespace,
            query="",  # Empty query gets all
            limit=1000  # Reasonable limit
        )
        
        # Count tokens
        total_tokens = 0
        for thought in all_thoughts:
            thought_text = thought.value.get('thought', '')
            tokens = count_tokens_approximately(thought_text)
            total_tokens += tokens
        
        return total_tokens
    except Exception as e:
        logger.error(f"Failed to count thought tokens: {e}")
        return 0


async def _summarize_old_thoughts(store, namespace, model):
    """
    OPTIONAL: Summarize old thoughts when token limit is reached.
    
    This is commented out in the main flow but can be enabled if needed.
    
    Strategy:
    1. Get all thoughts older than 7 days
    2. Group by phase
    3. Summarize each phase
    4. Replace old thoughts with phase summaries
    """
    logger.info("🔄 Starting thought summarization...")
    
    try:
        # Get thoughts older than 7 days
        cutoff_date = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        old_thoughts = await store.search(
            namespace,
            filter={"timestamp__lt": cutoff_date},
            limit=500
        )
        
        if len(old_thoughts) < 20:
            logger.info("Not enough old thoughts to summarize")
            return False
        
        # Group by phase
        phases = {}
        for thought in old_thoughts:
            phase = thought.value.get('phase', 'general')
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(thought)
        
        # Summarize each phase
        summaries_created = 0
        for phase, phase_thoughts in phases.items():
            # Build summarization prompt
            thoughts_text = "\n".join([
                f"- {t.value['thought']}" for t in phase_thoughts
            ])
            
            summary_prompt = f"""Summarize these thoughts from the {phase} phase:

{thoughts_text}

Create a concise summary that preserves:
- Key decisions made
- Files created/modified
- Problems solved
- Important technical details

Format as bullet points."""
            
            # Generate summary
            summary_response = await model.ainvoke([
                HumanMessage(content=summary_prompt)
            ])
            summary_text = summary_response.content
            
            # Save summary as a special thought
            await store.put(
                namespace,
                f"summary_{phase}_{datetime.now(UTC).isoformat()}",
                {
                    "thought": f"PHASE SUMMARY ({phase}):\n{summary_text}",
                    "phase": phase,
                    "type": "summary",
                    "original_thought_count": len(phase_thoughts),
                    "timestamp": datetime.now(UTC).isoformat()
                }
            )
            
            # Delete original thoughts
            for thought in phase_thoughts:
                await store.delete(namespace, thought.key)
            
            summaries_created += 1
            logger.info(f"✅ Summarized {len(phase_thoughts)} thoughts for phase: {phase}")
        
        logger.info(f"✅ Created {summaries_created} phase summaries")
        return True
        
    except Exception as e:
        logger.error(f"Failed to summarize thoughts: {e}")
        return False

# =============================================================================
# THINK TOOL
# =============================================================================

@tool
async def think(
    thought: str,
    config: RunnableConfig,
    thought_type: Optional[str] = "planning",
    phase: Optional[str] = None,
    milestone: Optional[str] = None,
    priority: Optional[str] = "normal"
) -> str:
    """
    Record your thoughts, plans, and observations to external memory.
    
    This is your EXTERNAL NOTEPAD - thoughts persist even when messages
    are trimmed or summarized. Use frequently to maintain context!
    
    **When to use:**
    - 🎯 Before starting work: Break down tasks and plan approach
    - 📝 After tool calls: Summarize what was accomplished
    - 🔍 When debugging: Analyze errors step-by-step
    - ✅ After tests: Evaluate results and plan fixes
    - 💡 For decisions: Document why you chose an approach
    - 🎉 At milestones: Mark important achievements
    
    Args:
        thought: Your reasoning or observation (be detailed!)
        config: LangGraph config (auto-provided)
        thought_type: Type - planning, decision, progress, debug, 
                     next_steps, remember, analysis, reflection
        phase: Current phase - planning, backend_dev, testing_backend,
               frontend_dev, testing_frontend, integration
        milestone: Mark achievements (e.g., "Backend Complete")
        priority: high/normal/low - for filtering critical thoughts
        
    Examples:
        think("Planning todo app: Need JWT auth, separate auth.py and todos.py. MongoDB with user isolation.")
        
        think("Backend auth complete! JWT working, all endpoints tested.", 
              milestone="Backend Auth Complete", phase="backend_dev")
        
        think("Fixed import error by switching to absolute imports instead of relative", 
              thought_type="debug", phase="backend_dev")
    """
    ctx = get_runtime_context(config)
    timestamp = datetime.now(UTC).isoformat()
    
    # Validate thought type
    try:
        tt = ThoughtType[thought_type.upper()]
    except (KeyError, AttributeError):
        tt = ThoughtType.PLANNING
    
    # Get store
    store = await _get_store_from_config(config)
    if not store:
        logger.warning("Store not available in config - saving to database only")
        # Fallback to database
        await db_service.save_thought(
            ctx.project_id,
            {
                "thought": thought,
                "thought_type": tt.value,
                "phase": phase,
                "milestone": milestone,
                "priority": priority,
                "timestamp": timestamp
            }
        )
        return "💭 Thought recorded (database only)"
    
    # Save to PostgresStore (with semantic search)
    namespace = ("projects", ctx.project_id)
    
    await store.put(
        namespace,
        f"thought_{timestamp}",
        {
            "thought": thought,
            "thought_type": tt.value,
            "phase": phase,
            "milestone": milestone,
            "priority": priority,
            "timestamp": timestamp
        }
    )
    
    # Also save to database for backup
    await db_service.save_thought(
        ctx.project_id,
        {
            "thought": thought,
            "thought_type": tt.value,
            "phase": phase,
            "milestone": milestone,
            "priority": priority,
            "timestamp": timestamp
        }
    )
    
    # =========================================================================
    # OPTIONAL: Automatic thought summarization (commented out by default)
    # =========================================================================
    # Uncomment this section to enable automatic thought summarization
    # when total thought tokens exceed threshold
    #
    # # Check if summarization needed (every N thoughts)
    # thought_count = await store.count(namespace)
    # 
    # if thought_count % THOUGHTS_PER_SUMMARY_CHECK == 0:
    #     # Count total tokens in thoughts
    #     total_tokens = await _count_thought_tokens(store, namespace)
    #     
    #     if total_tokens > THOUGHT_SUMMARIZATION_THRESHOLD:
    #         logger.info(f"🔄 Thought tokens ({total_tokens}) exceed threshold ({THOUGHT_SUMMARIZATION_THRESHOLD})")
    #         
    #         # Trigger summarization
    #         model = ChatAnthropic(model="claude-sonnet-4")
    #         summarized = await _summarize_old_thoughts(store, namespace, model)
    #         
    #         if summarized:
    #             logger.info("✅ Old thoughts summarized successfully")
    #         else:
    #             logger.warning("⚠️ Thought summarization failed")
    # =========================================================================
    
    # Format output (minimal to avoid context pollution)
    emoji_map = {
        "planning": "🎯",
        "decision": "💡",
        "progress": "✅",
        "debug": "🔍",
        "next_steps": "🔄",
        "remember": "📝",
        "analysis": "📊",
        "reflection": "💭"
    }
    emoji = emoji_map.get(tt.value, "💭")
    
    if milestone:
        return f"🎯 Milestone recorded: {milestone}"
    elif priority == "high":
        return f"{emoji} High-priority thought recorded"
    else:
        return f"{emoji} Thought recorded"

# =============================================================================
# RECALL TOOL
# =============================================================================

@tool
async def recall(
    query: str,
    config: RunnableConfig,
    limit: int = 5,
    phase_filter: Optional[str] = None
) -> str:
    """
    Recall your previous thoughts using semantic search.
    
    Use when you need to remember:
    - How you solved a similar problem before
    - What decisions you made earlier
    - Current project status and progress
    - Files you created
    
    Args:
        query: What you want to remember (natural language)
        config: LangGraph config (auto-provided)
        limit: Max thoughts to retrieve (default: 5)
        phase_filter: Filter by phase (optional)
        
    Examples:
        recall("authentication implementation")
        recall("how did I fix the import errors")
        recall("backend files I created")
        recall("testing strategy", phase_filter="testing_backend")
    """
    ctx = get_runtime_context(config)
    
    # Get store
    store = await _get_store_from_config(config)
    if not store:
        logger.warning("Store not available - falling back to database")
        # Fallback to database
        thoughts = await db_service.get_thoughts(
            ctx.project_id,
            filters={"phase": phase_filter} if phase_filter else {},
            limit=limit
        )
        
        if not thoughts:
            return f"No thoughts found for: {query}"
        
        output = f"📚 Found {len(thoughts)} thoughts (from database):\n\n"
        for i, t in enumerate(thoughts, 1):
            output += f"{i}. {t['thought']}\n\n"
        return output
    
    # Semantic search in store
    namespace = ("projects", ctx.project_id)
    
    # Build filter
    filter_dict = {}
    if phase_filter:
        filter_dict["phase"] = phase_filter
    
    # Search
    thoughts = await store.search(
        namespace,
        query=query,
        filter=filter_dict if filter_dict else None,
        limit=limit
    )
    
    if not thoughts:
        return f"No thoughts found for: {query}"
    
    # Format output
    output = f"📚 Found {len(thoughts)} relevant thoughts:\n\n"
    
    for i, t in enumerate(thoughts, 1):
        thought_data = t.value
        milestone = thought_data.get('milestone')
        phase = thought_data.get('phase')
        thought_type = thought_data.get('thought_type', 'general')
        
        # Header
        if milestone:
            prefix = f"🎯 {milestone}"
        elif phase:
            prefix = f"💭 {phase} ({thought_type})"
        else:
            prefix = f"💭 {thought_type}"
        
        output += f"{i}. {prefix}\n"
        output += f"   {thought_data['thought']}\n\n"
    
    return output

# =============================================================================
# EXPORT
# =============================================================================

THINK_TOOLS = [think, recall]

if __name__ == "__main__":
    print("Think Tool - Agent's External Memory System")
    print("=" * 60)
    print("Tools:", [t.name for t in THINK_TOOLS])
    print("\n✅ Persistent thought storage with semantic search")
    print("✅ Automatic summarization support (optional)")
    print("✅ Phase-based organization")
    print("✅ Milestone tracking")
