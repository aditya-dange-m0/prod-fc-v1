"""
Token Tracker Middleware - Track token usage using built-in usage_metadata.

Runs: after_model hook
Purpose: Track token consumption, estimate costs, log usage

Uses AIMessage.usage_metadata for accurate token counts (no approximation!).
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, UTC

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from agent_state import FullStackAgentState

logger = logging.getLogger(__name__)

# Pricing (as of 2025-10) - Update these as needed
MODEL_PRICING = {
    # Anthropic
    "claude-sonnet-4": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    "claude-sonnet-4-5": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    "claude-3-5-haiku": {
        "input": 0.80 / 1_000_000,
        "output": 4.00 / 1_000_000,
    },
    
    # OpenAI
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "o1": {
        "input": 15.00 / 1_000_000,
        "output": 60.00 / 1_000_000,
    },
    "o1-mini": {
        "input": 3.00 / 1_000_000,
        "output": 12.00 / 1_000_000,
    },
}

class TokenTrackerMiddleware(AgentMiddleware[FullStackAgentState]):
    """
    Track token usage using built-in AIMessage.usage_metadata.
    
    Features:
    - Use provider-reported token counts (accurate!)
    - Estimate costs per model
    - Detect anomalies (huge responses)
    - Aggregate across multiple models
    - Optional database logging
    
    Example usage:
        middleware = [
            TokenTrackerMiddleware(
                log_to_db=True,
                warn_threshold=50000
            )
        ]
    """
    
    state_schema = FullStackAgentState
    
    def __init__(
        self,
        log_to_db: bool = False,
        warn_threshold: int = 50000,
        enable_detailed_logging: bool = False
    ):
        """
        Initialize token tracker.
        
        Args:
            log_to_db: Whether to persist usage to database
            warn_threshold: Warn if response exceeds this many tokens
            enable_detailed_logging: Log detailed token breakdowns
        """
        super().__init__()
        self.log_to_db = log_to_db
        self.warn_threshold = warn_threshold
        self.enable_detailed_logging = enable_detailed_logging
    
    def after_model(
        self,
        state: FullStackAgentState,
        runtime: Runtime
    ) -> Optional[Dict[str, Any]]:
        """
        Track tokens after each model call.
        
        Extracts usage_metadata from AIMessage and updates state.
        """
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        
        # Extract usage metadata from AIMessage
        usage_metadata = self._extract_usage_metadata(last_message)
        
        if not usage_metadata:
            logger.debug("No usage_metadata found in response")
            return None
        
        # Extract token counts
        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
        
        # Get model name
        model_name = self._extract_model_name(last_message, usage_metadata)
        
        # Calculate cost
        cost = self._calculate_cost(model_name, input_tokens, output_tokens)
        
        # Get current token state
        tokens_used = state.get('tokens_used', self._create_empty_token_state())
        
        # Update totals
        tokens_used['total_input'] += input_tokens
        tokens_used['total_output'] += output_tokens
        tokens_used['total_cost'] += cost
        
        # Update per-model breakdown
        if model_name not in tokens_used['by_model']:
            tokens_used['by_model'][model_name] = {
                'input': 0,
                'output': 0,
                'cost': 0.0,
                'calls': 0
            }
        
        model_stats = tokens_used['by_model'][model_name]
        model_stats['input'] += input_tokens
        model_stats['output'] += output_tokens
        model_stats['cost'] += cost
        model_stats['calls'] += 1
        
        # Record last call details
        tokens_used['last_call'] = {
            'input': input_tokens,
            'output': output_tokens,
            'total': total_tokens,
            'model': model_name,
            'cost': cost,
            'timestamp': datetime.now(UTC).isoformat(),
            'raw_metadata': usage_metadata if self.enable_detailed_logging else None
        }
        
        # Log summary
        logger.info(
            f"💰 Tokens: {input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} total | "
            f"Cost: ${cost:.4f} | Model: {model_name}"
        )
        
        # Detailed logging (optional)
        if self.enable_detailed_logging:
            self._log_detailed_breakdown(usage_metadata, model_name)
        
        # Warn if response is huge
        if output_tokens > self.warn_threshold:
            logger.warning(
                f"⚠️ Large response: {output_tokens:,} tokens! "
                f"Consider enabling summarization or context editing."
            )
        
        # Log to database if enabled
        if self.log_to_db:
            self._log_to_database(runtime, tokens_used['last_call'])
        
        return {'tokens_used': tokens_used}
    
    def _extract_usage_metadata(self, message) -> Optional[Dict[str, Any]]:
        """
        Extract usage_metadata from AIMessage.
        
        Supports multiple formats:
        - message.usage_metadata (LangChain standard)
        - message.response_metadata['usage'] (provider-specific)
        """
        # Try standard usage_metadata field
        if hasattr(message, 'usage_metadata') and message.usage_metadata:
            return message.usage_metadata
        
        # Try response_metadata.usage (some providers)
        if hasattr(message, 'response_metadata'):
            metadata = message.response_metadata
            
            # OpenAI format
            if 'usage' in metadata:
                usage = metadata['usage']
                return {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
            
            # Anthropic format
            if 'usage' in metadata:
                usage = metadata['usage']
                return {
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'total_tokens': (
                        usage.get('input_tokens', 0) + 
                        usage.get('output_tokens', 0)
                    )
                }
        
        return None
    
    def _extract_model_name(self, message, usage_metadata: Dict[str, Any]) -> str:
        """Extract model name from message or metadata"""
        
        # Try usage_metadata first (some providers include it)
        if 'model' in usage_metadata:
            return usage_metadata['model']
        
        # Try response_metadata
        if hasattr(message, 'response_metadata'):
            metadata = message.response_metadata
            
            if 'model' in metadata:
                return metadata['model']
            if 'model_name' in metadata:
                return metadata['model_name']
        
        # Fallback
        return "unknown"
    
    def _calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost based on model pricing"""
        
        # Try exact match
        pricing = MODEL_PRICING.get(model_name)
        
        # Try fuzzy match
        if not pricing:
            for key in MODEL_PRICING.keys():
                if key in model_name.lower():
                    pricing = MODEL_PRICING[key]
                    break
        
        # Fallback to zero
        if not pricing:
            logger.debug(f"Unknown model pricing: {model_name}")
            return 0.0
        
        input_cost = input_tokens * pricing['input']
        output_cost = output_tokens * pricing['output']
        
        return input_cost + output_cost
    
    def _create_empty_token_state(self) -> Dict[str, Any]:
        """Create empty token state structure"""
        return {
            'total_input': 0,
            'total_output': 0,
            'total_cost': 0.0,
            'by_model': {},
            'last_call': {}
        }
    
    def _log_detailed_breakdown(
        self,
        usage_metadata: Dict[str, Any],
        model_name: str
    ) -> None:
        """Log detailed token breakdown (for debugging)"""
        
        logger.debug(f"📊 Detailed token breakdown for {model_name}:")
        
        for key, value in usage_metadata.items():
            if isinstance(value, (int, float)):
                logger.debug(f"  {key}: {value:,}")
            else:
                logger.debug(f"  {key}: {value}")
    
    def _log_to_database(self, runtime: Runtime, call_data: Dict[str, Any]) -> None:
        """
        Log token usage to database (async).
        
        TODO: Implement based on your database schema.
        """
        try:
            from context.runtime_context import RuntimeContext
            ctx: RuntimeContext = runtime.context
            
            # Example database logging:
            # await db_service.log_token_usage(
            #     project_id=ctx.project_id,
            #     user_id=ctx.user_id,
            #     model=call_data['model'],
            #     input_tokens=call_data['input'],
            #     output_tokens=call_data['output'],
            #     total_tokens=call_data['total'],
            #     cost=call_data['cost'],
            #     timestamp=call_data['timestamp']
            # )
            
            logger.debug(
                f"📊 Token usage logged: {call_data['model']} "
                f"({call_data['input']} + {call_data['output']} = {call_data['total']} tokens)"
            )
            
        except Exception as e:
            logger.error(f"Failed to log tokens to database: {e}")


# =============================================================================
# USAGE METADATA CALLBACK (Alternative Approach)
# =============================================================================

from langchain_core.callbacks import UsageMetadataCallbackHandler

class GlobalUsageTracker:
    """
    Global usage tracker using callback handler.
    
    Use this if you want to track usage across multiple agent runs
    without middleware.
    
    Example:
        tracker = GlobalUsageTracker()
        
        result = await agent.ainvoke(
            state,
            context=context,
            config={"callbacks": [tracker.callback]}
        )
        
        print(tracker.get_summary())
    """
    
    def __init__(self):
        self.callback = UsageMetadataCallbackHandler()
    
    def get_usage(self) -> Dict[str, Any]:
        """Get current usage metadata"""
        return self.callback.usage_metadata
    
    def get_summary(self) -> Dict[str, Any]:
        """Get formatted summary"""
        metadata = self.callback.usage_metadata
        
        return {
            'total_input': metadata.get('input_tokens', 0),
            'total_output': metadata.get('output_tokens', 0),
            'total_tokens': metadata.get('total_tokens', 0),
            'cost_estimate': self._estimate_cost(metadata)
        }
    
    def _estimate_cost(self, metadata: Dict[str, Any]) -> float:
        """Rough cost estimate"""
        # Would need model name to be accurate
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        
        # Assume average pricing ($3 input, $15 output per 1M tokens)
        return (input_tokens * 3 + output_tokens * 15) / 1_000_000
    
    def reset(self) -> None:
        """Reset tracker"""
        self.callback = UsageMetadataCallbackHandler()
