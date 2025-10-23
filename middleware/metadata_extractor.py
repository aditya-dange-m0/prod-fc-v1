# middleware/metadata_extractor.py - HANDLES OPTIONAL FIELDS

"""
Metadata Extractor Middleware - Minimal metadata extraction.
Only extracts fields when present - all fields are optional.
"""

import re
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
from datetime import datetime, UTC
import logging

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from agent_state import FullStackAgentState

logger = logging.getLogger(__name__)


class MetadataExtractorMiddleware(AgentMiddleware[FullStackAgentState]):
    """
    Extract metadata from agent responses.
    
    ALL FIELDS ARE OPTIONAL - only extract what's present.
    Minimal metadata for efficiency.
    """
    
    state_schema = FullStackAgentState
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, warn about missing metadata blocks.
                        If False, silently ignore missing metadata (default).
        """
        super().__init__()
        self.xml_pattern = re.compile(
            r"<agent_metadata>(.*?)</agent_metadata>", re.DOTALL | re.IGNORECASE
        )
        self.strict_mode = strict_mode
        self.warning_count = 0
        self.max_warnings = 3
    
    def after_model(
        self, state: FullStackAgentState, runtime: Runtime
    ) -> Optional[Dict[str, Any]]:
        """Extract metadata after LLM generates response."""
        
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        
        # Only process AI messages
        if not hasattr(last_message, "content"):
            return None
        
        content = last_message.content
        
        # Handle content that might be a list
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if isinstance(item, str):
                    text_content += item
                elif isinstance(item, dict) and "text" in item:
                    text_content += item["text"]
            content = text_content
        
        # Skip if content is not a string
        if not isinstance(content, str):
            return None
        
        # Extract XML block
        match = self.xml_pattern.search(content)
        if not match:
            # No metadata found - this is OK in minimal mode
            if self.strict_mode and self.warning_count < self.max_warnings:
                logger.debug("No metadata block in response (this is fine)")
                self.warning_count += 1
            return None
        
        xml_content = match.group(1)
        
        # Parse and build state update
        try:
            metadata = self._parse_metadata(xml_content)
            state_update = self._build_state_update(metadata, state)
            
            if state_update:
                logger.debug(f"📊 Extracted: {list(state_update.keys())}")
            
            return state_update
            
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {e}")
            return None
    
    def _parse_metadata(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse XML structure into metadata dict.
        
        ALL FIELDS OPTIONAL - only extract what's present.
        """
        
        # Wrap in root for parsing
        xml_str = f"<root>{xml_content}</root>"
        
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return {}
        
        metadata = {}
        
        # 1. Phase (optional)
        phase_elem = root.find("phase")
        if phase_elem is not None and phase_elem.text:
            text = phase_elem.text.strip()
            if text:  # Only if non-empty
                metadata["phase"] = text
        
        # 2. Next Phase (optional)
        next_phase_elem = root.find("next_phase")
        if next_phase_elem is not None and next_phase_elem.text:
            text = next_phase_elem.text.strip()
            if text:
                metadata["next_phase"] = text
        
        # 3. Thinking (optional)
        thinking_elem = root.find("thinking")
        if thinking_elem is not None and thinking_elem.text:
            text = thinking_elem.text.strip()
            if text:
                metadata["thinking"] = text
        
        # 4. Error (optional)
        error_elem = root.find("error")
        if error_elem is not None and error_elem.text:
            text = error_elem.text.strip()
            if text:
                metadata["error"] = {
                    "description": text,
                    "severity": error_elem.get("severity", "medium"),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
        
        # 5. Next steps (optional)
        steps_elem = root.find("next_steps")
        if steps_elem is not None:
            steps = []
            for step_elem in steps_elem.findall("step"):
                if step_elem.text:
                    text = step_elem.text.strip()
                    if text:  # Only non-empty steps
                        steps.append(text)
            
            if steps:  # Only if we found any steps
                metadata["next_steps"] = steps
        
        return metadata
    
    def _build_state_update(
        self, metadata: Dict[str, Any], current_state: FullStackAgentState
    ) -> Optional[Dict[str, Any]]:
        """Build state update dict from parsed metadata."""
        
        if not metadata:
            return None
        
        update = {}
        
        # 1. Phase update (only if present)
        if "phase" in metadata:
            current_phase = current_state.get("current_phase")
            new_phase = metadata["phase"]
            
            if new_phase != current_phase:
                logger.info(f"📍 Phase: {current_phase} → {new_phase}")
                update["current_phase"] = new_phase
        
        # 2. Next Phase (only if present)
        if "next_phase" in metadata:
            next_phase = metadata["next_phase"]
            current_next = current_state.get("next_phase")
            
            if next_phase != current_next:
                logger.info(f"🔮 Next phase: {next_phase}")
                update["next_phase"] = next_phase
        
        # 3. Thinking (only if present, keep last 3)
        if "thinking" in metadata:
            recent = current_state.get("recent_thinking", [])
            recent.append({
                "thinking": metadata["thinking"],
                "phase": metadata.get("phase", current_state.get("current_phase")),
                "timestamp": datetime.now(UTC).isoformat(),
            })
            update["recent_thinking"] = recent[-3:]  # Keep last 3 only
            logger.debug(f"💭 Thinking recorded")
        
        # 4. Error handling (only if present)
        if "error" in metadata:
            error_count = current_state.get("error_count", 0) + 1
            update["error_count"] = error_count
            update["last_error"] = metadata["error"]
            
            logger.warning(
                f"⚠️ Error #{error_count} [{metadata['error']['severity']}]: "
                f"{metadata['error']['description'][:60]}..."
            )
        
        # 5. Next steps (only if present)
        if "next_steps" in metadata:
            update["next_steps"] = metadata["next_steps"]
            logger.info(f"📋 Planned: {len(metadata['next_steps'])} steps")
        
        return update if update else None







# """
# Metadata Extractor Middleware - Extract XML metadata from agent responses.

# Runs: after_model hook
# Updates: current_phase, recent_thinking, error_count, last_error, next_steps
# """

# import re
# import xml.etree.ElementTree as ET
# from typing import Optional, Dict, Any
# from datetime import datetime, UTC
# import logging

# from langchain.agents.middleware import AgentMiddleware
# from langgraph.runtime import Runtime
# from agent.state import FullStackAgentState

# logger = logging.getLogger(__name__)


# class MetadataExtractorMiddleware(AgentMiddleware[FullStackAgentState]):
#     """
#     Extract structured metadata from agent responses via XML tags.

#     Expected XML format in agent responses:
#         <agent_metadata>
#           <phase>backend_dev</phase>
#           <thinking>Brief reasoning...</thinking>
#           <error severity="high">Error description</error>
#           <next_steps>
#             <step>Action 1</step>
#             <step>Action 2</step>
#           </next_steps>
#         </agent_metadata>
#     """

#     state_schema = FullStackAgentState

#     def __init__(self):
#         super().__init__()
#         self.xml_pattern = re.compile(
#             r"<agent_metadata>(.*?)</agent_metadata>", re.DOTALL | re.IGNORECASE
#         )
#         self.warning_count = 0
#         self.max_warnings = 3  # Stop warning after 3 times

#     def after_model(
#         self, state: FullStackAgentState, runtime: Runtime
#     ) -> Optional[Dict[str, Any]]:
#         """
#         Extract metadata after LLM generates response.

#         This hook runs after every model call, extracts XML metadata,
#         and returns state updates to be merged.
#         """
#         messages = state.get("messages", [])
#         if not messages:
#             return None

#         last_message = messages[-1]

#         # Only process AI messages
#         if not hasattr(last_message, "content"):
#             return None

#         content = last_message.content
#         #**************************************************************************************************
#         # Handle content that might be a list (e.g., with tool calls)
#         if isinstance(content, list):
#             # Extract text content from list
#             text_content = ""
#             for item in content:
#                 if isinstance(item, str):
#                     text_content += item
#                 elif isinstance(item, dict) and "text" in item:
#                     text_content += item["text"]
#             content = text_content

#         # Skip if content is not a string at this point
#         if not isinstance(content, str):
#             return None

#         # Extract XML block
#         match = self.xml_pattern.search(content)
#         if not match:
#             # Warn only first few times (agent might forget)
#             if self.warning_count < self.max_warnings:
#                 logger.warning("⚠️ No metadata block in response!")
#                 self.warning_count += 1
#             return None

#         xml_content = match.group(1)

#         # Parse and build state update
#         try:
#             metadata = self._parse_metadata(xml_content)
#             state_update = self._build_state_update(metadata, state)

#             if state_update:
#                 logger.debug(
#                     f"📊 Extracted metadata fields: {list(state_update.keys())}"
#                 )

#             return state_update

#         except Exception as e:
#             logger.error(f"Failed to parse metadata: {e}")
#             logger.debug(f"XML content: {xml_content}")
#             return None

#     def _parse_metadata(self, xml_content: str) -> Dict[str, Any]:
#         """Parse XML structure into metadata dict"""

#         # Wrap in root for parsing
#         xml_str = f"<root>{xml_content}</root>"
#         root = ET.fromstring(xml_str)

#         metadata = {}

#         # 1. Phase
#         phase_elem = root.find("phase")
#         if phase_elem is not None and phase_elem.text:
#             metadata["phase"] = phase_elem.text.strip()

#         # 2. Thinking
#         thinking_elem = root.find("thinking")
#         if thinking_elem is not None and thinking_elem.text:
#             metadata["thinking"] = thinking_elem.text.strip()

#         # 3. Error
#         error_elem = root.find("error")
#         if error_elem is not None and error_elem.text:
#             metadata["error"] = {
#                 "description": error_elem.text.strip(),
#                 "severity": error_elem.get("severity", "medium"),
#                 "timestamp": datetime.now(UTC).isoformat(),
#             }

#         # 4. Next steps
#         steps_elem = root.find("next_steps")
#         if steps_elem is not None:
#             metadata["next_steps"] = []
#             for step_elem in steps_elem.findall("step"):
#                 if step_elem.text:
#                     metadata["next_steps"].append(step_elem.text.strip())

#         return metadata

#     def _build_state_update(
#         self, metadata: Dict[str, Any], current_state: FullStackAgentState
#     ) -> Optional[Dict[str, Any]]:
#         """Build state update dict from parsed metadata"""

#         update = {}

#         # 1. Phase update
#         if "phase" in metadata:
#             current_phase = current_state.get("current_phase")
#             new_phase = metadata["phase"]

#             if new_phase != current_phase:
#                 logger.info(f"📍 Phase transition: {current_phase} → {new_phase}")
#                 update["current_phase"] = new_phase

#         # 2. Thinking (store in state, keep last 5)
#         if "thinking" in metadata:
#             recent = current_state.get("recent_thinking", [])
#             recent.append(
#                 {
#                     "thinking": metadata["thinking"],
#                     "phase": metadata.get("phase", current_state.get("current_phase")),
#                     "iteration": current_state.get("iteration_count", 0),
#                     "timestamp": datetime.now(UTC).isoformat(),
#                 }
#             )
#             update["recent_thinking"] = recent[-5:]  # Keep last 5
#             logger.debug(f"💭 Thinking: {metadata['thinking'][:60]}...")

#         # 3. Error handling (circuit breaker)
#         if "error" in metadata:
#             error_count = current_state.get("error_count", 0)
#             new_count = error_count + 1

#             update["error_count"] = new_count
#             update["last_error"] = metadata["error"]

#             logger.warning(
#                 f"⚠️ Error #{new_count} [{metadata['error']['severity']}]: "
#                 f"{metadata['error']['description'][:80]}..."
#             )

#             # Circuit breaker: log if approaching limit
#             if new_count >= 3:
#                 logger.error(f"🛑 High error count: {new_count} errors")
#         else:
#             # Reset error count if no error
#             if current_state.get("error_count", 0) > 0:
#                 logger.info("✅ Error count reset (no error in this iteration)")
#                 update["error_count"] = 0

#         # 4. Next steps
#         if "next_steps" in metadata:
#             update["next_steps"] = metadata["next_steps"]
#             logger.info(f"📋 Next steps: {len(metadata['next_steps'])} actions planned")

#         return update if update else None


# middleware/metadata_extractor.py - WITH NEXT_PHASE SUPPORT

# """
# Metadata Extractor Middleware - Extract XML metadata from agent responses.

# Runs: after_model hook
# Updates: current_phase, next_phase, recent_thinking, error_count, last_error, next_steps

# NEW: Extracts next_phase for proactive phase-aware prompting
# """

# import re
# import xml.etree.ElementTree as ET
# from typing import Optional, Dict, Any
# from datetime import datetime, UTC
# import logging

# from langchain.agents.middleware import AgentMiddleware
# from langgraph.runtime import Runtime
# from agent.state import FullStackAgentState

# logger = logging.getLogger(__name__)


# class MetadataExtractorMiddleware(AgentMiddleware[FullStackAgentState]):
#     """
#     Extract structured metadata from agent responses via XML tags.

#     Expected XML format in agent responses:
#     <agent_metadata>
#       <phase>backend_dev</phase>
#       <next_phase>frontend_dev</next_phase>  ← NEW!
#       <thinking>Brief reasoning...</thinking>
#       <error severity="high">Error description</error>
#       <next_steps>
#         <step>Action 1</step>
#         <step>Action 2</step>
#       </next_steps>
#     </agent_metadata>
#     """

#     state_schema = FullStackAgentState

#     def __init__(self):
#         super().__init__()
#         self.xml_pattern = re.compile(
#             r"<agent_metadata>(.*?)</agent_metadata>", re.DOTALL | re.IGNORECASE
#         )
#         self.warning_count = 0
#         self.max_warnings = 3

#     def after_model(
#         self, state: FullStackAgentState, runtime: Runtime
#     ) -> Optional[Dict[str, Any]]:
#         """
#         Extract metadata after LLM generates response.
#         """
#         messages = state.get("messages", [])
#         if not messages:
#             return None

#         last_message = messages[-1]

#         # Only process AI messages
#         if not hasattr(last_message, "content"):
#             return None

#         content = last_message.content

#         # Handle content that might be a list (e.g., with tool calls)
#         if isinstance(content, list):
#             text_content = ""
#             for item in content:
#                 if isinstance(item, str):
#                     text_content += item
#                 elif isinstance(item, dict) and "text" in item:
#                     text_content += item["text"]
#             content = text_content

#         # Skip if content is not a string
#         if not isinstance(content, str):
#             return None

#         # Extract XML block
#         match = self.xml_pattern.search(content)
#         if not match:
#             if self.warning_count < self.max_warnings:
#                 logger.warning("⚠️ No metadata block in response!")
#                 self.warning_count += 1
#             return None

#         xml_content = match.group(1)

#         # Parse and build state update
#         try:
#             metadata = self._parse_metadata(xml_content)
#             state_update = self._build_state_update(metadata, state)

#             if state_update:
#                 logger.debug(
#                     f"📊 Extracted metadata fields: {list(state_update.keys())}"
#                 )

#             return state_update

#         except Exception as e:
#             logger.error(f"Failed to parse metadata: {e}")
#             logger.debug(f"XML content: {xml_content}")
#             return None

#     def _parse_metadata(self, xml_content: str) -> Dict[str, Any]:
#         """Parse XML structure into metadata dict"""

#         # Wrap in root for parsing
#         xml_str = f"<root>{xml_content}</root>"
#         root = ET.fromstring(xml_str)

#         metadata = {}

#         # 1. Current Phase
#         phase_elem = root.find("phase")
#         if phase_elem is not None and phase_elem.text:
#             metadata["phase"] = phase_elem.text.strip()

#         # 2. Next Phase (NEW!)
#         next_phase_elem = root.find("next_phase")
#         if next_phase_elem is not None and next_phase_elem.text:
#             metadata["next_phase"] = next_phase_elem.text.strip()

#         # 3. Thinking
#         thinking_elem = root.find("thinking")
#         if thinking_elem is not None and thinking_elem.text:
#             metadata["thinking"] = thinking_elem.text.strip()

#         # 4. Error
#         error_elem = root.find("error")
#         if error_elem is not None and error_elem.text:
#             metadata["error"] = {
#                 "description": error_elem.text.strip(),
#                 "severity": error_elem.get("severity", "medium"),
#                 "timestamp": datetime.now(UTC).isoformat(),
#             }

#         # 5. Next steps
#         steps_elem = root.find("next_steps")
#         if steps_elem is not None:
#             metadata["next_steps"] = []
#             for step_elem in steps_elem.findall("step"):
#                 if step_elem.text:
#                     metadata["next_steps"].append(step_elem.text.strip())

#         return metadata

#     def _build_state_update(
#         self, metadata: Dict[str, Any], current_state: FullStackAgentState
#     ) -> Optional[Dict[str, Any]]:
#         """Build state update dict from parsed metadata"""

#         update = {}

#         # 1. Phase update
#         if "phase" in metadata:
#             current_phase = current_state.get("current_phase")
#             new_phase = metadata["phase"]

#             if new_phase != current_phase:
#                 logger.info(f"📍 Phase transition: {current_phase} → {new_phase}")

#             update["current_phase"] = new_phase

#         # 2. Next Phase (NEW!)
#         if "next_phase" in metadata:
#             next_phase = metadata["next_phase"]
#             update["next_phase"] = next_phase

#             logger.info(f"🔮 Planned next phase: {next_phase}")

#         # 3. Thinking (store in state, keep last 5)
#         if "thinking" in metadata:
#             recent = current_state.get("recent_thinking", [])
#             recent.append({
#                 "thinking": metadata["thinking"],
#                 "phase": metadata.get("phase", current_state.get("current_phase")),
#                 "iteration": current_state.get("iteration_count", 0),
#                 "timestamp": datetime.now(UTC).isoformat(),
#             })
#             update["recent_thinking"] = recent[-5:]  # Keep last 5
#             logger.debug(f"💭 Thinking: {metadata['thinking'][:60]}...")

#         # 4. Error handling
#         if "error" in metadata:
#             error_count = current_state.get("error_count", 0)
#             new_count = error_count + 1

#             update["error_count"] = new_count
#             update["last_error"] = metadata["error"]

#             logger.warning(
#                 f"⚠️ Error #{new_count} [{metadata['error']['severity']}]: "
#                 f"{metadata['error']['description'][:80]}..."
#             )

#             if new_count >= 3:
#                 logger.error(f"🛑 High error count: {new_count} errors")
#         else:
#             # Reset error count if no error
#             if current_state.get("error_count", 0) > 0:
#                 logger.info("✅ Error count reset (no error in this iteration)")
#                 update["error_count"] = 0

#         # 5. Next steps
#         if "next_steps" in metadata:
#             update["next_steps"] = metadata["next_steps"]
#             logger.info(f"📋 Next steps: {len(metadata['next_steps'])} actions planned")

#         return update if update else None


# # middleware/metadata_extractor.py - HANDLES OPTIONAL FIELDS

# """
# Metadata Extractor Middleware - Minimal metadata extraction.
# Only extracts fields when present - all fields are optional.
# """

# import re
# import xml.etree.ElementTree as ET
# from typing import Optional, Dict, Any
# from datetime import datetime, UTC
# import logging

# from langchain.agents.middleware import AgentMiddleware
# from langgraph.runtime import Runtime
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from agent.state import FullStackAgentState

# logger = logging.getLogger(__name__)


# class MetadataExtractorMiddleware(AgentMiddleware):
#     """
#     Extract metadata from agent responses.

#     ALL FIELDS ARE OPTIONAL - only extract what's present.
#     Minimal metadata for efficiency.
#     """

#     state_schema = FullStackAgentState

#     def __init__(self, strict_mode: bool = False):
#         """
#         Args:
#             strict_mode: If True, warn about missing metadata blocks.
#                         If False, silently ignore missing metadata (default).
#         """
#         super().__init__()
#         self.xml_pattern = re.compile(
#             r"<agent_metadata>(.*?)</agent_metadata>", re.DOTALL | re.IGNORECASE
#         )
#         self.strict_mode = strict_mode
#         self.warning_count = 0
#         self.max_warnings = 3

#     def after_model(
#         self, state: FullStackAgentState, runtime: Runtime
#     ) -> Optional[Dict[str, Any]]:
#         """Extract metadata after LLM generates response."""

#         messages = state.get("messages", [])
#         if not messages:
#             return None

#         last_message = messages[-1]

#         # Only process AI messages
#         if not hasattr(last_message, "content"):
#             return None

#         content = last_message.content

#         # Handle content that might be a list
#         if isinstance(content, list):
#             text_content = ""
#             for item in content:
#                 if isinstance(item, str):
#                     text_content += item
#                 elif isinstance(item, dict) and "text" in item:
#                     text_content += item["text"]
#             content = text_content

#         # Skip if content is not a string
#         if not isinstance(content, str):
#             return None

#         # Extract XML block
#         match = self.xml_pattern.search(content)
#         if not match:
#             # No metadata found - this is OK in minimal mode
#             if self.strict_mode and self.warning_count < self.max_warnings:
#                 logger.debug("No metadata block in response (this is fine)")
#                 self.warning_count += 1
#             return None

#         xml_content = match.group(1)

#         # Parse and build state update
#         try:
#             metadata = self._parse_metadata(xml_content)
#             state_update = self._build_state_update(metadata, state)

#             if state_update:
#                 logger.debug(f"📊 Extracted: {list(state_update.keys())}")

#             return state_update

#         except Exception as e:
#             logger.warning(f"Failed to parse metadata: {e}")
#             return None

#     def _parse_metadata(self, xml_content: str) -> Dict[str, Any]:
#         """
#         Parse XML structure into metadata dict.

#         ALL FIELDS OPTIONAL - only extract what's present.
#         """

#         # Wrap in root for parsing
#         xml_str = f"<root>{xml_content}</root>"

#         try:
#             root = ET.fromstring(xml_str)
#         except ET.ParseError as e:
#             logger.warning(f"XML parse error: {e}")
#             return {}

#         metadata = {}

#         # 1. Phase (optional)
#         phase_elem = root.find("phase")
#         if phase_elem is not None and phase_elem.text:
#             text = phase_elem.text.strip()
#             if text:  # Only if non-empty
#                 metadata["phase"] = text

#         # 2. Next Phase (optional)
#         next_phase_elem = root.find("next_phase")
#         if next_phase_elem is not None and next_phase_elem.text:
#             text = next_phase_elem.text.strip()
#             if text:
#                 metadata["next_phase"] = text

#         # 3. Thinking (optional)
#         thinking_elem = root.find("thinking")
#         if thinking_elem is not None and thinking_elem.text:
#             text = thinking_elem.text.strip()
#             if text:
#                 metadata["thinking"] = text

#         # 4. Error (optional)
#         error_elem = root.find("error")
#         if error_elem is not None and error_elem.text:
#             text = error_elem.text.strip()
#             if text:
#                 metadata["error"] = {
#                     "description": text,
#                     "severity": error_elem.get("severity", "medium"),
#                     "timestamp": datetime.now(UTC).isoformat(),
#                 }

#         # 5. Next steps (optional)
#         steps_elem = root.find("next_steps")
#         if steps_elem is not None:
#             steps = []
#             for step_elem in steps_elem.findall("step"):
#                 if step_elem.text:
#                     text = step_elem.text.strip()
#                     if text:  # Only non-empty steps
#                         steps.append(text)

#             if steps:  # Only if we found any steps
#                 metadata["next_steps"] = steps

#         return metadata

#     def _build_state_update(
#         self, metadata: Dict[str, Any], current_state: FullStackAgentState
#     ) -> Optional[Dict[str, Any]]:
#         """Build state update dict from parsed metadata."""

#         if not metadata:
#             return None

#         update = {}

#         # 1. Phase update (only if present)
#         if "phase" in metadata:
#             current_phase = current_state.get("current_phase")
#             new_phase = metadata["phase"]

#             if new_phase != current_phase:
#                 logger.info(f"📍 Phase: {current_phase} → {new_phase}")
#                 update["current_phase"] = new_phase

#         # 2. Next Phase (only if present)
#         if "next_phase" in metadata:
#             next_phase = metadata["next_phase"]
#             current_next = current_state.get("next_phase")

#             if next_phase != current_next:
#                 logger.info(f"🔮 Next phase: {next_phase}")
#                 update["next_phase"] = next_phase

#         # 3. Thinking (only if present, keep last 3)
#         if "thinking" in metadata:
#             recent = current_state.get("recent_thinking", [])
#             recent.append(
#                 {
#                     "thinking": metadata["thinking"],
#                     "phase": metadata.get("phase", current_state.get("current_phase")),
#                     "timestamp": datetime.now(UTC).isoformat(),
#                 }
#             )
#             update["recent_thinking"] = recent[-3:]  # Keep last 3 only
#             logger.debug(f"💭 Thinking recorded")

#         # 4. Error handling (only if present)
#         if "error" in metadata:
#             error_count = current_state.get("error_count", 0) + 1
#             update["error_count"] = error_count
#             update["last_error"] = metadata["error"]

#             logger.warning(
#                 f"⚠️ Error #{error_count} [{metadata['error']['severity']}]: "
#                 f"{metadata['error']['description'][:60]}..."
#             )

#         # 5. Next steps (only if present)
#         if "next_steps" in metadata:
#             update["next_steps"] = metadata["next_steps"]
#             logger.info(f"📋 Planned: {len(metadata['next_steps'])} steps")

#         return update if update else None