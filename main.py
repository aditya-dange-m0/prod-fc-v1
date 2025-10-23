"""
Start the Streaming Agent API Server
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Start the streaming API server"""

    # Check environment
    required_env_vars = [
        "ANTHROPIC_API_KEY",
        "E2B_API_KEY",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("Please set them in your .env file or environment")
        sys.exit(1)

    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    logger.info("🚀 Starting Full-Stack Agent Streaming API")
    logger.info(f"📡 Server: http://{host}:{port}")
    logger.info(f"🔄 Reload: {reload}")
    logger.info(f"📁 Project root: {project_root}")

    # Start server
    uvicorn.run(
        "api.fs_agent_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
