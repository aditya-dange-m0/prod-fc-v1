# Full-Stack AI Code Generation Platform

A **production-grade Full-Stack AI Code Generation Platform** built with Python, FastAPI, and LangGraph. This platform autonomously generates complete web applications using AI agents with persistent state management, multi-tenant sandbox isolation, and real-time streaming capabilities.

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI API   │    │  LangGraph      │    │  E2B Sandbox   │
│   (Streaming)   │◄──►│  Agent System   │◄──►│  Multi-Tenant   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PostgreSQL     │    │  Redis Cache    │    │  File System    │
│  (Checkpoints)  │    │  (Sessions)     │    │  (Projects)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

- **Backend Framework**: FastAPI with Server-Sent Events (SSE)
- **AI Agent**: LangGraph with Claude Sonnet 4.5
- **Database**: PostgreSQL with async SQLAlchemy
- **Caching**: Redis for session management
- **Sandboxes**: E2B for isolated code execution
- **Frontend**: Next.js (pre-configured in projects)
- **Database**: MongoDB for generated applications

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL database
- Redis server (optional, for caching)
- E2B API key
- Anthropic API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd full-stack-ai-platform
```

2. **Create virtual environment with uv**
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate virtual environment
# On Windows:
.venv/Scripts/activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
uv pip install -r requirements.txt
```

4. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - ANTHROPIC_API_KEY
# - E2B_API_KEY
# - DATABASE_URL
# - DIRECT_DATABASE_URL
```

5. **Database Setup**
```bash
# Run database migrations
alembic upgrade head
```

6. **Start the server**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📋 Environment Variables

### Required API Keys

```env
# Anthropic Claude API Key (Required)
ANTHROPIC_API_KEY=sk-ant-api03-your_anthropic_api_key_here

# E2B Sandbox API Key (Required)
E2B_API_KEY=e2b_your_e2b_api_key_here
```

### Database Configuration

```env
# PostgreSQL Database URL (Pooled connection via PgBouncer)
DATABASE_URL=postgresql://username:password@host-pooler.region.provider.com/database?sslmode=require

# Direct PostgreSQL Database URL (Direct connection, bypasses pooler)
DIRECT_DATABASE_URL=postgresql://username:password@host.region.provider.com/database?sslmode=require
```

### Optional Services

```env
# OpenAI API Key (Optional - for middleware summarization)
OPENAI_API_KEY=sk-proj-your_openai_api_key_here

# Redis URL for caching and session storage
REDIS_URL=redis://default:password@host:port

# LangSmith API Key (Optional - for tracing and monitoring)
LANGSMITH_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
```

## 🏛️ System Architecture

### Agent System (`agent/`)

The core AI agent system built on LangGraph:

- **`singleton_agent.py`**: Production singleton pattern - creates agent once, reuses for all requests
- **`prompts.py`**: Comprehensive system prompt defining Full-Stack Developer capabilities
- **`tools_loader.py`**: Centralized tool loading from multiple modules
- **`state.py`**: Agent state management with phase tracking and error handling

#### Agent Capabilities

- **Tech Stack**: FastAPI backend + Next.js frontend + MongoDB
- **Working Directory**: `/home/user/code/` with structured folders
- **Development Phases**: Planning → Backend → Frontend → Integration
- **Iteration Management**: 25 iteration limit with intelligent phase completion

### Multi-Tenant Sandbox Manager (`sandbox_manager.py`)

**Sophisticated E2B sandbox management system** with enterprise features:

#### Core Features
- **Multi-tenant Isolation**: Each (user_id, project_id) gets isolated sandbox
- **Resource Management**: Configurable per-user and total sandbox limits
- **Automatic Cleanup**: Idle timeout (500s) and max age (900s) management
- **Health Monitoring**: Sandbox health checks and reconnection logic
- **Retry Mechanisms**: Exponential backoff for failed operations

#### Caching System (Redis Integration)
```python
# Sandbox ID caching with TTL matching sandbox lifetime
_cache_sandbox_id(user_id, project_id, sandbox_id, ttl=900)

# Multi-step fallback: Memory → Redis → Create New
sandbox = await get_sandbox(user_id, project_id)
```

#### Key Configuration
```python
@dataclass
class SandboxConfig:
    template: str = "next-fast-mongo-pre-v2"
    timeout: int = 500
    max_sandboxes_per_user: int = 2
    max_total_sandboxes: int = 10
    idle_timeout: int = 500  # 500 seconds
    max_sandbox_age: int = 900  # 15 minutes
```

### Database Layer (`db/`)

#### Models (`models.py`)
**Simplified Architecture**: Project = Session (one-to-one mapping)

```python
class User(Base):
    id: str (Primary Key)
    email: str (Unique)
    username: str
    projects: List[Project] (Relationship)

class Project(Base):
    id: str (Primary Key) # Used as session_id everywhere
    user_id: str (Foreign Key)
    name: str
    active_sandbox_id: Optional[str]
    sandbox_state: SandboxState (running/paused/killed/none)
    status: SessionStatus (active/paused/ended)
    project_files: List[ProjectFile]
    thoughts: List[ProjectThought]
```

#### Service Layer (`service.py`)
Clean business logic layer with async operations:
- **User Management**: Create, get, delete with unique email handling
- **Project Operations**: CRUD with session integration
- **File Management**: Content storage and retrieval

### Checkpointing System (`checkpoint/`)

**Production PostgreSQL checkpointer** for LangGraph state persistence:

#### Features
- **Connection Pooling**: Optimized async connection management (2-20 connections)
- **Message Persistence**: Automatic conversation history storage
- **State Snapshots**: Agent state checkpoints at each step
- **Thread Management**: Project-based conversation threading
- **Health Monitoring**: Pool statistics and connection health

#### Configuration
```python
class CheckpointerService:
    min_pool_size: int = 2
    max_pool_size: int = 20
    pool_timeout: int = 30
    max_idle_time: int = 300  # 5 minutes
```

### API Layer (`api/fs_agent_api.py`)

**Production streaming API** with Server-Sent Events:

#### Endpoints

```http
POST /api/projects/{project_id}/chat
# Real-time streaming chat with agent

GET /api/projects/{project_id}/history
# Retrieve conversation history with message serialization

GET /api/projects/{project_id}/state
# Get current agent state and progress

GET /api/projects/{project_id}/files
# Get all project files

POST /api/projects/{project_id}/restore
# Restore and activate project session

GET /api/users/{user_id}/projects
# Get all projects for a user

GET /health
# System health check
```

#### Streaming Events
```javascript
// Event types sent via SSE
{
  "agent_start": { timestamp, project_id },
  "agent_thinking": { token, node },
  "tool_start": { tool_name, tool_id, tool_args },
  "tool_complete": { tool_name, output_preview },
  "agent_complete": { timestamp, project_id },
  "error": { message, type, timestamp }
}
```

## 🛠️ Tools System

### File Operations (`tools/file_tools_e2b.py`)
- **E2B Integration**: Sandbox file system operations
- **Batch Operations**: Efficient multi-file read/write
- **Database Tracking**: File changes stored in database
- **Error Handling**: Comprehensive error types and recovery

### Available Tools
- **File Tools**: `read_file`, `write_file`, `batch_read_files`, `batch_write_files`
- **Edit Tools**: `edit_file`, `smart_edit_file`
- **Command Tools**: `run_command`, `run_service`, `list_processes`
- **Web Search**: `search_web`
- **User Interaction**: `ask_user_tool`

## 🔄 Development Workflow

### Project Structure Generated
```
/home/user/code/
├── backend/              # FastAPI backend (generated)
│   ├── main.py          # FastAPI app + routes
│   ├── models.py        # Pydantic models
│   ├── database.py      # MongoDB connection
│   ├── requirements.txt # Dependencies
│   └── .env.example     # Config template
│
└── frontend/            # Next.js frontend (pre-configured)
    ├── package.json     # Already exists
    ├── pages/           # React pages
    ├── components/      # React components
    └── ...              # Next.js structure
```

### Development Phases

1. **Planning Phase** (2-3 iterations)
   - Define project structure
   - Create directories
   - Plan component architecture

2. **Backend Development** (8-10 iterations)
   - Create FastAPI application
   - Implement CRUD endpoints
   - Set up MongoDB integration
   - Add error handling and validation

3. **Frontend Development** (8-10 iterations)
   - Create Next.js pages and components
   - Implement API integration
   - Add user interface elements

4. **Integration Phase** (2-3 iterations)
   - Connect frontend ↔ backend
   - End-to-end testing
   - Final optimizations

### Standard Dependencies
```txt
# Backend (FastAPI)
fastapi==0.104.1
uvicorn==0.24.0
pymongo==4.6.0
pydantic==2.5.0
python-dotenv==1.0.0
python-multipart==0.0.6
```

## 📊 Monitoring & Observability

### Agent State Tracking
```python
class FullStackAgentState:
    current_phase: str           # Current development phase
    next_steps: List[str]        # Planned actions
    recent_thinking: List[dict]  # Last 5 thoughts
    error_count: int            # Consecutive errors
    active_files: List[str]     # Recently accessed files
    service_pids: Dict[str, int] # Running services
    tokens_used: Dict           # Token usage tracking
```

### Health Monitoring
- **Database Health**: Connection pool status
- **Sandbox Health**: Active sandbox monitoring
- **Agent Health**: State and iteration tracking
- **API Health**: Response times and error rates

### Logging
```python
# Comprehensive logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Log files
log/log_latest.log  # All application logs
```

## 🔧 Configuration

### Agent Configuration
```env
# AI Model Settings
DEFAULT_MODEL=claude-sonnet-4-5-20250929
MODEL_TEMPERATURE=0.1
MAX_TOKENS=8192
MODEL_TIMEOUT=120

# Agent Behavior
MAX_ITERATIONS=25
ENABLE_MEMORY=true
ENABLE_SUMMARIZATION=true
SUMMARIZATION_THRESHOLD=5000
```

### Sandbox Configuration
```env
# E2B Template ID
E2B_TEMPLATE_ID=next-fast-mongo-pre-v2

# Resource Limits
MAX_SANDBOXES_PER_USER=2
MAX_TOTAL_SANDBOXES=10
SANDBOX_IDLE_TIMEOUT=500
SANDBOX_MAX_AGE=900
```

### Database Pool Settings
```env
# Connection Pool Settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
CHECKPOINT_POOL_MIN=2
CHECKPOINT_POOL_MAX=20
```

## 🚀 Production Deployment

### Docker Setup (Recommended)
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install uv && uv pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Environment Variables for Production
```env
# Production Settings
ENV=production
DEBUG=False
RELOAD=false
LOG_LEVEL=WARNING
ENABLE_DEBUG_TOOLS=false

# Security
SSL_MODE=require
CORS_ORIGINS=https://yourdomain.com
```

### Database Migration
```bash
# Run migrations in production
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Description"
```

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
uv pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Test Configuration
```env
# Testing Environment
TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/ai_agent_test
ENABLE_TEST_MODE=true
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Issues**
```bash
# Check database connectivity
psql $DATABASE_URL -c "SELECT 1;"

# Verify pool settings
curl http://localhost:8000/health
```

2. **E2B Sandbox Issues**
```bash
# Check E2B API key
curl -H "Authorization: Bearer $E2B_API_KEY" https://api.e2b.dev/sandboxes

# Monitor sandbox usage
# Check logs for sandbox creation/cleanup
```

3. **Agent State Issues**
```bash
# Check agent state
curl http://localhost:8000/api/projects/{project_id}/state

# Clear agent state (if needed)
# Delete from checkpoints table
```

### Logging Levels
```env
# Debug logging
LOG_LEVEL=DEBUG
ENABLE_DB_LOGGING=true
ENABLE_AGENT_LOGGING=true
ENABLE_TOOL_LOGGING=true
```

## 📚 API Documentation

### Interactive API Docs
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Example Usage

#### Start a Chat Session
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/api/projects/my-project/chat',
  {
    method: 'POST',
    body: JSON.stringify({
      message: "Create a todo app with FastAPI and Next.js",
      user_id: "user123",
      project_id: "my-project",
      email_id: "user@example.com"
    })
  }
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent response:', data);
};
```

#### Get Project History
```javascript
const response = await fetch(
  'http://localhost:8000/api/projects/my-project/history'
);
const history = await response.json();
console.log('Conversation history:', history.messages);
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd full-stack-ai-platform

# Setup development environment
uv venv --python 3.12
.venv/Scripts/activate  # Windows
uv pip install -r requirements.txt

# Install development dependencies
uv pip install black isort flake8 mypy

# Run formatting
black .
isort .
```

### Code Style
- **Formatting**: Black
- **Import Sorting**: isort
- **Linting**: flake8
- **Type Checking**: mypy

## 📄 License

[Add your license information here]

## 🆘 Support

For support and questions:
- **Issues**: [GitHub Issues](link-to-issues)
- **Documentation**: [Full Documentation](link-to-docs)
- **Community**: [Discord/Slack](link-to-community)

---

**Built with ❤️ using LangGraph, FastAPI, and E2B**