"""Central settings loaded from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def _f(name, default): return float(os.getenv(name, default))
def _i(name, default): return int(os.getenv(name, default))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "llama-3.1-8b-instant")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIM_THRESHOLD = _f("SIM_THRESHOLD", 0.45)
MEMORY_THRESHOLD = _f("MEMORY_THRESHOLD", 0.3)             # looser than SIM_THRESHOLD: short memory sentences score lower than runbook paragraphs
SUMMARY_TRIGGER_TOKENS = _i("SUMMARY_TRIGGER_TOKENS", 1200)
SUMMARY_KEEP_LAST_N = _i("SUMMARY_KEEP_LAST_N", 2)
DEFER_MIN_TOOLS = _i("DEFER_MIN_TOOLS", 8)
PAGE_SIZE = _i("PAGE_SIZE", 50)
SUMMARY_EVERY_N_TURNS = _i("SUMMARY_EVERY_N_TURNS", 0)       # 0 = token trigger only
CONTEXT_BUDGET_TOKENS = _i("CONTEXT_BUDGET_TOKENS", 8000)    # for the UI budget bar
AUTO_PRELOAD_TOOLS = _i("AUTO_PRELOAD_TOOLS", 1)             # 1 = host picks top-2 tools up front (saves one LLM round trip)
CACHE_BLOCK_TOKENS = _i("CACHE_BLOCK_TOKENS", 64)            # KV-cache block size used by the hit-rate estimator
MAX_TOOL_OUTPUT_TOKENS = _i("MAX_TOOL_OUTPUT_TOKENS", 3000)  # hard cap on any tool result entering the prompt (engineered mode)
PERSISTENT_MCP = _i("PERSISTENT_MCP", 1)                     # 1 = keep MCP server processes alive between calls
UPLOAD_SIM_THRESHOLD = _f("UPLOAD_SIM_THRESHOLD", 0.2)       # looser than SIM_THRESHOLD: the user attached the file on purpose
UPLOAD_TOP_K = _i("UPLOAD_TOP_K", 6)                         # chunks of the attached file(s) sent per question
OVERVIEW_CHUNKS = _i("OVERVIEW_CHUNKS", 8)                   # chunks sampled across the file for "summarise this" questions
MAX_UPLOAD_MB = _i("MAX_UPLOAD_MB", 15)
MCP_REGISTRY = Path(os.getenv("MCP_REGISTRY", str(ROOT / "mcp_servers.json")))
MCP_TOOL_TIMEOUT = _f("MCP_TOOL_TIMEOUT", 8.0)               # circuit breaker: one hung external server must never stall the graph
MCP_START_TIMEOUT = _f("MCP_START_TIMEOUT", 45.0)            # per-server startup budget (uvx may download a package the first time)
RERANKER = os.getenv("RERANKER", "flashrank")                # flashrank | none (none = lexical/dense blend, used offline and in tests)
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
HYBRID_CANDIDATES = _i("HYBRID_CANDIDATES", 20)              # candidates pulled from EACH of dense and BM25 before RRF
ESCALATION_TOOLS = _i("ESCALATION_TOOLS", 3)                 # tools loaded by the deferred search on a RAG miss
MAX_AGENT_STEPS = _i("MAX_AGENT_STEPS", 6)
DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "OnCall-Engineer")
AGENT_ID = "opsgate-copilot"                                 # shared scope for procedural memory
MEM0_VECTOR_STORE = os.getenv("MEM0_VECTOR_STORE", "chroma") # chroma | qdrant (docker-compose runs qdrant)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost"); QDRANT_PORT = _i("QDRANT_PORT", 6333)
UPLOAD_TYPES = ["pdf", "txt", "md", "html", "htm", "csv", "json", "log", "docx"]

DATA = ROOT / "data"
DB_PATH = DATA / "mock_ops.db"
RUNBOOK_DIR = DATA / "runbooks"
CHROMA_DIR = DATA / "chroma"
MEM0_DIR = DATA / "mem0"
UPLOAD_DIR = DATA / "uploads"      # raw copies of attached files, per thread
THREADS_DIR = DATA / "threads"     # one JSON per user: thread titles + chat log
LOG_DIR = ROOT / "logs"
AUDIT_LOG = LOG_DIR / "audit.jsonl"
CHECKPOINT_DB = Path(os.getenv("CHECKPOINT_DB", str(DATA / "checkpoints.sqlite")))
GIT_JSON = DATA / "mock_git.json"
SERVICES = ["payment-worker", "order-api", "inventory-sync"]
