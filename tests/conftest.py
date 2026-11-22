"""Test environment: offline embedder, no reranker download, no network for utility tools, throwaway checkpoint DB. Set BEFORE any app import."""
import os
for k, v in {"EMBEDDING_PROVIDER": "hash", "SIM_THRESHOLD": "0.2", "RERANKER": "none", "GROQ_API_KEY": "", "CHECKPOINT_DB": ":memory:",
             "MEMORY_THRESHOLD": "0.15", "UTILITY_OFFLINE": "1", "MCP_START_TIMEOUT": "20"}.items(): os.environ.setdefault(k, v)
