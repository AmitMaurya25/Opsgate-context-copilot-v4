"""MCP server exposing the RAG retriever as a tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from core.rag_retriever import retrieve

mcp = FastMCP("runbook-rag")

@mcp.tool()
def search_runbooks(query: str, k: int = 5) -> list[dict]:
    """Search the on-call runbooks (error codes, retry scripts, escalation policy). Returns chunks above the similarity threshold."""
    return retrieve(query, k=k)

if __name__ == "__main__": mcp.run(transport="stdio")
