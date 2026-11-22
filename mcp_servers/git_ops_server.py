"""MCP server: git operations over mock data (5 tools)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from core import config

mcp = FastMCP("git-ops")
def _g(): return json.loads(config.GIT_JSON.read_text())

@mcp.tool()
def list_branches(service: str) -> list[str]:
    """List git branches of a service repository."""
    return _g()[service]["branches"]

@mcp.tool()
def get_recent_commits(service: str, limit: int = 10) -> list[dict]:
    """Get the most recent git commits (sha, author, message, time) for a service."""
    return _g()[service]["commits"][:limit]

@mcp.tool()
def get_commit_diff(service: str, sha: str) -> str:
    """Show the code diff of one commit in a service repository."""
    return _g()[service]["diffs"].get(sha, "commit not found")

@mcp.tool()
def list_open_prs(service: str, page: int = 1, page_size: int = 50) -> dict:
    """List open pull requests for a service, paginated."""
    prs = _g()[service]["prs"]; s = (page - 1) * page_size
    return {"rows": prs[s:s + page_size], "page": page, "total": len(prs), "more_pages_available": s + page_size < len(prs)}

@mcp.tool()
def blame_file(service: str, path: str) -> dict:
    """Show who last changed a file (git blame summary) in a service repository."""
    c = _g()[service]["commits"][0]
    return {"path": path, "last_author": c["author"], "sha": c["sha"], "message": c["message"]}

if __name__ == "__main__": mcp.run(transport="stdio")
