"""MCP Host: reads mcp_servers.json, connects to every ENABLED server (stdio), namespaces tools as <server>_<tool>, tags each tool with a
RBAC risk level, and calls tools behind a timeout circuit breaker.

Plug-and-play: adding a server = one JSON entry (or register_new_mcp_server()); no routing or graph code changes.
A server that fails to start (e.g. `uvx` not installed, no network) is skipped and reported in `host.failed`; the others keep working."""
import asyncio, atexit, json, os, sys, threading
from contextlib import AsyncExitStack
from langchain_mcp_adapters.client import MultiServerMCPClient
from core import config

CUSTOM_KEYS = ("enabled", "default_risk", "risk_overrides")        # registry-only keys, never passed to the MCP client
RISKS = ("READ_ONLY", "PRIVILEGED_WRITE")

class ToolTimeout(Exception): pass

def load_registry(path=None) -> dict:
    p = path or config.MCP_REGISTRY
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def build_connections(registry: dict) -> dict:
    """enabled servers only, custom keys stripped, `python` -> this interpreter, relative script paths -> absolute."""
    out = {}
    for name, cfg in registry.items():
        if not cfg.get("enabled", True): continue
        c = {k: v for k, v in cfg.items() if k not in CUSTOM_KEYS}
        if c.get("command") in ("python", "python3"):
            c["command"] = sys.executable
            c.setdefault("env", dict(os.environ))          # our own servers inherit the app settings (embedder, thresholds); third-party servers get a clean env
        c["args"] = [str(config.ROOT / a) if a.endswith(".py") and not str(a).startswith("/") and (config.ROOT / a).exists() else a for a in c.get("args", [])]
        c.setdefault("transport", "stdio")
        out[name] = c
    return out

def _norm(res):
    """Normalise a tool result to a Python object (dict/list/str)."""
    if isinstance(res, tuple): res = res[0]
    if isinstance(res, list) and res and isinstance(res[0], dict) and "text" in res[0]:
        texts = [b["text"] for b in res if "text" in b]
        if len(texts) > 1:                                   # list-returning tools arrive as one block per item
            try: return [json.loads(t) for t in texts]
            except Exception: res = "\n".join(texts)
        else: res = texts[0]
    if isinstance(res, str):
        try: return json.loads(res)
        except Exception: return res
    return res

class MCPHost:
    """persistent=True (default): ONE background event loop keeps every server process alive, so a tool call is a JSON-RPC message.
    If the persistent start fails entirely it falls back to the stateless mode."""
    def __init__(self, persistent: bool | None = None, registry_path=None):
        self.registry_path = registry_path or config.MCP_REGISTRY
        self.persistent = bool(config.PERSISTENT_MCP) if persistent is None else persistent
        self.registry: dict = {}; self.client = None
        self.tools: dict = {}                    # prefixed name -> langchain tool
        self.server_of: dict = {}                # prefixed name -> server name
        self.risk: dict = {}                     # prefixed name -> READ_ONLY | PRIVILEGED_WRITE
        self.failed: dict = {}                   # server -> reason it was skipped
        self._loop = self._thread = self._stop = self._task = None
        self._sessions: dict = {}
        self._lock = threading.Lock()

    # ---- registry ----
    def _read_registry(self):
        self.registry = load_registry(self.registry_path)
        self.connections = build_connections(self.registry)
        self.client = MultiServerMCPClient(self.connections)
    def _tag(self, server: str, tool):
        cfg = self.registry.get(server, {}); orig = tool.name
        if not orig.startswith(server + "_"): tool.name = f"{server}_{orig}"         # namespace: utility_tools_get_live_weather
        risk = (cfg.get("risk_overrides") or {}).get(orig) or cfg.get("default_risk", "READ_ONLY")
        self.tools[tool.name], self.server_of[tool.name], self.risk[tool.name] = tool, server, risk if risk in RISKS else "PRIVILEGED_WRITE"

    # ---- persistent mode ----
    async def _serve(self, ready: asyncio.Future):
        """ONE task opens and closes every async context (anyio requirement)."""
        from langchain_mcp_adapters.tools import load_mcp_tools
        try:
            async with AsyncExitStack() as stack:
                for name in self.connections:
                    try:
                        async with asyncio.timeout(config.MCP_START_TIMEOUT):          # same task, so anyio scopes stay valid
                            sess = await stack.enter_async_context(self.client.session(name))
                            loaded = await load_mcp_tools(sess)
                        self._sessions[name] = sess
                        for t in loaded: self._tag(name, t)
                    except BaseException as e:
                        if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
                        self.failed[name] = f"{type(e).__name__}: {str(e)[:160]}"; print(f"[mcp] skipped server '{name}': {self.failed[name]}")
                self._stop = asyncio.Event(); ready.set_result(True)
                await self._stop.wait()
        except Exception as e:
            if not ready.done(): ready.set_exception(e)
            else: raise

    def _start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-host"); self._thread.start()
        ready = asyncio.run_coroutine_threadsafe(self._make_ready(), self._loop).result()
        self._task = asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
        total = config.MCP_START_TIMEOUT * max(1, len(self.connections)) + 30
        res = asyncio.run_coroutine_threadsafe(self._await(ready), self._loop).result(timeout=total)
        if not getattr(self, "_atexit", False): atexit.register(self.close); self._atexit = True
        return res
    async def _make_ready(self): return asyncio.get_running_loop().create_future()
    @staticmethod
    async def _await(fut): return await fut

    def _run(self, coro, timeout: float = 120): return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self):
        if self._loop and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try: self._task.result(timeout=10)
            except Exception: pass
            self._stop = None
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ---- public API ----
    def load(self):
        self._read_registry(); self.tools, self.server_of, self.risk, self.failed, self._sessions = {}, {}, {}, {}, {}
        if self.persistent:
            try: self._start(); return self
            except Exception as e:
                print(f"[mcp] persistent sessions unavailable ({e!r}); falling back to per-call processes")
                self.persistent = False
        for name in self.connections:
            try:
                for t in asyncio.run(asyncio.wait_for(self.client.get_tools(server_name=name), config.MCP_START_TIMEOUT)): self._tag(name, t)
            except Exception as e: self.failed[name] = f"{type(e).__name__}: {str(e)[:160]}"; print(f"[mcp] skipped server '{name}': {self.failed[name]}")
        return self

    def reload(self):
        """Hot reload after the registry changed: restart sessions, rediscover tools."""
        with self._lock:
            self.close(); self._loop = self._thread = self._task = None
            return self.load()

    def servers(self) -> list[dict]:
        out = []
        for name, cfg in self.registry.items():
            tools = [n for n, s in self.server_of.items() if s == name]
            status = "disabled" if not cfg.get("enabled", True) else "failed" if name in self.failed else "connected"
            out.append({"name": name, "status": status, "tools": len(tools), "default_risk": cfg.get("default_risk", "READ_ONLY"),
                        "command": " ".join([cfg.get("command", "")] + [str(a) for a in cfg.get("args", [])]).strip(), "error": self.failed.get(name, "")})
        return out

    async def _guarded(self, name: str, args: dict):
        """Circuit breaker: hard timeout, ONE retry on a transient timeout, then a clean error instead of a stalled graph."""
        last = None
        for _attempt in range(2):
            try: return await asyncio.wait_for(self.tools[name].ainvoke(args), timeout=config.MCP_TOOL_TIMEOUT)
            except (asyncio.TimeoutError, TimeoutError) as e: last = e
        raise ToolTimeout(f"{name} timed out after {config.MCP_TOOL_TIMEOUT}s (2 attempts)") from last

    def call(self, name: str, args: dict | None = None):
        if name not in self.tools: raise KeyError(f"unknown tool: {name}")
        coro = self._guarded(name, args or {})
        return _norm(self._run(coro, timeout=config.MCP_TOOL_TIMEOUT * 2 + 10) if self.persistent else asyncio.run(coro))

    def read_resource(self, server: str, uri: str) -> str:
        async def go(s):
            r = await s.read_resource(uri); return "\n".join(c.text for c in r.contents if hasattr(c, "text"))
        return self._with_session(server, go)
    def get_prompt(self, server: str, name: str, arguments: dict) -> str:
        async def go(s):
            r = await s.get_prompt(name, arguments); return "\n".join(m.content.text for m in r.messages if hasattr(m.content, "text"))
        return self._with_session(server, go)
    def _with_session(self, server, fn):
        if self.persistent: return self._run(fn(self._sessions[server]))
        async def one():
            async with self.client.session(server) as s: return await fn(s)
        return asyncio.run(one())

def register_new_mcp_server(name: str, cfg: dict, host: MCPHost | None = None, path=None) -> dict:
    """Persist a new server entry to mcp_servers.json and (if a host is given) hot-reload it. Returns the stored entry."""
    import re
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", name or ""): raise ValueError("name must be lowercase letters, digits or underscore (2-41 chars)")
    if not isinstance(cfg, dict) or not cfg.get("command"): raise ValueError("config needs at least a 'command'")
    if cfg.get("default_risk", "READ_ONLY") not in RISKS: raise ValueError(f"default_risk must be one of {RISKS}")
    entry = {"transport": "stdio", "default_risk": "READ_ONLY", "enabled": True, **cfg}
    p = path or (host.registry_path if host else config.MCP_REGISTRY)
    reg = load_registry(p); reg[name] = entry
    p.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    if host is not None: host.registry_path = p; host.reload()
    return entry
