"""MCP server: live utility tools for demos that fall outside the internal runbooks (weather, FX, time). All READ_ONLY."""
import json, os, sys, urllib.parse, urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("utility-tools")
GEO = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
FX = "https://api.frankfurter.app/latest"

def _fixture(url: str, params: dict) -> dict:
    """UTILITY_OFFLINE=1: canned answers for tests and no-internet demos. Results are labelled so they are never mistaken for live data."""
    if url == GEO: return {"results": [{"name": str(params["name"]).title(), "country": "Offline fixture", "latitude": 28.61, "longitude": 77.21}]}
    if url == FORECAST: return {"current": {"time": "2026-01-01T00:00", "temperature_2m": 24.3, "relative_humidity_2m": 41, "wind_speed_10m": 9.0}}
    return {"date": "2026-01-01", "rates": {str(params["to"]).upper(): round(float(params["amount"]) * 83.95, 2)}}

def _get_json(url: str, params: dict) -> dict:
    """Single network seam. Short timeout: the host also enforces its own circuit breaker."""
    if os.getenv("UTILITY_OFFLINE") == "1": return _fixture(url, params)
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "opsgate-copilot/4.0"})
    with urllib.request.urlopen(req, timeout=6) as r: return json.loads(r.read().decode("utf-8"))

@mcp.tool()
def get_live_weather(city: str) -> dict:
    """Get the current live weather (temperature, humidity, wind) for a city."""
    geo = _get_json(GEO, {"name": city, "count": 1, "language": "en", "format": "json"}).get("results") or []
    if not geo: return {"error": f"city not found: {city}"}
    g = geo[0]
    cur = _get_json(FORECAST, {"latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto",
                               "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"}).get("current", {})
    return {"city": g.get("name", city), "country": g.get("country", ""), "temperature_c": cur.get("temperature_2m"),
            "humidity_pct": cur.get("relative_humidity_2m"), "wind_kmh": cur.get("wind_speed_10m"), "observed_at": cur.get("time")}

@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using live foreign-exchange rates (for example USD to INR)."""
    f, t = from_currency.strip().upper(), to_currency.strip().upper()
    d = _get_json(FX, {"amount": amount, "from": f, "to": t})
    if "rates" not in d or t not in d["rates"]: return {"error": f"no rate for {f}->{t}"}
    return {"from": f, "to": t, "input_amount": amount, "converted_amount": d["rates"][t], "rate_date": d.get("date")}

@mcp.tool()
def get_world_time(timezone: str) -> dict:
    """Get the current exact time and UTC offset in an IANA timezone, for example Asia/Kolkata."""
    try: now = datetime.now(ZoneInfo(timezone))
    except Exception: return {"error": f"unknown timezone: {timezone}"}
    off = now.utcoffset(); mins = int(off.total_seconds() // 60) if off else 0
    return {"timezone": timezone, "iso_timestamp": now.isoformat(timespec="seconds"), "utc_offset": f"{'+' if mins >= 0 else '-'}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"}

if __name__ == "__main__": mcp.run(transport="stdio")
