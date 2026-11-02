from datetime import date
from typing import Optional

import requests

from travel_agent import config

SERPAPI_URL = "https://serpapi.com/search"
MAX_EVENTS = 10


def _date_chip(start_date: Optional[str]) -> Optional[str]:
    """Map a trip start date to the closest Google Events date filter."""
    if not start_date:
        return None
    start = date.fromisoformat(start_date)
    today = date.today()
    if (start - today).days <= 6:
        return "date:week"
    if (start.year, start.month) == (today.year, today.month):
        return "date:month"
    next_month = (today.month % 12) + 1
    next_month_year = today.year + (1 if today.month == 12 else 0)
    if (start.year, start.month) == (next_month_year, next_month):
        return "date:next_month"
    return None


def _format_event(event: dict) -> str:
    when = (event.get("date") or {}).get("when") or (event.get("date") or {}).get("start_date", "")
    venue = (event.get("venue") or {}).get("name") or ", ".join(event.get("address") or [])
    description = (event.get("description") or "").strip().replace("\n", " ")[:160]
    return f"- {event.get('title', 'Untitled event')} | {when} | {venue} | {description}"


def fetch_events(city: str, start_date: Optional[str] = None) -> str:
    """Search upcoming events in a city through SerpApi's Google Events engine."""
    params = {
        "engine": "google_events",
        "q": f"Events in {city}",
        "hl": "en",
        "api_key": config.SERPAPI_API_KEY,
    }
    chip = _date_chip(start_date)
    if chip:
        params["htichips"] = chip

    response = requests.get(SERPAPI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        return f"Event search failed: {data['error']}"

    events = data.get("events_results") or []
    if not events:
        return f"No events found in {city}."
    return "\n".join(_format_event(event) for event in events[:MAX_EVENTS])
