from datetime import date, timedelta
from typing import Optional

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 15
DAILY_FIELDS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}


def _geocode(city: str) -> Optional[dict]:
    response = requests.get(
        GEOCODING_URL, params={"name": city, "count": 1, "language": "en"}, timeout=15
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    return results[0] if results else None


def _shift_year(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # Feb 29 on a non-leap year
        return day.replace(year=day.year + years, day=28)


def fetch_weather(city: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Get the daily weather for a city using the free Open-Meteo API.

    Dates within the forecast horizon return a real forecast. Dates further
    away fall back to the same dates last year as a historical reference.
    """
    location = _geocode(city)
    if location is None:
        return f"Could not find coordinates for '{city}'."

    start = date.fromisoformat(start_date) if start_date else date.today()
    end = date.fromisoformat(end_date) if end_date else start + timedelta(days=3)

    is_forecast = (end - date.today()).days <= FORECAST_HORIZON_DAYS
    url = FORECAST_URL if is_forecast else ARCHIVE_URL
    query_start, query_end = (start, end) if is_forecast else (_shift_year(start, -1), _shift_year(end, -1))

    response = requests.get(
        url,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": DAILY_FIELDS,
            "timezone": "auto",
            "start_date": query_start.isoformat(),
            "end_date": query_end.isoformat(),
        },
        timeout=15,
    )
    response.raise_for_status()
    daily = response.json().get("daily") or {}

    source = "Forecast" if is_forecast else "Historical reference (same dates last year, no forecast yet)"
    lines = [f"{source} for {location['name']}, {location.get('country', '')}:"]
    for i, day in enumerate(daily.get("time", [])):
        code = daily["weather_code"][i]
        lines.append(
            f"- {day}: {WEATHER_CODES.get(code, f'code {code}')}, "
            f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]} °C, "
            f"precipitation {daily['precipitation_sum'][i]} mm"
        )
    return "\n".join(lines)
