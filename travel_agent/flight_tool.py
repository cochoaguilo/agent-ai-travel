from typing import Optional

import numpy as np
import pandas as pd
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from travel_agent import config

SERPAPI_URL = "https://serpapi.com/search?engine=google_flights"


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Origin airport IATA code, e.g. MAD")
    destination: str = Field(description="Destination airport IATA code, e.g. EZE")
    outbound_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(
        default=None, description="Return date in YYYY-MM-DD format, omit for one-way"
    )
    adults: int = Field(default=1, description="Number of adult passengers")


def _flights_to_dataframe(flights: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in flights:
        legs = entry.get("flights", [])
        first_leg = legs[0] if legs else {}
        last_leg = legs[-1] if legs else {}
        rows.append(
            {
                "airline": first_leg.get("airline", "unknown"),
                "price": entry.get("price"),
                "duration_min": entry.get("total_duration"),
                "stops": max(len(legs) - 1, 0),
                "departure_time": first_leg.get("departure_airport", {}).get("time"),
                "arrival_time": last_leg.get("arrival_airport", {}).get("time"),
            }
        )
    return pd.DataFrame(rows)


def _format_results(df: pd.DataFrame) -> str:
    if df.empty:
        return "No flights found for that search."

    prices = df["price"].dropna().to_numpy(dtype=float)
    stats = (
        f"Price stats (USD) — min: {np.min(prices):.0f}, "
        f"mean: {np.mean(prices):.0f}, median: {np.median(prices):.0f}"
        if prices.size
        else "Price stats unavailable."
    )

    table = df.to_string(index=False)
    return f"{table}\n\n{stats}"


@tool("search_flights", args_schema=FlightSearchInput)
def search_flights(
    origin: str,
    destination: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
) -> str:
    """Search for flights between two airports on given dates using live flight data."""
    params = {
        "engine": "google_flights",
        "departure_id": origin.upper(),
        "arrival_id": destination.upper(),
        "outbound_date": outbound_date,
        "adults": adults,
        "currency": "USD",
        "hl": "en",
        "type": 1 if return_date else 2,
        "api_key": config.SERPAPI_API_KEY,
    }
    if return_date:
        params["return_date"] = return_date

    response = requests.get(SERPAPI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        return f"Flight search failed: {data['error']}"

    flights = data.get("best_flights") or data.get("other_flights") or []
    df = _flights_to_dataframe(flights)
    return _format_results(df)
