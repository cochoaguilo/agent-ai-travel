from typing import Optional

import numpy as np
import pandas as pd
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from travel_agent import config

SERPAPI_URL = "https://serpapi.com/search?engine=google_hotels"


class HotelSearchInput(BaseModel):
    destination: str = Field(description="Destination city or area, e.g. 'Buenos Aires'")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format")
    adults: int = Field(default=1, description="Number of adult guests")
    children: int = Field(default=0, description="Number of children")


def _hotels_to_dataframe(properties: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in properties:
        rate = entry.get("rate_per_night", {}) or {}
        rows.append(
            {
                "name": entry.get("name", "unknown"),
                "price_per_night": rate.get("extracted_lowest"),
                "hotel_class": entry.get("hotel_class"),
                "rating": entry.get("overall_rating"),
                "reviews": entry.get("reviews"),
            }
        )
    return pd.DataFrame(rows)


def _format_results(df: pd.DataFrame) -> str:
    if df.empty:
        return "No hotels found for that search."

    prices = df["price_per_night"].dropna().to_numpy(dtype=float)
    stats = (
        f"Price per night (USD) — min: {np.min(prices):.0f}, "
        f"mean: {np.mean(prices):.0f}, median: {np.median(prices):.0f}"
        if prices.size
        else "Price stats unavailable."
    )

    table = df.to_string(index=False)
    return f"{table}\n\n{stats}"


@tool("search_hotels", args_schema=HotelSearchInput)
def search_hotels(
    destination: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
) -> str:
    """Search for hotels in a destination city for given check-in/check-out dates using live hotel data."""
    params = {
        "engine": "google_hotels",
        "q": destination,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
        "currency": "USD",
        "hl": "en",
        "api_key": config.SERPAPI_API_KEY,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=30)

    response.raise_for_status()
    data = response.json()
    if "error" in data:
        return f"Hotel search failed: {data['error']}"

    properties = data.get("properties") or []
    df = _hotels_to_dataframe(properties)
    return _format_results(df)
