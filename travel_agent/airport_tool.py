import unicodedata

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Main commercial airport per city. Keys are normalized (lowercase, no accents).
CITY_TO_IATA = {
    "buenos aires": "EZE",
    "ciudad de buenos aires": "EZE",
    "caba": "EZE",
    "cordoba": "COR",
    "rosario": "ROS",
    "mendoza": "MDZ",
    "bariloche": "BRC",
    "mar del plata": "MDQ",
    "salta": "SLA",
    "madrid": "MAD",
    "barcelona": "BCN",
    "paris": "CDG",
    "londres": "LHR",
    "london": "LHR",
    "roma": "FCO",
    "rome": "FCO",
    "nueva york": "JFK",
    "new york": "JFK",
    "miami": "MIA",
    "los angeles": "LAX",
    "mexico": "MEX",
    "ciudad de mexico": "MEX",
    "mexico city": "MEX",
    "santiago": "SCL",
    "sao paulo": "GRU",
    "rio de janeiro": "GIG",
    "lima": "LIM",
    "bogota": "BOG",
    "montevideo": "MVD",
    "amsterdam": "AMS",
    "lisboa": "LIS",
    "lisbon": "LIS",
    "tokio": "HND",
    "tokyo": "HND",
    "dubai": "DXB",
}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class CityToAirportInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Buenos Aires'")


@tool("city_to_airport_code", args_schema=CityToAirportInput)
def city_to_airport_code(city: str) -> str:
    """Resolve a city name to its main airport IATA code, for use as departure_id/arrival_id in search_flights."""
    code = CITY_TO_IATA.get(_normalize(city))
    if code is None:
        return f"No known airport code for '{city}'. Ask the user for the IATA code directly."
    return code
