from typing import Optional, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from travel_agent import config

SERPAPI_URL = "https://serpapi.com/search"
MAX_ROUTES = 5
MAX_RESTAURANTS = 10
MAX_SNIPPETS = 8

TRANSFER_PROMPT = """You are an airport transfer specialist. Using ONLY the route
data provided, summarize the options to get from the airport to the traveler's
destination: mode, duration, distance, and cost when available. Recommend the
best option for a solo traveler and say why in one sentence. Do not invent
routes, prices, or lines that are not in the data. Keep it under 120 words."""

FOOD_TIPS_PROMPT = """You are a food culture guide. Using ONLY the search results
provided, list 5-7 typical dishes or food experiences to try in the city, one
line each. Do not name restaurants here and do not add dishes that are not
supported by the results. Keep it under 150 words."""

RESTAURANTS_PROMPT = """You are a restaurant critic. Using ONLY the restaurant
list provided, recommend 4-5 places, one line each with cuisine type, price
range, and rating. Prefer well-rated places with many reviews and include a
budget-friendly option. Never invent restaurants. Keep it under 150 words."""


def _serpapi(params: dict) -> dict:
    response = requests.get(
        SERPAPI_URL, params={**params, "hl": "en", "api_key": config.SERPAPI_API_KEY}, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def fetch_airport_transfers(airport: str, destination: str) -> str:
    """Route options between an airport and a destination via SerpApi Google Maps Directions."""
    data = _serpapi(
        {
            "engine": "google_maps_directions",
            "start_addr": f"{airport} airport",
            "end_addr": destination,
            "travel_mode": 6,  # best routes across all modes
        }
    )
    routes = data.get("directions") or []
    if not routes:
        return f"No routes found from {airport} airport to {destination}."
    lines = []
    for route in routes[:MAX_ROUTES]:
        cost = f", cost {route['cost']} {route.get('currency', '')}" if route.get("cost") else ""
        notes = "; ".join(route.get("extensions") or [])
        lines.append(
            f"- {route.get('travel_mode', 'unknown')} via {route.get('via', '?')}: "
            f"{route.get('formatted_duration', '?')}, {route.get('formatted_distance', '?')}{cost}"
            + (f" ({notes})" if notes else "")
        )
    return "\n".join(lines)


def fetch_food_tips(city: str) -> str:
    """Search snippets about local food via SerpApi Google Search."""
    data = _serpapi({"engine": "google", "q": f"traditional food to try in {city}"})
    results = data.get("organic_results") or []
    if not results:
        return f"No food tips found for {city}."
    return "\n".join(
        f"- {result.get('title', '')}: {result.get('snippet', '')}" for result in results[:MAX_SNIPPETS]
    )


def fetch_restaurants(city: str, near: Optional[str] = None) -> str:
    """Restaurant listings via SerpApi Google Local."""
    query = f"restaurants near {near}, {city}" if near else f"best restaurants in {city}"
    data = _serpapi({"engine": "google_local", "q": query})
    places = data.get("local_results") or []
    if not places:
        return f"No restaurants found for '{query}'."
    return "\n".join(
        f"- {place.get('title')} | {place.get('type', '?')} | rating {place.get('rating', '?')} "
        f"({place.get('reviews', 0)} reviews) | {place.get('price', 'price n/a')} | {place.get('address', '')}"
        for place in places[:MAX_RESTAURANTS]
    )


class LocalGuideState(TypedDict, total=False):
    city: str
    airport: str
    hotel: Optional[str]
    transfers: str
    food_tips: str
    restaurants: str
    guide: str


def _summarize(llm: ChatOpenAI, prompt: str, context: str) -> str:
    return llm.invoke([SystemMessage(content=prompt), HumanMessage(content=context)]).content


def build_local_guide_graph():
    """Fan out to the transfer, food tips, and restaurant agents in parallel, then join their sections."""
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.3)

    def transfer_agent(state: LocalGuideState) -> LocalGuideState:
        destination = f"{state['hotel']}, {state['city']}" if state.get("hotel") else f"{state['city']} city center"
        try:
            routes = fetch_airport_transfers(state["airport"], destination)
            context = f"Airport: {state['airport']}\nDestination: {destination}\n\nROUTES:\n{routes}"
            return {"transfers": _summarize(llm, TRANSFER_PROMPT, context)}
        except Exception as exc:  # degrade gracefully: the other sections still return
            return {"transfers": f"Airport transfer lookup failed: {exc}"}

    def food_tips_agent(state: LocalGuideState) -> LocalGuideState:
        try:
            snippets = fetch_food_tips(state["city"])
            return {"food_tips": _summarize(llm, FOOD_TIPS_PROMPT, f"City: {state['city']}\n\nRESULTS:\n{snippets}")}
        except Exception as exc:
            return {"food_tips": f"Food tips lookup failed: {exc}"}

    def restaurants_agent(state: LocalGuideState) -> LocalGuideState:
        try:
            places = fetch_restaurants(state["city"], state.get("hotel"))
            area = f" near {state['hotel']}" if state.get("hotel") else ""
            context = f"City: {state['city']}{area}\n\nRESTAURANTS:\n{places}"
            return {"restaurants": _summarize(llm, RESTAURANTS_PROMPT, context)}
        except Exception as exc:
            return {"restaurants": f"Restaurant search failed: {exc}"}

    def join(state: LocalGuideState) -> LocalGuideState:
        return {
            "guide": (
                f"AIRPORT TRANSFER\n{state['transfers']}\n\n"
                f"FOOD TIPS\n{state['food_tips']}\n\n"
                f"RESTAURANTS\n{state['restaurants']}"
            )
        }

    graph = StateGraph(LocalGuideState)
    graph.add_node("transfer_agent", transfer_agent)
    graph.add_node("food_tips_agent", food_tips_agent)
    graph.add_node("restaurants_agent", restaurants_agent)
    graph.add_node("join", join)

    for agent in ("transfer_agent", "food_tips_agent", "restaurants_agent"):
        graph.add_edge(START, agent)
    graph.add_edge(["transfer_agent", "food_tips_agent", "restaurants_agent"], "join")
    graph.add_edge("join", END)
    return graph.compile()


_local_guide_graph = None


class LocalGuideInput(BaseModel):
    city: str = Field(description="Destination city name, e.g. 'Paris'")
    airport: str = Field(description="Arrival airport IATA code or name, e.g. 'CDG'")
    hotel: Optional[str] = Field(
        default=None, description="Chosen hotel name, used for transfer routes and nearby restaurants"
    )


@tool("local_guide", args_schema=LocalGuideInput)
def local_guide(
    city: str,
    airport: str,
    hotel: Optional[str] = None,
    *,
    run_config: RunnableConfig,
) -> str:
    """Airport transfer options, local food tips, and restaurant recommendations for the destination."""
    global _local_guide_graph
    if _local_guide_graph is None:
        _local_guide_graph = build_local_guide_graph()
    result = _local_guide_graph.invoke({"city": city, "airport": airport, "hotel": hotel}, run_config)
    return result["guide"]
