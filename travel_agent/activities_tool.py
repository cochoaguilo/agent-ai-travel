from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from travel_agent import config
from travel_agent.events_tool import fetch_events
from travel_agent.weather_tool import fetch_weather

RECOMMENDER_PROMPT = """You are a local activities expert. Using ONLY the events
and weather data provided, recommend activities for the traveler's stay.

- Match each recommendation to a specific day when the weather data allows it.
- Prefer indoor events or activities on rainy, stormy, or very cold days, and
  outdoor ones on dry, mild days.
- Only mention events that appear in the events list; never invent events,
  venues, or dates. You may add general well-known sights of the city as
  weather-appropriate fillers, clearly labeled as general suggestions.
- If the events or weather data reports an error, say so briefly and work
  with whatever data is available.
- Keep it concise: a short weather summary, then a day-by-day plan."""


class ActivitiesState(TypedDict, total=False):
    city: str
    start_date: Optional[str]
    end_date: Optional[str]
    events: str
    weather: str
    recommendation: str


def _events_agent(state: ActivitiesState) -> ActivitiesState:
    try:
        return {"events": fetch_events(state["city"], state.get("start_date"))}
    except Exception as exc:  # degrade gracefully: the recommender still runs
        return {"events": f"Event search failed: {exc}"}


def _weather_agent(state: ActivitiesState) -> ActivitiesState:
    try:
        return {"weather": fetch_weather(state["city"], state.get("start_date"), state.get("end_date"))}
    except Exception as exc:
        return {"weather": f"Weather lookup failed: {exc}"}


def _build_recommender():
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.3)

    def recommender(state: ActivitiesState) -> ActivitiesState:
        dates = f"{state.get('start_date') or 'unspecified'} to {state.get('end_date') or 'unspecified'}"
        context = (
            f"City: {state['city']}\nTravel dates: {dates}\n\n"
            f"EVENTS:\n{state['events']}\n\nWEATHER:\n{state['weather']}"
        )
        response = llm.invoke([SystemMessage(content=RECOMMENDER_PROMPT), HumanMessage(content=context)])
        return {"recommendation": response.content}

    return recommender


def build_activities_graph():
    """Fan out to the events and weather agents in parallel, then join in the recommender."""
    graph = StateGraph(ActivitiesState)
    graph.add_node("events_agent", _events_agent)
    graph.add_node("weather_agent", _weather_agent)
    graph.add_node("recommender", _build_recommender())

    graph.add_edge(START, "events_agent")
    graph.add_edge(START, "weather_agent")
    graph.add_edge(["events_agent", "weather_agent"], "recommender")
    graph.add_edge("recommender", END)
    return graph.compile()


_activities_graph = None


class RecommendActivitiesInput(BaseModel):
    city: str = Field(description="Destination city name, e.g. 'Paris'")
    start_date: Optional[str] = Field(default=None, description="First day of the stay in YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="Last day of the stay in YYYY-MM-DD format")


@tool("recommend_activities", args_schema=RecommendActivitiesInput)
def recommend_activities(
    city: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    run_config: RunnableConfig,
) -> str:
    """Recommend activities in the destination city based on live local events and the weather forecast."""
    global _activities_graph
    if _activities_graph is None:
        _activities_graph = build_activities_graph()
    result = _activities_graph.invoke(
        {"city": city, "start_date": start_date, "end_date": end_date}, run_config
    )
    return result["recommendation"]
