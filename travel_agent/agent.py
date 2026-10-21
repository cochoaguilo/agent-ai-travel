from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from travel_agent import config
from travel_agent.airport_tool import city_to_airport_code
from travel_agent.date_tool import convert_date
from travel_agent.flight_tool import search_flights

SYSTEM_PROMPT = """You are a helpful travel agent assistant specialized in flights.

When the user wants to search for flights, you need origin, destination, and a date
 before calling the search_flights tool. If any of
these are missing, ask the user for them instead of guessing.
Don't ask for the IATA code. Whenever the user gives a city name instead of an
IATA code for origin or destination, call the city_to_airport_code tool to resolve
it before calling search_flights. Never guess or invent an IATA code yourself.

Whenever the user gives a date in natural language (e.g. "October 20th",
"next Friday", "tomorrow"), call the convert_date tool to turn it into
YYYY-MM-DD before calling search_flights. Never convert dates yourself.

Only report flight details that come from the search_flights tool results.
Never invent prices, airlines, or schedules."""


def build_agent_executor():
    config.validate()

    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
    tools = [search_flights, convert_date, city_to_airport_code]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=InMemorySaver())
