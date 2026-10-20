# Agent AI Travel

A conversational flight-search assistant built with LangChain and LangGraph. The agent chats with the user, extracts origin, destination, and travel dates from natural language, and queries live flight data to answer with real prices, airlines, and schedules — never invented ones.

## How it works

- **`travel_agent/agent.py`** — builds a LangGraph state graph around a `ChatOpenAI` model bound to two tools (`search_flights`, `convert_date`). The system prompt instructs the model to ask for missing trip details and to only report data returned by the tools.
- **`travel_agent/date_tool.py`** — converts natural-language dates (`"tomorrow"`, `"next Friday"`, `"in 2 weeks"`, `"October 20th"`) into `YYYY-MM-DD`.
- **`travel_agent/flight_tool.py`** — queries the [SerpApi Google Flights](https://serpapi.com/google-flights-api) endpoint and summarizes results (airline, price, duration, stops) with basic price statistics via pandas/numpy.
- **`travel_agent/config.py`** — loads and validates required environment variables.
- **`main.py`** — simple CLI loop to chat with the agent.

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [SerpApi](https://serpapi.com/) API key

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your-openai-key
SERPAPI_API_KEY=your-serpapi-key
OPENAI_MODEL=gpt-4o-mini
```

## Usage

```bash
python main.py
```

Type your travel request (e.g. "flights from Madrid to Buenos Aires next Friday") and type `exit` or `quit` to leave.
