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

## Deploy the API to Google Cloud Run

The HTTP API is served by `server.main:app` and streams responses from `POST /chat/stream`.
The container listens on Cloud Run's `PORT` environment variable. The `.env` file is
excluded from the build context; configure API keys as Secret Manager secrets instead.

### Prerequisites

- Google Cloud CLI installed and authenticated with `gcloud auth login`.
- A Google Cloud project with billing enabled.
- Permissions to enable Cloud Run, Cloud Build, and Artifact Registry APIs, deploy
  Cloud Run services, and act as the build/runtime service accounts.
- Secret Manager secrets named `openai-api-key` and `serpapi-api-key`, each containing
  the corresponding API key. Grant the Cloud Run service identity permission to access
  these secrets (`roles/secretmanager.secretAccessor`).

Enable the APIs (once per project):

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

In the Google Cloud Console, create Secret Manager secrets named `openai-api-key`
and `serpapi-api-key` and add the API keys as secret versions. Do not put keys in
the Docker image, source repository, or deployment command.

Create a dedicated runtime identity and allow it to read only those two secrets:

```bash
gcloud iam service-accounts create travel-agent-runtime --display-name="Travel Agent Cloud Run"
gcloud secrets add-iam-policy-binding openai-api-key --member="serviceAccount:travel-agent-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding serpapi-api-key --member="serviceAccount:travel-agent-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

Deploy from the repository root, replacing the project and region values:

```bash
gcloud run deploy travel-agent --source . --project YOUR_PROJECT_ID --region YOUR_REGION --service-account travel-agent-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com --min-instances 0 --max-instances 1 --set-env-vars OPENAI_MODEL=gpt-4o-mini --set-secrets OPENAI_API_KEY=openai-api-key:latest,SERPAPI_API_KEY=serpapi-api-key:latest
```

Pick a region eligible for the Cloud Run free tier (for example `us-central1`).
`--min-instances 0` lets the service scale to zero so idle time is not billed.
`--max-instances 1` caps cost and keeps every request on the same instance, which
the in-memory conversation store (`InMemorySaver`) requires. Conversation history
is lost when the instance scales to zero or is restarted; switch to a persistent
LangGraph checkpointer if threads must survive that.

To restrict browser callers, set `ALLOWED_ORIGINS` to a comma-separated list of
origins (for example `--set-env-vars OPENAI_MODEL=gpt-4o-mini,ALLOWED_ORIGINS=https://app.example.com`).
It defaults to `*`. `GET /health` returns `{"status": "ok"}` for smoke checks.

Cloud Run requires authentication by default. Keep it that way unless the API is
intended to be publicly callable; unauthenticated access can allow others to incur
OpenAI and SerpApi costs. If you choose to make it public, add
`--allow-unauthenticated` deliberately. The deployed service URL is printed when
the deployment completes.

To test the deployed endpoint, send a JSON request to
`https://YOUR_SERVICE_URL/chat/stream` with `message` and an optional `thread_id`.
The endpoint returns server-sent events (`text/event-stream`).
