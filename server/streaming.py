import json
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage

from travel_agent.agent import build_agent_executor

_agent_executor = build_agent_executor()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_agent_response(message: str, thread_id: str) -> AsyncIterator[str]:
    run_config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=message)]}

    async for event in _agent_executor.astream_events(inputs, run_config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield _sse("token", {"content": chunk.content})

        elif kind == "on_tool_start":
            yield _sse("tool_start", {"tool": event["name"], "input": event["data"].get("input")})

        elif kind == "on_tool_end":
            yield _sse("tool_end", {"tool": event["name"], "output": str(event["data"].get("output"))})

    yield _sse("done", {})
