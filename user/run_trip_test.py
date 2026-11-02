"""End-to-end test: a simulated curious traveler chats with the travel agent.

Run from the repository root:
    python -m user.run_trip_test

Writes user/report.html (monitoring tables) and user/report.json (raw data).
"""

import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from travel_agent import config
from travel_agent.agent import build_agent_executor
from user.monitor import MonitorCallback, estimate_cost
from user.report import render_report

MAX_TURNS = 6
END_TOKEN = "[END]"
OUTPUT_DIR = Path(__file__).parent

PERSONA_PROMPT = f"""You are role-playing a curious traveler chatting with a travel agent assistant.
Today is {datetime.now():%A, %B %d, %Y}.

Your situation: you live in Madrid and are planning a short trip to Paris,
leaving next Thursday and coming back the following Sunday. You travel alone.
You are curious and enthusiastic: you want to know about flights, a place to
stay, what to do there, what the weather will be like, how to get from the
airport to your hotel, what local food to try, and where to eat.

Rules:
- Write ONLY your next message to the agent, in English, 1-3 sentences.
- Reveal details naturally over the conversation instead of all at once.
- Ask follow-up questions about the answers you get (e.g. events, weather, budget).
- When you are satisfied (or after about {MAX_TURNS - 1} messages), say goodbye and
  end your message with {END_TOKEN}."""


def persona_reply(persona_llm: ChatOpenAI, history: list) -> tuple[str, dict]:
    response = persona_llm.invoke([SystemMessage(content=PERSONA_PROMPT), *history])
    return response.content.strip(), response.usage_metadata or {}


def run() -> dict:
    """Run the conversation. Always returns the collected data, even after a failure."""
    config.validate()
    agent = build_agent_executor()
    persona_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.7, timeout=60, max_retries=2)
    monitor = MonitorCallback()
    run_config = {"configurable": {"thread_id": f"user-test-{int(time.time())}"}, "callbacks": [monitor]}

    # From the persona's point of view the agent is the "human" speaking to it.
    persona_history = [HumanMessage(content="Hi! I'm your travel assistant. How can I help you?")]
    turns, persona_usage = [], {"input_tokens": 0, "output_tokens": 0}
    started = datetime.now()

    for turn in range(1, MAX_TURNS + 1):
        try:
            user_message, usage = persona_reply(persona_llm, persona_history)
        except Exception as exc:
            error = f"Simulated traveler (OpenAI) failed: {type(exc).__name__}: {exc}"
            turns.append({"turn": turn, "user": "", "agent": "", "latency_ms": 0.0, "error": error})
            sys.stdout.write(f"\n[{turn}] {error}\n")
            break
        persona_usage["input_tokens"] += usage.get("input_tokens", 0)
        persona_usage["output_tokens"] += usage.get("output_tokens", 0)
        is_last = END_TOKEN in user_message
        user_message = user_message.replace(END_TOKEN, "").strip()
        persona_history.append(AIMessage(content=user_message))
        sys.stdout.write(f"\n[{turn}] Traveler: {user_message}\n")

        monitor.turn = turn
        t0 = time.perf_counter()
        error = None
        try:
            result = agent.invoke({"messages": [HumanMessage(content=user_message)]}, run_config)
            reply = result["messages"][-1].content
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reply = ""
            traceback.print_exc()
        latency_ms = (time.perf_counter() - t0) * 1000

        turns.append({"turn": turn, "user": user_message, "agent": reply, "latency_ms": latency_ms, "error": error})
        sys.stdout.write(f"[{turn}] Agent ({latency_ms / 1000:.1f}s): {reply}\n")

        if is_last or error:
            break
        persona_history.append(HumanMessage(content=reply))

    spans = [
        {**asdict(span), "duration_ms": span.duration_ms, "cost_usd": span.cost_usd,
         "start_offset_ms": (span.started_at - monitor.origin) * 1000}
        for span in monitor.spans
    ]
    return {
        "started_at": started.isoformat(timespec="seconds"),
        "model": config.OPENAI_MODEL,
        "turns": turns,
        "spans": spans,
        "persona_usage": {
            **persona_usage,
            "cost_usd": estimate_cost(config.OPENAI_MODEL, persona_usage["input_tokens"], persona_usage["output_tokens"]),
        },
    }


def main() -> None:
    data = run()
    (OUTPUT_DIR / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "report.html").write_text(render_report(data), encoding="utf-8")
    sys.stdout.write(f"\nReport written to {OUTPUT_DIR / 'report.html'}\n")


if __name__ == "__main__":
    main()
