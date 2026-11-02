import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per 1M tokens (input, output). Unknown models report cost as None.
MODEL_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

# Tool outputs that signal a handled failure instead of a raised exception.
SOFT_ERROR_MARKERS = ("failed", "could not", "no known airport", "no flights found", "no hotels found")

TRACKED_NODES = {
    "events_agent",
    "weather_agent",
    "recommender",
    "transfer_agent",
    "food_tips_agent",
    "restaurants_agent",
}


def estimate_cost(model: Optional[str], input_tokens: int, output_tokens: int) -> Optional[float]:
    if not model:
        return None
    for name, (price_in, price_out) in sorted(MODEL_PRICING.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(name):
            return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return None


@dataclass
class Span:
    kind: str  # "llm" | "tool" | "node"
    name: str
    turn: int
    started_at: float
    node: Optional[str] = None
    ended_at: Optional[float] = None
    status: str = "ok"  # "ok" | "error" | "soft_error"
    error: Optional[str] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    tool_calls_requested: int = 0
    detail: str = ""

    @property
    def duration_ms(self) -> Optional[float]:
        return None if self.ended_at is None else (self.ended_at - self.started_at) * 1000

    @property
    def cost_usd(self) -> Optional[float]:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


@dataclass(eq=False)  # callbacks must stay hashable: LangChain merges handlers in a set
class MonitorCallback(BaseCallbackHandler):
    """Records latency, tokens, and errors for every LLM call, tool call, and activities node."""

    spans: list[Span] = field(default_factory=list)
    turn: int = 0
    origin: float = field(default_factory=time.perf_counter)
    _open: dict[UUID, Span] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _start(self, run_id: UUID, span: Span) -> None:
        with self._lock:
            self._open[run_id] = span
            self.spans.append(span)

    def _finish(self, run_id: UUID) -> Optional[Span]:
        with self._lock:
            span = self._open.pop(run_id, None)
        if span is not None:
            span.ended_at = time.perf_counter()
        return span

    # LLM calls -----------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs: Any) -> None:
        metadata = metadata or {}
        params = kwargs.get("invocation_params") or {}
        self._start(
            run_id,
            Span(
                kind="llm",
                name="chat_model",
                turn=self.turn,
                started_at=time.perf_counter(),
                node=metadata.get("langgraph_node"),
                model=params.get("model_name") or params.get("model") or metadata.get("ls_model_name"),
                detail=f"{sum(len(batch) for batch in messages)} prompt messages",
            ),
        )

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs: Any) -> None:
        span = self._finish(run_id)
        if span is None:
            return
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        span.input_tokens = usage.get("input_tokens", 0)
        span.output_tokens = usage.get("output_tokens", 0)
        span.cached_tokens = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        tool_calls = getattr(message, "tool_calls", None) or []
        span.tool_calls_requested = len(tool_calls)
        if tool_calls:
            span.detail = "requested: " + ", ".join(call["name"] for call in tool_calls)
        else:
            span.detail = "final answer"

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs: Any) -> None:
        span = self._finish(run_id)
        if span is not None:
            span.status, span.error = "error", repr(error)

    # Tool calls ----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._start(
            run_id,
            Span(
                kind="tool",
                name=name,
                turn=self.turn,
                started_at=time.perf_counter(),
                node=(metadata or {}).get("langgraph_node"),
                detail=str(kwargs.get("inputs") or input_str)[:200],
            ),
        )

    def on_tool_end(self, output: Any, *, run_id, **kwargs: Any) -> None:
        span = self._finish(run_id)
        if span is None:
            return
        text = str(getattr(output, "content", output))
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if any(marker in first_line.lower() for marker in SOFT_ERROR_MARKERS):
            span.status, span.error = "soft_error", first_line[:200]

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs: Any) -> None:
        span = self._finish(run_id)
        if span is not None:
            span.status, span.error = "error", repr(error)[:300]

    # Activities sub-graph nodes (to prove the parallel fan-out) ----------
    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs: Any) -> None:
        name = kwargs.get("name")
        if name in TRACKED_NODES and (metadata or {}).get("langgraph_node") == name:
            self._start(run_id, Span(kind="node", name=name, turn=self.turn, started_at=time.perf_counter(), node=name))

    def on_chain_end(self, outputs, *, run_id, **kwargs: Any) -> None:
        span = self._finish(run_id)
        if span is None:
            return
        text = " ".join(str(v) for v in outputs.values()) if isinstance(outputs, dict) else str(outputs)
        if "failed" in text.lower()[:200]:
            span.status, span.error = "soft_error", text[:200]

    def on_chain_error(self, error: BaseException, *, run_id, **kwargs: Any) -> None:
        span = self._finish(run_id)
        if span is not None:
            span.status, span.error = "error", repr(error)[:300]
