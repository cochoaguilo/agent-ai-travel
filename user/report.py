from html import escape
from typing import Optional

# Which external API each tool/node depends on, so failures point at a provider.
API_BY_COMPONENT = {
    "search_flights": "SerpApi · Google Flights",
    "search_hotels": "SerpApi · Google Hotels",
    "events_agent": "SerpApi · Google Events",
    "weather_agent": "Open-Meteo (geocoding + forecast/archive)",
    "recommend_activities": "SerpApi Events + Open-Meteo + OpenAI",
    "recommender": "OpenAI",
    "local_guide": "SerpApi Maps/Search/Local + OpenAI",
    "transfer_agent": "SerpApi · Google Maps Directions + OpenAI",
    "food_tips_agent": "SerpApi · Google Search + OpenAI",
    "restaurants_agent": "SerpApi · Google Local + OpenAI",
    "chat_model": "OpenAI",
    "convert_date": "local (no API)",
    "city_to_airport_code": "local (no API)",
}


PARALLEL_GROUPS = {
    "activities (events ‖ weather)": {"events_agent", "weather_agent"},
    "local guide (transfer ‖ food ‖ restaurants)": {"transfer_agent", "food_tips_agent", "restaurants_agent"},
}


def _ms(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:,.0f} ms"


def _usd(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:.5f}"


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _status_badge(status: str) -> str:
    label = {"ok": "OK", "error": "ERROR", "soft_error": "DEGRADED"}.get(status, status)
    return f'<span class="badge {escape(status)}">{label}</span>'


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{cell}</td>' for i, cell in enumerate(row)) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_report(data: dict) -> str:
    turns, spans = data["turns"], data["spans"]
    llm_spans = [s for s in spans if s["kind"] == "llm"]
    tool_spans = [s for s in spans if s["kind"] == "tool"]
    node_spans = [s for s in spans if s["kind"] == "node"]
    failures = [s for s in spans if s["status"] != "ok"]
    turn_errors = [t for t in turns if t["error"]]

    latencies = [t["latency_ms"] for t in turns if t["user"]]  # skip turns the traveler never sent
    tokens_in = sum(s["input_tokens"] for s in llm_spans)
    tokens_out = sum(s["output_tokens"] for s in llm_spans)
    costs = [s["cost_usd"] for s in llm_spans if s["cost_usd"] is not None]
    agent_cost = sum(costs) if costs else None
    persona = data["persona_usage"]

    kpis = [
        ("Turns", str(len(turns))),
        ("Total latency", f"{sum(latencies) / 1000:.1f} s"),
        ("Avg turn latency", f"{(sum(latencies) / len(latencies) / 1000) if latencies else 0:.1f} s"),
        ("P95 turn latency", f"{(_pct(latencies, 0.95) or 0) / 1000:.1f} s"),
        ("Agent tokens (in / out)", f"{tokens_in:,} / {tokens_out:,}"),
        ("Agent est. cost", _usd(agent_cost)),
        ("LLM calls", str(len(llm_spans))),
        ("Tool calls", str(len(tool_spans))),
        ("Errors / degraded", f"{len(turn_errors) + sum(s['status'] == 'error' for s in spans)} / "
                              f"{sum(s['status'] == 'soft_error' for s in spans)}"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>' for k, v in kpis)

    # Per-turn table
    turn_rows = []
    for t in turns:
        t_llm = [s for s in llm_spans if s["turn"] == t["turn"]]
        t_tools = [s for s in tool_spans if s["turn"] == t["turn"]]
        t_bad = [s for s in spans if s["turn"] == t["turn"] and s["status"] != "ok"]
        t_in = sum(s["input_tokens"] for s in t_llm)
        t_out = sum(s["output_tokens"] for s in t_llm)
        t_cost = sum(s["cost_usd"] or 0 for s in t_llm)
        llm_ms = sum(s["duration_ms"] or 0 for s in t_llm)
        status = "error" if t["error"] else ("soft_error" if t_bad else "ok")
        turn_rows.append([
            str(t["turn"]),
            escape(t["user"][:140]),
            _ms(t["latency_ms"]),
            _ms(llm_ms),
            str(len(t_llm)),
            escape(", ".join(s["name"] for s in t_tools) or "—"),
            f"{t_in:,}",
            f"{t_out:,}",
            _usd(t_cost),
            f"{len(t['agent']):,}",
            _status_badge(status),
        ])

    # Per-API health table
    api_rows = []
    for component in sorted({s["name"] for s in tool_spans + node_spans}):
        group = [s for s in tool_spans + node_spans if s["name"] == component]
        durations = [s["duration_ms"] for s in group if s["duration_ms"] is not None]
        bad = [s for s in group if s["status"] != "ok"]
        api_rows.append([
            escape(component),
            escape(API_BY_COMPONENT.get(component, "unknown")),
            str(len(group)),
            str(len(bad)),
            _ms(sum(durations) / len(durations) if durations else None),
            _ms(max(durations) if durations else None),
            _status_badge("ok" if not bad else ("error" if any(s["status"] == "error" for s in bad) else "soft_error")),
        ])
    if llm_spans:
        durations = [s["duration_ms"] for s in llm_spans if s["duration_ms"] is not None]
        bad = [s for s in llm_spans if s["status"] != "ok"]
        api_rows.append([
            "chat_model", escape(API_BY_COMPONENT["chat_model"] + f" ({data['model']})"),
            str(len(llm_spans)), str(len(bad)),
            _ms(sum(durations) / len(durations) if durations else None),
            _ms(max(durations) if durations else None),
            _status_badge("error" if bad else "ok"),
        ])

    # Parallelism check for each fan-out sub-graph
    parallel_rows = []
    for group, members in PARALLEL_GROUPS.items():
        for turn in sorted({s["turn"] for s in node_spans}):
            branch = [s for s in node_spans if s["turn"] == turn and s["name"] in members and s["ended_at"] is not None]
            if len(branch) < 2:
                continue
            start_spread = (max(s["started_at"] for s in branch) - min(s["started_at"] for s in branch)) * 1000
            overlaps = min(s["ended_at"] for s in branch) > max(s["started_at"] for s in branch)
            wall = (max(s["ended_at"] for s in branch) - min(s["started_at"] for s in branch)) * 1000
            sequential = sum(s["duration_ms"] or 0 for s in branch)
            parallel_rows.append([
                str(turn),
                escape(group),
                "<br>".join(f"{escape(s['name'])}: {_ms(s['duration_ms'])}" for s in sorted(branch, key=lambda s: s["name"])),
                _ms(start_spread), _ms(wall), _ms(sequential), _ms(sequential - wall),
                _status_badge("ok" if overlaps else "soft_error"),
            ])

    # Every span, chronologically
    span_rows = [[
        str(s["turn"]),
        escape(s["kind"]),
        escape(s["name"]),
        escape(s["node"] or "—"),
        _ms(s["start_offset_ms"]),
        _ms(s["duration_ms"]),
        f"{s['input_tokens']:,}" if s["kind"] == "llm" else "—",
        f"{s['output_tokens']:,}" if s["kind"] == "llm" else "—",
        f"{s['cached_tokens']:,}" if s["kind"] == "llm" else "—",
        _usd(s["cost_usd"]) if s["kind"] == "llm" else "—",
        escape(s["detail"][:120]),
        _status_badge(s["status"]),
    ] for s in sorted(spans, key=lambda s: s["start_offset_ms"])]

    error_rows = [
        [str(s["turn"]), escape(s["name"]), escape(API_BY_COMPONENT.get(s["name"], "unknown")),
         _status_badge(s["status"]), escape(s["error"] or "")]
        for s in failures
    ] + [
        [str(t["turn"]), "agent.invoke", "—", _status_badge("error"), escape(t["error"])] for t in turn_errors
    ]

    transcript = "".join(
        f'<div class="msg user"><b>Traveler · turn {t["turn"]}</b><p>{escape(t["user"])}</p></div>'
        f'<div class="msg agent"><b>Agent · {t["latency_ms"] / 1000:.1f}s</b><p>{escape(t["agent"] or "(no reply)")}</p></div>'
        for t in turns
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Travel Agent Monitoring</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0; --ok:#1f7a4d; --okbg:#e3f4ea;
        --err:#b42318; --errbg:#fde8e6; --warn:#8a5a00; --warnbg:#fdf1d8; --accent:#3d5afe; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --card:#1d1d20; --fg:#ececee; --muted:#9a9aa2; --line:#2e2e33;
        --ok:#6fd49c; --okbg:#173325; --err:#ff8a80; --errbg:#3a1a18; --warn:#f5c26b; --warnbg:#3a2d12; --accent:#8c9eff; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ margin:0 0 4px; font-size:24px }} h2 {{ margin:32px 0 8px; font-size:17px }}
.sub {{ color:var(--muted); margin:0 0 20px }} .note {{ color:var(--muted); font-size:13px; margin:0 0 8px }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px }}
.kpi .k {{ color:var(--muted); font-size:12px }} .kpi .v {{ font-size:20px; font-weight:600; font-variant-numeric:tabular-nums }}
.scroll {{ overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px }}
table {{ border-collapse:collapse; width:100%; font-size:13px }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top }}
th {{ color:var(--muted); font-weight:600; white-space:nowrap; position:sticky; top:0; background:var(--card) }}
td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap }}
tbody tr:last-child td {{ border-bottom:0 }}
.badge {{ display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px; font-weight:700 }}
.badge.ok {{ color:var(--ok); background:var(--okbg) }} .badge.error {{ color:var(--err); background:var(--errbg) }}
.badge.soft_error {{ color:var(--warn); background:var(--warnbg) }}
details {{ margin-top:32px }} summary {{ cursor:pointer; font-weight:600 }}
.msg {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin:8px 0 }}
.msg.user {{ border-left:3px solid var(--accent) }} .msg p {{ margin:4px 0 0; white-space:pre-wrap }}
</style></head><body><main>
<h1>Travel Agent Monitoring</h1>
<p class="sub">Simulated traveler session · {escape(data['started_at'])} · model {escape(data['model'])} ·
persona overhead (excluded from agent totals): {persona['input_tokens']:,} in / {persona['output_tokens']:,} out, {_usd(persona['cost_usd'])}</p>
<div class="kpis">{kpi_html}</div>

<h2>Per turn</h2>
<p class="note">Latency is the wall-clock time of one agent.invoke; LLM time is the sum of model calls in that turn (tools account for the rest).</p>
{_table(["#", "Traveler message", "Latency", "LLM time", "LLM calls", "Tools used", "Tokens in", "Tokens out", "Est. cost", "Reply chars", "Status"], turn_rows, {0, 2, 3, 4, 6, 7, 8, 9})}

<h2>API / component health</h2>
<p class="note">DEGRADED means the call returned a handled failure (e.g. "Event search failed") instead of raising.</p>
{_table(["Component", "External API", "Calls", "Failures", "Avg latency", "Max latency", "Status"], api_rows, {2, 3, 4, 5})}

<h2>Errors</h2>
{_table(["Turn", "Component", "External API", "Severity", "Message"], error_rows) if error_rows else '<p class="note">No errors recorded.</p>'}

<h2>Parallel sub-agents</h2>
<p class="note">Start spread = gap between the first and last branch starting (≈0 means they started together). Saved = sequential time minus actual wall time.</p>
{_table(["Turn", "Sub-graph", "Branch durations", "Start spread", "Wall time", "If sequential", "Saved", "Ran in parallel"], parallel_rows, {0, 3, 4, 5, 6}) if parallel_rows else '<p class="note">No parallel sub-graph was called in this session.</p>'}

<h2>All spans (chronological)</h2>
{_table(["Turn", "Kind", "Name", "Graph node", "Start", "Duration", "In", "Out", "Cached", "Cost", "Detail", "Status"], span_rows, {0, 4, 5, 6, 7, 8, 9})}

<details><summary>Conversation transcript</summary>{transcript}</details>
</main></body></html>
"""
