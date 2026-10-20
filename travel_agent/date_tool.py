import re
from datetime import date, datetime

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_IN_N_UNITS_RE = re.compile(r"^in\s+(\d+)\s+(day|week|month|year)s?$")
_NEXT_WEEKDAY_RE = re.compile(r"^next\s+(" + "|".join(WEEKDAYS) + r")$")


class ConvertDateInput(BaseModel):
    text: str = Field(description="A date expressed in natural language, e.g. 'October 20th', 'tomorrow'")


@tool("convert_date", args_schema=ConvertDateInput)
def convert_date(text: str) -> str:
    """Convert a natural-language date into YYYY-MM-DD format for use with the flight search tool.

    If the text has no year, the next upcoming occurrence of that date is used
    (rolling over to next year if the date already passed this year).
    """
    today = date.today()
    normalized = text.strip().lower()

    relative_days = {"today": 0, "tomorrow": 1, "yesterday": -1}
    if normalized in relative_days:
        return (today + relativedelta(days=relative_days[normalized])).isoformat()

    in_n_units_match = _IN_N_UNITS_RE.match(normalized)
    if in_n_units_match:
        amount, unit = int(in_n_units_match.group(1)), in_n_units_match.group(2)
        return (today + relativedelta(**{f"{unit}s": amount})).isoformat()

    next_weekday_match = _NEXT_WEEKDAY_RE.match(normalized)
    if next_weekday_match:
        target = WEEKDAYS.index(next_weekday_match.group(1))
        days_ahead = (target - today.weekday()) % 7 or 7
        return (today + relativedelta(days=days_ahead)).isoformat()

    try:
        parsed = dateutil_parser.parse(
            text, default=datetime.combine(today, datetime.min.time()), fuzzy=True
        ).date()
    except (ValueError, OverflowError):
        return f"Could not parse '{text}' as a date."

    has_year = any(token.isdigit() and len(token) == 4 for token in text.split())
    if not has_year and parsed < today:
        parsed += relativedelta(years=1)

    return parsed.isoformat()
