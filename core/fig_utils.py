"""Serialize Plotly figures to plain dicts safe for JSON transport."""
import re
from core.numpy_encoder import numpy_safe

# Safari throws "The string did not match the expected pattern" whenever Plotly.js
# tries to auto-detect a string axis as dates and calls Date.parse() on the values.
# Fix: force type='category' for ANY string-valued axis so Safari never date-parses them.
# Exception: full ISO-8601 date strings (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) are left
# alone so time-series charts keep their date axis and proper tick formatting.
_FULL_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}(T.*)?$')


def _safari_safe_layout(d: dict) -> dict:
    """Walk traces and force xaxis/yaxis type='category' for any non-ISO string axis."""
    traces = d.get("data", [])
    layout = d.get("layout", {})

    needs_category_x = False
    needs_category_y = False

    for trace in traces:
        for axis_key in ("x", "y"):
            vals = trace.get(axis_key, [])
            if vals and isinstance(vals, list) and isinstance(vals[0], str):
                # Only leave as-is if it's a proper full ISO date (YYYY-MM-DD...)
                if not _FULL_DATE_RE.match(vals[0]):
                    if axis_key == "x":
                        needs_category_x = True
                    else:
                        needs_category_y = True

    if needs_category_x:
        layout.setdefault("xaxis", {})["type"] = "category"
    if needs_category_y:
        layout.setdefault("yaxis", {})["type"] = "category"

    d["layout"] = layout
    return d


def fig_to_dict(fig) -> dict | None:
    if fig is None:
        return None
    try:
        raw = fig.to_dict()
        safe = numpy_safe(raw)
        return _safari_safe_layout(safe)
    except Exception:
        return None
