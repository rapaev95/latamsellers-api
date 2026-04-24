"""Normalize cell values that often arrive as Excel floats (MLB, SKU, etc.)."""
from __future__ import annotations

import math
from typing import Any


def excel_scalar_to_clean_str(val: Any) -> str:
    """4516196937.0 → '4516196937'; 'MLB123' unchanged; NaN/None → ''."""
    if val is None:
        return ""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return ""
        if val == int(val):
            return str(int(val))
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    if "." in s and "e" not in s.lower() and "E" not in s:
        head, _, tail = s.partition(".")
        if head.lstrip("-").isdigit() and tail.strip("0") == "":
            return head
    return s
