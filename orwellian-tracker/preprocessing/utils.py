from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


DECADE_ORDER = [f"{year}s" for year in range(1870, 2020, 10)]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def year_to_decade(year: int) -> str:
    if year < 1870:
        year = 1870
    if year > 2019:
        year = 2010
    return f"{(year // 10) * 10}s"


def decade_to_index(decade: str) -> int:
    try:
        return DECADE_ORDER.index(decade) + 1
    except ValueError:
        return math.nan


def safe_corr(xs: Iterable[float], ys: Iterable[float]) -> float:
    import numpy as np

    xa = np.asarray(list(xs), dtype=float)
    ya = np.asarray(list(ys), dtype=float)
    mask = ~np.isnan(xa) & ~np.isnan(ya)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(xa[mask], ya[mask])[0, 1])
