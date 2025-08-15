
# analysis/coverage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from analysis.sensitivity import metric_array


@dataclass(frozen=True)
class DSCRSummary:
    min_p5: float
    min_p50: float
    min_p95: float
    mean_p50: float
    breach_prob: float
    years_breached_p50: float

def dscr_summary(
    model: Any,
    *,
    threshold: float = 1.2,
) -> DSCRSummary:
    """
    Summaries for DSCR across time & iterations:
      - p5/p50/p95 of per-iteration min DSCR
      - median of per-iteration mean DSCR
      - breach_prob: share of iterations where min DSCR < threshold
      - years_breached_p50: median count of years with DSCR < threshold
    """
    dscr = getattr(model, "dscr").data  # (I, T)
    I, T = dscr.shape
    min_per_iter = np.min(dscr, axis=1)
    mean_per_iter = np.mean(dscr, axis=1)
    years_breached = np.sum(dscr < threshold, axis=1)

    return DSCRSummary(
        min_p5=float(np.percentile(min_per_iter, 5)),
        min_p50=float(np.percentile(min_per_iter, 50)),
        min_p95=float(np.percentile(min_per_iter, 95)),
        mean_p50=float(np.percentile(mean_per_iter, 50)),
        breach_prob=float(np.mean(min_per_iter < threshold)),
        years_breached_p50=float(np.percentile(years_breached, 50)),
    )
