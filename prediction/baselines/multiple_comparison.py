"""
multiple_comparison.py - koreksi family-wise error rate untuk p-value DM.

Menyediakan Holm (1979) step-down. Dependency-free (tidak butuh statsmodels);
hasil identik dgn statsmodels.stats.multitest.multipletests(method='holm').

References
----------
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  Scandinavian Journal of Statistics, 6(2), 65-70.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def holm_bonferroni(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Holm (1979) step-down FWER correction.

    p_value NaN (mis. DM gagal fit) DIKECUALIKAN dari family dan tetap NaN.

    Returns
    -------
    (adjusted_p, reject_flags) sejajar dgn input. reject = adjusted_p < alpha.
    """
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    idx_valid = np.where(valid)[0]
    m = idx_valid.size
    adj = np.full(p.shape, np.nan)
    if m == 0:
        return adj, np.zeros(p.shape, dtype=bool)
    order = idx_valid[np.argsort(p[idx_valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    reject = np.where(valid, adj < alpha, False)
    return adj, reject
