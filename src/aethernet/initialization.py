"""Bootstrap a position without GNSS: Gauss-Newton multilateration against surveyed anchors."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def multilaterate(anchors_xyz: Sequence[Sequence[float]], ranges: Sequence[float], own_alt: float,
                  sigma: float = 0.15, x0: Optional[Sequence[float]] = None, iters: int = 25
                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Horizontal position + 2x2 covariance from >=3 slant ranges to surveyed anchors (known own altitude).

    Returns None when geometry is degenerate (collinear anchors / too few ranges).  With 3 anchors there can be a
    mirror ambiguity; pass ``x0`` (any prior guess on the correct side) to select the branch.
    """
    A = np.asarray(anchors_xyz, float)
    r = np.asarray(ranges, float)
    if A.ndim != 2 or A.shape[0] < 3 or A.shape[0] != r.shape[0]:
        return None
    c = A[:, :2] - A[:, :2].mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    if sv[0] <= 0 or sv[-1] / sv[0] < 0.1:      # (near-)collinear anchors: mirror ambiguity, refuse
        return None
    dz = own_alt - A[:, 2]
    r_h2 = r * r - dz * dz
    if np.any(r_h2 < 0):                       # noise can push slightly negative when nearly overhead
        r_h2 = np.maximum(r_h2, 0.0)
    rh = np.sqrt(r_h2)
    p = np.asarray(x0, float).copy() if x0 is not None else A[:, :2].mean(axis=0) + 1.0
    for _ in range(iters):
        d = p - A[:, :2]
        n = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        J = d / n[:, None]
        res = rh - n
        JTJ = J.T @ J
        if np.linalg.cond(JTJ) > 1e6:
            return None
        step = np.linalg.solve(JTJ, J.T @ res)
        p = p + step
        if np.linalg.norm(step) < 1e-4:
            break
    d = p - A[:, :2]
    n = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    J = d / n[:, None]
    JTJ = J.T @ J
    if np.linalg.cond(JTJ) > 1e6:
        return None
    return p, np.linalg.inv(JTJ) * sigma ** 2
