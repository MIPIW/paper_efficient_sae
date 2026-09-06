"""Correctness check for the CAGrad implementation (arXiv:2110.14048, Algorithm 1).

Verifies, on manually constructed gradient triples/pairs:
  1. d stays inside the trust region:  ||d - g_0|| <= c*||g_0|| (+ float tolerance).
  2. d beats the naive average on the worst-case improvement rate:
       min_i <d, g_i>  >=  min_i <g_0, g_i>.
  3. d is the constrained optimum: no random feasible direction in the ball attains a
     larger min_i <d, g_i>.
  4. c = 0 reproduces the average gradient exactly (no-op property).
  5. d differs from the naive average whenever the tasks conflict and c > 0.

Run:  python src/check_cagrad.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dictionary_learning"))
from dictionary_learning.trainers.kron_top_k import cagrad_direction  # noqa: E402


CASES: dict[str, list[list[float]]] = {
    # Two directly opposed tasks + one neutral: the classic conflict case.
    "opposed_pair_plus_neutral": [[1.0, 0.0], [-1.0, 0.2], [0.0, 1.0]],
    # One huge task (reconstruction-like) dominating two small ones -- the regime
    # REPORT.md 9.2/9.3 measured (recon ~45-400x the supervision gradients).
    "dominant_recon": [[20.0, 0.0], [0.0, 0.4], [-0.1, 0.35]],
    # Mildly conflicting, comparable magnitudes.
    "mild_conflict": [[1.0, 0.3], [0.4, -0.9], [-0.6, 0.5]],
    # Fully aligned tasks: no conflict, CAGrad should just push along the common direction.
    "aligned": [[1.0, 0.1], [0.9, 0.15], [1.1, 0.05]],
    # Higher-dimensional conflicting triple.
    "highdim": [
        [1.0, 0.0, 0.5, -0.2],
        [-0.8, 0.6, 0.0, 0.3],
        [0.1, -0.7, 0.9, 0.4],
    ],
}


def random_feasible_dirs(g0: np.ndarray, c: float, n: int, rng) -> np.ndarray:
    radius = c * np.linalg.norm(g0)
    v = rng.normal(size=(n, g0.size))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    scale = radius * rng.random(size=(n, 1)) ** (1.0 / g0.size)
    return g0[None, :] + v * scale


def main() -> int:
    rng = np.random.default_rng(0)
    report: list[dict] = []
    ok = True
    for c in (0.1, 0.25, 0.5):
        for name, grads in CASES.items():
            G = np.asarray(grads, dtype=np.float64)
            d, info = cagrad_direction(t.tensor(G, dtype=t.float64), c)
            d = d.numpy()
            g0 = G.mean(axis=0)

            min_d = float((G @ d).min())
            min_g0 = float((G @ g0).min())
            in_ball = bool(float(np.linalg.norm(d - g0)) <= c * np.linalg.norm(g0) + 1e-8)

            cand = random_feasible_dirs(g0, c, 20000, rng)
            best_random = float((cand @ G.T).min(axis=1).max())
            optimal = bool(min_d >= best_random - 1e-8)

            improves = bool(min_d >= min_g0 - 1e-12)
            row = {
                "case": name,
                "c": c,
                "min_i<d,g_i>": round(min_d, 6),
                "min_i<g0,g_i>": round(min_g0, 6),
                "best_random_feasible": round(best_random, 6),
                "||d-g0||": round(float(np.linalg.norm(d - g0)), 6),
                "c*||g0||": round(c * float(np.linalg.norm(g0)), 6),
                "w": [round(info[f"w{i}"], 4) for i in range(G.shape[0])],
                "in_trust_region": in_ball,
                "beats_average": improves,
                "matches_constrained_optimum": optimal,
                "differs_from_average": bool(np.linalg.norm(d - g0) > 1e-9),
            }
            ok = ok and in_ball and improves and optimal
            report.append(row)

    # c = 0 must be an exact no-op.
    for name, grads in CASES.items():
        G = np.asarray(grads, dtype=np.float64)
        d, _ = cagrad_direction(t.tensor(G, dtype=t.float64), 0.0)
        same = bool(np.allclose(d.numpy(), G.mean(axis=0), atol=1e-12))
        ok = ok and same
        report.append({"case": name, "c": 0.0, "c0_equals_average": same})

    print(json.dumps(report, indent=2))
    print("\nCAGRAD CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
