"""Merge the per-arm JSON shards of the supervision control experiment into one file
and print the summary table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

SHARDS = [
    RUNS / "_armA_partial.json",
    RUNS / "supervision_control_baseline_BC.json",
    RUNS / "supervision_control_baseline_flat.json",
]
OUT = RUNS / "supervision_control_baseline.json"

ORDER = ["A_raw", "B_randinit_kron", "C_trained", "C_trained_flat"]


def main() -> None:
    merged = None
    for p in SHARDS:
        if not p.exists():
            print(f"WARNING: missing shard {p}", file=sys.stderr)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if merged is None:
            merged = {"settings": d["settings"], "arms": {}}
            merged["settings"]["shards"] = []
        merged["settings"]["shards"].append(str(p.name))
        for arm, reps in d["arms"].items():
            merged["arms"].setdefault(arm, {}).update(reps)

    assert merged is not None, "no shards found"
    merged["settings"]["dimensionality_note"] = (
        "The linear up-projection control (raw 1024 -> 2048 Gaussian) was dropped: any linear map "
        "out of a 1024-dim space has rank <= 1024, so it cannot give a probe genuine extra capacity "
        "and it made the L2-regularised logistic solver ill-conditioned (>70 min/fit). The "
        "matched-dimension controls actually used are (i) Arm B's random-init, untrained KronSAE "
        "Q branch, a full-rank 2048-dim random-ReLU-feature expansion of the same 1024-dim input, "
        "and (ii) train-split PCA of every 2048-dim Q branch (and 16384-dim flat vector) down to "
        "1024 dims, matching the raw activation dimensionality exactly."
    )
    merged["arms"] = {k: merged["arms"][k] for k in ORDER if k in merged["arms"]}
    OUT.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    hdr = f"{'arm':<18}{'representation':<36}{'dim':>6} | {'sent logreg':>16} {'sent mlp':>9} {'sent rawAdam':>16} | {'topic logreg':>16} {'topic mlp':>9} {'topic rawAdam':>16}"
    print(hdr)
    print("-" * len(hdr))
    for arm, reps in merged["arms"].items():
        for key, v in reps.items():
            s, tp = v["labels"]["sentiment"], v["labels"]["topic"]
            def f(r, k):
                return f"{r[k]['mean']:.3f}+-{r[k]['std']:.3f}"
            print(
                f"{arm:<18}{key:<36}{s['probe_dim']:>6} | "
                f"{f(s,'std_logreg'):>16} {s['std_mlp']['mean']:>9.3f} {f(s,'raw_adam'):>16} | "
                f"{f(tp,'std_logreg'):>16} {tp['std_mlp']['mean']:>9.3f} {f(tp,'raw_adam'):>16}"
            )
    print("\nchance: sentiment 0.500, topic 0.167")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
