"""Build untrained (random-init) KronSAE/flat checkpoints for the Part B random-init
control (flow_7_plus_8.md RQ5), matching the exact architecture of the trained
pilot_balanced_v2_420k / joint_dpo_cross_full420k checkpoints, so eval_absorption_hedging.py
can load them via the same trainer_config.json + ae_step_N.pt convention used for
real trained checkpoints (see src/eval_saebench_absorption.py:load_checkpoint_spec).

No training occurs here -- this is exactly what REPORT.md §9.4 calls the random-init
control: same architecture, same seed convention, zero optimizer steps.
"""
import json
import sys
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dictionary_learning"))

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.dictionary_kron import KronAutoEncoderTopK

SEED = 42
ACTIVATION_DIM = 1024
DICT_SIZE = 16384
K = 24
H, M, N = 128, 8, 16
COMBINE_RULE = "mand"

OUT_ROOT = ROOT / "runs" / "absorption_hedging_randinit" / "checkpoints"


def save(name: str, dict_class: str, cfg: dict, model: t.nn.Module) -> None:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t.save(model.state_dict(), out_dir / "ae_step_0.pt")
    full_cfg = {"dict_class": dict_class, **cfg}
    (out_dir / "trainer_config.json").write_text(json.dumps(full_cfg, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


def main() -> None:
    t.manual_seed(SEED)
    kron = KronAutoEncoderTopK(
        activation_dim=ACTIVATION_DIM, h=H, m=M, n=N, k=K, combine_rule=COMBINE_RULE,
    )
    save(
        "kron_randinit",
        "KronAutoEncoderTopK",
        {
            "activation_dim": ACTIVATION_DIM, "dict_size": H * M * N, "k": K,
            "h": H, "m": M, "n": N, "combine_rule": COMBINE_RULE, "seed": SEED,
            "note": "untrained random-init control, REPORT.md §9.4 standard",
        },
        kron,
    )

    t.manual_seed(SEED)
    flat = AutoEncoderTopK(activation_dim=ACTIVATION_DIM, dict_size=DICT_SIZE, k=K)
    save(
        "flat_randinit",
        "AutoEncoderTopK",
        {
            "activation_dim": ACTIVATION_DIM, "dict_size": DICT_SIZE, "k": K, "seed": SEED,
            "note": "untrained random-init control, REPORT.md §9.4 standard",
        },
        flat,
    )


if __name__ == "__main__":
    main()
