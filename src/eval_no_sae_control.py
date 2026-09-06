"""Existence-proof control: can collect+suppress happen at all WITHOUT any SAE?

Motivation (see REPORT.md §9.4-9.5, REPORT1.md, .claude/documents/flow_7_plus_8.md):
every collect/suppress attempt in this project so far has been tested INSIDE an
SAE's training loop -- a shared reconstruction objective, top-k sparsity, and a
DPO-pairwise/SupCon-style contrastive supervision loss, all at once. §9.5 found
that even fully zeroing the reconstruction weight (lambda_recon=0.0) did not let
P or Q clear their random-init baselines -- P actually got WORSE (0.911 vs.
control 0.967) and Q-top collapsed (0.367 vs. control 0.973). That rules out
"reconstruction competes with supervision" as the *sole* explanation, since
removing reconstruction didn't fix it either.

This script isolates a different, more standard variable: does the DPO-cross
loss formulation itself (used everywhere else in this project) need an SAE's
machinery to work at all, or does plain, standard supervised objectives --
ordinary cross-entropy for collect, a textbook gradient-reversal-layer (GRL,
Ganin & Lempitsky 2016) discriminator for suppress -- succeed on raw
activations with NO SAE, NO shared decoder, NO top-k, NO dictionary at all?

Two fully independent small MLP heads are trained directly on the synthetic
buffer's raw 1024-dim activations:
  - head_sent: predicts sentiment (cross-entropy) -- the "collect" objective.
  - head_topic: predicts topic (cross-entropy) + adversarially suppresses
    sentiment via a GRL-fed discriminator -- "collect X while suppressing Y."
Every probe is read against a matched random-init (0 training steps) control,
per REPORT.md §9.4's mandatory standard, and uses the standardized
LogisticRegression protocol from §9.3 (not the retracted unstandardized probe).

If this succeeds cleanly (large excess-over-random-init collect accuracy for
both heads, suppress accuracy near chance), the conclusion is: decomposition
into "collect A, ignore B" IS achievable in this data -- the failure documented
throughout this project is specific to routing that objective through an SAE's
reconstruction+dictionary machinery, not a fact about the data or the task
itself. If it also fails, that reopens the question of whether the DPO-cross
loss family (used everywhere else) or something about this data prevents any
method from achieving it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import SyntheticJointActivationBuffer, standardized_logreg_accuracy  # noqa: E402


class _GradReverse(t.autograd.Function):
    """Identity forward, -lambda*grad backward (Ganin & Lempitsky 2016)."""

    @staticmethod
    def forward(ctx, x: t.Tensor, lam: float) -> t.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: t.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: t.Tensor, lam: float) -> t.Tensor:
    return _GradReverse.apply(x, lam)


class MLPHead(nn.Module):
    """Two-layer MLP: activation_dim -> hidden -> n_classes. This is the
    "representation" being probed for collect/suppress -- deliberately generic
    (no top-k, no dictionary, no reconstruction), so any collect/suppress
    result traces to the supervised objective alone."""

    def __init__(self, activation_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(activation_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_classes)

    def features(self, x: t.Tensor) -> t.Tensor:
        return self.trunk(x)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.head(self.features(x))


class Discriminator(nn.Module):
    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, feats: t.Tensor) -> t.Tensor:
        return self.net(feats)


@dataclass
class RunResult:
    sent_head_collect_sent_acc: float
    sent_head_collect_sent_ctrl: float
    topic_head_collect_topic_acc: float
    topic_head_collect_topic_ctrl: float
    topic_head_suppress_sent_acc: float
    topic_head_suppress_sent_ctrl: float
    n_train_steps: int


def _collect_eval_batch(buf: SyntheticJointActivationBuffer, n_docs: int):
    xs, sents, tops = [], [], []
    got = 0
    while got < n_docs:
        batch = next(buf)
        xs.append(batch.activations.detach().cpu())
        # activations are token-level here; buffer already pools per synthetic
        # "document" == one row, so topic/sentiment labels align 1:1 with rows.
        sents.append(batch.sentiment_labels.detach().cpu())
        tops.append(batch.topic_labels.detach().cpu())
        got += batch.activations.shape[0]
    return t.cat(xs)[:n_docs], t.cat(sents)[:n_docs], t.cat(tops)[:n_docs]


def run(
    activation_dim: int = 1024,
    num_topics: int = 6,
    num_sentiments: int = 2,
    doc_batch_size: int = 64,
    hidden: int = 128,
    total_steps: int = 3000,
    grl_lambda: float = 1.0,
    lr: float = 1e-3,
    n_eval_docs: int = 4000,
    seed: int = 42,
    skip_training: bool = False,
    disc_steps_per_iter: int = 3,
) -> RunResult:
    t.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    buf = SyntheticJointActivationBuffer(
        activation_dim=activation_dim,
        doc_batch_size=doc_batch_size,
        num_topics=num_topics,
        num_sentiments=num_sentiments,
        topic_signal_scale=1.0,
        sentiment_signal_scale=1.0,
        noise_std=1.0,
        seed=seed,
        device=device,
    )

    sent_head = MLPHead(activation_dim, hidden, num_sentiments)
    topic_head = MLPHead(activation_dim, hidden, num_topics)
    disc = Discriminator(hidden, num_sentiments)

    opt_sent = t.optim.Adam(sent_head.parameters(), lr=lr)
    opt_topic = t.optim.Adam(topic_head.parameters(), lr=lr)
    opt_disc = t.optim.Adam(disc.parameters(), lr=lr)

    n_steps = 0 if skip_training else total_steps
    for step in range(n_steps):
        batch = next(buf)
        x = batch.activations
        sentiment = batch.sentiment_labels.long()
        topic = batch.topic_labels.long()

        # Head A: plain supervised collect-sentiment. Fully independent
        # parameters from head_topic/disc -- no shared trunk, no shared loss.
        opt_sent.zero_grad()
        logits_sent = sent_head(x)
        loss_sent = F.cross_entropy(logits_sent, sentiment)
        loss_sent.backward()
        opt_sent.step()

        # Head B: collect-topic (cross-entropy) + adversarially suppress
        # sentiment via GRL into `disc`. Standard DANN practice (Ganin et al.
        # 2016) trains the discriminator to convergence relative to the
        # feature extractor and ramps lambda over training, rather than a
        # single joint step at a fixed lambda from step 0 -- a discriminator
        # that is itself weak/undertrained gives a useless adversarial signal
        # to reverse. `disc_steps_per_iter` and the lambda ramp implement that.
        for _ in range(disc_steps_per_iter):
            opt_disc.zero_grad()
            with t.no_grad():
                feats_topic_detached = topic_head.features(x)
            disc_logits_only = disc(feats_topic_detached)
            loss_disc_only = F.cross_entropy(disc_logits_only, sentiment)
            loss_disc_only.backward()
            opt_disc.step()

        progress = step / max(1, n_steps - 1)
        lam = grl_lambda * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)  # DANN-style ramp 0->grl_lambda

        opt_topic.zero_grad()
        opt_disc.zero_grad()
        feats_topic = topic_head.features(x)
        logits_topic = topic_head.head(feats_topic)
        loss_collect_topic = F.cross_entropy(logits_topic, topic)

        reversed_feats = grad_reverse(feats_topic, lam)
        disc_logits = disc(reversed_feats)
        loss_adv = F.cross_entropy(disc_logits, sentiment)

        (loss_collect_topic + loss_adv).backward()
        opt_topic.step()
        opt_disc.step()

    # --- Evaluation: standardized probes on freshly-sampled activations from
    # the SAME buffer (same seed -> same sentiment_dirs/topic_dirs QR basis).
    # A different seed would regenerate an entirely different, uncorrelated
    # random orthonormal basis (the buffer draws sentiment_dirs/topic_dirs from
    # a seed-determined QR decomposition), silently turning this into an
    # evaluate-on-the-wrong-direction bug rather than a held-out-data split.
    # Held-out-ness here comes from drawing NEW noise/label draws post-training,
    # not from a new seed.
    x_all, sent_all, top_all = _collect_eval_batch(buf, n_eval_docs)
    n_train = int(0.7 * x_all.shape[0])

    with t.no_grad():
        sent_feats = sent_head.features(x_all)
        topic_feats = topic_head.features(x_all)

    def acc(feats: t.Tensor, labels: t.Tensor) -> float:
        a, converged = standardized_logreg_accuracy(
            X_train=feats[:n_train], y_train=labels[:n_train],
            X_test=feats[n_train:], y_test=labels[n_train:],
        )
        if not converged:
            print("WARNING: probe did not converge", file=sys.stderr)
        return a

    return RunResult(
        sent_head_collect_sent_acc=acc(sent_feats, sent_all),
        sent_head_collect_sent_ctrl=float("nan"),  # filled by caller
        topic_head_collect_topic_acc=acc(topic_feats, top_all),
        topic_head_collect_topic_ctrl=float("nan"),
        topic_head_suppress_sent_acc=acc(topic_feats, sent_all),
        topic_head_suppress_sent_ctrl=float("nan"),
        n_train_steps=n_steps,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--total_steps", type=int, default=3000)
    ap.add_argument("--grl_lambda", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_eval_docs", type=int, default=4000)
    ap.add_argument("--disc_steps_per_iter", type=int, default=3)
    ap.add_argument("--output", type=str, default=str(ROOT.parent / "runs" / "no_sae_control_results.json"))
    args = ap.parse_args()

    print("Training arm...")
    trained = run(
        total_steps=args.total_steps, grl_lambda=args.grl_lambda, hidden=args.hidden,
        lr=args.lr, seed=args.seed, n_eval_docs=args.n_eval_docs, skip_training=False,
        disc_steps_per_iter=args.disc_steps_per_iter,
    )
    print("Random-init control arm (0 steps)...")
    control = run(
        total_steps=args.total_steps, grl_lambda=args.grl_lambda, hidden=args.hidden,
        lr=args.lr, seed=args.seed, n_eval_docs=args.n_eval_docs, skip_training=True,
        disc_steps_per_iter=args.disc_steps_per_iter,
    )

    summary = {
        "collect_sentiment": {"trained": trained.sent_head_collect_sent_acc, "random_init": control.sent_head_collect_sent_acc,
                               "delta": trained.sent_head_collect_sent_acc - control.sent_head_collect_sent_acc},
        "collect_topic": {"trained": trained.topic_head_collect_topic_acc, "random_init": control.topic_head_collect_topic_acc,
                           "delta": trained.topic_head_collect_topic_acc - control.topic_head_collect_topic_acc},
        "suppress_sentiment_on_topic_head": {"trained": trained.topic_head_suppress_sent_acc, "random_init": control.topic_head_suppress_sent_acc,
                                              "delta": trained.topic_head_suppress_sent_acc - control.topic_head_suppress_sent_acc},
        "chance": {"sentiment": 1.0 / 2, "topic": 1.0 / 6},
        "args": vars(args),
    }
    print(json.dumps(summary, indent=2))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
