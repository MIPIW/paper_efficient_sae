import sys
sys.path.insert(0, "src")
import torch as t
import numpy as np
from sklearn.linear_model import LogisticRegression
from eval_concept_erasure_control import leace_fit, leace_erase

def mean_pool_by_doc(x, ids, n_docs):
    out = t.zeros(n_docs, x.shape[-1])
    counts = t.zeros(n_docs)
    out.index_add_(0, ids, x)
    counts.index_add_(0, ids, t.ones_like(ids, dtype=t.float))
    return out / counts.clamp_min(1).unsqueeze(-1)

batches = t.load("runs/pythia_erasure_activation_cache.pt", map_location="cpu")
x_list, sent_list = [], []
for b in batches:
    x = b["x"].to(dtype=t.float32)
    ids = b["token_doc_ids"]
    pooled = mean_pool_by_doc(x, ids, b["n_docs"]).float()
    x_list.append(pooled)
    sent_list.append(b["sentiment"])
x_all = t.cat(x_list)[:20000]
sent_all = t.cat(sent_list)[:20000]

n_fit = 10000
x_fit, sent_fit = x_all[:n_fit], sent_all[:n_fit]
x_probe, sent_probe = x_all[n_fit:], sent_all[n_fit:]

fit = leace_fit(x_fit, sent_fit)
x_probe_erased = leace_erase(x_probe, fit)
x_fit_erased = leace_erase(x_fit, fit)

n_split = int(0.65 * x_probe_erased.shape[0])
x_tr, x_te = x_probe_erased[:n_split], x_probe_erased[n_split:]
y_tr, y_te = sent_probe[:n_split], sent_probe[n_split:]

mu = x_tr.mean(dim=0, keepdim=True)
sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
x_tr_std = ((x_tr - mu) / sd).numpy()
x_te_std = ((x_te - mu) / sd).numpy()
y_tr_np, y_te_np = y_tr.numpy(), y_te.numpy()

print(f"raw (unerased) sentiment acc, C=1.0 baseline for reference:")
mu_raw = x_probe[:n_split].mean(dim=0, keepdim=True)
sd_raw = x_probe[:n_split].std(dim=0, keepdim=True).clamp_min(1e-6)
clf_raw = LogisticRegression(max_iter=5000, C=1.0).fit(
    ((x_probe[:n_split] - mu_raw) / sd_raw).numpy(), y_tr_np)
acc_raw = clf_raw.score(((x_probe[n_split:] - mu_raw) / sd_raw).numpy(), y_te_np)
print(f"  raw acc={acc_raw:.4f}")

print(f"\nchance = {max(np.bincount(y_te_np))/len(y_te_np):.4f} (majority-class baseline, since sentiment is imbalanced)")
print(f"erased-sentiment accuracy vs regularization strength C:")
for C in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001):
    clf = LogisticRegression(max_iter=5000, C=C).fit(x_tr_std, y_tr_np)
    acc = clf.score(x_te_std, y_te_np)
    n_iter = int(np.max(clf.n_iter_))
    print(f"  C={C:8.4f}  acc={acc:.4f}  n_iter={n_iter}  converged={n_iter < 5000}")

print("\nAlso: majority-class-only baseline accuracy on this test split (sanity floor):")
maj = max(np.bincount(y_te_np)) / len(y_te_np)
print(f"  majority_acc={maj:.4f}")
