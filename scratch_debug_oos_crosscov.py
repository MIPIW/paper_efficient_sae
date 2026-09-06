import sys
sys.path.insert(0, "src")
import torch as t
from eval_concept_erasure_control import leace_fit, leace_erase

def mean_pool_by_doc(x, ids, n_docs):
    out = t.zeros(n_docs, x.shape[-1])
    counts = t.zeros(n_docs)
    out.index_add_(0, ids, x)
    counts.index_add_(0, ids, t.ones_like(ids, dtype=t.float))
    return out / counts.clamp_min(1).unsqueeze(-1)

batches = t.load("runs/pythia_erasure_activation_cache.pt", map_location="cpu")
x_list, top_list, sent_list = [], [], []
for b in batches:
    x = b["x"].to(dtype=t.float32)
    ids = b["token_doc_ids"]
    pooled = mean_pool_by_doc(x, ids, b["n_docs"]).float()
    x_list.append(pooled)
    top_list.append(b["topic"])
    sent_list.append(b["sentiment"])
x_all = t.cat(x_list)[:20000]
top_all = t.cat(top_list)[:20000]
sent_all = t.cat(sent_list)[:20000]

n_fit = 10000
x_fit, top_fit, sent_fit = x_all[:n_fit], top_all[:n_fit], sent_all[:n_fit]
x_probe, top_probe, sent_probe = x_all[n_fit:], top_all[n_fit:], sent_all[n_fit:]

def cross_cov_magnitude(x, y, n_classes):
    if n_classes <= 2:
        z = y.float().unsqueeze(-1)
    else:
        z = t.nn.functional.one_hot(y.long(), num_classes=n_classes).float()
    z = z - z.mean(dim=0, keepdim=True)
    xc = x - x.mean(dim=0, keepdim=True)
    cc = (xc.T @ z) / (x.shape[0] - 1)
    return cc.abs().max().item(), cc.norm().item()

for name, y_fit, y_probe, n_classes in (("sentiment", sent_fit, sent_probe, 2), ("topic", top_fit, top_probe, 6)):
    fit = leace_fit(x_fit, y_fit)
    x_fit_erased = leace_erase(x_fit, fit)
    x_probe_erased = leace_erase(x_probe, fit)

    max_abs_in, norm_in = cross_cov_magnitude(x_fit_erased, y_fit, n_classes)
    max_abs_oos, norm_oos = cross_cov_magnitude(x_probe_erased, y_probe, n_classes)
    max_abs_raw, norm_raw = cross_cov_magnitude(x_probe, y_probe, n_classes)

    print(f"=== {name} (rank={fit['rank']}) ===")
    print(f"  cross-cov on x_probe BEFORE erasure:            max_abs={max_abs_raw:.4f} norm={norm_raw:.4f}")
    print(f"  cross-cov on x_fit  AFTER erasure (in-sample):   max_abs={max_abs_in:.6f} norm={norm_in:.6f}")
    print(f"  cross-cov on x_probe AFTER erasure (out-of-samp):max_abs={max_abs_oos:.4f} norm={norm_oos:.4f}")
    print(f"  ratio oos/raw (fraction of raw cross-cov surviving): {norm_oos/max(norm_raw,1e-12):.4f}")
