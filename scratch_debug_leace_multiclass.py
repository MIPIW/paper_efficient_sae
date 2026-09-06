import sys
sys.path.insert(0, "src")
import torch as t
import numpy as np
from sklearn.linear_model import LogisticRegression
from eval_concept_erasure_control import leace_fit, leace_erase

batches = t.load("runs/pythia_erasure_activation_cache.pt", map_location="cpu")

# need pooling helper without importing dictionary_learning.trainers (broken path);
# just reimplement mean pooling directly here.
def mean_pool_by_doc(x, ids, n_docs):
    out = t.zeros(n_docs, x.shape[-1])
    counts = t.zeros(n_docs)
    out.index_add_(0, ids, x)
    counts.index_add_(0, ids, t.ones_like(ids, dtype=t.float))
    return out / counts.clamp_min(1).unsqueeze(-1)

x_list, top_list = [], []
for b in batches:
    x = b["x"].to(dtype=t.float32)
    ids = b["token_doc_ids"]
    pooled = mean_pool_by_doc(x, ids, b["n_docs"]).float()
    x_list.append(pooled)
    top_list.append(b["topic"])
x_all = t.cat(x_list)[:20000]
top_all = t.cat(top_list)[:20000]

n_fit = 10000
x_fit, y_fit = x_all[:n_fit], top_all[:n_fit]
print("y_fit bincount:", t.bincount(y_fit))

fit = leace_fit(x_fit, y_fit)
print("rank:", fit["rank"])
x_erased = leace_erase(x_fit, fit)

# sanity: cross-covariance should be ~0 in-sample
z_onehot = t.nn.functional.one_hot(y_fit.long(), num_classes=6).float()
z_onehot -= z_onehot.mean(dim=0, keepdim=True)
xc_erased = x_erased - x_erased.mean(dim=0, keepdim=True)
cross_cov = (xc_erased.T @ z_onehot) / (x_fit.shape[0]-1)
print("max abs cross-cov after erasure (should be ~0):", cross_cov.abs().max().item())
print("max abs cross-cov BEFORE erasure (raw x_fit):", (( (x_fit - x_fit.mean(0,keepdim=True)).T @ z_onehot)/(x_fit.shape[0]-1)).abs().max().item())

# now the actual probe check, replicated manually with prints
n_split = int(0.8 * x_erased.shape[0])
x_tr, x_te = x_erased[:n_split], x_erased[n_split:]
y_tr, y_te = y_fit[:n_split], y_fit[n_split:]
print("train bincount:", t.bincount(y_tr))
print("test bincount:", t.bincount(y_te))

mu = x_tr.mean(dim=0, keepdim=True)
sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
x_tr_std = ((x_tr - mu)/sd).numpy()
x_te_std = ((x_te - mu)/sd).numpy()
clf = LogisticRegression(max_iter=5000, C=1.0).fit(x_tr_std, y_tr.numpy())
preds = clf.predict(x_te_std)
acc = (preds == y_te.numpy()).mean()
print("test acc:", acc)
print("pred bincount:", np.bincount(preds, minlength=6))
print("n_iter:", clf.n_iter_)
