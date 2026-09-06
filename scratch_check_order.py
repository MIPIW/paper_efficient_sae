import torch as t

batches = t.load("runs/pythia_erasure_activation_cache.pt", map_location="cpu")
top_list, sent_list = [], []
for b in batches:
    top_list.append(b["topic"])
    sent_list.append(b["sentiment"])
top_all = t.cat(top_list)
sent_all = t.cat(sent_list)
print("total docs:", top_all.shape[0])
print("first 60 topic labels:", top_all[:60].tolist())
print("topic labels 9900:9960:", top_all[9900:9960].tolist())
print("topic labels 19900:19960:", top_all[19900:19960].tolist())
print("topic bincount first half:", t.bincount(top_all[:10000]))
print("topic bincount second half:", t.bincount(top_all[10000:20000]))
print("sentiment bincount first half:", t.bincount(sent_all[:10000]))
print("sentiment bincount second half:", t.bincount(sent_all[10000:20000]))
