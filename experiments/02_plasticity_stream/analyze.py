"""Summarize and plot exp 02 results.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("results.json")); cfgs = ["sgd_lr0.01", "sgd_lr0.1", "adam_lr1e-3"]; seeds = R["config"]["seeds"]
C = {"sgd_lr0.01": "#42a5f5", "sgd_lr0.1": "#4caf50", "adam_lr1e-3": "#e53935"}
def arr(c, k): return np.array([[d[k] for d in R[f"{c}|{s}"]["stream"]] for s in seeds])
def smooth(x, w=10): return np.convolve(x, np.ones(w) / w, mode="valid")
summ = {}
for c in cfgs:
    te = arr(c, "test"); T = te.shape[1]
    fresh = np.array([[d["test"] for d in R[f"{c}|{s}"]["fresh"]] for s in seeds])
    peak_idx = smooth(te.mean(0), 20).argmax()
    summ[c] = dict(first10=te[:, :10].mean(), peak20=smooth(te.mean(0), 20).max(), peak_at=int(peak_idx + 10),
                   last20=te[:, -20:].mean(), last20_sd=te[:, -20:].mean(1).std(), fresh=fresh.mean(), fresh_sd=fresh.mean(1).std(),
                   dead_first10=arr(c, "dead")[:, :10].mean(), dead_last20=arr(c, "dead")[:, -20:].mean(),
                   srank_first10=arr(c, "srank")[:, :10].mean(), srank_last20=arr(c, "srank")[:, -20:].mean(),
                   wnorm_first=arr(c, "wnorm")[:, 0].mean(), wnorm_last=arr(c, "wnorm")[:, -1].mean(),
                   train_last20=arr(c, "train")[:, -20:].mean())
    print(c, {k: round(float(v), 3) for k, v in summ[c].items()})
json.dump({c: {k: float(v) for k, v in d.items()} for c, d in summ.items()}, open("summary.json", "w"), indent=1)
fig, ax = plt.subplots(2, 2, figsize=(12, 8)); ax = ax.ravel()
for c in cfgs:
    for i, (k, title) in enumerate([("test", "New-task test acc after training (10-task avg)"), ("dead", "Dead hidden units (fraction)"),
                                    ("srank", "Effective rank, last hidden layer (of 256)"), ("wnorm", "Total weight norm")]):
        a = arr(c, k).mean(0); x = np.arange(len(a)) + 1
        if k == "test":
            ax[i].plot(x[9:], smooth(a), c=C[c], label=c); ax[i].axhline(summ[c]["fresh"], c=C[c], ls="--", lw=1)
        else: ax[i].plot(x, a, c=C[c], label=c)
        ax[i].set(title=title, xlabel="task")
ax[0].legend(title="dashed = fresh-init net"); plt.tight_layout(); plt.savefig("plot.png", dpi=120)
