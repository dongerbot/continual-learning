"""Exp 01: Permuted-MNIST baseline. 10 tasks, MLP 784-256-256-10, SGD, 1 epoch/task.
Methods: naive sequential fine-tuning, EWC (diag Fisher, lambda=50), experience replay (1000-sample reservoir buffer).
Also: plasticity probe = accuracy on each new task right after learning it, vs a fresh-init net on that task."""
import json, time, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
torch.set_num_threads(6)
root = "../../data"
tr, te = MNIST(root, download=True), MNIST(root, train=False, download=True)
Xtr = tr.data.float().view(-1, 784) / 255; Ytr = tr.targets; Xte = te.data.float().view(-1, 784) / 255; Yte = te.targets
T, BS, LR = 10, 128, 0.05
def net(): return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
def acc(m, X, Y):
    with torch.no_grad(): return float((m(X).argmax(1) == Y).float().mean())
def train_task(m, X, Y, method, state, g):
    opt = torch.optim.SGD(m.parameters(), lr=LR)
    perm = torch.randperm(len(X), generator=g)
    for i in range(0, len(X), BS):
        idx = perm[i:i+BS]; x, y = X[idx], Y[idx]
        loss = F.cross_entropy(m(x), y)
        if method == "replay" and state.get("buf_x") is not None:
            j = torch.randint(len(state["buf_x"]), (BS,), generator=g)
            loss = loss + F.cross_entropy(m(state["buf_x"][j]), state["buf_y"][j])
        if method == "ewc":
            for F_, P_ in state.get("fishers", []):
                loss = loss + 50 * sum((f * (p - p0)**2).sum() for f, p0, p in zip(F_, P_, m.parameters()))
        opt.zero_grad(); loss.backward(); opt.step()
    if method == "replay":  # add 100 samples per task, cap 1000 (balanced reservoir)
        k = 1000 // T; idx = torch.randperm(len(X), generator=g)[:k]
        bx = X[idx]; by = Y[idx]
        state["buf_x"] = bx if state.get("buf_x") is None else torch.cat([state["buf_x"], bx])
        state["buf_y"] = by if state.get("buf_y") is None else torch.cat([state["buf_y"], by])
    if method == "ewc":  # empirical diag Fisher on 1000 samples
        fs = [torch.zeros_like(p) for p in m.parameters()]
        idx = torch.randperm(len(X), generator=g)[:1000]
        for i in range(0, 1000, 50):
            m.zero_grad(); F.cross_entropy(m(X[idx[i:i+50]]), Y[idx[i:i+50]]).backward()
            for f, p in zip(fs, m.parameters()): f += p.grad.detach()**2 * 50 / 1000
        state.setdefault("fishers", []).append((fs, [p.detach().clone() for p in m.parameters()]))
res = {}; t0 = time.time()
for seed in [0, 1]:
    g = torch.Generator().manual_seed(seed); perms = [torch.arange(784)] + [torch.randperm(784, generator=g) for _ in range(T - 1)]
    for method in ["naive", "ewc", "replay", "fresh"]:
        torch.manual_seed(seed); m = net(); state = {}; R = np.zeros((T, T))
        for t in range(T):
            if method == "fresh": torch.manual_seed(seed * 100 + t); m = net()
            train_task(m, Xtr[:, perms[t]], Ytr, "naive" if method == "fresh" else method, state, g)
            for s in range(T): R[t, s] = acc(m, Xte[:, perms[s]], Yte)
        res[(method, seed)] = R
        print(method, seed, "final avg", R[-1].mean().round(4), "new-task acc", np.diag(R).round(3), round(time.time() - t0), flush=True)
out = {}
for method in ["naive", "ewc", "replay", "fresh"]:
    Rs = np.stack([res[(method, s)] for s in [0, 1]])
    fin = Rs[:, -1, :].mean(1); diag = np.stack([np.diag(r) for r in Rs])
    forget = np.stack([[r[:, s].max() - r[-1, s] for s in range(T - 1)] for r in Rs]).mean(1)
    out[method] = dict(final_avg_acc=[float(fin.mean()), float(fin.std())], avg_forgetting=[float(forget.mean()), float(forget.std())],
                       new_task_acc=diag.mean(0).tolist(), acc_on_task1_over_time=Rs[:, :, 0].mean(0).tolist())
out["runtime_s"] = time.time() - t0
json.dump(out, open("results.json", "w"), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); C = dict(naive="#e53935", ewc="#ff9800", replay="#4caf50", fresh="#888")
for mth in ["naive", "ewc", "replay"]:
    ax[0].plot(range(1, T + 1), out[mth]["acc_on_task1_over_time"], "o-", c=C[mth], label=mth)
    ax[1].bar(mth, out[mth]["final_avg_acc"][0], yerr=out[mth]["final_avg_acc"][1], color=C[mth])
for mth in ["naive", "ewc", "replay", "fresh"]:
    ax[2].plot(range(1, T + 1), out[mth]["new_task_acc"], "o-", c=C[mth], label=mth)
ax[0].set(title="Accuracy on task 1 as tasks accumulate", xlabel="tasks trained", ylabel="test acc"); ax[0].legend()
ax[1].set(title="Final avg acc over 10 tasks (2 seeds)", ylim=(0, 1))
ax[2].set(title="Plasticity: acc on each new task after learning it", xlabel="task", ylabel="test acc"); ax[2].legend()
plt.tight_layout(); plt.savefig("plot.png", dpi=130)
