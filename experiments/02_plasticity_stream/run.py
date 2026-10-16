"""Exp 02: Loss of plasticity over a long permuted-MNIST stream.
MLP 784-256-256-10 (ReLU), each task = a fresh random pixel permutation of a fresh random 5k-image train subset,
3 epochs per task, batch 64. Optimizers: SGD lr 0.01, SGD lr 0.1, Adam lr 1e-3 (default betas). 2 seeds, 200 tasks.
Per task we log: test acc on 2k permuted test images right after training (plasticity = can it still learn the new task?),
online train acc in the last epoch, fraction of dead hidden units (never active on 2k probe inputs), effective rank (srank,
delta=0.01, Kumar et al. 2021) of the last hidden layer's features, and total weight norm.
Reference: fresh-init net trained the same way on a task (mean over 10 tasks)."""
import json, time, sys, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
torch.set_num_threads(6)
root = "../../data"
tr, te = MNIST(root, download=False), MNIST(root, train=False, download=False)
Xtr = tr.data.float().view(-1, 784) / 255; Ytr = tr.targets; Xte = te.data.float().view(-1, 784) / 255; Yte = te.targets
T = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NTR, EP, BS, NTE = 5000, 3, 64, 2000
CFGS = {"sgd_lr0.01": ("sgd", 0.01), "sgd_lr0.1": ("sgd", 0.1), "adam_lr1e-3": ("adam", 1e-3)}
SEEDS = [0, 1]
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(784, 256); s.l2 = nn.Linear(256, 256); s.out = nn.Linear(256, 10)
    def forward(s, x, feats=False):
        h1 = F.relu(s.l1(x)); h2 = F.relu(s.l2(h1)); o = s.out(h2)
        return (o, h1, h2) if feats else o
def make_opt(m, kind, lr):
    return torch.optim.SGD(m.parameters(), lr=lr) if kind == "sgd" else torch.optim.Adam(m.parameters(), lr=lr)
def srank(H, delta=0.01):
    s = torch.linalg.svdvals(H); c = torch.cumsum(s, 0) / s.sum()
    return int((c < 1 - delta).sum()) + 1
def run_task(m, opt, g):
    idx = torch.randperm(60000, generator=g)[:NTR]; perm = torch.randperm(784, generator=g)
    X, Y = Xtr[idx][:, perm], Ytr[idx]
    tidx = torch.randperm(10000, generator=g)[:NTE]; Xt, Yt = Xte[tidx][:, perm], Yte[tidx]
    for ep in range(EP):
        p = torch.randperm(NTR, generator=g); corr = 0
        for i in range(0, NTR, BS):
            b = p[i:i+BS]; o = m(X[b]); loss = F.cross_entropy(o, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep == EP - 1: corr += int((o.argmax(1) == Y[b]).sum())
    with torch.no_grad():
        o, h1, h2 = m(Xt, feats=True)
        te_acc = float((o.argmax(1) == Yt).float().mean())
        dead = float(torch.cat([(h1 > 0).sum(0) == 0, (h2 > 0).sum(0) == 0]).float().mean())
        sr = srank(h2); wn = float(sum(p.norm()**2 for p in m.parameters()).sqrt())
    return dict(test=te_acc, train=corr / NTR, dead=dead, srank=sr, wnorm=wn)
out = {"config": dict(T=T, NTR=NTR, EP=EP, BS=BS, NTE=NTE, seeds=SEEDS, arch="784-256-256-10 ReLU")}; t0 = time.time()
for name, (kind, lr) in CFGS.items():
    for seed in SEEDS:
        torch.manual_seed(seed); g = torch.Generator().manual_seed(1000 + seed)
        m = MLP(); opt = make_opt(m, kind, lr); logs = []
        for t in range(T):
            logs.append(run_task(m, opt, g))
            if t % 25 == 0 or t == T - 1:
                print(name, seed, t, logs[-1], round(time.time() - t0), flush=True)
        # fresh-init reference: 10 tasks, new net + optimizer each time
        fresh = []
        for k in range(10):
            torch.manual_seed(5000 + 10 * seed + k); fm = MLP(); fresh.append(run_task(fm, make_opt(fm, kind, lr), g))
        out[f"{name}|{seed}"] = dict(stream=logs, fresh=fresh)
        json.dump(out, open("results.json", "w"))
out["runtime_s"] = time.time() - t0; json.dump(out, open("results.json", "w"))
print("done", round(time.time() - t0))
