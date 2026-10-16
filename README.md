# continual-learning

Small-scale continual-learning (CL) experiments on an Apple Silicon Mac (PyTorch, MPS/CPU). Queue: [QUEUE.md](QUEUE.md).
Setup: `python3 -m venv .venv && .venv/bin/pip install torch torchvision numpy matplotlib`.

## Literature anchors
- Catastrophic forgetting: McCloskey & Cohen 1989; Ratcliff 1990; French 1999 (*Trends Cogn. Sci.*).
- Regularization: EWC (Kirkpatrick et al. 2017, *PNAS*); SI (Zenke, Poole & Ganguli 2017, ICML); LwF (Li & Hoiem 2017, TPAMI).
- Replay/projection: GEM (Lopez-Paz & Ranzato 2017, NeurIPS); A-GEM (Chaudhry et al. 2019, ICLR); "Tiny episodic memories" ER (Chaudhry et al. 2019); iCaRL (Rebuffi et al. 2017).
- Scenarios: van de Ven & Tolias 2019, "Three scenarios for continual learning"; van de Ven, Tuytelaars & Tolias 2022 (*Nat. Mach. Intell.*).
- Loss of plasticity: Dohare et al. 2024, *Nature* 632 (continual backprop); Ash & Adams 2020, shrink-and-perturb (NeurIPS); Lyle et al. 2023, "Understanding plasticity in neural networks" (ICML); Sokar et al. 2023, dormant neurons (ICML); Kumar et al. 2023, "Maintaining plasticity via regenerative regularization" (L2-Init).
- LLM forgetting: Luo et al. 2023, "An empirical study of catastrophic forgetting in LLMs during continual fine-tuning" (arXiv:2308.08747); Biderman et al. 2024, "LoRA learns less and forgets less" (TMLR); Ibrahim et al. 2024, "Simple and scalable strategies to continually pre-train LLMs" (arXiv:2403.08763).
- 2025–26: Zheng et al. 2025, "Spurious forgetting in continual learning of LMs" (arXiv:2501.13453, ICLR 2025); "Spectral collapse drives loss of plasticity" (arXiv:2509.22335); GeRe general-sample replay (arXiv:2508.04676); GORP (ACL 2025); CLoRA (ACL 2025); "Forgetting, plasticity, and co-observation" (arXiv:2608.18803); Ren, Deng, Hong, Lyle & Gal, "Learning dynamics of continual learning" (arXiv:2609.33620); OSW (arXiv:2601.18255).

## Experiment log

### 01 — Permuted-MNIST baseline: naive vs EWC vs replay, plus plasticity probe (2026-10-15)
- **Hypothesis:** naive sequential SGD forgets heavily; small replay (1k buffer) beats EWC at equal effort; over only 10 tasks there is *no* measurable loss of plasticity (Dohare et al. see it over hundreds or thousands of tasks).
- **Method:** 10 permuted-MNIST tasks (task 1 = identity), MLP 784-256-256-10, SGD lr 0.05, batch 128, 1 epoch per task. EWC: diag empirical Fisher (1k samples), λ=50, untuned. Replay: 100 samples per task into a buffer (≤1k), one replay batch per step. "fresh" = new random init per task (plasticity reference). 2 seeds, CPU, 55 s total. `experiments/01_permuted_mnist/`.
- **Result (mean ± sd over 2 seeds):**

  | method | final avg acc (10 tasks) | avg forgetting | task-1 acc at end |
  |---|---|---|---|
  | naive | 0.722 ± 0.010 | 0.223 ± 0.011 | 0.551 |
  | EWC λ=50 | 0.750 ± 0.014 | 0.186 ± 0.016 | 0.666 |
  | replay 1k | **0.873 ± 0.002** | **0.062** | 0.777 |

  Plasticity: accuracy on each new task after learning it *rises* from 0.895 (task 1) to 0.937 (task 10) for naive, against a flat 0.89 for fresh init. That is positive forward transfer, not plasticity loss.
- **Interpretation:** forgetting and the large replay advantage reproduce the standard picture. EWC helps only modestly here, but λ wasn't tuned, so this isn't a fair EWC verdict. As expected, there's no plasticity loss at this horizon; the queue's plasticity experiments need long task streams.
