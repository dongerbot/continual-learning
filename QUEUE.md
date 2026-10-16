# Experiment queue (prioritized; each ≤ ~20 min on the Mac)

1. **[DONE] 01 Permuted-MNIST baseline** — naive/EWC/replay + plasticity probe. See README.
2. **02 Loss of plasticity over a long stream** — *H:* with plain SGD/Adam, new-task accuracy on a long stream (200 tasks of permuted MNIST on a 2k-image subset, or Dohare-style random-label MNIST) degrades, with rising dormant-unit fraction and falling effective rank. *Method:* small MLP, 200 tasks × short training; log new-task acc, % dead ReLUs, feature srank. ~15 min.
3. **03 Plasticity fixes head-to-head** — *H:* continual backprop (Dohare 2024), shrink-and-perturb (Ash & Adams 2020) and L2-Init (Kumar 2023) all preserve plasticity; L2-Init is cheapest at equal effect. *Method:* rerun 02 with each fix plus a small hyperparameter sweep, 2 seeds. ~20 min.
4. **04 Is forgetting "spurious"? (alignment vs knowledge)** — *H (Zheng et al. 2025):* much apparent forgetting is head/readout misalignment; a linear probe retrained on frozen features recovers most old-task accuracy. *Method:* after split-MNIST/permuted runs, compare old-task acc with original head vs a re-fit linear probe; also freeze bottom layer. ~10 min.
5. **05 Replay as co-observation** — *H (arXiv:2608.18803):* replay helps beyond preventing forgetting; with a no-shift chunked stream (iid MNIST in 10 chunks), replay still beats naive and approaches joint training. *Method:* chunked iid stream: naive vs replay vs joint, equal compute. ~10 min.
6. **06 Small-LM fine-tuning forgetting** — *H (Biderman 2024):* LoRA forgets less but learns less than full FT. *Method:* Pythia-70m fine-tuned on a narrow corpus (code or one Wikipedia domain); measure Pile-10k loss increase vs target-loss decrease for full FT vs LoRA ranks 4/16. ~20 min, uses cached weights.

## Open questions these address
- Does plasticity loss appear in tiny settings within minutes, and which cheap fix is enough? (02, 03)
- How much "forgetting" is readout misalignment rather than lost features? (04)
- Is replay's benefit mostly anti-forgetting, or the co-observation effect? (05)
- Does the LoRA forget-less/learn-less trade-off hold at 70M scale, and is it the same thing as a smaller effective learning rate? (06)
