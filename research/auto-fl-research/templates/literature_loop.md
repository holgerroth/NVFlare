# Literature loop worksheet

## Trigger

- Reason: two consecutive same-budget batches without improvement (server hp refinement around (lr=2.0, m=0.2); FedProx + step-mode at the kept stack).
- Current best: 0.8836 — `--aggregator fedavgm --server_lr 2.0 --server_momentum 0.2 --weight_decay 2e-4 --aggregation_epochs 5` on top of moderate_cnn / 8 sites / 20 rounds / alpha=0.5 / seed=0.
- Recent symptoms: FedProx mu in [1e-5, 5e-5] hurts; step-mode (lts=500) ~0.881; lr=2.25 NaN'd; high client momentum 0.95 collapsed; client lr in [0.03, 0.10] worse than 0.05; cosine with eta_min_factor=0.01 best.
- Confirmed null/worse: FedAdam (NaN at slr=1.0); SCAFFOLD (0.8548 with no enhancements); high client momentum; client lr deviations from 0.05; constant lr; FedProx alone with FedAvgM stack; weight_decay above 4e-4 at ae=5; aggregation_epochs ≤3.
- Candidate width: PARALLEL_CANDIDATES=4 on local H100 (CUDA_VISIBLE_DEVICES=0).
- Ledger event: timer started.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| "label smoothing federated learning non-IID CIFAR-10" | CE overconfidence under heterogeneity may be sharpening local minima | arXiv, OpenAlex (memory) | Reddi21 FedOpt explicitly uses LS; Müller19 NeurIPS shows LS regularizes calibration |
| "gradient clipping federated averaging client drift" | client gradient outliers under non-IID magnify update mismatch with FedAvgM momentum buffer | arXiv | Pascanu13 RNN; Geyer17 DP-FedAvg; FedAvgClip variants in DP literature |
| "MOON contrastive client representation federated learning" | client representation drift not captured by simple proximal term | arXiv (CVPR 2021) | Li21 MOON arXiv:2103.16257 |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Szegedy16 | "Rethinking the Inception Architecture for CV" 2016 | arXiv:1512.00567 | overconfident CE softmax | label smoothing | keep |
| Müller19 | "When Does Label Smoothing Help?" NeurIPS 2019 | arXiv:1906.02629 | calibration / generalization | label smoothing analysis | keep |
| Reddi21 | "Adaptive Federated Optimization" ICLR 2021 | arXiv:2003.00295 | server-side adaptivity in FL; uses LS in CIFAR experiments | FedOpt + LS | keep |
| Pascanu13 | "On the difficulty of training RNNs" ICML 2013 | arXiv:1211.5063 | exploding gradients | gradient norm clipping | keep |
| Li21 | "Model-Contrastive Federated Learning (MOON)" CVPR 2021 | arXiv:2103.16257 | client representation drift on non-IID | contrastive client loss | keep (reserve) |
| Zhang17 | "mixup: Beyond Empirical Risk Minimization" ICLR 2018 | arXiv:1710.09412 | sharpness / generalization | mixup augmentation | reject (changes data path; defer) |

## Challenge cards

| id | challenge | paper evidence | results.tsv symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | local overfitting from long local training (5 epochs × 20 rounds = 100 epochs of local SGD) | Szegedy16, Müller19 (LS regularizes overconfident logits and reduces overfitting) | ae=4 → ae=5 helped; ae=6 dipped (mild overfit); wd helped a lot | client.py loss function |
| C2 | client gradient outliers / variance under alpha=0.5 amplify FedAvgM momentum drift | Pascanu13 (clip explosive grads); Geyer17 (clipping under DP-FL) | lr=2.25 server NaN'd; high client momentum 0.95 collapsed; FedAdam diverged | client.py per-step optimizer hook |
| C3 | client representation drift not captured by parameter-space proximal term (FedProx) | Li21 MOON (contrastive on representations, not parameters) | FedProx mu in [1e-5..5e-5] all worse; SCAFFOLD didn't beat FedAvgM-only baseline | client.py loss function with previous local model state |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | label smoothing 0.1 on CrossEntropyLoss | Szegedy16, Müller19, Reddi21 | `nn.CrossEntropyLoss(label_smoothing=ls)`; new `--label_smoothing` arg, default 0.0 | regularize overconfident logits, +0.2-0.5pp | LS=0.1 < 0.8836 | none — no protocol change |
| P2 | global gradient clipping at max_norm=1.0 | Pascanu13, Geyer17 | `torch.nn.utils.clip_grad_norm_(params, max_norm)` between backward and step; new `--grad_clip_max_norm` arg, 0 disables | bound update magnitude, smooths FedAvgM accumulation, +0.1-0.4pp | clip < 0.8836 | none |
| P3 | combined LS=0.1 + clip=1.0 | both above | both args set together | additive +0.3-0.7pp | combined < 0.8836 | none |
| P4 | label smoothing 0.05 (lighter) | Müller19 | `--label_smoothing 0.05` | gentler regularizer for case where 0.1 is too strong | LS=0.05 worse than LS=0.1 | none |
| P5 (reserve) | MOON-style contrastive loss | Li21 | new feature class in client.py: keep prev local model; loss += contrastive(z, z_glob, z_prev_local) with τ=0.5 | +0.5-1.0pp on non-IID | < 0.8836 | low — adds memory & forward passes; no protocol change |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | none | not in ledger | keep |
| P2 | none | not in ledger | keep |
| P3 | none | combination, not in ledger | keep |
| P4 | partial of P1 | not in ledger | keep |
| P5 | none | distinct from FedProx (which failed at this stack) | reserve for next batch |

## Proposal scoring

Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 LS=0.1 | 3 | 5 | 5 | 4 | 2 | 1 | 26 |
| P2 clip=1.0 | 3 | 5 | 5 | 4 | 2 | 1 | 26 |
| P3 LS+clip | 4 | 5 | 4 | 4 | 2 | 1 | 27 |
| P4 LS=0.05 | 2 | 5 | 5 | 3 | 1 | 1 | 23 |
| P5 MOON | 4 | 4 | 2 | 4 | 4 | 3 | 23 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | ls01 | `--label_smoothing 0.1` |
| 2 | P2 | clip10 | `--grad_clip_max_norm 1.0` |
| 3 | P3 | ls01_clip10 | `--label_smoothing 0.1 --grad_clip_max_norm 1.0` |
| 4 | P4 | ls005 | `--label_smoothing 0.05` |

P5 (MOON) held in reserve as a follow-up if this batch doesn't improve and complexity is worth the risk.

## Reflective memory

- Keep: program.md is right that FedProx is a client-local term; FedAvgM hp is sensitive to local-compute and weight-decay settings.
- Discard: server_lr ≥ 2.25 with (m=0.1, ae=5) destabilizes; high client momentum 0.95 collapses.
- Do not retry: FedAdam at slr=1.0 (NaN); SCAFFOLD without enhancements; constant lr.
- Sources to carry forward: Szegedy16 LS, Pascanu13 clip, Li21 MOON, Reddi21 FedOpt LS settings.
