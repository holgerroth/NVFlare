# Literature loop worksheet

## Trigger

- Reason: 4+ consecutive same-budget batches yielded ≤0.001 absolute change vs best; the FedAvgM lr=1.9 m=0.4 + LS 0.175 plateau is saturated across server_lr, server_momentum, train_lr, train_momentum, weight_decay, cosine_eta_min, grad_clip, mixup, and FedProx axes.
- Current best: 0.8378 — FedAvgM lr=1.9 m=0.4 + LS=0.175 (FedAvg recipe with server momentum + cross-entropy label smoothing).
- Recent symptoms from `results.tsv`: smooth ridge ~0.83-0.84, sharp cliff above server_lr=2.0, mixup regresses, grad_clip neutral or worse, FedProx atop FedAvgM regresses.
- Confirmed null/worse: FedProx on top of FedAvgM, mixup, grad_clip, weight_decay, train_momentum != 0.9, no_lr_scheduler, cosine_eta_min outside [0.005, 0.01], server_lr > 2.0, FedAdam (any lr).
- Candidate width: `PARALLEL_CANDIDATES=2` (effective; 4 hits 64 MB /dev/shm contention during cross-site eval), staggered 90 s.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| "federated learning short rounds CIFAR-10 non-IID server momentum" | confirm regimes that match our 10-round budget | arXiv | Hsu19 FedAvgM is our base; Wang20 FedNova adjusts heterogeneous step counts. |
| "federated averaging bias correction server learning rate warmup" | server-side warmup on top of momentum is rarely tested in short-budget regimes | arXiv | Reddi20 FedOpt uses linear server lr; FedExP (Jhunjhunwala23) adds extrapolation step. |
| "ema teacher self distillation federated learning regularization" | cheap regularizer that adds no protocol fields | arXiv | FedDistill / SelfReg-FL families typically need shared logits — incompatible. KD via local EMA target stays local. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Hsu19 | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (2019) | arXiv:1909.06335 | non-IID drift | server momentum | already used |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization (FedNova) | arXiv:2007.07481 | heterogeneous local steps bias | normalized averaging | reject — same per-site step count, less applicable |
| Jhunjhunwala23 | FedExP: Speeding up Federated Averaging via Extrapolation | arXiv:2301.09604 | slow convergence | server-side extrapolation | keep — small change to aggregator |
| Yang22 | Federated Learning with Class Imbalance Reduction | arXiv:2210.* | class imbalance under non-IID | re-weighted per-class loss | reject — needs shared label histograms |
| Tarvainen17 | Mean teachers are better role models | arXiv:1703.01780 | regularization via EMA teacher | EMA self-KD | keep — purely client-local |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Server-side updates plateau early under fixed LR; extrapolation can speed late-round progress | FedExP (Jhunjhunwala23) | Score curve flatlines from round ~7-8 in tb_events; sharp lr=2.0 cliff suggests we are near the stable rate | aggregator-only, preserves DIFF | custom_aggregators.py |
| C2 | Local overfitting to skewed class slice causes confident wrong predictions; soft targets reduce it | Tarvainen17 (Mean Teacher), Hinton15 KD | LS already gives +0.02; further regularization may stack | client.py train loop, no shared state | client.py |
| C3 | Server lr warmup avoids early-round divergence on non-IID | Reddi20 FedOpt warmup recipe | Cliff at lr=2.5 (collapse) and lr=2.2 (regression) shows fragility at higher lr; warmup may unlock larger asymptote | aggregator-only, no contract change | custom_aggregators.py |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Server-side extrapolation: aggregator returns `mean_diff * eta_g` with `eta_g = 1 + extrap_step * (round_idx)` | Jhunjhunwala23 FedExP | new --server_extrap arg in aggregator. eta_g scales the aggregated diff before adding to global model. Just multiply update vector | break the lr=2.0 ceiling later in training | post-round score does not improve | low — pure aggregator math, no new fields |
| P2 | EMA teacher self-distillation: maintain client-local EMA model and add KL-divergence(student soft, teacher soft) loss | Tarvainen17 | new --ema_kd_alpha + --ema_decay; teacher = EMA of student; KL between softmaxed logits | regularize student beyond LS+aug | KL term overpowers task loss → worse | low — purely client-local |
| P3 | Server lr warmup over first K rounds for FedAvgM | Reddi20 (server-side schedule) | new --server_lr_warmup_rounds K with linear ramp from 0 to server_lr | avoid early divergence, allow higher final lr | regression on early rounds slow | low — aggregator math |
| P4 | Client-local SGD scheduler restart per round | Loshchilov17 SGDR | --cosine_lr_restart action; restart cosine each round | escape local minima | overheats first epoch each round | low — client local |
| P5 | Stochastic depth / dropout in classifier head | Huang16 | --classifier_dropout p; insert dropout in fc head | regularization | underfits in 10 rounds | medium — touches model arch — would need new registered variant |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 FedExP | new | not yet tested | run |
| P2 EMA-KD | new | extends LS axis but mechanism distinct | run |
| P3 server lr warmup | new | not in null set | run |
| P4 cosine restart | partially overlaps no_lr_scheduler null result | could differ from constant lr | run with caveat |
| P5 dropout | architecture change | requires new registered model_arch — defer | defer |

## Proposal scoring

Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 FedExP extrapolation | 3 | 5 | 4 | 4 | 3 | 1 | 2*3 + 2*5 + 4 + 4 + 3 - 1 = 26 |
| P2 EMA-KD | 3 | 5 | 3 | 4 | 4 | 2 | 2*3 + 2*5 + 3 + 4 + 4 - 2 = 25 |
| P3 server lr warmup | 2 | 5 | 4 | 3 | 3 | 1 | 2*2 + 2*5 + 4 + 3 + 3 - 1 = 23 |
| P4 cosine restart | 2 | 5 | 4 | 3 | 2 | 1 | 2*2 + 2*5 + 4 + 3 + 2 - 1 = 22 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 FedExP step 1.5 | fedexp_eta15 | --aggregator fedavgm + new server_extrap=1.5 (or implement fedexp aggregator) |
| 2 | P3 warmup R=2 | fedavgm_warmup2 | server_lr ramp 0→1.9 over rounds 0-1, then 1.9 |
| 3 | P2 EMA-KD alpha=0.3 | emakd_a03 | client EMA teacher; KL coefficient 0.3 |
| 4 | P4 per-round cosine | cosine_per_round | restart cosine schedule each round |

## Reflective memory

- Keep: server momentum (FedAvgM) + LS as the strong pair. Don't drop either without re-testing.
- Discard: mixup (regression), grad_clip (neutral/regression), FedProx atop FedAvgM, FedAdam at any lr ≤ 1.0.
- Do not retry: train_lr outside [0.04, 0.06], train_momentum != 0.9, weight_decay > 0 with this aggregator, cosine_eta_min outside [0.005, 0.01].
- Sources to carry forward: Jhunjhunwala23 FedExP arXiv:2301.09604; Tarvainen17 Mean Teacher arXiv:1703.01780; Reddi20 FedOpt arXiv:2003.00295.
