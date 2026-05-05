# Literature loop worksheet (cycle 2)

## Trigger

- Reason: two consecutive same-budget batches without improvement after the LS=0.1 keep (server hp re-tune around m=0.25, then architecture audit).
- Current best: 0.8936 — `--aggregator fedavgm --server_lr 2.0 --server_momentum 0.25 --weight_decay 2e-4 --aggregation_epochs 5 --label_smoothing 0.1` on top of moderate_cnn.
- Recent symptoms: arch_norm regresses ~2pp; small_head ~0.5pp behind; m=0.225/m=0.275 within ±0.01pp of m=0.25; lr=1.75 NaN'd.
- Confirmed null/worse: gradient clipping (any norm) interferes with LS gains; FedProx; FedAdam; SCAFFOLD without enhancements; alternative architectures; constant lr; high client momentum.
- Candidate width: 4 on local H100 (CUDA_VISIBLE_DEVICES=0).
- Ledger event: timer started.

## Search queries

| query | rationale | source(s) | notes |
| --- | --- | --- | --- |
| "mixup data augmentation cifar federated learning" | label-side regularizer (LS) is the latest big win — explore complementary input-space regularizer | arXiv | Zhang17 mixup; combines well with LS in CV literature |
| "MOON contrastive federated representation drift" | client representation drift may not be captured by parameter-space proximal terms | arXiv (CVPR) | Li21 MOON paper |
| "AdamW client optimizer cifar non-iid federated" | only SGD has been used; alternative optimizer family may help non-IID convergence | arXiv | Loshchilov19; Reddi21 used Adam server-side |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zhang17 | mixup: Beyond Empirical Risk Minimization, ICLR 2018 | arXiv:1710.09412 | sharp minima / data augmentation | input mixing | keep |
| Li21 | Model-Contrastive Federated Learning (MOON), CVPR 2021 | arXiv:2103.16257 | client representation drift | contrastive client loss | keep (reserve) |
| Loshchilov19 | Decoupled Weight Decay Regularization, ICLR 2019 | arXiv:1711.05101 | proper weight decay with adaptive optimizers | AdamW | keep |
| Yin18 | Byzantine-Robust Distributed Learning (median), ICML 2018 | arXiv:1803.01498 | robust aggregation | coordinate-wise median | keep (free try via aggregator=median) |

## Challenge cards

| id | challenge | paper evidence | results.tsv symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | local optimizer family unexplored — only SGD with momentum tried | Loshchilov19 AdamW outperforms Adam-with-L2 on CV | optimizer hp axis is saturated | client.py optimizer construction |
| C2 | input-space regularization unexplored; LS regularizes label side only | Zhang17 mixup gives +0.5-1pp on CIFAR-10 with no extra params | LS=0.1 helped +0.94pp; complementary axis untested | client.py training loop |
| C3 | aggregation operator unexplored beyond FedAvg/FedAvgM/FedAdam/SCAFFOLD; robust median may help under non-IID outlier clients | Yin18 median; Pillutla22 robust FL surveys | non-IID alpha=0.5 — some clients have rare classes | custom_aggregators.py / `--aggregator median` (already wired) |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | mixup augmentation alpha=0.2 in training loop | Zhang17 arXiv:1710.09412 | new `--mixup_alpha` arg; lambda~Beta(a,a); inputs=lam*x_a+(1-lam)*x_b; loss=lam*ce(p,y_a)+(1-lam)*ce(p,y_b) | input regularization, +0.3-0.7pp on top of LS | < 0.8936 | none — pure local op |
| P2 | mixup alpha=0.4 (heavier mixing) | Zhang17 | same with alpha=0.4 | possibly better for non-IID where mixing reduces drift | < 0.8936 | none |
| P3 | AdamW client optimizer with lr=1e-3 | Loshchilov19 arXiv:1711.05101 | new `--client_optimizer {sgd,adamw}`; build `optim.AdamW(...)` when adamw | different optimizer family; AdamW gradient updates may handle non-IID variance better | < 0.8936 | none — same DIFF contract |
| P4 | median aggregator at LS=0.1 stack | Yin18 arXiv:1803.01498 | `--aggregator median` (free toggle, already wired) | robust to outlier client gradients | < 0.8936 | none — already in custom_aggregators |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 mixup α=0.2 | none | not in ledger | keep |
| P2 mixup α=0.4 | partial of P1 | not in ledger | keep |
| P3 AdamW | none | not in ledger | keep |
| P4 median | none | not in ledger | keep |

## Proposal scoring

Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 mixup α=0.2 | 4 | 5 | 4 | 5 | 4 | 1 | 30 |
| P2 mixup α=0.4 | 3 | 5 | 4 | 4 | 3 | 1 | 26 |
| P3 AdamW lr=1e-3 | 2 | 5 | 5 | 4 | 3 | 1 | 25 |
| P4 median | 2 | 5 | 5 | 3 | 3 | 1 | 24 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | mix02 | `--mixup_alpha 0.2` |
| 2 | P2 | mix04 | `--mixup_alpha 0.4` |
| 3 | P3 | adamw_lr1e3 | `--client_optimizer adamw --lr 1e-3 --weight_decay 1e-2` |
| 4 | P4 | agg_median | `--aggregator median` |

MOON proposal (Li21 contrastive) held in reserve as a higher-cost follow-up if mixup/optimizer/median fail to improve.

## Reflective memory

- Keep: LS=0.1 is a strong regularizer on this stack; exploration of `client_optimizer` and `aggregator` axes still open.
- Discard: gradient clipping interferes with LS gains; SCAFFOLD without enhancements; FedAdam; lr=1.75 NaN.
- Sources to carry forward: Zhang17 mixup, Loshchilov19 AdamW, Yin18 median.
