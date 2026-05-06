# Literature loop worksheet

Use this worksheet only when progress stalls or the next axis is unclear. Keep entries short and source-backed.

## Trigger

- Reason: two consecutive same-budget FedAvgM momentum batches failed after best score `0.864700`.
- Current best: FedAvgM, `server_lr=1.5`, `server_momentum=0.2`, score `0.864700`.
- Recent symptoms from `results.tsv`: FedAvgM momentum `0.0`, `0.1`, `0.15`, and `0.25` all underperformed `0.2`; FedAdam at `server_lr=1.0`, `tau=1e-3` produced NaN diffs; SCAFFOLD was stable but lower at `0.854800`.
- Confirmed null/worse ideas to avoid: tiny FedProx with weighted FedAvg (`1e-5`, `1e-4`), FedAvgM `server_lr>=2.25` at momentum `0.4`, FedAvgM momentum jitter away from `0.2` at `server_lr=1.5`, high-LR FedAdam.
- Candidate width: `PARALLEL_CANDIDATES=2` after shared-memory contention at width 4; use unique `PYTHONPYCACHEPREFIX` per candidate.
- Ledger event: started with `scripts/log_literature_review.py --start`; finish with `--finish` before launching the next batch.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning client drift non-IID CIFAR-10 server momentum FedAvgM local epochs` | Explain why FedAvgM helped then plateaued. | arXiv, Hugging Face Papers | Found Reddi21 FedOpt and Cheng23 momentum analysis. |
| `FedOpt adaptive federated optimization server learning rate FedAdam non-IID CIFAR-10` | Recover from FedAdam NaN with safer adaptive settings. | arXiv, Hugging Face Papers | Reddi21 supports adaptive server optimizers but tuning is sensitive. |
| `federated learning label distribution skew logits calibration CIFAR10 FedLC` | Address Dirichlet label skew directly rather than only server optimizer jitter. | arXiv, PMLR | FedLC targets local label-skew bias with client-local loss calibration. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | FedAvg can be hard to tune under heterogeneity; adaptive/server optimizers can help. | FedOpt, FedAvgM, FedAdam | keep |
| Li20 | Federated Optimization in Heterogeneous Networks / 2020 | https://arxiv.org/abs/1812.06127 | Statistical and systems heterogeneity destabilize local updates. | FedProx proximal local loss | keep |
| Karimireddy20 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning / 2020 | https://arxiv.org/abs/1910.06378 | FedAvg client drift under non-IID data. | Control variates | keep |
| Zhang22 | Federated Learning with Label Distribution Skew via Logits Calibration / 2022 | https://arxiv.org/abs/2209.00189 | Label distribution skew causes biased local objectives. | FedLC client-local logit calibration | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Variable local updates can create objective inconsistency. | FedNova normalized aggregation | keep in reserve |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Momentum can improve FedAvg/SCAFFOLD convergence under heterogeneity. | Momentum variants | keep as interpretation |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client drift and unstable local objectives | SCAFFOLD frames non-IID FedAvg as client drift; FedProx frames heterogeneity as requiring local stabilization. | FedAvgM helped; tiny FedProx under weighted FedAvg did not; SCAFFOLD stable but below best. | Dirichlet `alpha=0.5` split and 4 local epochs create drift pressure. | `client.py` via `--fedproxloss_mu`; `custom_aggregators.py` via existing SCAFFOLD/FedAvgM. |
| C2 | Adaptive server update instability | FedOpt proposes adaptive server optimizers, but our high-LR FedAdam produced NaN diffs. | FedAdam crash at `server_lr=1.0`, `tau=1e-3`. | Safer FedAdam might work if server update scale is reduced. | Existing `--aggregator fedadam`, `--server_lr`, `--fedopt_tau`. |
| C3 | Label-skew local overfitting | FedLC argues softmax cross-entropy can overfit skewed/missing local classes and proposes logit calibration. | Server optimizer jitter plateaued after `0.864700`; data split is label-skewed Dirichlet. | A client-local loss change could attack the data symptom directly. | `client.py` optional loss flag; no protocol change. |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Combine current best FedAvgM with stronger FedProx regularization. | Li20; Reddi21 | CLI-only: `--aggregator fedavgm --server_lr 1.5 --server_momentum 0.2 --fedproxloss_mu 1e-3` | Reduce client drift without changing server contract. | Score below `0.864700` or slower with no gain. | Low. |
| P2 | Retry FedAdam with safer update scale. | Reddi21 | CLI-only: `--aggregator fedadam --server_lr 0.1 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-2` | Test adaptive server optimizer without NaN-scale update. | Crash again or score below FedAvgM. | Medium due prior NaN. |
| P3 | Add optional FedLC-style logit calibration. | Zhang22 | Code: client-local `--fedlc_tau` loss adjustment from site label frequencies; default off. | Directly counter label-skew bias. | No improvement over best or unstable loss. | Medium; client-only but new code. |
| P4 | FedNova-like normalization of weighted DIFFs. | Wang20 | Code: normalize client updates by local step count before aggregation. | Reduce objective inconsistency from varying client data sizes/steps. | No improvement or incompatibility with existing weighting. | Medium-high; aggregator math change. |
| P5 | Robust median aggregation audit. | Client drift/outlier literature; existing implementation | CLI-only: `--aggregator median` | Check whether outlier client updates are hurting weighted/FedAvgM. | Score below best. | Low, but weaker evidence. |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | FedProx tiny-mu rows only partially overlap. | `1e-5` and `1e-4` failed with weighted FedAvg, not current FedAvgM and not stronger mu. | keep |
| P2 | FedAdam crash at high LR. | Prior crash indicates scale risk; safer LR/tau is distinct. | keep |
| P3 | None. | Requires code; defer unless CLI proposals fail. | reserve |
| P4 | None. | More invasive than current need. | reserve |
| P5 | None. | Evidence weaker than P1/P2. | reserve |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 3 | 5 | 5 | 4 | 4 | 2 | 27 |
| P2 | 3 | 4 | 5 | 4 | 4 | 2 | 25 |
| P3 | 4 | 4 | 2 | 5 | 5 | 2 | 26 |
| P4 | 3 | 3 | 2 | 4 | 5 | 2 | 21 |
| P5 | 2 | 5 | 5 | 2 | 4 | 2 | 23 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P4 | `fedavgm_lr15_m02_steps300` | Exact local-step variant: `--aggregator fedavgm --server_lr 1.5 --server_momentum 0.2 --local_train_steps 300` |
| 2 | P4 | `fedavgm_lr15_m02_steps400` | Exact local-step variant: `--aggregator fedavgm --server_lr 1.5 --server_momentum 0.2 --local_train_steps 400` |

## Reflective memory

- Keep:
- Keep: FedAvgM `server_lr=1.5`, `server_momentum=0.2` remains current best.
- Discard: local FedAvgM momentum jitter around `0.2` unless a new mechanism changes the context; FedAvgM+FedProx `mu=1e-3`; safer FedAdam `server_lr=0.1`, `tau=1e-2`; FedLC `tau=0.5` and `1.0`.
- Do not retry: FedAdam `server_lr=1.0`, `tau=1e-3`; FedAdam low-LR/tau retry without a new stabilizer; FedProx `1e-5`/`1e-4` with weighted FedAvg; FedProx `1e-3` with current best FedAvgM.
- Sources to carry forward: Wang20 FedNova motivates exact local-step runs as a low-risk proxy for reducing update-count inconsistency before implementing aggregator normalization.
