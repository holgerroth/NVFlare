# Literature loop worksheet

Use this worksheet only when progress stalls or the next axis is unclear. Keep entries short and source-backed.

## Trigger

- Reason:
- Current best:
- Recent symptoms from `results.tsv`:
- Confirmed null/worse ideas to avoid:
- Candidate width: `PARALLEL_CANDIDATES` (default 4 on one local 80 GB H100; lower if memory or host contention appears)
- Ledger event: start with `scripts/log_literature_review.py --start`; finish with `--finish` before launching the next batch.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
|  |  |  |  |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 |  |  |  |  |  |
| C2 |  |  |  |  |  |
| C3 |  |  |  |  |  |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
|  |  |  |  |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |

## Reflective memory

- Keep:
- Discard:
- Do not retry:
- Sources to carry forward:

---

# Literature loop 2026-05-07 plateau after row 80

## Trigger

- Reason: watchdog recommendation=literature after 33 scored candidates since the last material improvement reset.
- Current best: 0.898900, FedAvgM server_lr 1.8, server_momentum 0.35, client lr 0.04, client momentum 0.925, weight_decay 5e-4, aggregation_epochs 7, local_train_steps 0.
- Recent symptoms from `results.tsv`: server_lr, server_momentum, client lr, weight_decay, epoch count, exact local steps, and client momentum fine sweeps mostly regress; best gains since 0.898500 are small.
- Confirmed null/worse ideas to avoid: exact local_train_steps 512/768, server_lr 1.75/1.85 on current stack, server_momentum 0.325/0.375, client lr 0.038/0.042, weight_decay 4.5e-4/5.5e-4, aggregation_epochs 6/8.
- Candidate width: 2 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0` after prior `/dev/shm` contention at width 4.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| FedProx federated optimization heterogeneous networks client drift non IID arXiv 1812.06127 | local drift / non-IID plateau with many near-miss optimizer jitters | web search, arXiv | FedProx maps to existing `--fedproxloss_mu` without protocol changes. |
| Adaptive Federated Optimization FedAdam FedYogi server learning rate tau arXiv | prior FedAdam crashed; source may justify safer server adaptivity settings | web search, arXiv | FedAdam exists in `custom_aggregators.py`; use conservative `server_lr` and larger `tau`. |
| SCAFFOLD stochastic controlled averaging federated learning client drift arXiv | explicit client-drift correction is already implemented as an opt-in mode | web search, arXiv | Strong evidence but previous default SCAFFOLD was poor; keep as reserve. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Li20 | Federated Optimization in Heterogeneous Networks / 2020 | https://arxiv.org/abs/1812.06127 | statistical and systems heterogeneity destabilize FedAvg-style local training | FedProx proximal local objective | keep |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | FedAvg is hard to tune and can converge poorly under heterogeneous data | server adaptive optimizers FedAdam/FedYogi | keep |
| Karimireddy20 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning / 2020 | https://arxiv.org/abs/1910.06378 | client drift from heterogeneous data | control variates | keep as reserve |
| Zhao18 | Federated Learning with Non-IID Data / 2018 | https://arxiv.org/abs/1806.00582 | non-IID label skew causes weight divergence | shared global data subset | reject: changes data substrate |
| Zhang21 | Understanding Clipping for Federated Learning / 2021 | https://arxiv.org/abs/2106.13673 | update clipping can still work under heterogeneity | client update clipping | reserve: requires code and DP-focused evidence |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client drift from non-IID local training | Li20 and Karimireddy20 both target heterogeneity/client drift | high local compute works, but small optimizer changes plateau near 0.899 | proximal correction or control variates may stabilize current high-LR stack | CLI-only `--fedproxloss_mu`; `--aggregator scaffold` reserve |
| C2 | Server optimizer sensitivity | Reddi21 motivates adaptive server optimizers for heterogeneous FL | FedAvgM is best but server_lr/momentum brackets are sharp; FedAdam crashed when aggressive | retry FedAdam conservatively with larger stabilizer | CLI-only `--aggregator fedadam`, `--server_lr`, `--fedopt_tau` |
| C3 | Update norm imbalance | Zhang21 shows clipped FedAvg can work under heterogeneity | exact-step and epoch sweeps imply client update scales are uneven | optional update clipping could regularize without changing params | future `client.py` arg; not first batch |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Add light FedProx proximal term to current best stack | Li20 | add `--fedproxloss_mu 1e-5` to best FedAvgM/client settings | reduce drift while preserving successful optimizer settings | score <= 0.898900 or runtime/instability cost dominates | low, existing client-local loss |
| P2 | Retry FedAdam with conservative adaptivity | Reddi21 | `--aggregator fedadam --server_lr 0.2 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 0.1` plus best client settings | adaptive server normalization may improve heterogeneous convergence without exploding | crash/NaN or score below FedAvgM plateau | medium, previous FedAdam crashed at aggressive settings |
| P3 | Tuned SCAFFOLD protocol mode | Karimireddy20 | `--aggregator scaffold` plus best client lr/momentum/wd/epochs | explicit control variates may reduce client drift | repeats low SCAFFOLD default result | medium, opt-in metadata mode but implemented |
| P4 | Client update/gradient clipping | Zhang21 | add bounded clipping arg in `client.py` or aggregator update clipping | suppress outlier local updates from skewed clients | no score gain or underfitting | medium, code change and clipping threshold search |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | FedProx mu 1e-5 on older m0.40/client-momentum0.9 stack scored 0.898300 | not tested with current m0.35/client-momentum0.925 best | select |
| P2 | FedAdam aggressive `server_lr=1.0,tau=1e-3` crashed | conservative tau/lr changes root instability | select |
| P3 | Default SCAFFOLD scored 0.854800 | not tested with tuned client settings, but prior null is strong | reserve |
| P4 | no direct duplicate | requires code edit and threshold search | reserve |

## Proposal scoring

Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 FedProx current stack | 4 | 5 | 5 | 5 | 3 | 2 | 29 |
| P2 conservative FedAdam | 3 | 3 | 4 | 5 | 4 | 2 | 23 |
| P3 tuned SCAFFOLD | 2 | 3 | 4 | 5 | 3 | 2 | 20 |
| P4 clipping | 2 | 4 | 2 | 3 | 5 | 2 | 20 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m035_epochs7_clientlr004_cm0925_wd5e4_fedprox1e5` | best stack plus `--fedproxloss_mu 1e-5` |
| 2 | P2 | `fedadam_lr02_tau01_epochs7_clientlr004_cm0925_wd5e4` | best client/local settings plus conservative FedAdam `--server_lr 0.2 --fedopt_tau 0.1` |

## Reflective memory

- Keep: FedProx is the highest-priority source-backed near miss because it is CLI-only and had a previous near-best score.
- Discard: shared-data non-IID remedies are out of budget because `data/*` and evaluation/data substrate must remain fixed.
- Do not retry: aggressive FedAdam `server_lr=1.0,tau=1e-3` without stabilizer.
- Sources to carry forward: Li20 FedProx, Reddi21 FedOpt, Karimireddy20 SCAFFOLD, Zhang21 clipping.

### Batch outcome

- FedProx `mu=1e-5` on the current best stack scored 0.897600, below the 0.898900 best; do not rerun that exact setting.
- Conservative FedAdam `server_lr=0.2,tau=0.1` completed without NaNs but scored 0.807800; adaptive Adam is a poor fit for the current custom aggregator/settings.
- Carry forward SCAFFOLD as the remaining source-backed drift correction, but treat it as a labeled protocol-mode comparison because the default SCAFFOLD audit was weak.
- Tuned SCAFFOLD with lr 0.04 and 0.02 scored 0.886800 and 0.884000, so do not retry SCAFFOLD on this optimizer/local budget without a stronger implementation change.
