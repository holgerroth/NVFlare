# Literature loop worksheet

Use this worksheet only when progress stalls or the next axis is unclear. Keep entries short and source-backed.

## Trigger

- Reason: after batch 27, every allowed CLI axis (aggregator family, server_lr, server_momentum, epochs at bound, client lr/momentum, weight decay, eta floor, prox mu, label smoothing, clipping, arch variants) is resolved or re-checked at the current stack; single-knob moves bounce within +/-0.003.
- Current best: 0.9044 — FedAvgM slr 1.75 / sm 0.2 over DIFFs; client SGD lr 0.06, mom 0.9, wd 2.5e-4, cosine floor 1e-4, prox mu 1e-4, label smoothing 0.05, 8 local epochs, 20 rounds, alpha 0.5.
- Recent symptoms from `results.tsv`: rows 77-82 all discards within noise of best; last material gains came from local compute (e8), regularization (wd/ls/prox), and scheduler depth.
- Confirmed null/worse ideas to avoid: FedAdam (diverges/weak within server_lr>=0.1 bound), SCAFFOLD (0.8569), grad clipping (hurts), no-scheduler (0.77), arch variants norm/small_head (worse), server_lr>=2.0 at e8 (NaN), client lr>=0.09 (NaN), client momentum!=0.9, shallow eta floors.
- Candidate width: 2 for 8-epoch stacks (cap headroom; e8 runs ~1100-1150s of the 1200s cap).
- Ledger event: timer started via `scripts/log_literature_review.py --start` (plateau after batch 27).

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| FL server-side model averaging SWA/EMA global model few rounds | improve final persisted global model without protocol change | WebSearch + arXiv | found Server Averaging, WiMA |
| mixup local training FL non-IID CIFAR-10 client drift | cheap client-loop augmentation; harness has only crop/flip | WebSearch + arXiv | ~2% CIFAR-10 gains reported in FL settings |
| sharpness-aware minimization FedSAM flat minima FL generalization | plateau may be a sharp-minimum generalization limit | WebSearch + arXiv | FedSAM strong evidence; 2x per-step cost |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| R1 | Server Averaging for Federated Learning, 2021 | arXiv:2103.11619 | late-round oscillation of global model | server-side model averaging (fed back) | keep |
| R2 | Window-based Model Averaging Improves Generalization in Heterogeneous FL, 2023 | arXiv:2310.01366 | SWA-style averaging unstable early | windowed average of round-wise globals | keep |
| R3 | Generalized FL via Sharpness Aware Minimization (FedSAM), 2022 | arXiv:2206.02618 | local sharp minima generalize poorly globally | client-side SAM | keep (reserve; runtime) |
| R4 | mixup: Beyond Empirical Risk Minimization, 2018 | arXiv:1710.09412 | overfitting/memorization in local training | batch-level mixup | keep |
| R5 | FedDC: local drift decoupling, CVPR 2022 | CVPR 2022 open access | client drift under non-IID | persistent local drift variable | reject (complexity; scaffold-family already below FedAvgM) |
| R6 | Beyond Local Sharpness (FedGloSS), 2024 | arXiv:2412.03752 | local vs global flatness mismatch | server-side sharpness | reject (needs extra communication/protocol coupling) |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | round-to-round oscillation of the aggregated global model wastes late rounds | R1/R2: averaging past globals smooths trajectory and improves generalization | eta-floor rows oscillate 0.8988-0.9044 on tiny changes = trajectory noise at convergence | ranked artifact IS the final persisted global model | tasks/shared/custom_aggregators.py |
| C2 | local training memorizes small non-IID shards | R4: mixup regularizes beyond crop/flip; ~2% FL CIFAR-10 reports | wd/ls both gave real gains -> regularization axis is productive | harness has only crop/flip; loss-level smoothing kept | tasks/cifar10/client.py loop |
| C3 | sharp local minima degrade the aggregated global model | R3: FedSAM +1-2% non-IID CIFAR | plateau despite tuned optimizer; flatness never addressed | SAM is client-loop-only but doubles per-step cost vs 1200s cap | tasks/cifar10/client.py loop (reserve) |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | server-side window average of round-wise global models, fed back as the broadcast/persisted global | R1 arXiv:2103.11619, R2 arXiv:2310.01366 | FedAvgMAggregator tracks cumulative emitted updates; emits adjusted DIFF so state = mean of last W round models; `--server_window_avg W` | smoother late rounds; +0.002-0.01 | window-averaged run scores <= 0.9044 | low: aggregator-only, keys/DIFF preserved |
| P2 | batch-level mixup in the client loop (Beta(a,a), per-site seeded RNG) | R4 arXiv:1710.09412 | client.py `--mixup_alpha 0.2`; mixed CE on permuted batch, works with label smoothing | stronger local regularization; +0.003-0.01 | mixup run scores <= 0.9044 | low-med: training-loop only; eval untouched |
| P3 | SAM local steps (rho 0.05) at reduced epochs to fit cap | R3 arXiv:2206.02618 | client.py SAM wrapper around SGD; run at e4 (2x step cost ~ e8 wall time) | flatter local minima; +0.005-0.015 | SAM@e4 <= 0.9044 | med: loop change + local-compute mode shift |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | none (server momentum tunes step, not trajectory averaging) | none | run |
| P2 | none (only loss-level smoothing kept; input-level untested) | none | run |
| P3 | none | clipping-hurts suggests large steps useful; SAM perturbs, not clips | reserve |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 window avg | 3 | 5 | 4 | 4 | 4 | 1 | 27 |
| P2 mixup | 3 | 4 | 5 | 4 | 4 | 1 | 26 |
| P3 FedSAM | 4 | 4 | 3 | 5 | 4 | 4 | 24 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | wima_w5 | `--server_window_avg 5` at best stack |
| 2 | P2 | mixup_02 | `--mixup_alpha 0.2` at best stack |
| 3 | P1 variant (next batch if P1 leads) | wima_w3 / wima_w8 | window sensitivity |
| 4 | P3 (reserve) | fedsam_e4 | SAM rho 0.05 at aggregation_epochs 4 |

## Reflective memory

- Keep: whichever of P1/P2 beats 0.9044; combine if both do.
- Discard: proposals losing to best under identical budget.
- Do not retry: FedAdam-family, clipping, shallow floors, arch variants (this campaign).
- Sources to carry forward: arXiv:2103.11619, 2310.01366, 1710.09412, 2206.02618.
