# Literature loop worksheet

Use this worksheet only when progress stalls or the next axis is unclear. Keep entries short and source-backed.

## Trigger

- Reason: `plateau_watchdog.py` printed `recommendation=literature` after the corrected scheduler floor sweep reached 33 scored candidates since the last material improvement.
- Current best: `0.899900`, FedAvgM, `aggregation_epochs=8`, `server_lr=1.75`, `server_momentum=0.15`, `weight_decay=4e-4`, `model_arch=moderate_cnn`, `alpha=0.5`, final eval `site-1`.
- Recent symptoms from `results.tsv`: FedAvgM is strong but saturated; server LR/momentum micro sweeps tied or missed; scheduler floor, client LR/momentum, exact steps, robust median, FedProx light/medium, and extra weight-decay refinements all missed.
- Confirmed null/worse ideas to avoid: no scheduler (`0.705300`), median aggregation (`0.787900`), weighted FedAvg under ep8 (`0.881100`), FedProx `1e-5`/`1e-4` under the best stack (`0.893600`/`0.888500`), exact local steps `768-1000`, lower server momentum `0.05/0.10`, and the wrong `--eta_min_factor` flag.
- Candidate width: 2 on one local 80 GB H100, pinned with `CUDA_VISIBLE_DEVICES=0`, reduced from the profile default because ep8 candidates are long and previous wider batches caused resource pressure.
- Ledger event: started with `scripts/log_literature_review.py --start`; finish with `--finish` before launching the next batch.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| FedAvgM server momentum non-IID CIFAR-10 visual classification | Current best is FedAvgM; need understand whether remaining improvement likely comes from momentum or a different drift-control mechanism. | Google Research, arXiv mirrors, FedLab docs | Hsu19 supports server momentum and notes tuning sensitivity. |
| adaptive federated optimization FedAdam FedYogi heterogeneous non-IID | FedAdam crashed in calibration at an aggressive setting; check whether a damped source-backed retry is justified. | ICLR, OpenReview link from ICLR, arXiv mirrors | Reddi21 motivates adaptive server optimizers but prior crash requires low server LR and larger tau. |
| SCAFFOLD client drift local steps non-IID federated learning | Plateau looks like client drift after long local training; check whether existing SCAFFOLD mode deserves a tuned retry. | PMLR, arXiv mirrors, ADS | Karimireddy20 directly targets client drift and heterogeneity. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://proceedings.mlr.press/v54/mcmahan17a.html | Communication-limited FL with local model averaging under non-IID and unbalanced data. | FedAvg/local epochs | Keep as baseline context. |
| Hsu19 | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification / 2019 | https://research.google/pubs/measuring-the-effects-of-non-identical-data-distribution-for-federated-visual-classification/ | CIFAR-10 visual FL degrades under non-identical data; server momentum mitigates. | FedAvgM/server momentum | Keep; explains current best and sensitivity. |
| Karimireddy20 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning / 2020 | https://proceedings.mlr.press/v119/karimireddy20a.html | FedAvg suffers client drift under heterogeneity; control variates correct drift. | SCAFFOLD/control variates | Keep; implemented and profile-supported. |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://iclr.cc/virtual/2021/poster/2691 | FedAvg can be hard to tune and has unfavorable convergence under heterogeneity. | FedAdam/FedOpt | Keep; implemented, but needs damped settings after prior crash. |
| Li20 | Federated Optimization in Heterogeneous Networks / 2020 | https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html | Statistical and systems heterogeneity destabilize FedAvg; proximal loss can stabilize. | FedProx | Keep as reserve; low/medium mu already missed here. |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://papers.nips.cc/paper_files/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html | Variable local work creates objective inconsistency and biased aggregation. | FedNova/normalized averaging | Reject for next batch: all clients use equal local work and implementation would add a new aggregator. |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client drift from non-IID local training remains after FedAvgM tuning. | Karimireddy20 says heterogeneity causes local-update drift; Hsu19 shows non-identical CIFAR-10 strongly degrades FedAvg. | Ep8 improves a lot, but client LR/momentum, exact steps, scheduler floors, and FedProx light/medium all miss the `0.899900` FedAvgM best. | Existing SCAFFOLD mode can correct drift without changing DIFF parameter keys. | `tasks/cifar10/client.py`, `tasks/shared/custom_aggregators.py`, `--aggregator scaffold` |
| C2 | Server update rule may need per-coordinate adaptivity, but instability must be damped. | Reddi21 proposes federated adaptive optimizers for heterogeneous non-convex FL. | FedAdam calibration crashed at `server_lr=1.0`, `tau=1e-3`; FedAvgM micro-tuning now only ties or misses. | Current FedAdam implementation preserves DIFF keys and can be retried with lower LR and larger tau. | `tasks/shared/custom_aggregators.py`, `--aggregator fedadam` |
| C3 | Objective/step normalization is a theoretical issue, but this campaign currently uses equal local work. | Wang20 shows variable local updates can bias naive weighted averaging. | Exact local step sweeps did not beat epoch mode, and all clients share the same local-compute rule each round. | FedNova-like changes are lower priority and would require new aggregator code. | Defer; not selected without stronger symptom. |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Tuned SCAFFOLD under the best local-compute and regularization stack. | Karimireddy20, Hsu19 | `--aggregator scaffold --aggregation_epochs 8 --local_train_steps 0 --weight_decay 4e-4` plus fixed CIFAR budget. | Reduce non-IID client drift while preserving the successful ep8/weight-decay regime. | Score does not beat `0.899900`, or SCAFFOLD metadata mode crashes. | Medium: uses existing profile-supported control-variate metadata. |
| P2 | Damped FedAdam server optimizer retry. | Reddi21 | `--aggregator fedadam --server_lr 0.1 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-2 --aggregation_epochs 8 --weight_decay 4e-4`. | Per-coordinate server adaptivity may escape the FedAvgM plateau; low LR and larger tau should reduce NaN risk. | Crash/NaN repeats, or score remains below `0.898000`. | Medium: previous aggressive FedAdam crashed, but this uses implemented bounds. |
| P3 | Stronger FedProx reserve candidate. | Li20 | `--aggregator fedavgm --server_lr 1.75 --server_momentum 0.15 --fedproxloss_mu 1e-3 --aggregation_epochs 8 --weight_decay 4e-4`. | Stronger proximal pull may reduce local drift more than `1e-5`/`1e-4`. | Score stays below the lower-mu FedProx rows or under `0.895000`. | Low, but novelty is weak because smaller mu already missed. |
| P4 | FedNova-style normalized aggregation. | Wang20 | New normalizing aggregator over client deltas and local step counts. | Address objective inconsistency if variable local work becomes active. | Equal-step setting shows no symptom; implementation adds complexity without immediate need. | High for this loop; reject for next batch. |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | Calibration SCAFFOLD used default ep4/no regularization, not the best stack. | Prior SCAFFOLD `0.854800` is not same stack. | Launch. |
| P2 | Calibration FedAdam was aggressive and unregularized. | Prior FedAdam crash at `server_lr=1.0`, `tau=1e-3`; retry only with damping. | Launch with low LR/high tau. |
| P3 | FedProx light/medium under best stack. | `mu=1e-5` and `1e-4` were worse. | Reserve, do not launch unless P1/P2 fail and watchdog resets through literature. |
| P4 | New normalized aggregation. | Equal local work makes objective-inconsistency diagnosis weak. | Reject for this batch. |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 4 | 4 | 4 | 5 | 4 | 3 | 26 |
| P2 | 4 | 3 | 4 | 4 | 4 | 3 | 23 |
| P3 | 2 | 5 | 5 | 3 | 2 | 3 | 21 |
| P4 | 2 | 2 | 1 | 4 | 5 | 4 | 14 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `r32_lit_scaffold_ep8_wd4e4` | `--cross_site_eval --n_clients 8 --num_rounds 20 --aggregation_epochs 8 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1 --aggregator scaffold --weight_decay 4e-4` |
| 2 | P2 | `r32_lit_fedadam_slr01_tau1e2_wd4e4` | `--cross_site_eval --n_clients 8 --num_rounds 20 --aggregation_epochs 8 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1 --aggregator fedadam --server_lr 0.1 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-2 --weight_decay 4e-4` |
| 3 | P3 | reserve | Do not launch in this batch. |
| 4 | P4 | rejected | Do not implement without a variable-local-work symptom or human-approved protocol subcampaign. |

## Reflective memory

- Keep: FedAvgM ep8 `server_lr=1.75`, `server_momentum=0.15`, `weight_decay=4e-4` remains the score target.
- Discard: routine jitter around server LR/momentum, client LR/momentum, scheduler floors, exact local steps, and weight decay is exhausted for now.
- Do not retry: invalid `--eta_min_factor`; use `--cosine_lr_eta_min_factor`.
- Sources to carry forward: Hsu19 for FedAvgM/server momentum; Karimireddy20 for SCAFFOLD/client drift; Reddi21 for damped adaptive server optimization; Li20 for FedProx reserve; Wang20 only if local work becomes variable.
