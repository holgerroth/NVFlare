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
- Discard: local FedAvgM momentum jitter around `0.2` unless a new mechanism changes the context; FedAvgM+FedProx `mu=1e-3`; safer FedAdam `server_lr=0.1`, `tau=1e-2`; FedLC `tau=0.5` and `1.0`; exact local steps `300` and `400`.
- Do not retry: FedAdam `server_lr=1.0`, `tau=1e-3`; FedAdam low-LR/tau retry without a new stabilizer; FedProx `1e-5`/`1e-4` with weighted FedAvg; FedProx `1e-3` with current best FedAvgM.
- Sources to carry forward: source-backed CLI probes did not beat FedAvgM; next exploration should move to the registered architecture calibration path before considering more invasive FedNova-style aggregation.

## Second Literature Loop

### Trigger

- Reason: two batches failed after the new best FedAvgM stack `server_lr=1.5`, `server_momentum=0.4`, `weight_decay=3e-4`, score `0.881400`.
- Recent symptoms: weight-decay retune and server-LR revisit both regressed; prior FedLC was close but not worth keeping.
- Candidate width: `PARALLEL_CANDIDATES=2`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedSAM sharpness aware minimization federated learning non-IID CIFAR-10 arXiv` | Explore flat-minima methods after weight decay helped. | arXiv, Hugging Face Papers | FedSAM-style methods are relevant but more invasive and costly. |
| `label smoothing federated learning non-IID arXiv` | Find simpler overconfidence regularization compatible with current client loop. | arXiv, MDPI, PyTorch docs | Label smoothing is client-local and supported by `CrossEntropyLoss`. |
| `federated learning robust aggregation median non-IID CIFAR-10 arXiv` | Consider robust aggregation if client updates are outlier-prone. | arXiv, paper indexes | Existing `median` aggregator is available but evidence is weaker for benign non-IID CIFAR. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Cho26 | FedENLC: An End-to-End Noisy Label Correction Framework in Federated Learning / 2026 | https://www.mdpi.com/2227-7390/14/2/290 | Non-IID plus noisy/biased local labels can overfit and become overconfident. | SCE plus label smoothing | keep |
| Soltany24 | Federated Domain Generalization with Label Smoothing and Balanced Decentralized Training / 2024 | https://arxiv.org/abs/2412.11408 | Heterogeneous client domains hurt generalization. | Label smoothing plus balanced training | keep |
| Foret20 | Sharpness-Aware Minimization for Efficiently Improving Generalization / 2020 | https://arxiv.org/abs/2010.01412 | Sharp minima can generalize poorly. | SAM | reserve |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | Non-IID FL benefits from flatness-aware local optimization. | FedSAM | reserve |
| PyTorch | `torch.nn.CrossEntropyLoss(label_smoothing=...)` | https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html | Implementation support for soft targets without custom loss code. | Client-local loss flag | keep |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Overconfident local classifiers | FedENLC and FedSB use label smoothing to stabilize heterogeneous FL training. | Weight decay helped, suggesting regularization matters; FedLC came close. | Label smoothing is a simpler client-local regularizer than FedLC. | `client.py`, `job.py`. |
| C2 | Sharp-minima generalization | SAM/FedSAM papers target flatter minima for non-IID FL. | Current best emerged from regularization plus momentum. | Potential next code path if label smoothing fails. | `client.py`, but higher compute. |
| C3 | Outlier updates | Robust aggregation papers target harmful client updates. | Median has not been audited under the current best client regularization. | CLI-only existing `median` aggregator can be tested later. | `custom_aggregators.py` via `--aggregator median`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P6 | Add optional label smoothing to local CE loss. | Cho26; Soltany24; PyTorch CE docs | Code: `--label_smoothing`; candidates `0.05`, `0.1` with current best FedAvgM stack. | Reduce local overconfidence and improve generalization. | Both scores below `0.881400`. | Low; client-local, default off. |
| P7 | FedSAM-style client optimizer. | Foret20; Qu22 | Code: SAM perturbation around local SGD. | Flatter minima under heterogeneity. | Runtime too high or no gain. | Medium-high; extra backward pass. |
| P8 | Median aggregation audit with best client regularization. | Robust aggregation literature | CLI-only: `--aggregator median --weight_decay 3e-4`. | Reduce effect of outlier client updates. | Score below current best. | Low but weaker evidence. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P6 | 3 | 5 | 5 | 4 | 4 | 2 | 27 |
| P7 | 4 | 3 | 2 | 4 | 5 | 4 | 23 |
| P8 | 2 | 5 | 5 | 2 | 3 | 2 | 22 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P6 | `fedavgm_lr15_m04_wd3e4_ls005` | Code variant: optional `--label_smoothing`; args `--server_lr 1.5 --server_momentum 0.4 --weight_decay 3e-4 --label_smoothing 0.05` |
| 2 | P6 | `fedavgm_lr15_m04_wd3e4_ls010` | Code variant: optional `--label_smoothing`; args `--server_lr 1.5 --server_momentum 0.4 --weight_decay 3e-4 --label_smoothing 0.1` |

## Third Literature Loop

### Trigger

- Reason: SCAFFOLD/median and scheduler-toggle batches failed after the second literature loop; current best remains FedAvgM `server_lr=1.5`, `server_momentum=0.4`, `weight_decay=3e-4`, score `0.881400`.
- Recent symptoms: regularization helped, but label smoothing did not; scheduler and alternative aggregators regressed.
- Candidate width: `PARALLEL_CANDIDATES=2`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning gradient clipping non-IID CIFAR-10 arXiv client drift` | Look for client-local stabilization lower risk than SAM. | arXiv, paper indexes | Found FedZMG/zero-mean gradients as a parameter-free client-side method. |
| `federated learning exponential moving average client models non-IID arXiv` | Consider EMA/teacher mechanisms after weight decay helped. | paper indexes | Most EMA/distill ideas need extra logits, proxy data, or protocol fields. |
| `FedSAM sharpness aware minimization federated learning non-IID CIFAR-10 arXiv` | Revisit flatness methods as a reserve. | arXiv | FedSAM is relevant but requires extra backward pass and more code. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zantalis26 | FedZMG: Efficient Client-Side Optimization in Federated Learning / 2026 | https://arxiv.org/abs/2602.18384 | Non-IID client drift and optimizer complexity. | Zero-mean gradients / gradient centralization | keep |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | Sharp global minima from local ERM under non-IID data. | FedSAM | reserve |
| Foret20 | Sharpness-Aware Minimization for Efficiently Improving Generalization / 2020 | https://arxiv.org/abs/2010.01412 | Sharp minima and label-noise robustness. | SAM | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client drift from biased local gradients | FedZMG argues zero-mean gradients reduce intensity/bias shifts without extra communication. | FedAvgM plus weight decay helped, but many client/server jitter sweeps plateaued. | A local gradient transform can change client updates without protocol change. | `client.py`. |
| C2 | Flatness/generalization plateau | SAM/FedSAM target sharp minima and non-IID local ERM. | Current best relies on regularization; label smoothing was close but not better. | SAM is plausible but higher compute and more code. | `client.py`, reserve. |
| C3 | Protocol complexity | EMA/distillation methods often require logits, proxy data, or server state. | The harness contract must preserve DIFF uploads and existing metadata. | Reject methods needing new datasets or metadata. | n/a. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P9 | Optional gradient centralization before local SGD step. | Zantalis26 FedZMG | Code: `--gradient_centralization`; candidate with current best stack. | Reduce biased local gradients under label skew. | Score below `0.881400`. | Low; client-local, no new state. |
| P10 | Gradient centralization with lighter weight decay. | Zantalis26 FedZMG | Code flag plus `--weight_decay 1e-4`. | Check whether zero-mean gradients replace some L2 regularization. | Score below current best. | Low-medium; varies regularization too. |
| P11 | Local SAM/FedSAM. | Foret20; Qu22 | Code: `--sam_rho`, extra backward pass. | Improve flatness/generalization. | Runtime near timeout or no gain. | Medium-high; reserve. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P9 | 3 | 5 | 4 | 4 | 5 | 2 | 27 |
| P10 | 3 | 4 | 4 | 4 | 4 | 2 | 24 |
| P11 | 4 | 4 | 2 | 5 | 5 | 4 | 24 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P9 | `fedavgm_lr15_m04_wd3e4_gc` | Code variant: optional `--gradient_centralization`; args `--server_lr 1.5 --server_momentum 0.4 --weight_decay 3e-4 --gradient_centralization` |
| 2 | P10 | `fedavgm_lr15_m04_wd1e4_gc` | Same code flag with `--weight_decay 1e-4` |

## Fourth Literature Loop

### Trigger

- Reason: two consecutive same-budget batches failed after the gradient-centralized FedAvgM best: `server_lr=1.5`, `server_momentum=0.35`, `weight_decay=3.5e-4`, score `0.904600`.
- Recent symptoms: extra server-momentum neighbors (`0.30`, `0.375`) and weight-decay retune (`3e-4`, `4e-4`) regressed. The score surface is narrow around the current optimizer stack.
- Confirmed null/worse ideas to avoid unless context changes: FedProx, FedAdam safe retry, FedLC, label smoothing, median aggregation, SCAFFOLD, scheduler toggles, exact local steps `300/400`, registered architecture variants, and tighter jitter around the same weight-decay/momentum values.
- Candidate width: `PARALLEL_CANDIDATES=2`; use `CUDA_VISIBLE_DEVICES=0` and unique `PYTHONPYCACHEPREFIX`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning local epochs client drift non-IID CIFAR-10 FedAvg local steps arXiv` | Check whether local compute, not another optimizer scalar, should be swept after the optimizer stack plateaued. | arXiv, Hugging Face Papers | FedAvg and FedNova both frame local update count as central to the communication/drift tradeoff. |
| `FedSAM sharpness aware minimization federated learning non-IID CIFAR-10 arXiv` | Revisit the higher-cost flatness reserve after gradient regularization helped. | arXiv, paper indexes | FedSAM is relevant but needs client-side code and extra backward passes. |
| `client-side momentum federated learning non-IID local momentum arXiv` | Look for safe client optimizer variants after server momentum improved but local momentum failed earlier. | arXiv, paper indexes | Client-level momentum/adaptive step-size papers are relevant, but several require algorithmic state not represented by plain SGD momentum. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Communication rounds are costly; FedAvg trades more local work against fewer rounds under non-IID/unbalanced data. | FedAvg local epochs | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Variable local update counts and data sizes can bias naive averaging. | FedNova / local-step normalization | keep |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | Local ERM under non-IID can find sharp valleys and produce local-client deviation. | FedSAM | reserve |
| Kim24 | Adaptive Federated Learning with Auto-Tuned Clients / 2024 | https://arxiv.org/abs/2306.11201 | Client-side hyperparameter tuning is hard under heterogeneous local objectives. | Adaptive client step size | keep in reserve |
| Xu21 | FedCM: Federated Learning with Client-level Momentum / 2021 | https://arxiv.org/abs/2106.10874 | Client heterogeneity and partial participation can bias local SGD; client-level momentum can stabilize it. | Client-level momentum | reserve |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Momentum can improve FedAvg/SCAFFOLD under heterogeneity. | Momentum analysis | keep as interpretation |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local update budget may be mis-sized | FedAvg explicitly trades local work for communication; FedNova shows local update counts can change objective bias under heterogeneity. | Current runs finish in about 6.3 minutes, exact steps failed, but epoch count has not been swept under the gradient-centralized best. | `aggregation_epochs` is an allowed local-compute knob while `num_rounds=20` stays fixed. | CLI-only via `--aggregation_epochs`; no protocol change. |
| C2 | Sharpness/generalization plateau | FedSAM argues local ERM in non-IID FL can drive the global model into sharp valleys. | Gradient centralization and weight decay helped, suggesting optimizer geometry matters. | SAM is plausible if cheaper CLI sweeps fail. | `client.py`, optional flag, reserve due runtime/code cost. |
| C3 | Client optimizer mismatch | Auto-tuned clients and FedCM target local objective heterogeneity with client-side adaptation or momentum-like correction. | Server momentum helped, but old client momentum settings failed before gradient centralization. | Current gradient-centralized stack may shift the viable client LR/momentum range. | CLI `--lr`/`--momentum` for simple probes; richer FedCM state would be rejected. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P12 | Sweep epoch-based local compute around 4. | McMahan17; Wang20 | CLI-only: current best stack with `--aggregation_epochs 3` and `5`, keep `--local_train_steps 0`. | Test whether less drift or more local fitting improves the fixed 20-round score. | Both scores below `0.904600` or timeout. | Low; existing budget knob. |
| P13 | Retune server learning rate under gradient centralization. | Reddi21; Cheng23 | CLI-only: current best stack with `--server_lr 1.25` and `1.75`. | Check whether the new momentum/GC stack changes the best server step scale. | Both below `0.904600`. | Low. |
| P14 | Narrow client learning rate under the new client-gradient geometry. | Kim24 | CLI-only: current best stack with `--lr 0.04` and `0.06`. | Adjust client update scale after gradient centralization changed gradients. | Both below `0.904600`. | Low. |
| P15 | Local SAM/FedSAM. | Qu22 | Code: optional SAM perturbation, e.g. `--sam_rho 0.02/0.05`. | Improve flatness/generalization. | Runtime near cap, instability, or no gain. | Medium-high; extra backward pass. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P12 | Exact local-step rows `300/400` only partially overlap. | Epoch count under gradient centralization has not been tested; keep only one local-compute mode in the sweep. | keep |
| P13 | Older server-LR revisit. | Prior `1.25/1.75` used `server_momentum=0.4`, `weight_decay=3e-4`, no gradient centralization; context differs. | keep in reserve |
| P14 | Older client-LR sweep. | Prior `0.03/0.07` used a weaker pre-GC stack; choose narrower values if used. | keep in reserve |
| P15 | Prior SAM reserve. | Still untested but more invasive than CLI local-compute sweep. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P12 | 3 | 5 | 5 | 4 | 4 | 3 | 26 |
| P13 | 2 | 5 | 5 | 4 | 3 | 2 | 24 |
| P14 | 2 | 5 | 5 | 3 | 3 | 2 | 23 |
| P15 | 4 | 4 | 2 | 5 | 5 | 4 | 24 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P12 | `fedavgm_lr15_m035_wd35e5_gc_ep3` | CLI-only: current best stack, `--aggregation_epochs 3 --local_train_steps 0` |
| 2 | P12 | `fedavgm_lr15_m035_wd35e5_gc_ep5` | CLI-only: current best stack, `--aggregation_epochs 5 --local_train_steps 0` |

### Reflective Memory

- Keep: FedAvgM `server_lr=1.5`, `server_momentum=0.35`, `weight_decay=3.5e-4`, `--gradient_centralization`, epoch-based `aggregation_epochs=4` remains the current best until a local-compute run beats `0.904600`.
- Discard: further tight jitter around `server_momentum=0.35` or `weight_decay=3.5e-4` without a new mechanism.
- Reserve next: if epoch sweep fails, try source-backed `server_lr` retune under the GC best, then narrower client LR; keep SAM as the higher-cost code mutation.
- Outcome: `aggregation_epochs=5` beat the previous best with `0.906500`; `aggregation_epochs=3` regressed to `0.895500`. Continue local-compute narrowing upward while staying within `RUN_TIMEOUT_SECONDS=1200`.
- Follow-up: `aggregation_epochs=6` scored `0.904600`; `7` scored `0.906300`. Treat `5` as the current local-compute peak and retune optimizer knobs in that context.
- Follow-up: epoch-5 server momentum `0.40` scored `0.905600`; `0.30` scored `0.904700`. Keep `server_momentum=0.35` unless a new source-backed mechanism changes context.

## Fifth Literature Loop

### Trigger

- Reason: two post-epoch-5 batches failed after best score `0.906500`: upward local epochs and epoch-5 server momentum.
- Current best: FedAvgM `server_lr=1.5`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`.
- Recent symptoms: local epochs `7` came close but did not improve; server momentum around `0.35` did not help at epoch 5.
- Candidate width: `PARALLEL_CANDIDATES=2`; use unique `RUN_LOG`, `--name`, and `PYTHONPYCACHEPREFIX`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Adaptive Federated Optimization server learning rate FedAvgM local epochs non-IID arXiv` | Server step scale may need retuning after the local epoch increase. | arXiv | Reddi21 supports FedOpt/FedAvgM tuning under heterogeneity. |
| `Adaptive Federated Learning with Auto-Tuned Clients learning rate heterogeneous federated arXiv` | More local epochs make the client step size more important. | ICLR proceedings, arXiv mirrors | Kim24 motivates client-side step-size adaptation for heterogeneous local objectives. |
| `federated learning SAM local optimizer non-IID CIFAR10 FedSAM arXiv` | Keep a higher-cost code reserve if simple step-size probes fail. | arXiv | FedSAM remains relevant but costs extra backward passes. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | FedAvg can be difficult to tune; server optimizers interact with heterogeneity and communication efficiency. | FedOpt / FedAvgM | keep |
| Kim24 | Adaptive Federated Learning with Auto-Tuned Clients / 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/d850b7e0cdc7f1c0820c6ad85405ae94-Abstract-Conference.html | Client-side hyperparameter tuning is hard when local data and smoothness differ. | Client step-size adaptation | keep |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | Non-IID local ERM can push global models toward sharp valleys. | FedSAM | reserve |
| Zantalis26 | FedZMG / 2026 | https://arxiv.org/abs/2602.18384 | Client drift can be reduced by zero-mean gradient projection. | Gradient centralization | already kept |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client step size may be off after more local work | Kim24 targets client-side step-size adaptation under heterogeneous local objectives. | `aggregation_epochs=5` improved, but more epochs and momentum changes did not. | A narrower LR around default `0.05` may improve the stronger local training budget. | CLI `--lr`; `client.py`. |
| C2 | Server step scale may be stale | Reddi21 shows server optimization scale matters under heterogeneity. | Momentum retune at epoch 5 failed, but server LR has not been retuned in this context. | Existing FedAvgM `--server_lr` can be swept without protocol changes. | CLI `--server_lr`; `custom_aggregators.py`. |
| C3 | Flatness remains a reserve mechanism | Qu22 argues SAM improves local learning generality under non-IID. | Regularization and gradient centralization helped, but simple knobs are still available. | Implement only if CLI step-size probes stall. | `client.py`, reserve. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P16 | Narrow client LR around default under epoch 5. | Kim24 | CLI-only: current best stack with `--lr 0.04` and `0.06`. | Adjust local update scale after stronger local compute and gradient centralization. | Both scores below `0.906500`. | Low. |
| P17 | Retune server LR under epoch 5. | Reddi21 | CLI-only: current best stack with `--server_lr 1.25` and `1.75`. | Check whether server step scale should follow the local epoch increase. | Both scores below best. | Low. |
| P18 | Local SAM/FedSAM. | Qu22 | Code: optional SAM perturbation with small `rho`. | Improve flatness/generalization. | Runtime too high or score below best. | Medium-high. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P16 | 3 | 5 | 5 | 4 | 4 | 2 | 27 |
| P17 | 2 | 5 | 5 | 4 | 3 | 2 | 24 |
| P18 | 4 | 4 | 2 | 5 | 5 | 4 | 24 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P16 | `fedavgm_lr15_m035_wd35e5_gc_ep5_lr004` | CLI-only: current best stack with `--lr 0.04` |
| 2 | P16 | `fedavgm_lr15_m035_wd35e5_gc_ep5_lr006` | CLI-only: current best stack with `--lr 0.06` |

### Reflective Memory

- Keep current best unchanged until an epoch-5 client LR candidate beats `0.906500`.
- Do not retry broad client LR values `0.03` or `0.07`; use narrow probes around default `0.05`.
- If client LR fails, run source-backed server LR retune before considering SAM code.
- Outcome: client LR `0.06` scored `0.901400`; `0.04` scored `0.900600`. Keep default `lr=0.05` and proceed to the server-LR reserve.
- Outcome: server LR `1.75` improved to `0.907300`; `1.25` regressed to `0.898900`. Continue with a narrow server-LR sweep before trying SAM code.
- Outcome: server LR `1.875` improved further to `0.909900`; `1.625` scored `0.907100`. Continue upward carefully while keeping FedAdam crash history in mind.
- Follow-up: server LR `2.0` scored `0.908800`; `2.125` scored `0.908200`. Treat `1.875` as a local peak and tighten on both sides.
- Follow-up: server LR `1.8125` scored `0.909800`; `1.9375` scored `0.909600`. Keep `1.875` and move to a new source-backed mechanism.

## Sixth Literature Loop

### Trigger

- Reason: two server-LR batches failed after best `server_lr=1.875`, score `0.909900`.
- Current best: FedAvgM `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`.
- Recent symptoms: server-LR neighbors around the peak are close but do not improve; client LR and local epoch narrowing already failed.
- Candidate width: `PARALLEL_CANDIDATES=2`; SAM candidates may run longer because each local step uses two backward passes.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedSAM sharpness aware minimization federated learning non-IID CIFAR-10 arXiv 2206.02618` | Check whether a client-local optimizer change is justified after optimizer-scalar tuning stalls. | arXiv, paper indexes | FedSAM directly targets non-IID local ERM sharpness. |
| `Sharpness Aware Minimization efficient improving generalization arXiv 2010.01412` | Confirm base SAM implementation mechanism and cost. | arXiv, Hugging Face Papers | SAM is a min-max optimizer with an extra backward pass. |
| `federated learning SAM local optimizer non-IID CIFAR10 FedSAM arXiv` | Look for FL-specific framing and falsifiers. | arXiv, ResearchTrend | FedSAM reports reduced client deviation but increases local compute. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Foret20 | Sharpness-Aware Minimization for Efficiently Improving Generalization / 2020 | https://arxiv.org/abs/2010.01412 | ERM can find sharp minima with poor generalization; SAM minimizes loss and sharpness. | SAM | keep |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | Non-IID FL local ERM can create sharp valleys and client deviation. | FedSAM / MoFedSAM | keep |
| Zantalis26 | FedZMG / 2026 | https://arxiv.org/abs/2602.18384 | Client-side gradient geometry matters under non-IID data. | Gradient centralization | already kept |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Server optimizer tuning helped but now has a local peak. | FedOpt | already exploited |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Sharp local minima after stronger local training | SAM and FedSAM target sharpness from local ERM. | `aggregation_epochs=5` and FedAvgM tuning improved, but scalar retunes are now near-flat. | A client-local sharpness step may improve generalization without changing DIFF uploads. | `client.py`, `job.py`. |
| C2 | Runtime cost from extra backward pass | SAM requires a perturb-and-second-gradient step. | Normal epoch-5 runs take about 7.5 minutes with width 2. | SAM must stay under `RUN_TIMEOUT_SECONDS=1200`; small `rho` values first. | `client.py`. |
| C3 | Protocol preservation | FedSAM can be expressed as a local optimizer and does not require new server tensors for plain SAM. | SCAFFOLD/FedAdam protocol variants underperformed or were risky. | Use default-off `--sam_rho`; no new metadata or dependencies. | `client.py`, `job.py`, `mutation_schema.yaml`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P19 | Optional local SAM with small radius. | Foret20; Qu22 | Code: `--sam_rho`; candidates `0.01` and `0.02` with current best stack. | Improve flatness/generalization after local ERM tuning stalls. | Runtime exceeds cap, crash, or both scores below `0.909900`. | Medium; client-local extra backward. |
| P20 | Retune weight decay under `server_lr=1.875`. | FedZMG; Reddi21 | CLI-only: current best stack with `weight_decay=3e-4` and `4e-4`. | New server LR may shift regularization optimum. | Both below best. | Low, but more jitter. |
| P21 | Retune server momentum under `server_lr=1.875`. | Cheng23; Reddi21 | CLI-only: current best stack with `server_momentum=0.30` and `0.40`. | New server LR may shift momentum optimum. | Both below best. | Low, but repeated axis. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P19 | 4 | 4 | 2 | 5 | 5 | 4 | 24 |
| P20 | 2 | 5 | 5 | 3 | 2 | 2 | 22 |
| P21 | 2 | 5 | 5 | 3 | 2 | 2 | 22 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P19 | `fedavgm_lr1875_m035_wd35e5_gc_ep5_sam001` | Code variant: optional `--sam_rho`; args `--sam_rho 0.01` with current best stack |
| 2 | P19 | `fedavgm_lr1875_m035_wd35e5_gc_ep5_sam002` | Same code flag with `--sam_rho 0.02` |

### Reflective Memory

- Keep current best unchanged until a SAM candidate beats `0.909900`.
- If SAM fails or times out, revert the optional SAM code and return to lower-risk regularization retunes under `server_lr=1.875`.
- Outcome: SAM/FedSAM underperformed and added runtime cost. `sam_rho=0.01` scored `0.907600` in 830 seconds; `sam_rho=0.02` scored `0.907500` in 832 seconds.
- Action: reverted the optional SAM code and returned to P20, the lower-risk weight-decay retune under `server_lr=1.875`.
- Outcome: P20 did not improve. `weight_decay=3e-4` scored `0.907500`; `weight_decay=4e-4` scored `0.903900`.
- Action: proceed to P21, the server-momentum retune under `server_lr=1.875`, before starting another literature loop.
- Outcome: P21 did not improve. `server_momentum=0.40` scored `0.908800`; `server_momentum=0.30` scored `0.906600`.
- Action: sixth-loop proposals are exhausted; start a new literature loop before further local jitter.

## Seventh Literature Loop

### Trigger

- Reason: sixth-loop proposals were exhausted after SAM, weight-decay, and server-momentum probes all missed the `0.909900` best.
- Current best: FedAvgM `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`.
- Recent symptoms: stronger local training helped, but scalar retunes around the FedAvgM/GC optimum are now below best.
- Candidate width: `PARALLEL_CANDIDATES=2`; prefer contract-safe aggregation changes that reuse existing DIFF params and metadata.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedNova Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization arXiv 2007.07481 NUM_STEPS aggregation` | Revisit local-step normalization after epoch-5 local training became the best regime. | arXiv, Flower baseline docs | FedNova targets objective inconsistency from unequal local update counts and uses normalized averaging. |
| `Adaptive Federated Optimization FedYogi FedAdam FedAdagrad arXiv 2003.00295 server optimizer federated learning` | Check safer adaptive-server reserves after FedAdam crashed at high LR and underperformed at low LR. | arXiv, paper indexes | Reddi21 includes FedAdagrad and FedYogi in addition to FedAdam/FedAvgM. |
| `Federated Optimization with Doubly Regularized Drift Correction arXiv 2404.08447 client drift` | Look for newer client-drift corrections. | arXiv | FedRed/DANE-style correction is relevant but needs a more invasive local objective/protocol path. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Naive averaging can converge to a mismatched objective when local update counts differ. | FedNova normalized averaging | keep |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | FedAvg-style methods can be difficult to tune under heterogeneous data; adaptive server optimizers may help. | FedOpt / FedYogi / FedAdagrad | reserve |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | Client drift increases communication cost and can hurt performance. | FedRed / DANE-style drift correction | reject for now |
| Flower26 | FedNova baseline documentation / 2026 | https://flower.ai/docs/baselines/fednova.html | Non-IID CIFAR-10 baselines report FedNova improvements over FedAvg across local solvers. | Implementation reference | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Objective inconsistency from local-step heterogeneity | Wang20 frames mismatched objectives from differing local update counts and proposes normalized averaging. | `aggregation_epochs=5` improved best, so local training is useful but may amplify step-weight bias. | Existing `NUM_STEPS_CURRENT_ROUND` is already sent in `FLModel.meta`; no new client metadata needed. | `custom_aggregators.py`, `job.py`. |
| C2 | Adaptive-server alternatives remain partially unexplored | Reddi21 includes Yogi/Adagrad in addition to Adam and FedAvgM. | FedAdam crashed at high LR and scored poorly at low LR, but FedAvgM is strong. | Possible reserve if normalized FedNova fails. | `custom_aggregators.py`, `job.py`. |
| C3 | Drift-correction methods can be too invasive | Jiang24/FedRed addresses drift but changes local objectives and solver assumptions. | SCAFFOLD and SAM code paths did not beat the current simpler stack. | Reject until simpler DIFF-only mechanisms are exhausted. | Higher protocol/code risk. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P22 | FedNova-style normalized DIFF aggregation with optional server momentum. | Wang20; Flower26 | Code: add `--aggregator fednova` that normalizes each client DIFF by `NUM_STEPS_CURRENT_ROUND`, re-scales by weighted average local steps, then applies optional server momentum. Candidates: current FedAvgM LR/momentum and a pure normalized `server_lr=1.0, momentum=0.0` control. | Reduce local-step weighting bias under non-IID epoch-5 training while preserving the DIFF contract. | Both below `0.909900`, NaN, or timeout. | Medium-low; server-only code using existing meta. |
| P23 | Add FedYogi/FedAdagrad server optimizers. | Reddi21 | Code: extend FedOpt second-moment update beyond FedAdam; test conservative server LR. | Adaptive server normalization may stabilize heterogeneous updates. | Repeats FedAdam instability or underperforms FedAvgM. | Medium; server-only but more optimizer state. |
| P24 | Exact local-step retune under current best. | Wang20; McMahan17 | CLI-only: `local_train_steps=500` and `600` with epoch mode disabled by positive steps. | Reduce per-client update-count variation directly. | Both below best or runtime grows too much. | Low, but previous exact-step probes under older stack failed. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P22 | 4 | 5 | 4 | 5 | 4 | 2 | 24 |
| P23 | 3 | 5 | 3 | 4 | 3 | 2 | 20 |
| P24 | 2 | 5 | 5 | 3 | 2 | 2 | 19 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P22 | `fednova_lr1875_m035_wd35e5_gc_ep5` | Code variant: `--aggregator fednova --server_lr 1.875 --server_momentum 0.35` with current best client/local stack |
| 2 | P22 | `fednova_lr10_m00_wd35e5_gc_ep5` | Same code variant with pure normalized averaging, `--server_lr 1.0 --server_momentum 0.0` |

### Reflective Memory

- Keep the current FedAvgM best unchanged unless a FedNova candidate beats `0.909900`.
- If FedNova fails, revert the optional aggregator code unless it is needed for a follow-up source-backed candidate, then move to either FedYogi/FedAdagrad or exact local-step retuning.
- Outcome: FedNova with current server settings improved the best to `0.910300`; pure normalized FedNova regressed to `0.894300`.
- Action: keep the FedNova aggregator code and narrow server LR around `1.875` before trying adaptive FedYogi/FedAdagrad reserves.
- Outcome: FedNova server LR neighbors did not improve. `server_lr=2.0` scored `0.909300`; `server_lr=1.75` scored `0.908700`.
- Action: keep `server_lr=1.875` and retune FedNova server momentum around `0.35`.
- Outcome: FedNova server-momentum neighbors did not improve. `server_momentum=0.40` scored `0.910000`; `server_momentum=0.30` scored `0.909100`.
- Action: keep `server_momentum=0.35` and test whether FedNova shifts the weight-decay optimum.
- Outcome: FedNova weight-decay retune did not improve. `weight_decay=4e-4` scored `0.905800`; `weight_decay=3e-4` scored `0.905700`.
- Action: move to P24 exact local-step retuning under the kept FedNova stack.
- Outcome: exact local-step retuning did not improve. `local_train_steps=600` scored `0.910000`; `local_train_steps=500` scored `0.906100`.
- Action: move to P23, the FedYogi/FedAdagrad adaptive-server reserve.
- Outcome: P23 crashed. FedYogi and FedAdagrad both produced NaN client diffs in round 1 at conservative `server_lr=0.1`, `beta1=0.0`, `tau=1e-2`.
- Action: reverted the optional adaptive-server code and keep the FedNova `0.910300` stack as the active best before a new literature loop.

## Eighth Literature Loop

### Trigger

- Reason: seventh-loop reserves are exhausted; FedNova remains best at `0.910300`, while FedYogi/FedAdagrad crashed with NaN client diffs.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`.
- Recent symptoms: FedNova tuning is near-flat and adaptive optimizers are unstable, suggesting a softer server-side stabilization mechanism.
- Candidate width: `PARALLEL_CANDIDATES=2`; prefer default-off server-only changes.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning adaptive clipping client updates server aggregation non-IID CIFAR arXiv` | Look for update-scale stabilization that does not change client training. | OpenReview, NeurIPS, arXiv indexes | Adaptive clipping tracks update norm quantiles and avoids fixed-threshold guessing. |
| `robust federated aggregation norm clipping client updates non-IID image classification arXiv` | Check whether clipping is used outside privacy-only settings for robust aggregation. | arXiv/paper indexes | Robust FL papers often clip entire update vectors by L2 norm before aggregation. |
| `federated learning clipped averaging client updates robust aggregation non-IID` | Find lower-risk alternatives to median aggregation after median failed. | paper indexes | Clipping is less destructive than coordinate-wise median and keeps weighted averaging. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Andrew21 | Differentially Private Learning with Adaptive Clipping / 2021 | https://openreview.net/forum?id=RUQ1zwZR8_ | Update norm scale depends on architecture, data, LR, and training stage; fixed clipping can be hard to tune. | Adaptive median/quantile clipping | keep |
| McMahan18 | Learning Differentially Private Recurrent Language Models / 2018 | https://arxiv.org/abs/1710.06963 | User-level updates can be large; clipping bounds contribution before averaging/noising. | Update clipping | context |
| Wang20 | FedNova / 2020 | https://arxiv.org/abs/2007.07481 | Local update normalization helps objective mismatch. | FedNova | already kept |
| robust-clipping papers | Robust FL norm clipping variants / 2020-2026 | mixed | Norm clipping can limit amplified or outlier updates. | robust aggregation | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client update scale heterogeneity | Andrew21 notes update norm distributions depend on model, data, LR, and other settings. | FedNova improved normalization but neighboring hyperparameters are near-flat. | Server can observe full DIFFs and compute per-client norms without new metadata. | `custom_aggregators.py`, `job.py`. |
| C2 | Robustness without replacing FedNova | Robust clipping scales entire update vectors rather than using coordinate-wise median. | Median aggregation scored `0.747000`; FedNova averaging is strong. | Clip before FedNova normalized averaging to keep the successful mechanism. | `custom_aggregators.py`. |
| C3 | Threshold selection risk | Andrew21 motivates adaptive quantile clipping because fixed norms are hard to choose. | No current norm telemetry in logs. | Use per-round median update norm times a factor; candidates `1.5` and `2.0`. | Server-only default-off flag. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P25 | FedNova with median-norm update clipping. | Andrew21; Wang20 | Code: `--server_clip_norm_factor`; clip each raw client DIFF to `factor * median(norms)` before FedNova normalization. Candidates `1.5` and `2.0`. | Damp outlier client updates while preserving FedNova. | Both below `0.910300`, crash, or runtime penalty. | Medium-low; server-only and default off. |
| P26 | Fixed norm clipping. | McMahan18; robust FL clipping papers | Code: explicit norm threshold. | Similar outlier damping. | Threshold tuning blind without norm telemetry. | Medium; more brittle. |
| P27 | Drift-correction objective. | Jiang24/FedRed | Code: new local objective. | Reduce client drift directly. | Protocol/local objective complexity. | Higher; defer. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P25 | 3 | 5 | 4 | 4 | 4 | 2 | 22 |
| P26 | 2 | 5 | 3 | 3 | 3 | 2 | 18 |
| P27 | 3 | 3 | 2 | 3 | 4 | 3 | 17 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P25 | `fednova_lr1875_m035_wd35e5_gc_clip15_ep5` | Code variant: `--server_clip_norm_factor 1.5` with the current FedNova best stack |
| 2 | P25 | `fednova_lr1875_m035_wd35e5_gc_clip20_ep5` | Same code variant with `--server_clip_norm_factor 2.0` |

### Reflective Memory

- Keep the current FedNova best unchanged until clipped FedNova beats `0.910300`.
- If clipping fails, revert the optional clipping code unless a follow-up clipped FedNova candidate is explicitly justified by the observed scores.
- Outcome: clipping tied but did not improve. Factors `1.5` and `2.0` both scored `0.910300`, equal to the unclipped FedNova best.
- Action: reverted the optional clipping code because it added server logic without a score gain.

## Ninth Literature Loop

### Trigger

- Reason: clipped FedNova tied but did not improve, and adding more server code has repeatedly failed or tied.
- Current best: unclipped FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, client `lr=0.05`, `--gradient_centralization`.
- Recent symptoms: server-side mechanisms around FedNova are flat; adaptive server optimizers are unstable.
- Candidate width: `PARALLEL_CANDIDATES=2`; prefer CLI-only probes before new code.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Adaptive Federated Learning with Auto-Tuned Clients arXiv 2306.11201 client learning rate federated learning` | Revisit client step-size sensitivity under the new FedNova aggregator. | ICLR proceedings, arXiv mirrors | Client-side step size is a central tuning challenge in FL. |
| `federated learning client learning rate tuning non-IID CIFAR FedAvgM arXiv` | Check whether local LR remains an accepted low-risk axis. | paper indexes | Prior client LR probes were before FedNova and may not transfer. |
| `federated learning cosine learning rate local optimizer tuning non-IID CIFAR arXiv` | Compare scheduler/client step-size mechanisms. | paper indexes | Scheduler toggles underperformed earlier, so use direct LR retune first. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Kim24 | Adaptive Federated Learning with Auto-Tuned Clients / ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/d850b7e0cdc7f1c0820c6ad85405ae94-Abstract-Conference.html | Client-side hyperparameter tuning is difficult in heterogeneous FL. | Client LR adaptation / Delta-SGD | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Local SGD step size and local work control FedAvg behavior. | FedAvg local SGD | context |
| FedZMG26 | FedZMG / 2026 | https://arxiv.org/abs/2602.18384 | Client-side gradient geometry matters under non-IID data. | Gradient centralization | already kept |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | FedNova may shift the client LR optimum | Kim24 emphasizes client-side step-size tuning under FL heterogeneity. | Prior client LR probes were under FedAvgM `server_lr=1.5`, not FedNova. | CLI-only `--lr`; no code or protocol change. | `client.py` args. |
| C2 | Scheduler changes were too coarse | Earlier scheduler toggles regressed. | Current best uses default cosine schedule with `lr=0.05`. | Narrow LR probes preserve scheduler shape. | `client.py`. |
| C3 | Avoid more brittle server code | Adaptive server optimizers crashed; clipping tied. | Best server mechanism is now FedNova. | Use client LR before new mechanisms. | CLI-only. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P28 | Narrow client LR retune under FedNova. | Kim24; McMahan17 | CLI-only: `--lr 0.045` and `0.055` with current FedNova best. | New aggregator may prefer slightly different local SGD step size. | Both below `0.910300`. | Low. |
| P29 | Broader client LR retune. | Kim24 | CLI-only: `--lr 0.04` and `0.06`. | Catch a larger shift. | Prior similar values regressed under FedAvgM. | Low, but repeated. |
| P30 | Implement adaptive client LR. | Kim24 | Code: Delta-SGD-like per-client rule. | Reduce manual tuning. | More code and per-client instability. | Medium-high; reserve. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P28 | 3 | 5 | 5 | 4 | 3 | 2 | 23 |
| P29 | 2 | 5 | 5 | 3 | 2 | 2 | 20 |
| P30 | 4 | 4 | 2 | 4 | 4 | 3 | 21 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P28 | `fednova_lr1875_m035_wd35e5_gc_ep5_lr0045` | CLI-only: `--lr 0.045` with current FedNova best stack |
| 2 | P28 | `fednova_lr1875_m035_wd35e5_gc_ep5_lr0055` | CLI-only: `--lr 0.055` with current FedNova best stack |

### Reflective Memory

- Keep FedNova `lr=0.05` unless a client-LR probe beats `0.910300`.
- If both narrow LR probes fail, either broaden once (`0.04/0.06`) or start a new literature loop before implementing adaptive client LR.
- Outcome: narrow LR probes regressed. `lr=0.045` scored `0.906600`; `lr=0.055` scored `0.906100`.
- Action: skip broader LR jitter and start a new literature loop.

## Tenth Literature Loop

### Trigger

- Reason: FedNova scalar retunes, clipping, adaptive server optimizers, and client LR probes have not improved beyond `0.910300`.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, client `lr=0.05`, `--gradient_centralization`, `model_arch=moderate_cnn`.
- Recent symptoms: optimizer-local space is flat or unstable; registered architecture variants have not been tested under FedNova.
- Candidate width: `PARALLEL_CANDIDATES=2`; label this as an architecture subcampaign because `model_arch` changes.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning normalization layers non-IID convolutional networks FedBN arXiv` | Check whether normalization architecture changes are source-backed under non-IID FL. | OpenReview, arXiv/paper indexes | FedBN motivates normalization handling under feature/non-IID shifts. |
| `federated learning model architecture classifier head parameter efficiency non-IID CIFAR arXiv` | Check whether smaller heads are a plausible regularization architecture axis. | paper indexes | Smaller heads reduce parameters and may regularize, but evidence is weaker. |
| `federated learning architecture search non-IID CIFAR convolutional neural network arXiv` | Confirm architecture calibration is a reasonable next axis after optimizer plateau. | paper indexes | Architecture is higher-level than scalar optimizer jitter but registered locally. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Li21 | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization / ICLR 2021 | https://openreview.net/forum?id=6YEQUn0QICG | Normalization behavior matters under non-IID FL, especially feature shift. | Normalization-aware architecture | keep |
| FedZMG26 | FedZMG / 2026 | https://arxiv.org/abs/2602.18384 | Client-side gradient geometry matters. | Gradient centralization | already kept |
| Wang20 | FedNova / 2020 | https://arxiv.org/abs/2007.07481 | Objective inconsistency from local updates. | FedNova | already kept |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Normalization under non-IID data | FedBN shows normalization choices can affect FL convergence/performance. | FedNova+GC best may interact differently with GroupNorm than older FedAvgM runs. | `moderate_cnn_norm` is registered and under cap. | `model.py` registered variant only. |
| C2 | Over-parameterized classifier head | Smaller heads can regularize and reduce parameter count. | Prior small-head run under older stack scored `0.860900`, but FedNova stack is different. | `moderate_cnn_small_head` is registered and under cap. | `model.py` registered variant only. |
| C3 | Comparability risk | Program requires architecture scores be labeled separately. | Existing best is optimizer-only `moderate_cnn`. | Run as labeled architecture subcampaign and do not silently mix budgets. | Description/ledger labeling. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P31 | Registered GroupNorm architecture audit. | Li21; FedZMG26 | CLI-only: `--model_arch moderate_cnn_norm` with current FedNova best stack. | Normalization may improve non-IID robustness under FedNova. | Score below current best or runtime/cap failure. | Medium; architecture subcampaign. |
| P32 | Registered small-head architecture audit. | model regularization context | CLI-only: `--model_arch moderate_cnn_small_head` with current FedNova best stack. | Less classifier overfitting and fewer parameters. | Score below current best. | Medium; architecture subcampaign. |
| P33 | New architecture variant. | architecture search literature | Code: add a new registered model. | Find better capacity/normalization. | Too broad without audits. | Higher; defer. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P31 | 3 | 4 | 5 | 4 | 3 | 2 | 21 |
| P32 | 2 | 4 | 5 | 2 | 3 | 2 | 18 |
| P33 | 4 | 3 | 2 | 3 | 4 | 3 | 19 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P31 | `arch_fednova_norm_lr1875_m035_wd35e5_gc_ep5` | Architecture subcampaign: `--model_arch moderate_cnn_norm` with current FedNova best optimizer/client stack |
| 2 | P32 | `arch_fednova_smallhead_lr1875_m035_wd35e5_gc_ep5` | Architecture subcampaign: `--model_arch moderate_cnn_small_head` with current FedNova best optimizer/client stack |

### Reflective Memory

- Treat these as architecture-subcampaign rows; do not silently mix them with optimizer-only `moderate_cnn` rows.
- If both registered variants underperform, return to literature before adding a new architecture variant.
- Outcome: both registered variants underperformed under FedNova. `moderate_cnn_small_head` scored `0.909200`; `moderate_cnn_norm` scored `0.905300`.
- Action: keep `moderate_cnn` for the active FedNova stack and return to literature before any new architecture code.

## Eleventh Literature Loop

### Trigger

- Reason: registered architecture variants underperformed and the active FedNova stack is still on the inherited client momentum `0.9`.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, client `lr=0.05`, client momentum `0.9`, `--gradient_centralization`.
- Recent symptoms: client LR retune regressed, but client momentum has not been narrowly retuned under FedNova.
- Candidate width: `PARALLEL_CANDIDATES=2`; use CLI-only retune.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Momentum Benefits Non-IID Federated Learning Simply and Provably arXiv 2306.16504` | Check theoretical support for local momentum under non-IID FL. | arXiv mirrors, paper indexes | Momentum can help FedAvg/SCAFFOLD convergence under heterogeneity. |
| `FedCM Federated Learning with Client-level Momentum arXiv 2106.10874` | Check FL-specific client momentum mechanisms. | arXiv/paper indexes | FedCM modifies clients with momentum-like correction; reserve code path. |
| `federated learning client momentum local SGD non-IID CIFAR arXiv` | Validate client momentum as an accepted optimizer axis. | paper indexes | Supports narrow CLI retune before new client-momentum code. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Non-IID FL convergence can benefit from momentum. | Momentum | keep |
| Xu21 | FedCM: Federated Learning with Client-level Momentum / 2021 | https://arxiv.org/abs/2106.10874 | Client heterogeneity and drift can be addressed with client-level momentum. | Client momentum correction | reserve |
| FedZMG26 | FedZMG / 2026 | https://arxiv.org/abs/2602.18384 | Client-side optimization geometry matters. | Gradient centralization | already kept |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Momentum under new server dynamics | Cheng23 supports momentum in non-IID FL. | FedNova changed server aggregation, but client momentum remains inherited. | CLI-only `--momentum`; no protocol change. | `client.py` args. |
| C2 | Avoid broad unstable momentum values | Prior `0.8` and `0.95` under old stack regressed. | Need narrow probes around `0.9`. | Test `0.875` and `0.925`. | Low risk. |
| C3 | FedCM is more invasive | Xu21 uses momentum-like correction involving prior global information. | Adaptive server code just crashed. | Keep FedCM as reserve, not first move. | Higher code risk. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P34 | Narrow client momentum retune under FedNova. | Cheng23; Xu21 | CLI-only: `--momentum 0.875` and `0.925`. | Tune local SGD inertia for FedNova normalized aggregation. | Both below `0.910300`. | Low. |
| P35 | FedCM-style client momentum correction. | Xu21 | Code: add client correction term. | Correct drift with historical global direction. | Complexity or instability. | Medium-high. |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P34 | 2 | 5 | 5 | 4 | 2 | 2 | 21 |
| P35 | 3 | 3 | 2 | 4 | 4 | 3 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P34 | `fednova_lr1875_m035_wd35e5_gc_ep5_m0875` | CLI-only: `--momentum 0.875` with current FedNova best stack |
| 2 | P34 | `fednova_lr1875_m035_wd35e5_gc_ep5_m0925` | CLI-only: `--momentum 0.925` with current FedNova best stack |

### Reflective Memory

- Keep client momentum `0.9` unless a narrow retune beats `0.910300`.
- If both fail, start a new literature loop before FedCM-style code.
- Outcome: narrow client momentum probes underperformed. `momentum=0.925` scored `0.909200`; `momentum=0.875` scored `0.906000`.
- Action: keep client momentum `0.9` and return to literature before any FedCM-style code.

## Twelfth Literature Loop

### Trigger

- Reason: registered architecture variants, client LR, client momentum, FedNova clipping, and adaptive server optimizers have not improved beyond `0.910300`.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, client `lr=0.05`, client momentum `0.9`, `--gradient_centralization`, `model_arch=moderate_cnn`.
- Recent symptoms: the optimizer-local surface is narrow; invasive adaptive-server variants crashed, and FedCM-style code would add more state before retesting simpler drift regularization under the current FedNova context.
- Candidate width: `PARALLEL_CANDIDATES=2`; prefer CLI-only candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedProx Federated Optimization in Heterogeneous Networks proximal term non-IID arXiv 1812.06127` | Re-check proximal local regularization after FedNova changed the server aggregation context. | arXiv, paper indexes | FedProx is low-risk and already available through `--fedproxloss_mu`. |
| `FedNova objective inconsistency heterogeneous federated optimization FedProx local steps arXiv 2007.07481` | Check compatibility between normalized aggregation and local solvers such as FedProx. | arXiv, paper indexes | FedNova explicitly frames a general heterogeneous optimization setting that includes FedAvg/FedProx-style local solvers. |
| `federated learning drift correction proximal dynamic regularization FedDyn FedDC arXiv non-IID CIFAR` | Compare lightweight FedProx against more invasive drift-correction objectives. | arXiv, OpenReview | FedDyn, FedDC, FedCM, and FedRed are relevant but require extra client/server state or larger code changes. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Li20 | Federated Optimization in Heterogeneous Networks / 2020 | https://arxiv.org/abs/1812.06127 | Statistical and systems heterogeneity destabilize local updates. | FedProx proximal local loss | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local update counts and data heterogeneity can create objective inconsistency. | FedNova normalized aggregation | keep |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Local-device empirical optima can be inconsistent with global optima. | FedDyn dynamic regularization | reserve |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://arxiv.org/abs/2203.11751 | Local drift makes optimized client models inconsistent. | Local drift correction | reserve |
| Xu21 | FedCM: Federated Learning with Client-level Momentum / 2021 | https://arxiv.org/abs/2106.10874 | Client heterogeneity and partial participation bias local SGD. | Client-level momentum correction | reserve |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | FedAvg client drift can worsen communication-computation trade-offs. | FedRed / DANE-style drift correction | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Client drift remains after FedNova | FedProx targets heterogeneity with a proximal local term; FedNova handles objective inconsistency in aggregation, not necessarily local overfitting. | FedNova improved to `0.910300`, but LR, momentum, clipping, and architecture probes underperformed. | A small proximal term may damp client movement without new server metadata. | CLI-only `--fedproxloss_mu` in `client.py`. |
| C2 | Dynamic drift correction is promising but stateful | FedDyn/FedDC/FedCM/FedRed all add correction state or modified local objectives. | Recent adaptive-server code crashed, and clipping tied best without enough gain to keep code. | Reserve higher-risk code paths until the existing FedProx hook is retested in the current context. | `client.py`/`custom_aggregators.py`, but not first. |
| C3 | Avoid duplicate old nulls | Early FedProx `1e-5`/`1e-4` failed with weighted FedAvg, not with FedNova plus gradient centralization and epoch 5. | The active stack differs substantially from the initial calibration rows. | Retesting tiny FedProx is not duplicate jitter because the aggregation/local geometry changed. | CLI-only; same fixed budget. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P36 | FedNova plus light FedProx regularization. | Li20; Wang20 | CLI-only: current FedNova best stack with `--fedproxloss_mu 1e-5`. | Reduce client drift while preserving normalized DIFF aggregation. | Score below `0.910300` or runtime penalty without gain. | Low. |
| P37 | FedNova plus medium FedProx regularization. | Li20; Wang20 | CLI-only: current FedNova best stack with `--fedproxloss_mu 1e-4`. | Test whether the stronger current local training context needs more proximal pull. | Score below `0.910300`. | Low. |
| P38 | FedDyn-style dynamic regularizer. | Acar21 | Code: maintain/update per-client dynamic regularization state. | Align local and global stationary points under heterogeneity. | Complexity, protocol/state risk, or no gain. | Medium-high; reserve. |
| P39 | FedDC/FedRed drift correction. | Gao22; Jiang24 | Code: local drift variable or doubly regularized correction. | Correct drift more directly than FedProx. | Requires stateful client/server changes and careful validation. | Higher; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P36 | Early FedProx calibration only partially overlaps. | Prior `1e-5` was weighted FedAvg at epoch 4 without FedNova/GC. | keep |
| P37 | Early FedProx calibration only partially overlaps. | Prior `1e-4` was weighted FedAvg at epoch 4 without FedNova/GC. | keep |
| P38 | No exact prior row. | More invasive than available CLI hook. | reserve |
| P39 | No exact prior row. | Requires new drift state; defer until CLI FedProx fails. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P36 | 2 | 5 | 5 | 4 | 3 | 2 | 22 |
| P37 | 2 | 5 | 5 | 4 | 3 | 2 | 22 |
| P38 | 3 | 3 | 2 | 4 | 4 | 3 | 19 |
| P39 | 3 | 2 | 2 | 4 | 4 | 3 | 17 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P36 | `fednova_lr1875_m035_wd35e5_gc_fedprox1e5_ep5` | CLI-only: `--fedproxloss_mu 1e-5` with current FedNova best stack |
| 2 | P37 | `fednova_lr1875_m035_wd35e5_gc_fedprox1e4_ep5` | CLI-only: `--fedproxloss_mu 1e-4` with current FedNova best stack |

### Reflective Memory

- Keep FedProx only if it beats or clearly simplifies the current `0.910300` FedNova stack.
- If both FedProx probes fail, do not repeat FedProx under this stack without a new mechanism; move to a carefully bounded drift-correction code proposal or another distinct literature-backed axis.
- Outcome: FedProx under the kept FedNova stack did not improve. `mu=1e-5` scored `0.907300`; `mu=1e-4` scored `0.909900`.
- Action: do not repeat FedProx under this stack. Before adding FedDyn/FedDC/FedCM-style state, run a distinct FedNova local-compute audit at `aggregation_epochs=4` and `6` because FedNova itself changed the epoch-count context.
- Follow-up outcome: FedNova epoch neighbors also underperformed. `aggregation_epochs=4` scored `0.905500`; `6` scored `0.908200`.
- Follow-up action: with two non-improving batches after this literature reset, start a new literature loop before implementing stateful drift correction.

## Thirteenth Literature Loop

### Trigger

- Reason: two consecutive batches after the Twelfth loop failed: FedProx and FedNova epoch-neighbor audits both underperformed the `0.910300` best.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, client `lr=0.05`, client momentum `0.9`, `--gradient_centralization`.
- Recent symptoms: FedNova is still the strongest mechanism, but step-count, LR, momentum, architecture, clipping, and FedProx probes did not improve. Stateful FedDyn/FedDC/FedCM ideas are plausible but higher risk.
- Candidate width: `PARALLEL_CANDIDATES=2`; use default-preserving code and two same-budget candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedDyn Federated Learning Based on Dynamic Regularization OpenReview client drift non-IID` | Revisit dynamic regularization after FedProx failed. | OpenReview, arXiv/paper indexes | Relevant but requires per-client correction state and a changed local objective. |
| `FedDC Federated Learning with Non-IID Data via Local Drift Decoupling and Correction arXiv 2203.11751` | Compare stateful drift decoupling against lighter aggregation changes. | arXiv, CVF Open Access | Strong drift-correction motivation, but requires auxiliary local drift variables. |
| `federated learning client weighting non-IID unbalanced data aggregation FedNova FedLAW arXiv` | Look for lower-risk aggregation-weight refinements before stateful drift correction. | arXiv, paper indexes | FedNova and FedLAW both highlight aggregation weighting as a meaningful axis under heterogeneity. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Heterogeneous local update counts and naive weighted aggregation can bias the objective. | FedNova normalized aggregation | keep |
| Li23 | Revisiting Weighted Aggregation in Federated Learning with Neural Networks / 2023 | https://arxiv.org/abs/2302.10911 | Aggregation weights and shrinkage affect FL generalization under heterogeneity and local epochs. | Learnable/modified aggregation weights | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | FedAvg must handle unbalanced and non-IID client data. | Federated averaging | keep |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Device-level empirical minima can be inconsistent with global minima. | FedDyn dynamic regularization | reserve |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://arxiv.org/abs/2203.11751 | Local drift and residual parameter deviation accumulate under non-IID data. | Local drift correction | reserve |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | FedAvg client drift worsens communication-computation trade-offs. | FedRed/DANE-style drift correction | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | FedNova weighting may still overemphasize step-heavy clients | FedNova fixes local-update normalization; FedLAW argues relative aggregation weights affect generalization. | Current FedNova uses `NUM_STEPS_CURRENT_ROUND` as both tau and aggregation weight; neighboring local compute and FedProx failed. | The server already sees step counts, so a bounded weight exponent can alter weighting without new metadata. | `custom_aggregators.py`, `job.py`. |
| C2 | Dynamic drift correction needs more state | FedDyn/FedDC/FedRed use client/server correction terms or auxiliary local variables. | Recent new server code either crashed or tied; keep code risk low first. | A default-off aggregation-weight knob is safer than new correction tensors. | Reserve `client.py`/`custom_aggregators.py`. |
| C3 | Uniform weighting is not a new data budget | McMahan17 treats unbalanced/non-IID data as core FL; changing aggregation weights leaves data, rounds, model, and evaluation fixed. | Weighted and normalized variants differ only server-side. | This preserves comparability with the current FedNova rows. | `custom_aggregators.py`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P40 | FedNova square-root step weighting. | Wang20; Li23 | Code: `--fednova_weight_power 0.5`; current best stack. | Reduce dominance of step-heavy clients while retaining partial size weighting. | Score below `0.910300` or no difference. | Low-medium; server-only, default preserves old behavior. |
| P41 | FedNova uniform client weighting. | Wang20; Li23; McMahan17 | Code: `--fednova_weight_power 0.0`; current best stack. | Test whether equal client influence improves label-skew generalization. | Score below `0.910300`. | Low-medium; server-only. |
| P42 | FedDyn-lite client dynamic regularization. | Acar21 | Code: maintain a per-client correction vector in the client process. | Align client and global stationary points. | Complexity, instability, or no gain. | Medium-high; reserve. |
| P43 | FedDC/FedRed drift correction. | Gao22; Jiang24 | Code: auxiliary drift variables or doubly regularized local objective. | Reduce residual parameter drift. | Requires stateful correction design. | Higher; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P40 | Current FedNova with weight power `1.0` only. | No previous partial-weight FedNova row. | keep |
| P41 | Current FedNova with weight power `1.0` only. | No previous uniform-weight FedNova row; median aggregation failure is not the same. | keep |
| P42 | FedProx and SCAFFOLD only partially overlap. | More invasive than P40/P41. | reserve |
| P43 | No exact prior row. | Higher code risk; defer until weight-power audit. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P40 | 3 | 5 | 4 | 4 | 4 | 2 | 26 |
| P41 | 3 | 5 | 4 | 4 | 4 | 2 | 26 |
| P42 | 3 | 3 | 2 | 4 | 4 | 3 | 19 |
| P43 | 3 | 2 | 2 | 4 | 4 | 3 | 17 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P40 | `fednova_lr1875_m035_wd35e5_gc_wpow05_ep5` | Code variant: `--fednova_weight_power 0.5` with current FedNova best stack |
| 2 | P41 | `fednova_lr1875_m035_wd35e5_gc_wpow00_ep5` | Code variant: `--fednova_weight_power 0.0` with current FedNova best stack |

### Reflective Memory

- Keep the weight-power code only if one of these candidates beats or ties the current best with a defensible simplicity/runtime trade-off.
- If both fail, revert the optional weight-power code before moving to FedDyn/FedDC-style stateful drift correction.
- Outcome: both weight-power candidates underperformed. Uniform weighting scored `0.904000`; square-root weighting scored `0.904800`.
- Action: revert the optional weight-power code and do not repeat FedNova client-weight flattening under this stack without a new signal.
- Follow-up action: launch the reserved FedDyn-style dynamic regularizer with conservative `--feddyn_alpha` values before broader FedDC/FedRed code.
- Follow-up outcome: FedDyn-style `alpha=1e-4` improved the best to `0.910900`; `alpha=5e-4` regressed to `0.907300`.
- Follow-up action: keep the default-off FedDyn-style code and narrow alpha around `1e-4`.
- Alpha-neighbor outcome: `alpha=5e-5` scored `0.906600`; `alpha=2e-4` scored `0.909600`.
- Alpha-neighbor action: keep `alpha=1e-4` and retune server LR under the new FedDyn-enabled stack.
- Server-LR outcome: `server_lr=1.9375` scored `0.910500`; `1.8125` scored `0.908300`.
- Server-LR action: keep `server_lr=1.875`. Two batches have failed after the FedDyn improvement, so start a new literature loop before more local jitter.

## Fourteenth Literature Loop

### Trigger

- Reason: two FedDyn-enabled follow-up batches failed after the new best `0.910900`: alpha neighbors and server-LR neighbors both underperformed.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`.
- Recent symptoms: FedDyn helps at a narrow alpha, but nearby alpha and server-LR changes do not. The next axis should test an interaction supported by the drift/momentum literature rather than more blind scalar jitter.
- Candidate width: `PARALLEL_CANDIDATES=2`; use CLI-only candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedDyn dynamic regularization federated learning hyperparameter weight decay non-IID CIFAR` | Understand how dynamic local regularization interacts with other stabilizers. | OpenReview, arXiv/paper indexes | FedDyn frames dynamic regularization as aligning client/global optima, but hyperparameters remain sensitive. |
| `FedDyn federated learning server momentum local regularization non-IID` | Check whether momentum remains a justified interaction axis after dynamic regularization. | arXiv, paper indexes | Momentum analysis supports FL convergence under heterogeneity. |
| `federated learning dynamic regularization local regularization weight decay client drift non-IID` | Compare weight decay/proximal retunes against momentum retunes. | arXiv, paper indexes | Weight decay remains plausible but is lower priority because recent FedProx and alpha neighbors already probed local regularization strength. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Device-level minima are inconsistent with global minima under heterogeneity. | FedDyn dynamic regularization | keep |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Momentum can improve FedAvg/SCAFFOLD convergence under non-IID data. | Momentum analysis | keep |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Server optimizer hyperparameters interact with heterogeneity and communication efficiency. | FedOpt/FedAvgM | keep |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | Drift correction can improve communication-computation trade-offs. | FedRed/DANE-style drift correction | reserve |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://arxiv.org/abs/2203.11751 | Residual drift variables can correct local drift. | FedDC | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | FedDyn shifted the optimizer context | Acar21 dynamic regularization changes the local objective to align client/global optima. | `alpha=1e-4` improved, but alpha neighbors failed. | Server optimizer settings should be rechecked under this changed local objective. | CLI `--server_momentum`, `--server_lr`. |
| C2 | Momentum can help non-IID convergence but is narrow | Cheng23 supports momentum under heterogeneity; Reddi21 highlights server optimizer tuning sensitivity. | Old FedNova momentum neighbors were close but below best before FedDyn. | Test only near current `server_momentum=0.35`. | CLI-only. |
| C3 | More drift-correction code is premature | FedRed/FedDC add more state/objective machinery. | A small FedDyn variant just worked; immediate code expansion would add risk. | Prefer momentum retune before another stateful code path. | Reserve `client.py`/`custom_aggregators.py`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P44 | FedDyn-enabled server momentum retune. | Acar21; Cheng23; Reddi21 | CLI-only: `--server_momentum 0.30` and `0.40` with `--feddyn_alpha 1e-4`. | Check whether dynamic local correction changes the best server inertia. | Both below `0.910900`. | Low. |
| P45 | FedDyn-enabled weight decay retune. | Acar21; FedProx/FedDyn regularization context | CLI-only: `--weight_decay 3e-4` and `4e-4` with `--feddyn_alpha 1e-4`. | Balance static L2 with dynamic local regularization. | Both below best. | Low. |
| P46 | FedRed/FedDC-style drift correction. | Jiang24; Gao22 | Code: add stronger drift-correction state/objective. | Further reduce client drift. | Complexity or no gain. | Medium-high; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P44 | Pre-FedDyn server momentum rows only partially overlap. | Context changed after FedDyn improvement. | keep |
| P45 | Pre-FedDyn weight decay rows only partially overlap. | Regularization context changed but less directly than momentum. | reserve |
| P46 | FedDyn-style code only partially overlaps. | More invasive; defer. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P44 | 3 | 5 | 5 | 4 | 3 | 3 | 25 |
| P45 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P46 | 3 | 3 | 2 | 4 | 4 | 4 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P44 | `fednova_lr1875_m030_wd35e5_gc_feddyn1e4_ep5` | CLI-only: `--server_momentum 0.30 --feddyn_alpha 1e-4` |
| 2 | P44 | `fednova_lr1875_m040_wd35e5_gc_feddyn1e4_ep5` | CLI-only: `--server_momentum 0.40 --feddyn_alpha 1e-4` |

### Reflective Memory

- Keep `server_momentum=0.35` unless a neighbor beats `0.910900`.
- If both momentum neighbors fail, try the reserved FedDyn-enabled weight-decay retune before adding more drift-correction code.
- Outcome: momentum neighbors missed the best. `server_momentum=0.30` scored `0.910800`; `0.40` scored `0.910400`.
- Action: keep `server_momentum=0.35` and run the reserved FedDyn-enabled weight-decay retune.
- Weight-decay outcome: `weight_decay=4e-4` scored `0.910400`; `3e-4` scored `0.907700`.
- Weight-decay action: keep `weight_decay=3.5e-4`. Two post-loop batches have failed, so start a new literature loop before selecting another mechanism.

## Fifteenth Literature Loop

### Trigger

- Reason: two batches after the Fourteenth loop failed: FedDyn-enabled server momentum and weight decay both missed the `0.910900` best.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`.
- Recent symptoms: scalar optimizer retunes around the new FedDyn stack are close but not improving. Local-compute neighbors were tested before FedDyn, not after the local objective changed.
- Candidate width: `PARALLEL_CANDIDATES=2`; use CLI-only candidates and keep `local_train_steps=0`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedDyn dynamic regularization local epochs communication efficiency federated learning` | Check whether FedDyn motivates retesting local compute. | OpenReview, paper indexes | FedDyn is communication-oriented and allows more local device computation. |
| `federated learning local epochs dynamic regularization client drift FedDyn FedNova` | Connect local epochs to drift correction and FedNova normalization. | arXiv, paper indexes | Local epochs remain central to the drift/communication trade-off. |
| `FedDyn CIFAR local epochs hyperparameter federated learning` | Look for practical FedDyn local epoch sensitivity. | paper indexes, implementation docs | Implementations expose epochs as a core FedDyn training knob. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Dynamic regularization aligns device/global optima and is communication-oriented. | FedDyn | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Local epochs trade communication for client drift/local fitting. | FedAvg local epochs | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local update counts affect objective consistency under heterogeneity. | FedNova | keep |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Momentum helps non-IID convergence but recent momentum retune failed. | Momentum | reserve |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://arxiv.org/abs/2203.11751 | Stronger local drift correction is possible but stateful. | FedDC | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | FedDyn may shift the local-compute optimum | Acar21 targets communication efficiency by allowing more local computation with dynamic regularization. | Epoch `4/6` failed before FedDyn; `alpha=1e-4` then changed the local objective. | `aggregation_epochs` is mutable with `local_train_steps=0`. | CLI-only `--aggregation_epochs`. |
| C2 | Too much local work can still drift | McMahan17 and Wang20 frame local update counts as a core trade-off. | Epoch `6` without FedDyn and exact-step variants regressed. | Test only immediate epoch neighbors around the current `5`. | CLI-only. |
| C3 | More stateful correction is not yet needed | FedDC/FedRed add additional drift state. | FedDyn produced one small gain; optimizer retunes were close misses. | Exhaust the local-compute interaction before adding more code. | Reserve code paths. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P47 | FedDyn-enabled epoch-count audit. | Acar21; McMahan17; Wang20 | CLI-only: `--aggregation_epochs 4` and `6`, `--feddyn_alpha 1e-4`, `--local_train_steps 0`. | Check whether dynamic regularization wants less/more local work than the pre-FedDyn optimum. | Both below `0.910900` or timeout. | Low. |
| P48 | FedDyn-enabled exact-step audit. | Acar21; Wang20 | CLI-only: `--local_train_steps 500/600`, no epoch sweep in same batch. | Revisit exact steps under FedDyn. | Prior exact steps were weak; reserve after epoch audit. | Low-medium. |
| P49 | Stronger FedDC/FedRed drift correction. | Gao22; Jiang24 | Code: add additional correction state. | Improve beyond FedDyn-lite. | Complexity or no gain. | Medium-high; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P47 | Pre-FedDyn epoch audit only partially overlaps. | FedDyn changed the local objective; not duplicate. | keep |
| P48 | Pre-FedDyn exact-step rows only partially overlap. | Do not vary exact steps in the same narrow sweep as epochs. | reserve |
| P49 | FedDyn-lite only partially overlaps. | More code risk. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P47 | 3 | 5 | 5 | 4 | 3 | 3 | 25 |
| P48 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P49 | 3 | 3 | 2 | 4 | 4 | 4 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P47 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_ep4` | CLI-only: `--aggregation_epochs 4 --feddyn_alpha 1e-4` |
| 2 | P47 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_ep6` | CLI-only: `--aggregation_epochs 6 --feddyn_alpha 1e-4` |

### Reflective Memory

- Keep `aggregation_epochs=5` unless a FedDyn-enabled neighbor beats `0.910900`.
- If both fail, consider FedDyn-enabled exact local steps before adding FedDC/FedRed-style code.
- Outcome: FedDyn-enabled epoch neighbors underperformed. `aggregation_epochs=4` scored `0.908800`; `6` scored `0.909500`.
- Action: keep epoch-based `aggregation_epochs=5` and run the reserved exact local-step audit as a separate local-compute sweep.
- Exact-step outcome: `local_train_steps=500` and `600` both hit the 1200-second timeout with NVFlare target-unreachable/get-task failures.
- Exact-step action: do not retry exact local steps at width 2 without a new reliability mitigation; return to literature before FedDC/FedRed-style code.

## Sixteenth Literature Loop

### Trigger

- Reason: the reserved FedDyn-enabled exact-step audit failed operationally rather than producing comparable scores. Both `local_train_steps=500` and `600` timed out at 1200 seconds with NVFlare target-unreachable/get-task failures while running at width 2.
- Current best: score `0.910900` from FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, and `--feddyn_alpha 1e-4`.
- Recent symptoms: epoch neighbors under FedDyn were valid but below best; exact-step runs may have suffered from local simulation contention rather than pure algorithm failure.
- Candidate width: lower to `PARALLEL_CANDIDATES=1` for the next exact-step reliability audit; keep `CUDA_VISIBLE_DEVICES=0` and `RUN_TIMEOUT_SECONDS=1200`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedDyn dynamic regularization local epochs communication efficiency federated learning` | Recheck whether FedDyn justifies more local device computation despite the timeout. | OpenReview, paper indexes | FedDyn is explicitly communication-oriented and allows more device-level computation. |
| `FedNova heterogeneous federated optimization local update counts exact steps objective inconsistency` | Confirm exact update counts remain algorithmically relevant with FedNova normalization. | arXiv indexes, paper pages | FedNova targets inconsistency from different local update counts, so exact-step audits are still plausible. |
| `FedDC FedRed drift correction non-IID federated learning local drift correction` | Compare a reliability-mitigated exact-step audit against adding stronger drift-correction code. | CVF/arXiv/dblp pages | FedDC/FedRed are relevant reserves but add more state and implementation risk. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Local device optima can be inconsistent with global optima; dynamic regularization supports more local computation. | FedDyn | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local update counts can bias the global objective unless normalized. | FedNova | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Local computation trades communication for client drift and runtime. | FedAvg local epochs | keep |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf | Residual parameter drift can accumulate under non-IID data. | FedDC | reserve |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | FedAvg-style local drift can hurt communication-computation trade-offs. | FedRed/DANE-style drift correction | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Exact-step signal is confounded by contention | Acar21 and McMahan17 motivate local compute as a real algorithmic knob, but our failed rows ended as simulator reachability failures. | `local_train_steps=500/600` both crashed at 1200 seconds with width 2. | A width-1 retry can separate runtime contention from algorithm quality without code changes. | CLI-only plus launch width. |
| C2 | Local compute may still help under FedDyn | FedDyn changes the local objective and is designed around more device-level computation; FedNova normalizes local update counts. | Pre-FedDyn exact-step rows were valid but below best; post-FedDyn exact-step rows did not complete. | One single-lane exact-step audit is still justified before abandoning the axis. | CLI-only `--local_train_steps`. |
| C3 | Stronger drift correction is more invasive | FedDC and FedRed add additional correction state/objective machinery. | FedDyn-lite already improved once; simple optimizer/local-compute follow-ups have mostly missed. | FedDC/FedRed should follow only after the no-code reliability audit fails or underperforms. | `client.py` and possibly `custom_aggregators.py` reserve. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P50 | Single-lane FedDyn exact-step reliability audit. | Acar21; Wang20; McMahan17 | CLI-only: `PARALLEL_CANDIDATES=1`, `--local_train_steps 600`, current best FedNova/FedDyn stack, unique run log/name. | Determine whether width-2 failures were host/NVFlare contention and recover a comparable exact-step score. | Another timeout/crash or score below `0.910900`. | Low-medium runtime risk; no code change. |
| P51 | Single-lane lower exact-step reserve. | Acar21; Wang20 | CLI-only: `--local_train_steps 500` only if P50 completes and is close. | Check whether slightly less exact local compute is more reliable. | Timeout or worse score. | Low-medium runtime risk; reserve. |
| P52 | FedDC/FedRed-style stronger drift correction. | Gao22; Jiang24 | Code: add a default-off local drift-correction variant after exact-step evidence is exhausted. | Reduce residual drift beyond FedDyn-lite. | Complexity, protocol pressure, or no score gain. | Medium-high; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P50 | Prior exact-step rows only partially overlap. | Width-2 crashes were not comparable algorithm outcomes. | keep |
| P51 | Prior exact-step rows and P50. | Reserve only; do not run both until P50 provides a signal. | reserve |
| P52 | FedDyn-style code only partially overlaps. | More invasive and not needed before one reliability-controlled audit. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P50 | 2 | 5 | 5 | 4 | 3 | 4 | 22 |
| P51 | 2 | 5 | 5 | 3 | 2 | 4 | 20 |
| P52 | 3 | 3 | 2 | 4 | 4 | 4 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P50 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_steps600_solo` | CLI-only: `PARALLEL_CANDIDATES=1 --local_train_steps 600 --feddyn_alpha 1e-4` with current best FedNova stack |

### Reflective Memory

- Treat the width-2 exact-step failures as an operational contention signal, not a scored algorithm result.
- If the width-1 `steps600` audit also times out or underperforms, abandon exact local steps under the current FedDyn/FedNova stack and move to a source-backed FedDC/FedRed-style code proposal.
- Outcome: the width-1 `steps600` audit completed successfully with score `0.909300`, below the `0.910900` best.
- Action: abandon exact local steps under this FedNova/FedDyn stack. The reliability mitigation worked, but the algorithmic result did not improve; move to the FedDC/FedRed drift-correction reserve next.
- Follow-up outcome: the FedDC/FedRed-inspired client EMA drift correction improved the best. `feddrift_mu=5e-5, beta=0.9` scored `0.911400`; `mu=1e-4` scored `0.910300`.
- Follow-up action: keep the default-off FedDrift code and narrow `feddrift_mu` around `5e-5` before changing `feddrift_beta`.
- Narrowing outcome: `feddrift_mu=2.5e-5, beta=0.9` improved the best to `0.913200`; `mu=7.5e-5` regressed to `0.905300`.
- Narrowing action: keep `feddrift_mu=2.5e-5` and test lower-side neighbors before changing `feddrift_beta`.
- Lower-side outcome: `feddrift_mu=1.25e-5` and `3.75e-5` both scored `0.909700`.
- Lower-side action: keep `feddrift_mu=2.5e-5`; switch the next narrow sweep to `feddrift_beta`.
- Beta-sweep outcome: `feddrift_beta=0.8` scored `0.908900`; `0.95` scored `0.908100`.
- Beta-sweep action: keep `feddrift_beta=0.9`. Two batches after the `0.913200` improvement failed, so start a fresh literature loop before more local jitter.

## Seventeenth Literature Loop

### Trigger

- Reason: two same-budget follow-up batches failed after the FedDrift improvement to `0.913200`: lower `feddrift_mu` neighbors and `feddrift_beta` neighbors both underperformed.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: the client-side drift correction is useful but narrow; more local drift-state jitter is now producing worse scores.
- Candidate width: `PARALLEL_CANDIDATES=2`; prefer CLI-only candidates before more code.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Adaptive Federated Optimization server learning rate momentum client heterogeneity` | Check whether a changed local objective should trigger server optimizer retuning. | arXiv/paper indexes, paper pages | FedOpt frames server optimizer hyperparameters as central under heterogeneity. |
| `FedDC local drift correction server optimizer learning rate federated learning non-IID` | Connect local drift correction to optimizer context. | CVF/arXiv/paper indexes | FedDC uses local drift variables; our FedDrift-lite shifted the best stack. |
| `server momentum federated learning non-IID FedAvgM FedNova` | Compare server LR retune against server momentum retune. | ICLR/AAAI/arXiv pages | Momentum remains relevant but prior FedDyn momentum retunes were close misses. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Server optimizer parameters interact with client heterogeneity and communication efficiency. | FedOpt/FedAvgM | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local updates and normalization affect the server-side update scale. | FedNova | keep |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf | Auxiliary drift variables can improve non-IID convergence but change local update geometry. | FedDC | keep |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | Drift correction targets communication-computation trade-offs. | FedRed/DANE-style drift correction | keep |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Momentum can help non-IID convergence but may be sensitive to the local correction context. | Momentum analysis | keep |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | FedDrift changed the server update scale | Reddi21 emphasizes server optimizer tuning under heterogeneity; Wang20 shows local-update normalization changes objective consistency. | `feddrift_mu=2.5e-5` improved to `0.913200`, while nearby drift coefficients regressed. | Retune server LR under the changed local objective before adding code. | CLI `--server_lr`. |
| C2 | Momentum remains plausible but secondary | Cheng23 supports momentum for non-IID FL, but previous FedDyn momentum retunes were close misses. | FedDyn momentum `0.30` nearly tied (`0.910800`) before FedDrift, but did not beat the then-best. | Test momentum after server LR if LR neighbors fail. | CLI `--server_momentum`. |
| C3 | More drift-state code risks overfitting the local correction | FedDC/FedRed motivate correction but also add objective/state complexity. | `mu=7.5e-5`, `3.75e-5`, `1.25e-5`, beta `0.8/0.95` all regressed. | Stop local correction jitter and change a server optimizer knob. | Reserve `client.py`; prefer CLI. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P53 | FedDrift-enabled server LR retune. | Reddi21; Wang20; Gao22 | CLI-only: current best stack with `--server_lr 1.8125` and `1.9375`. | Check whether the drift-corrected local updates want a smaller/larger server step. | Both below `0.913200`. | Low. |
| P54 | FedDrift-enabled server momentum retune. | Cheng23; Reddi21 | CLI-only: current best stack with `--server_momentum 0.30` and `0.40`. | Recheck server inertia under the shifted local correction. | Both below `0.913200`. | Low; reserve. |
| P55 | FedDrift-enabled client LR retune. | FedDC/FedOpt context | CLI-only: current best stack with `--lr 0.045` and `0.055`. | Local correction may shift stable client step size. | Both below best or slower without gain. | Low; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P53 | FedDyn-only server LR sweep. | FedDrift changed the local objective; not a duplicate. | keep |
| P54 | FedDyn-only server momentum sweep. | Context changed, but LR is the more direct scale retune. | reserve |
| P55 | FedNova client-LR sweep before FedDyn/FedDrift. | Context changed, but client-LR retunes have repeatedly regressed. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P53 | 3 | 5 | 5 | 4 | 3 | 3 | 25 |
| P54 | 2 | 5 | 5 | 4 | 3 | 3 | 23 |
| P55 | 2 | 5 | 5 | 3 | 2 | 3 | 21 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P53 | `fednova_lr18125_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_ep5` | CLI-only: `--server_lr 1.8125 --feddrift_mu 2.5e-5 --feddrift_beta 0.9` |
| 2 | P53 | `fednova_lr19375_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_ep5` | CLI-only: `--server_lr 1.9375 --feddrift_mu 2.5e-5 --feddrift_beta 0.9` |

### Reflective Memory

- Keep `feddrift_mu=2.5e-5` and `feddrift_beta=0.9` unless a server optimizer retune beats `0.913200`.
- If server LR neighbors fail, use the reserved server momentum retune before returning to client-local jitter.
- Outcome: server LR neighbors failed under the FedDrift best stack. `server_lr=1.9375` scored `0.909300`; `1.8125` scored `0.908800`.
- Action: keep `server_lr=1.875` and launch the reserved FedDrift-enabled server momentum retune.
- Momentum outcome: server momentum neighbors also failed. `server_momentum=0.40` scored `0.910200`; `0.30` scored `0.909500`.
- Momentum action: keep `server_momentum=0.35`. Two post-loop server optimizer batches failed, so start a new literature loop before selecting another axis.

## Eighteenth Literature Loop

### Trigger

- Reason: two post-loop server optimizer batches failed under the FedDrift best stack: server LR and server momentum neighbors both scored below `0.913200`.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: server-side scale/inertia retunes are not helping after FedDrift; the useful signal is a narrow client-local drift/regularization interaction.
- Candidate width: `PARALLEL_CANDIDATES=2`; prefer CLI-only candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning weight decay local regularization non-IID FedDyn FedDC` | Check whether local regularization is the next plausible axis after drift correction. | paper indexes, arXiv/CVF pages | Drift-correction methods alter the local objective, so static regularization may need retuning. |
| `FedDyn dynamic regularization weight decay hyperparameter federated learning` | Connect FedDyn/FedDrift dynamic regularization to static L2 regularization. | OpenReview, paper indexes | FedDyn alpha and FedDrift mu introduce local regularization-like forces. |
| `federated learning client learning rate local regularization drift correction non-IID` | Compare weight decay against client LR as local objective retune axes. | arXiv/paper indexes | Client LR has repeatedly been a weak axis in this campaign; weight decay was a stronger historical axis. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf | Auxiliary local drift variables change the local objective and can reduce inconsistency. | FedDC | keep |
| Jiang24 | Federated Optimization with Doubly Regularized Drift Correction / 2024 | https://arxiv.org/abs/2404.08447 | Doubly regularized correction highlights the value of local regularization balance. | FedRed/DANE-style drift correction | keep |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Dynamic regularization aligns client/global optima under heterogeneity. | FedDyn | keep |
| Loshchilov19 | Decoupled Weight Decay Regularization / 2019 | https://arxiv.org/abs/1711.05101 | Weight decay is a distinct regularization control from optimizer step size. | Weight decay | keep as general optimizer context |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Server optimizer retuning was plausible but just failed empirically. | FedOpt | reject for next batch |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local regularization balance shifted | Acar21, Gao22, and Jiang24 all alter local objectives to control drift. | FedDrift improved to `0.913200`; nearby mu/beta settings and server retunes regressed. | Static weight decay may now be miscentered around `3.5e-4`. | CLI `--weight_decay`. |
| C2 | Server optimizer retune is a current null | Reddi21 supports server tuning, but both server LR and momentum neighbors failed under the new stack. | LR neighbors scored `0.909300/0.908800`; momentum scored `0.910200/0.909500`. | Move away from server knobs until a new signal appears. | Avoid server knobs next. |
| C3 | Client LR is less promising than weight decay | Local LR affects optimization scale, but this campaign has multiple client-LR misses. | FedNova client-LR retunes previously regressed; FedDrift server retunes also missed. | Weight decay has historically been a stronger axis and is a narrower regularization retune. | CLI `--weight_decay`; reserve `--lr`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P56 | FedDrift-enabled weight-decay retune. | Acar21; Gao22; Jiang24; Loshchilov19 | CLI-only: current best stack with `--weight_decay 3.25e-4` and `3.75e-4`. | Recenter static L2 around the new dynamic drift corrections. | Both below `0.913200`. | Low. |
| P57 | FedDrift-enabled client LR retune. | Gao22; FedOpt/local optimizer context | CLI-only: current best stack with `--lr 0.045` and `0.055`. | Check local step size only if weight decay fails. | Both below best. | Low; reserve. |
| P58 | FedDrift alpha interaction retune. | Acar21; Gao22 | CLI-only: current best stack with `--feddyn_alpha 5e-5` and `2e-4`. | FedDrift may change the best FedDyn alpha. | Both below best or instability. | Low; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P56 | Pre-FedDrift weight-decay sweeps. | FedDrift changed the local objective; not duplicate. | keep |
| P57 | Earlier client-LR sweeps. | Context changed but prior results were weak. | reserve |
| P58 | FedDyn alpha neighbor sweep before FedDrift. | Context changed; still likely after weight decay. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P56 | 3 | 5 | 5 | 4 | 3 | 3 | 25 |
| P57 | 2 | 5 | 5 | 3 | 2 | 3 | 21 |
| P58 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P56 | `fednova_lr1875_m035_wd325e5_gc_feddyn1e4_feddrift2p5e5_ep5` | CLI-only: `--weight_decay 3.25e-4 --feddrift_mu 2.5e-5 --feddrift_beta 0.9` |
| 2 | P56 | `fednova_lr1875_m035_wd375e5_gc_feddyn1e4_feddrift2p5e5_ep5` | CLI-only: `--weight_decay 3.75e-4 --feddrift_mu 2.5e-5 --feddrift_beta 0.9` |

### Reflective Memory

- Keep `weight_decay=3.5e-4` unless a FedDrift-enabled neighbor beats `0.913200`.
- If both weight-decay neighbors fail, use the reserved FedDyn-alpha interaction before trying client LR.
- Outcome: weight-decay neighbors failed. `3.75e-4` scored `0.909700`; `3.25e-4` scored `0.906800`.
- Action: keep `weight_decay=3.5e-4` and launch the reserved FedDyn-alpha interaction under the FedDrift best stack.
- FedDyn-alpha outcome: `feddyn_alpha=2e-4` scored `0.911900`; `5e-5` scored `0.909800`, both below the `0.913200` best.
- FedDyn-alpha action: keep `feddyn_alpha=1e-4`. Two post-loop local-regularization batches failed, so start a new literature loop.

## Nineteenth Literature Loop

### Trigger

- Reason: two post-loop local regularization batches failed: FedDrift-enabled weight decay and FedDyn-alpha interactions both missed the `0.913200` best.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: server optimizer, static weight decay, and FedDyn-alpha retunes failed after the FedDrift improvement. The next no-code axis should be local optimizer step size before additional code.
- Candidate width: `PARALLEL_CANDIDATES=2`; use CLI-only candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning client learning rate local optimizer drift correction non-IID FedDC` | Check whether local optimizer scale is relevant after drift correction. | arXiv/CVF/paper indexes | FedDC-style methods alter local objective geometry, making client LR plausible. |
| `FedDyn local optimizer learning rate CIFAR non-IID federated learning` | Connect dynamic regularization with local learning-rate sensitivity. | OpenReview, paper indexes | FedDyn changes local loss curvature and may shift LR. |
| `local learning rate federated learning client drift FedAvg non-IID` | Compare LR retune against local-compute changes. | paper indexes | Local LR remains a direct client-drift/optimization knob. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf | Local correction changes the client objective and local optimizer behavior. | FedDC | keep |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Dynamic regularization can shift local optimization sensitivity. | FedDyn | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Client local optimization controls the communication/local-drift trade-off. | FedAvg | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local update scale and count affect objective consistency. | FedNova | keep |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Server optimizer retunes failed empirically under current stack. | FedOpt | reject for next batch |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local objective curvature changed | FedDyn/FedDC add local correction terms, changing the effective local objective. | FedDrift improved strongly, but mu/beta/weight decay/alpha neighbors are narrow. | Client LR may need recentering under the changed objective. | CLI `--lr`. |
| C2 | Server and regularization axes are current nulls | FedOpt/regularization ideas were source-backed but recently underperformed. | Server LR, momentum, weight decay, and FedDyn-alpha retunes all missed. | Avoid repeating those axes immediately. | Choose client LR next. |
| C3 | Local compute is riskier than local LR | McMahan17/Wang20 support local compute as important, but exact steps were unreliable and epoch neighbors failed. | Exact steps crashed/underperformed; epoch `4/6` under FedDyn missed. | Try LR before more local-compute audits. | CLI `--lr`; reserve epochs. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P59 | FedDrift-enabled client LR retune. | Gao22; Acar21; McMahan17; Wang20 | CLI-only: current best stack with `--lr 0.045` and `0.055`. | Recenter local optimizer step size after FedDrift/FedDyn corrections. | Both below `0.913200`. | Low. |
| P60 | FedDrift-enabled epoch audit. | McMahan17; Wang20; Acar21 | CLI-only: current best stack with `--aggregation_epochs 4` and `6`. | Test whether FedDrift shifted local compute optimum. | Both below best or slow. | Low; reserve. |
| P61 | FedDrift EMA-state code variant. | Gao22; Jiang24 | Code: add a second drift-state update form or clipping. | Better residual-drift control. | Complexity or no gain. | Medium; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P59 | Earlier client-LR sweeps before FedDyn/FedDrift. | Context changed materially after FedDrift; not duplicate. | keep |
| P60 | FedDyn epoch audit. | Context changed after FedDrift, but local compute has several misses. | reserve |
| P61 | Current FedDrift code. | More code risk; not before CLI LR. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P59 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P60 | 2 | 5 | 5 | 3 | 2 | 3 | 21 |
| P61 | 3 | 3 | 2 | 4 | 3 | 4 | 17 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P59 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_clr045_ep5` | CLI-only: `--lr 0.045 --feddrift_mu 2.5e-5 --feddrift_beta 0.9` |
| 2 | P59 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_clr055_ep5` | CLI-only: `--lr 0.055 --feddrift_mu 2.5e-5 --feddrift_beta 0.9` |

### Reflective Memory

- Keep client LR at the default `0.05` unless a FedDrift-enabled LR neighbor beats `0.913200`.
- If both client-LR neighbors fail, use the reserved epoch audit before adding more FedDrift code.
- Outcome: client-LR neighbors failed. `lr=0.055` scored `0.911500`; `lr=0.045` scored `0.906900`.
- Action: keep default client LR and launch the reserved FedDrift-enabled epoch audit with `aggregation_epochs=4` and `6`.
- Outcome: the reserved epoch audit failed. `aggregation_epochs=6` scored `0.911500`; `aggregation_epochs=4` scored `0.905000`.
- Action: use reserve P61 and test default-off FedDrift EMA-state clipping before another literature reset.
- Outcome: FedDrift EMA-state clipping failed. `clip_norm=2.0` scored `0.910900`; `clip_norm=1.0` scored `0.906500`.
- Action: revert the optional clipping code path and start a Twentieth literature loop before the next candidate batch.

## Twentieth Literature Loop

### Trigger

- Reason: Nineteenth-loop reserves are exhausted. FedDrift-enabled client LR, epoch count, and EMA-state clipping all missed the `0.913200` best.
- Current best: FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: static scalar retunes around the best local objective are mostly null. Client LR neighbors failed, but scheduler shape has not been retested after FedDyn/FedDrift changed the local loss.
- Candidate width: `PARALLEL_CANDIDATES=2`; use CLI-only candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Adaptive Federated Learning with Auto-Tuned Clients arXiv 2306.11201 learning rate federated` | Check whether client step-size adaptation remains a central FL issue. | arXiv, paper indexes | Kim24 supports client-side step-size sensitivity but full Delta-SGD is more code than needed next. |
| `FedNova objective inconsistency local steps learning rate arXiv 2007.07481` | Tie local step scale to the current FedNova/FedDrift stack. | arXiv | FedNova addresses local update-count inconsistency; schedule shape is a low-risk local update-scale axis. |
| `Local Adaptivity in Federated Learning convergence consistency arXiv 2106.02305` | Check risk of local adaptive optimizers before adding more optimizer code. | arXiv | Local adaptive methods can introduce solution bias; prefer scheduler-floor CLI probes. |
| `Adaptive Federated Optimization client heterogeneity communication efficiency arXiv 2003.00295` | Compare scheduler retune against server adaptive methods. | arXiv | FedOpt is source-backed but prior adaptive server variants crashed or underperformed here. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Kim24 | Adaptive Federated Learning with Auto-Tuned Clients / 2024 revision | https://arxiv.org/abs/2306.11201 | Client-side hyperparameter tuning is difficult under heterogeneous FL. | client step size | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Local SGD step size and local training trade communication for drift. | FedAvg | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local update counts and heterogeneity can bias the effective objective. | FedNova | keep |
| Wang21 | Local Adaptivity in Federated Learning: Convergence and Consistency / 2021 | https://arxiv.org/abs/2106.02305 | Local adaptive optimizers can accelerate but also create non-vanishing solution bias. | local adaptivity | reserve/reject code next |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Adaptive FL optimizers help heterogeneity but require careful tuning. | FedOpt | reject for next batch due local crashes/nulls |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Step-size schedule may be stale after FedDrift | Kim24 emphasizes client step-size tuning; FedDyn/FedDrift changed the local objective. | Direct client LR neighbors failed, but schedule floor was last tested before FedDyn/FedDrift. | Cosine eta floor changes late-round local step size without changing communication or local epoch budget. | CLI `--cosine_lr_eta_min_factor`. |
| C2 | Avoid new adaptive optimizer code | Wang21 warns local adaptivity can be inconsistent without correction; FedOpt variants were unstable here. | FedAdam/FedYogi/FedAdagrad variants crashed or underperformed. | Prefer existing scheduler knob before new optimizer state. | CLI only. |
| C3 | Local compute axes are exhausted | McMahan17/Wang20 support local update-scale importance, but exact steps and epoch neighbors failed. | Exact steps were unreliable; epoch `4/6` failed under FedDrift. | Scheduler floor is a distinct local update-scale axis while keeping `aggregation_epochs=5`. | CLI only. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P62 | FedDrift-enabled cosine scheduler floor sweep. | Kim24; McMahan17; Wang20 | CLI-only: current best stack with `--cosine_lr_eta_min_factor 0.003` and `0.03`. | Rebalance late-round local update size after drift-corrected objective. | Both below `0.913200`. | Low. |
| P63 | Disable client LR scheduler. | Kim24; McMahan17 | CLI-only: `--no_lr_scheduler`. | Test if decay is over-damping late rounds. | Prior scheduler-off rows and a new score below best. | Low but duplicate/null-prone. |
| P64 | Local adaptive optimizer rule. | Kim24; Wang21 | Code: Delta-SGD-like client step-size adaptation. | Per-client smoothness adaptation. | Bias/complexity or no score gain. | Medium-high; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P62 | Earlier scheduler floor sweeps before FedNova/FedDyn/FedDrift. | Context changed materially; avoid old crashed `0.001`. | keep |
| P63 | Earlier scheduler-off run. | Prior no-scheduler score was poor and context change is weaker than for floor. | reject for next batch |
| P64 | New optimizer code. | Local adaptive theory is mixed and current code mutations are high-risk after clipping failed. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P62 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P63 | 1 | 5 | 5 | 2 | 1 | 3 | 18 |
| P64 | 3 | 3 | 2 | 4 | 3 | 4 | 17 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P62 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_eta003_ep5` | CLI-only: `--cosine_lr_eta_min_factor 0.003` |
| 2 | P62 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_eta03_ep5` | CLI-only: `--cosine_lr_eta_min_factor 0.03` |

### Reflective Memory

- Keep the default cosine floor `0.01` unless a FedDrift-enabled scheduler-floor neighbor beats `0.913200`.
- Do not test `0.001` again because an earlier low-floor run timed out; use bounded floor neighbors first.
- If both scheduler-floor candidates fail, return to literature before adding local adaptive optimizer code.
- Outcome: scheduler-floor neighbors failed. `eta_min_factor=0.003` scored `0.910900`; `0.03` scored `0.906100`.
- Action: keep the default `cosine_lr_eta_min_factor=0.01` and return to literature before adding local adaptive optimizer code.

## Twenty-first Literature Loop

### Trigger

- Reason: the Twentieth scheduler-floor batch missed, and adding local adaptive optimizer code is higher-risk than re-auditing already registered architectures under the changed FedDyn/FedDrift objective.
- Current best: `moderate_cnn` with FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR/scheduler, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: client LR, scheduler floor, epoch count, drift-state clipping, weight decay, and FedDyn-alpha interactions all missed. The earlier registered-architecture audit was before FedDyn/FedDrift, so the objective context has changed materially.
- Candidate width: `PARALLEL_CANDIDATES=2`; label this as an architecture subcampaign because `model_arch` changes while `max_model_params=5000000` remains fixed.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedBN Federated Learning on Non-IID Features via Local Batch Normalization OpenReview ICLR 2021` | Check source backing for normalization-aware architecture changes in FL. | OpenReview, ICLR pages | FedBN motivates normalization choices under non-IID FL. |
| `federated learning normalization non-IID convolutional networks GroupNorm CIFAR` | Connect registered GroupNorm architecture to FL robustness. | paper indexes | GroupNorm avoids batch-stat buffers and keeps server/client state schema explicit. |
| `federated learning model architecture capacity regularization non-IID CIFAR` | Check whether smaller classifier heads are a plausible low-code capacity axis. | paper indexes/local prior rows | Evidence weaker than normalization, but registered small head is under cap and no new code. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Li21 | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization / 2021 | https://openreview.net/forum?id=6YEQUn0QICG | Non-IID feature distributions can make normalization behavior important. | normalization-aware FL | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | FedAvg-style local training is sensitive to model and local update behavior. | FedAvg baseline | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Objective consistency depends on local update geometry; architecture can interact with update scale. | FedNova | keep |
| Wang21 | Local Adaptivity in Federated Learning / 2021 | https://arxiv.org/abs/2106.02305 | Local adaptive optimizer code can introduce consistency risk. | local adaptivity | reject next |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Normalization context changed | FedBN shows normalization choices can affect non-IID FL. | `moderate_cnn_norm` failed before FedDyn/FedDrift, but the current local objective is materially different. | `moderate_cnn_norm` is registered, buffer-free GroupNorm and under the parameter cap. | CLI `--model_arch moderate_cnn_norm`; architecture subcampaign. |
| C2 | Capacity/regularization may interact with drift correction | Smaller heads can reduce overfitting capacity; evidence is weaker but code already exists. | `moderate_cnn_small_head` was close at `0.909200` before FedDyn/FedDrift. | Registered architecture avoids new code and stays under cap. | CLI `--model_arch moderate_cnn_small_head`; architecture subcampaign. |
| C3 | Comparability risk | `program.md` requires architecture scores be labeled separately. | Best score uses `moderate_cnn`. | Use explicit architecture-subcampaign descriptions and do not silently treat it as the same optimizer-only budget. | Ledger/report labeling. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P65 | Re-audit registered GroupNorm architecture under FedDrift. | Li21; Wang20 | CLI-only architecture subcampaign: current best stack with `--model_arch moderate_cnn_norm`. | Normalization may interact better with FedDyn/FedDrift local corrections. | Score below `0.913200` or cap/runtime issue. | Medium; model schema changes by registered variant. |
| P66 | Re-audit registered small-head architecture under FedDrift. | McMahan17; local prior | CLI-only architecture subcampaign: current best stack with `--model_arch moderate_cnn_small_head`. | Reduce classifier overfitting/capacity under drift-corrected local objective. | Score below best. | Medium; model schema changes by registered variant. |
| P67 | Add new architecture variant. | architecture search literature | Code: new registered model under cap. | Search beyond current variants. | Too broad before re-audits. | High; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P65 | Tenth-loop architecture audit. | Context changed after FedDyn/FedDrift improvement; not a strict duplicate. | keep |
| P66 | Tenth-loop architecture audit. | Context changed; small-head was the closer previous architecture row. | keep |
| P67 | New architecture code. | Existing registered variants need current-stack audit first. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P65 | 2 | 4 | 5 | 3 | 3 | 3 | 20 |
| P66 | 2 | 4 | 5 | 2 | 3 | 3 | 19 |
| P67 | 3 | 2 | 1 | 2 | 4 | 3 | 14 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P65 | `arch_feddrift_norm_lr1875_m035_wd35e5_gc_feddyn1e4_ep5` | Architecture subcampaign: current best stack with `--model_arch moderate_cnn_norm` |
| 2 | P66 | `arch_feddrift_smallhead_lr1875_m035_wd35e5_gc_feddyn1e4_ep5` | Architecture subcampaign: current best stack with `--model_arch moderate_cnn_small_head` |

### Reflective Memory

- Treat these as architecture-subcampaign rows; keep `max_model_params=5000000`.
- Keep `moderate_cnn` unless a registered variant beats `0.913200` clearly enough to justify the schema change.
- If both registered variants fail again, return to literature before adding a new architecture or local adaptive optimizer code.
- Outcome: both registered architecture variants missed. `moderate_cnn_small_head` scored `0.912100`; `moderate_cnn_norm` scored `0.904600`.
- Action: keep `moderate_cnn` for the active best stack and return to literature before adding a new architecture or local adaptive optimizer code.

## Twenty-second Literature Loop

### Trigger

- Reason: the registered architecture subcampaign missed, and local adaptive optimizer code remains higher-risk. Revisit a low-risk existing local regularizer under the current FedDyn/FedDrift objective before new optimizer code.
- Current best: `moderate_cnn` with FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR/scheduler, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: direct FedDyn-alpha, FedDrift-mu/beta, weight decay, LR, scheduler, epoch, architecture, and clipping retunes missed. FedProx was tested before FedDyn/FedDrift, but not as a very light interaction with both drift corrections enabled.
- Candidate width: `PARALLEL_CANDIDATES=2`; use CLI-only candidates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedProx Federated Optimization in Heterogeneous Networks MLSys 2020 arXiv 1812.06127` | Confirm proximal local regularization source and heterogeneity motivation. | MLSys, arXiv/paper indexes | FedProx is a direct client-local heterogeneity stabilizer. |
| `FedDyn dynamic regularization FedProx federated learning non-IID local regularization` | Compare proximal regularization with the current dynamic correction. | arXiv/OpenReview/paper indexes | FedDyn changes the local objective; a smaller FedProx coefficient may interact differently than prior rows. |
| `FedDC FedProx client drift regularization federated learning non-IID` | Check whether local drift-correction context still supports proximal anchoring. | arXiv/CVF/paper indexes | FedDC/FedDrift-style correction and FedProx both target client drift from different angles. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Li20 | Federated Optimization in Heterogeneous Networks / 2020 | https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html | Statistical heterogeneity and variable local behavior destabilize FedAvg-style training. | FedProx | keep |
| Acar21 | Federated Learning Based on Dynamic Regularization / 2021 | https://openreview.net/forum?id=B7v4QMR6Z9w | Dynamic regularization aligns local/global objectives. | FedDyn | keep |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://openaccess.thecvf.com/content/CVPR2022/html/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.html | Client drift correction can improve non-IID training. | FedDC | keep |
| Wang21 | Local Adaptivity in Federated Learning / 2021 | https://arxiv.org/abs/2106.02305 | New local adaptive optimizers can add consistency risk. | local adaptivity | reject next |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Proximal anchor may need to be much smaller now | Li20 supports proximal stabilization; FedDyn/FedDrift already add local corrections. | Prior FedProx `1e-5/1e-4` under pre-FedDrift stack missed; current objective is different and likely needs smaller coefficients. | Existing `--fedproxloss_mu` is client-local and default-off. | CLI `--fedproxloss_mu`. |
| C2 | Drift corrections are narrow | Acar21/Gao22 support drift correction, but alpha/mu/beta neighbors failed. | Best came from `feddyn_alpha=1e-4` plus `feddrift_mu=2.5e-5`, but nearby regularization was worse. | Use tiny FedProx values to avoid overwhelming the kept corrections. | CLI only. |
| C3 | Avoid local adaptive optimizer code for one more batch | Wang21 flags consistency risk for local adaptivity. | Architecture and scheduler no-code audits just missed, but no-code FedProx interaction is still untested in current context. | Prefer existing proximal hook before adding optimizer state. | CLI only. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P68 | Very-light FedProx interaction under FedDrift. | Li20; Acar21; Gao22 | CLI-only: current best stack with `--fedproxloss_mu 1e-6`. | Add a weak global anchor without overpowering FedDyn/FedDrift. | Score below `0.913200`. | Low. |
| P69 | Light FedProx interaction under FedDrift. | Li20; Acar21; Gao22 | CLI-only: current best stack with `--fedproxloss_mu 5e-6`. | Test a slightly stronger proximal anchor below prior failed `1e-5`. | Score below `0.913200`. | Low. |
| P70 | Local adaptive optimizer code. | Kim24; Wang21 | Code: client-side adaptive LR/optimizer. | Per-client update adaptation. | Bias/complexity or no gain. | Medium-high; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P68 | Pre-FedDyn/FedDrift FedProx rows. | Context changed and coefficient is lower. | keep |
| P69 | Pre-FedDyn/FedDrift FedProx rows. | Context changed and coefficient is below prior failed `1e-5`. | keep |
| P70 | New optimizer code. | Higher risk than existing FedProx hook. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P68 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P69 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P70 | 3 | 3 | 2 | 3 | 4 | 4 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P68 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_fedprox1e6_ep5` | CLI-only: `--fedproxloss_mu 1e-6` |
| 2 | P69 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_fedprox5e6_ep5` | CLI-only: `--fedproxloss_mu 5e-6` |

### Reflective Memory

- Keep `fedproxloss_mu=0` unless a very-light FedProx interaction beats `0.913200`.
- Do not repeat `1e-5` or `1e-4` unless one of the lower values improves; both missed before FedDyn/FedDrift.
- If both FedProx interactions fail, return to literature before adding local adaptive optimizer code.
- Outcome: very-light FedProx interactions failed. `mu=1e-6` scored `0.908500`; `mu=5e-6` scored `0.906000`.
- Action: keep `fedproxloss_mu=0` and return to literature before adding local adaptive optimizer code.

## Twenty-third Literature Loop

### Trigger

- Reason: light FedProx under the FedDyn/FedDrift best stack missed, and the remaining no-code retunes are mostly exhausted or recently null. A bounded local optimizer-family mutation is now justified if default behavior remains SGD.
- Current best: SGD with FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR/scheduler, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: FedProx, scheduler floor, architecture, epoch count, client LR, weight decay, FedDyn-alpha, and FedDrift-mu/beta retunes all missed. Local adaptive optimizer code has been reserved until simpler axes failed.
- Candidate width: `PARALLEL_CANDIDATES=2`; add a default-off local optimizer selector, validate, and test two conservative AdamW client learning rates.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Local Adaptivity in Federated Learning convergence consistency arXiv 2106.02305 Adam local optimizer federated` | Check risk/benefit of adaptive local client optimizers. | arXiv/paper indexes | Wang21 warns local adaptivity can bias the global solution, so test conservatively and keep default SGD. |
| `Adaptive Federated Learning with Auto-Tuned Clients local learning rate optimizer arXiv 2306.11201` | Confirm client-side optimizer/step-size adaptation remains relevant. | arXiv/paper indexes | Kim24 motivates client-side adaptivity under heterogeneous FL. |
| `Decoupled Weight Decay Regularization AdamW arXiv 1711.05101 image classification` | Choose AdamW over Adam when weight decay is already a kept regularizer. | arXiv/paper indexes | AdamW decouples weight decay for adaptive optimizers and is available in PyTorch. |
| `Adaptive Federated Optimization Reddi 2021 federated learning client heterogeneity adaptive optimizer` | Compare adaptive method risk against prior server-adaptive failures. | arXiv/paper indexes | FedOpt supports adaptive ideas but previous server-adaptive variants were unstable here. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Wang21 | Local Adaptivity in Federated Learning: Convergence and Consistency / 2021 | https://arxiv.org/abs/2106.02305 | Local adaptive optimizers can accelerate but may introduce solution bias. | local adaptivity | keep with caution |
| Kim24 | Adaptive Federated Learning with Auto-Tuned Clients / 2024 revision | https://arxiv.org/abs/2306.11201 | Client-side hyperparameter tuning is difficult under heterogeneity. | client adaptivity | keep |
| Loshchilov17 | Decoupled Weight Decay Regularization / 2017 | https://arxiv.org/abs/1711.05101 | Adam-style adaptive optimizers need decoupled weight decay for reliable regularization. | AdamW | keep |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Adaptive FL optimizers can help heterogeneity but tuning is sensitive. | adaptive FL optimizers | keep as context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | SGD local optimizer may be exhausted | Kim24 argues client-side step-size tuning is a central FL challenge. | Many SGD scalar retunes around the best missed. | A client optimizer-family flag is within `client.py` and does not change FLModel fields. | Default-off `--optimizer adamw`. |
| C2 | Local adaptivity can bias the global objective | Wang21 warns adaptive local methods can introduce non-vanishing bias. | Server adaptive variants crashed or regressed; local code must be bounded. | Test only conservative AdamW LRs and revert if not better. | Default SGD remains unchanged. |
| C3 | Weight decay semantics differ under adaptive optimizers | Loshchilov17 supports AdamW instead of Adam when using weight decay. | Current best relies on `weight_decay=3.5e-4`. | AdamW preserves a meaningful decoupled decay knob. | `torch.optim.AdamW`; no new dependency. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P71 | Default-off local AdamW optimizer at conservative LR. | Wang21; Kim24; Loshchilov17 | Code: add `--optimizer {sgd,adamw}`; candidate `--optimizer adamw --lr 0.001`. | Test local adaptivity while preserving FedNova/FedDyn/FedDrift contract. | Score below `0.913200` or instability. | Medium; client-local optimizer only. |
| P72 | Lower-LR local AdamW optimizer. | Wang21; Kim24; Loshchilov17 | Same code; candidate `--optimizer adamw --lr 0.0005`. | Reduce local adaptivity bias/instability. | Score below `0.913200`. | Medium. |
| P73 | Implement Delta-SGD-style adaptive step rule. | Kim24 | Code: per-client step-size adaptation. | More tailored client adaptivity. | Complexity or no gain. | Higher; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P71 | Server-adaptive FedAdam/FedYogi/FedAdagrad. | Local adaptivity is different but risky; use low LR. | keep |
| P72 | Server-adaptive FedAdam/FedYogi/FedAdagrad. | Lower LR reduces instability risk. | keep |
| P73 | New adaptive rule. | More invasive than optimizer-family hook. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P71 | 3 | 4 | 4 | 3 | 4 | 3 | 23 |
| P72 | 2 | 4 | 4 | 3 | 4 | 3 | 21 |
| P73 | 3 | 3 | 2 | 4 | 4 | 4 | 20 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P71 | `fednova_adamw1e3_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_ep5` | Code variant: `--optimizer adamw --lr 0.001` |
| 2 | P72 | `fednova_adamw5e4_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_ep5` | Code variant: `--optimizer adamw --lr 0.0005` |

### Reflective Memory

- Keep `--optimizer sgd` as default and current best unless an AdamW candidate beats `0.913200`.
- If both AdamW candidates fail, revert the optional optimizer code path and return to literature before implementing Delta-SGD.
- Outcome: local AdamW failed badly. `lr=0.0005` scored `0.252000`; `lr=0.001` scored `0.100000`.
- Action: revert the optional AdamW optimizer code path and keep SGD as the only local optimizer.

## Twenty-fourth Literature Loop

### Trigger

- Reason: local AdamW failed badly, so do not add more adaptive optimizer code yet. Try a simpler client-local stabilization mechanism that preserves SGD and the current protocol.
- Current best: SGD with FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR/scheduler, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: adaptive optimizer code was unstable/low-scoring, server-side update clipping did not help, but client-side gradient clipping has not been tested under the current SGD/FedDrift stack.
- Candidate width: `PARALLEL_CANDIDATES=2`; add a default-off client gradient clip norm.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning client gradient clipping non-IID local training arXiv` | Look for client-local clipping as a drift/stability mechanism. | arXiv/paper indexes | Clipping is common in FL, usually on client updates; client-gradient clipping is a lower-level bounded variant. |
| `Differentially Private Learning with Adaptive Clipping Andrew Thakkar McMahan Ramaswamy NeurIPS 2021 OpenReview` | Confirm FL clipping sensitivity and tuning caveats. | NeurIPS/OpenReview | Andrew21 shows clipping norms depend on model/loss/data/client LR, so fixed values should be tested cautiously. |
| `FedZMG Efficient Client-Side Optimization gradient centralization non-IID federated learning` | Compare gradient clipping with current gradient centralization. | arXiv/paper indexes | FedZMG supports client-side gradient geometry changes without protocol changes. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Andrew21 | Differentially Private Learning with Adaptive Clipping / 2021 | https://openreview.net/forum?id=RUQ1zwZR8_ | Clipping norm choice depends on model, loss, data, LR, and other settings. | clipping | keep with caution |
| Zantalis26 | FedZMG: Efficient Client-Side Optimization in Federated Learning / 2026 | https://arxiv.org/abs/2602.18384 | Client-side gradient geometry can reduce non-IID drift without communication changes. | gradient geometry | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Local updates and aggregation normalization interact. | FedNova | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local gradient spikes may still hurt SGD | Andrew21 highlights clipping sensitivity; FedZMG supports client-local gradient transforms. | AdamW failed; SGD best may still benefit from bounded local gradients. | Clip gradients before optimizer step without changing DIFF metadata. | `client.py` default-off `--gradient_clip_norm`. |
| C2 | Fixed clipping is hard to tune | Andrew21 warns norms depend on task settings. | Server median clipping failed, so fixed clipping can hurt. | Use two conservative norms and revert if no gain. | CLI after code. |
| C3 | Distinct from failed server update clipping | Prior clipping acted on complete client DIFFs server-side. | FedNova median clipping tied/missed. | Client gradient clipping changes local optimization path instead of aggregation. | Client-local only. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P74 | Default-off client gradient clipping, norm 1.0. | Andrew21; Zantalis26 | Code: `--gradient_clip_norm`; candidate `1.0`. | Stabilize local SGD after gradient centralization and drift regularization. | Score below `0.913200` or slow/unstable. | Low-medium; client-local. |
| P75 | Default-off client gradient clipping, norm 5.0. | Andrew21; Zantalis26 | Same code; candidate `5.0`. | Less aggressive clipping if norm 1.0 over-constrains. | Score below best. | Low-medium. |
| P76 | Adaptive clip norm. | Andrew21 | Code: estimate/update clip norm online. | Avoid fixed-norm tuning. | Complexity/extra state. | Medium; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P74 | Server-side FedNova median clipping. | Different mechanism and location; not duplicate. | keep |
| P75 | Server-side FedNova median clipping. | Different mechanism and less aggressive option. | keep |
| P76 | Adaptive clipping. | More code than needed before fixed client clip audit. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P74 | 2 | 5 | 4 | 3 | 3 | 3 | 21 |
| P75 | 2 | 5 | 4 | 3 | 3 | 3 | 21 |
| P76 | 3 | 3 | 2 | 4 | 3 | 4 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P74 | `fednova_lr1875_m035_wd35e5_gc_clip1_feddyn1e4_feddrift2p5e5_ep5` | Code variant: `--gradient_clip_norm 1.0` |
| 2 | P75 | `fednova_lr1875_m035_wd35e5_gc_clip5_feddyn1e4_feddrift2p5e5_ep5` | Code variant: `--gradient_clip_norm 5.0` |

### Reflective Memory

- Keep no gradient clipping unless a client clipping candidate beats `0.913200`.
- If both fixed client clipping candidates fail, revert the optional code path and return to literature before adaptive clipping.
- Outcome: client gradient clipping failed. `clip_norm=5.0` scored `0.908700`; `1.0` scored `0.898400`.
- Action: revert the optional client gradient clipping code path and return to literature before adaptive clipping.

## Twenty-fifth Literature Loop

### Trigger

- Reason: client gradient clipping missed and was reverted. The closest recent non-kept signal is the architecture subcampaign `moderate_cnn_small_head` at `0.912100`, only `0.001100` below the current best.
- Current best: `moderate_cnn` with FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR/scheduler, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: broad new code paths have regressed. The registered small-head architecture is already under cap and close enough to justify a small subcampaign retune.
- Candidate width: `PARALLEL_CANDIDATES=2`; label rows as architecture subcampaign and keep `max_model_params=5000000`.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning architecture capacity regularization non-IID classifier head weight decay` | Check whether architecture capacity and regularization interaction is plausible. | local prior rows, paper indexes | Evidence is weaker than optimizer papers, but the local signal is close. |
| `FedBN normalization architecture federated learning non-IID model capacity` | Keep architecture-context source trail from the prior loop. | OpenReview, paper indexes | Normalization variant failed; small-head was the stronger architecture signal. |
| `decoupled weight decay regularization neural network capacity smaller model` | Confirm weight decay is a capacity-sensitive regularizer. | optimizer/regularization papers | Weight decay may need retuning after reducing classifier-head capacity. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Local model training and regularization affect federated generalization. | FedAvg baseline | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Model/local update geometry interacts with normalized aggregation. | FedNova | keep |
| Li21 | FedBN / 2021 | https://openreview.net/forum?id=6YEQUn0QICG | Architecture/normalization choices matter under non-IID FL. | architecture context | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Small-head capacity changed regularization needs | Smaller classifier head reduces parameter count and may need less or more L2. | `moderate_cnn_small_head` scored `0.912100`, close to best at current `weight_decay=3.5e-4`. | Weight decay is an existing CLI knob and architecture rows are labeled. | CLI `--weight_decay` inside architecture subcampaign. |
| C2 | Avoid new architecture code | Program prefers registered variants before new model code. | Registered norm failed; small-head is the only close architecture signal. | Retune the registered variant before adding architecture code. | No code changes. |
| C3 | Comparability risk | Architecture scores must be labeled separately. | Best remains `moderate_cnn`. | Keep explicit architecture-subcampaign descriptions. | Ledger/report labeling. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P77 | Small-head lower weight decay. | McMahan17; Wang20 | Architecture subcampaign: `--model_arch moderate_cnn_small_head --weight_decay 2.5e-4`. | Reduce over-regularization in the smaller classifier head. | Score below `0.913200`. | Medium; registered model schema. |
| P78 | Small-head higher weight decay. | McMahan17; Wang20 | Architecture subcampaign: `--model_arch moderate_cnn_small_head --weight_decay 4.5e-4`. | Test whether the close small-head row benefits from stronger regularization. | Score below best. | Medium. |
| P79 | New architecture variant. | architecture search literature | Code: add another registered model. | Broader capacity search. | Too broad before retuning the close variant. | High; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P77 | Moderate-CNN FedDrift weight-decay retunes. | Model capacity changed; not duplicate. | keep |
| P78 | Moderate-CNN FedDrift weight-decay retunes. | Model capacity changed; not duplicate. | keep |
| P79 | New architecture code. | Existing close variant has not been retuned. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P77 | 2 | 4 | 5 | 2 | 3 | 3 | 19 |
| P78 | 2 | 4 | 5 | 2 | 3 | 3 | 19 |
| P79 | 3 | 2 | 1 | 2 | 4 | 3 | 14 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P77 | `arch_smallhead_lr1875_m035_wd25e5_gc_feddyn1e4_feddrift2p5e5_ep5` | Architecture subcampaign: `--model_arch moderate_cnn_small_head --weight_decay 2.5e-4` |
| 2 | P78 | `arch_smallhead_lr1875_m035_wd45e5_gc_feddyn1e4_feddrift2p5e5_ep5` | Architecture subcampaign: `--model_arch moderate_cnn_small_head --weight_decay 4.5e-4` |

### Reflective Memory

- Keep `moderate_cnn` and `weight_decay=3.5e-4` unless a labeled small-head retune beats `0.913200`.
- If both small-head weight-decay retunes fail, return to literature before adding a new architecture variant.
- Outcome: small-head weight-decay retunes failed. `weight_decay=4.5e-4` scored `0.910300`; `2.5e-4` scored `0.907600`.
- Action: keep `moderate_cnn` and return to literature before adding a new architecture variant.

## Twenty-sixth Literature Loop

### Trigger

- Reason: small-head weight-decay retunes failed, but the original small-head architecture score `0.912100` remains the closest recent non-kept result.
- Current best: `moderate_cnn` with FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: small-head capacity is close, while small-head weight decay worsened. Server update scale may need retuning under the altered model geometry.
- Candidate width: `PARALLEL_CANDIDATES=2`; architecture subcampaign rows, no code changes.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `Adaptive Federated Optimization server learning rate model architecture federated learning` | Check whether server step scale should be retuned after context changes. | arXiv/paper indexes | FedOpt literature treats server LR as a sensitive heterogeneity knob. |
| `FedNova objective inconsistency server learning rate local update normalization architecture` | Tie FedNova server step scale to changed model/update geometry. | arXiv/paper indexes | FedNova normalized updates can still require server scale tuning. |
| `federated learning architecture subcampaign optimizer retune model capacity` | Check whether architecture changes warrant local optimizer retunes. | local prior rows, paper indexes | Program architecture guidance supports optimizer retunes around promising registered variants. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | Server optimizer scale is important under heterogeneity. | FedOpt/FedAvgM | keep |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | Normalized local updates interact with server aggregation scale. | FedNova | keep |
| McMahan17 | Communication-Efficient Learning of Deep Networks from Decentralized Data / 2017 | https://arxiv.org/abs/1602.05629 | Local model training and aggregation trade off communication and convergence. | FedAvg baseline | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Small-head update scale may differ | Reddi21/Wang20 support tuning server step scale under changed local update geometry. | Small-head base was close; weight decay did not help. | Server LR is an existing CLI knob and architecture row is labeled. | CLI `--server_lr`. |
| C2 | Avoid broader architecture code | Registered small-head is already close. | New architecture code has lower evidence than retuning the close registered variant. | No code change. | Existing `model_arch`. |
| C3 | Tight bounds only | Server LR neighbors under main architecture failed. | Context differs because model architecture changed, but avoid broad retunes. | Use immediate neighbors around `1.875`. | CLI only. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P80 | Small-head lower server LR. | Reddi21; Wang20 | Architecture subcampaign: `--model_arch moderate_cnn_small_head --server_lr 1.8125`. | Reduce server step scale for changed model geometry. | Score below `0.913200`. | Medium; registered model schema. |
| P81 | Small-head higher server LR. | Reddi21; Wang20 | Architecture subcampaign: `--model_arch moderate_cnn_small_head --server_lr 1.9375`. | Test whether the smaller head can take a slightly larger normalized server step. | Score below best. | Medium. |
| P82 | New architecture variant. | architecture search literature | Code: add another registered model. | Broader architecture search. | Too broad before server-step retune. | High; reserve. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P80 | Moderate-CNN FedDrift server-LR retune. | Model architecture changed; not a strict duplicate. | keep |
| P81 | Moderate-CNN FedDrift server-LR retune. | Model architecture changed; not a strict duplicate. | keep |
| P82 | New architecture code. | Existing close architecture still has one optimizer scale axis left. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P80 | 2 | 4 | 5 | 3 | 3 | 3 | 20 |
| P81 | 2 | 4 | 5 | 3 | 3 | 3 | 20 |
| P82 | 3 | 2 | 1 | 2 | 4 | 3 | 14 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P80 | `arch_smallhead_lr18125_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_ep5` | Architecture subcampaign: `--model_arch moderate_cnn_small_head --server_lr 1.8125` |
| 2 | P81 | `arch_smallhead_lr19375_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_ep5` | Architecture subcampaign: `--model_arch moderate_cnn_small_head --server_lr 1.9375` |

### Reflective Memory

- Keep `moderate_cnn` and `server_lr=1.875` unless a labeled small-head server-LR neighbor beats `0.913200`.
- If both small-head server-LR retunes fail, stop the small-head branch and return to literature before new architecture code.
- Outcome: small-head server-LR retunes failed. `server_lr=1.8125` scored `0.908500`; `1.9375` scored `0.908400`.
- Action: keep `moderate_cnn`, close the current small-head branch, and do not add new architecture code without another literature-backed rationale.

## Twenty-seventh Literature Loop

### Trigger

- Reason: small-head server-LR retunes failed and the architecture branch is closed. Watchdog printed `recommendation=continue`, but the next safe axis should be literature-backed rather than a new architecture variant.
- Current best: `0.913200` with `moderate_cnn`, FedNova `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, default client LR/scheduler, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: architecture, FedProx, scheduler-floor, client-LR, weight-decay, epoch, and server-scale neighbors failed under the drift-corrected stack. The client momentum axis was tested earlier under a weaker pre-FedDyn/FedDrift objective but not under the current best stack.
- Candidate width: `PARALLEL_CANDIDATES=2`; no code changes, one local optimizer knob.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning non-IID client momentum SGD momentum benefits local training heterogeneity arXiv` | Check whether client-side momentum remains a plausible drift-control knob under non-IID local training. | arXiv, Hugging Face papers | Cheng23 directly supports momentum for non-IID FedAvg/SCAFFOLD-style local training. |
| `federated learning CIFAR non-IID data augmentation mixup client local training arXiv` | Look for low-protocol client-only regularizers after optimizer retunes stalled. | arXiv, OpenReview/project pages | FedMix supports mixup-style augmentation, but its mean-sharing mechanism is a protocol change; local-only mixup is a possible later code mutation. |
| `federated learning classifier calibration non-IID CIFAR classifier bias arXiv` | Revisit classifier-head bias after small-head architecture nearly matched best. | arXiv, paper indexes | CCVR supports classifier-bias diagnosis, but its post-training calibration needs new calibration logic and likely a separate subcampaign. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Cheng23 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2023 | https://arxiv.org/abs/2306.16504 | Momentum can improve convergence under non-IID local training without extra protocol state. | client optimizer momentum | keep |
| Yoon21 | FedMix: Approximation of Mixup under Mean Augmented Federated Learning / 2021 | https://arxiv.org/abs/2107.00233 | Heterogeneous clients benefit from augmentation, but MAFL shares averaged local data. | augmentation | reserve |
| Luo21 | No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data / 2021 | https://arxiv.org/abs/2106.05001 | Classifier layers can be more biased than representation layers under non-IID FL. | classifier calibration | reserve |
| Gao22 | FedDC / 2022 | https://arxiv.org/abs/2203.11751 | Client drift correction is already the best local stateful mechanism here. | drift correction | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local objective changed momentum needs | Cheng23 motivates momentum as a non-IID local-training stabilizer. | Client momentum was only tested before FedDyn/FedDrift; current objective now includes two local drift terms. | `--momentum` is already a client CLI knob and does not alter FLModel metadata. | CLI `--momentum`. |
| C2 | Avoid protocol-changing augmentation | FedMix improves non-IID FL through mean-augmented mixup, but mean sharing is not in the current protocol. | Broad code mutations such as AdamW and clipping failed. | Keep augmentation as reserve unless CLI-only candidates fail. | Possible future `client.py` local-only mixup, not this batch. |
| C3 | Classifier bias is plausible but heavier | Luo21 identifies classifier bias in non-IID classifiers. | Small-head came close but retunes missed. | Calibration would require new logic and careful comparability labeling. | Reserve; avoid before simpler momentum audit. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P83 | Lower client momentum under FedDrift best stack. | Cheng23 | CLI only: `--momentum 0.85`. | Reduce overshoot when FedDyn/FedDrift already supply local correction. | Score below `0.913200`. | Low. |
| P84 | Higher client momentum under FedDrift best stack. | Cheng23 | CLI only: `--momentum 0.95`. | Test whether stronger local momentum complements drift correction. | Score below `0.913200`. | Low. |
| P85 | Local-only mixup. | Yoon21 | Code: add default-off `--mixup_alpha`, local batch mixup only. | Improve generalization under label skew without sharing data. | No gain or unstable training. | Medium; reserve. |
| P86 | Classifier calibration. | Luo21 | Code: post-training or classifier-head calibration. | Reduce non-IID classifier bias. | Requires evaluation/procedure changes or added complexity. | High; reject for now. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P83 | Pre-FedDrift client-momentum retune. | Not strict duplicate because current local objective changed. | keep |
| P84 | Pre-FedDrift client-momentum retune. | Not strict duplicate; tests the other side around default `0.9`. | keep |
| P85 | Label smoothing / clipping / AdamW code paths. | Different mechanism but code-bearing; reserve behind CLI audit. | reserve |
| P86 | Small-head architecture subcampaign. | Too much new calibration surface after architecture branch failed. | reject |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P83 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P84 | 2 | 5 | 5 | 3 | 3 | 3 | 22 |
| P85 | 3 | 4 | 2 | 3 | 4 | 3 | 20 |
| P86 | 3 | 2 | 1 | 3 | 3 | 4 | 12 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P83 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_mom085_ep5` | Current best stack plus `--momentum 0.85` |
| 2 | P84 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_mom095_ep5` | Current best stack plus `--momentum 0.95` |

### Reflective Memory

- The older momentum miss is not decisive under the current FedDyn/FedDrift local objective.
- If both momentum neighbors fail, do not continue momentum jitter; promote the reserved local-only mixup idea only after another literature check confirms it can be implemented without protocol or evaluation changes.
- Outcome: current-stack client momentum neighbors failed. `momentum=0.85` scored `0.906100`; `0.95` scored `0.905800`.
- Action: keep client momentum at default `0.9`; do not continue momentum jitter under the current stack.

## Twenty-eighth Literature Loop

### Trigger

- Reason: current-stack client momentum missed, and the reserved local-only mixup idea needs a protocol-safety check before code changes.
- Current best: `0.913200` with `moderate_cnn`, FedNova `server_lr=1.875`, `server_momentum=0.35`, default client LR/momentum/scheduler, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`.
- Recent symptoms: optimizer-scale and architecture retunes failed. The current CIFAR training path already uses crop/flip augmentation, but not label-space interpolation.
- Candidate width: `PARALLEL_CANDIDATES=2`; add default-off client-local mixup only, no shared data, no new metadata, no evaluation changes.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `FedMix Approximation of Mixup under Mean Augmented Federated Learning local mixup non-IID federated learning` | Separate FedMix's protocol-coupled MAFL idea from local-only mixup. | arXiv, OpenReview, KAIST page | FedMix supports mixup in FL but full MAFL sends averaged local data, so only local mixup is compatible here. |
| `mixup Beyond Empirical Risk Minimization arXiv CIFAR regularization` | Check the base method and CIFAR evidence for local implementation. | arXiv, Meta AI page | Mixup is a local convex-combination training principle with CIFAR evidence and no dependency needs. |
| `federated learning local mixup non-IID label skew CIFAR client-side augmentation arXiv` | Look for non-IID FL motivation for local augmentation. | OpenReview, paper indexes | Local-only mixup is weaker than FedMix but remains protocol-safe. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zhang17 | mixup: Beyond Empirical Risk Minimization / 2017 | https://arxiv.org/abs/1710.09412 | Neural nets can overfit and benefit from convex input/label interpolation. | local augmentation | keep |
| Yoon21 | FedMix: Approximation of Mixup under Mean Augmented Federated Learning / 2021 | https://arxiv.org/abs/2107.00233 | Non-IID FL can benefit from mixup-style augmentation, but MAFL exchanges averaged data. | FL mixup | keep for motivation; reject MAFL protocol |
| OpenReview21 | FedMix ICLR 2021 page | https://openreview.net/forum?id=Ogga20D2HO- | Confirms FedMix is peer-reviewed and targets difficult non-IID settings. | FL mixup provenance | context |
| Luo21 | CCVR / 2021 | https://arxiv.org/abs/2106.05001 | Classifier bias remains plausible but heavier than local augmentation. | classifier calibration | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local overfitting under heterogeneity | Zhang17 shows mixup improves CIFAR generalization by interpolating examples and labels. | Current best has plateaued after many optimizer retunes. | Local batch mixup changes only client training loss. | `client.py`, `job.py` CLI forwarding. |
| C2 | FedMix protocol boundary | Yoon21 uses averaged local data exchange in MAFL. | Protocol-changing ideas are forbidden in this campaign. | Implement local-only mixup, not MAFL/FedMix metadata or shared data. | `client.py` only; no FLModel fields. |
| C3 | Loss interaction risk | FedDyn/FedDrift already add local regularization. | Heavy code mutations such as AdamW and clipping regressed. | Keep mixup default-off and sweep two conservative alphas. | CLI `--mixup_alpha`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P87 | Conservative local mixup. | Zhang17; Yoon21 | Add default-off `--mixup_alpha`; candidate `--mixup_alpha 0.1`. | Mild label-space interpolation may improve non-IID generalization without destabilizing drift terms. | Score below `0.913200`. | Medium-low; client-local code. |
| P88 | Standard local mixup. | Zhang17; Yoon21 | Same code; candidate `--mixup_alpha 0.2`. | Stronger regularization closer to common CIFAR mixup settings. | Score below best or unstable loss. | Medium-low. |
| P89 | FedMix/MAFL averaged-data exchange. | Yoon21 | Add exchanged averaged local data. | Closer to FedMix paper. | Requires new protocol data exchange. | Reject. |
| P90 | CCVR classifier calibration. | Luo21 | Add post-training classifier calibration. | Address classifier bias directly. | Requires calibration/evaluation procedure changes. | Reserve high risk. |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P87 | Label smoothing. | Different mechanism; local interpolation of inputs and labels. | keep |
| P88 | Label smoothing. | Stronger interpolation may conflict with existing regularizers but is source-backed. | keep |
| P89 | None. | Violates no new protocol/shared data rule. | reject |
| P90 | Small-head architecture branch. | Too heavy after architecture branch closed. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P87 | 3 | 4 | 3 | 4 | 4 | 3 | 22 |
| P88 | 3 | 4 | 3 | 4 | 4 | 3 | 22 |
| P90 | 3 | 2 | 1 | 3 | 3 | 4 | 12 |
| P89 | 4 | 1 | 1 | 4 | 4 | 3 | 12 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P87 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_mixup01_ep5` | Current best stack plus `--mixup_alpha 0.1` |
| 2 | P88 | `fednova_lr1875_m035_wd35e5_gc_feddyn1e4_feddrift2p5e5_mixup02_ep5` | Current best stack plus `--mixup_alpha 0.2` |

### Reflective Memory

- Only local mixup is compatible; do not implement FedMix/MAFL averaged-data exchange without human approval for a protocol upgrade.
- If both conservative local mixup alphas fail, revert the optional code path and return to literature instead of broadening augmentation immediately.
- Outcome: local-only mixup improved the best. `mixup_alpha=0.2` scored `0.914100`; `0.1` scored `0.913700`.
- Action: keep the default-off `--mixup_alpha` code path and narrow around `0.2` before moving to another mechanism.
- Follow-up: mixup narrowing missed. `mixup_alpha=0.15` scored `0.911800`; `0.3` scored `0.906100`.
- Action: keep `mixup_alpha=0.2`; stop alpha-only jitter and test one interaction axis under the new local objective.
- Follow-up: mixup-enabled weight-decay interaction missed. `weight_decay=3.0e-4` scored `0.912100`; `4.0e-4` scored `0.906600`.
- Action: keep `weight_decay=3.5e-4` with `mixup_alpha=0.2`.
- Follow-up: mixup-enabled server-LR interaction missed. `server_lr=1.9375` scored `0.913200`; `1.8125` scored `0.910000`.
- Action: keep `server_lr=1.875` with `mixup_alpha=0.2` and return to literature before more post-mixup local jitter.

## Twenty-ninth Literature Loop

### Trigger

- Reason: local-only mixup improved the best, but alpha narrowing plus weight-decay and server-LR interactions failed. The next idea should build on the augmentation signal without changing the FL protocol.
- Current best: `0.914100` with `moderate_cnn`, FedNova `server_lr=1.875`, `server_momentum=0.35`, default client LR/momentum/scheduler, `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5`, `--feddrift_beta 0.9`, and local-only `--mixup_alpha 0.2`.
- Recent symptoms: mixup is useful, but more alpha/regularization/server-step jitter is not. The next augmentation should be default-off and client-local.
- Candidate width: `PARALLEL_CANDIDATES=2`; add default-off CutMix only, no shared data, no new FLModel metadata, no evaluation changes.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `CutMix federated learning non-IID CIFAR client augmentation arXiv` | Find an augmentation adjacent to mixup that remains local. | arXiv, paper indexes | CutMix is not FL-specific but is local and CIFAR-backed. |
| `CutMix Regularization Strategy to Train Strong Classifiers with Localizable Features arXiv` | Confirm the primary method details and CIFAR evidence. | arXiv | CutMix mixes patches and labels by patch area. |
| `RandAugment practical automated data augmentation CIFAR arXiv` | Compare another augmentation family. | arXiv/paper indexes | RandAugment would require transform-policy edits and broader search; reserve. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Yun19 | CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features / 2019 | https://arxiv.org/abs/1905.04899 | Improve image classifier generalization by replacing local patches and mixing labels by area. | local augmentation | keep |
| Zhang17 | mixup / 2017 | https://arxiv.org/abs/1710.09412 | Prior local interpolation improved this harness. | local augmentation | context |
| DeVries17 | Improved Regularization of Convolutional Neural Networks with Cutout / 2017 | https://arxiv.org/abs/1708.04552 | Regional dropout is simple and CIFAR-backed, but removes pixels rather than reusing them. | local augmentation | reserve |
| Cubuk19 | RandAugment / 2019 | https://arxiv.org/abs/1909.13719 | Strong augmentation can improve CIFAR, but policy magnitude search is broader. | augmentation policy | reserve |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Augmentation signal is real | Mixup improved to `0.914100`. | Post-mixup retunes failed, but augmentation itself helped. | Try a neighboring local augmentation mechanism rather than optimizer jitter. | `client.py`, `job.py`. |
| C2 | Preserve protocol | CutMix can be applied entirely inside each local batch. | FedMix/MAFL shared-data exchange was rejected. | Add no FLModel fields and share no data. | Client-local code. |
| C3 | Avoid broad transform policy search | RandAugment is stronger but opens a wider transform/magnitude surface. | Candidate width is 2 and repeated broad code paths have regressed. | Test CutMix before policy-search augmentation. | Default-off `--cutmix_alpha`. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P91 | Conservative CutMix. | Yun19; Zhang17 | Add default-off `--cutmix_alpha`; candidate `--cutmix_alpha 0.5` with `mixup_alpha=0`. | Patch-level interpolation may regularize differently from mixup. | Score below `0.914100`. | Medium-low; client-local code. |
| P92 | Standard CutMix. | Yun19; Zhang17 | Same code; candidate `--cutmix_alpha 1.0` with `mixup_alpha=0`. | Test common CutMix strength. | Score below best or unstable loss. | Medium-low. |
| P93 | Cutout. | DeVries17; Yun19 | Add random erasing/cutout. | Simpler regional dropout. | CutMix is better evidenced after mixup because it preserves pixels. | reserve |
| P94 | RandAugment. | Cubuk19 | Add transform-policy knobs. | Stronger augmentation family. | Too broad for the next branch. | reserve |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P91 | Mixup alpha sweep. | Different spatial mixing mechanism, still local. | keep |
| P92 | Mixup alpha sweep. | Different spatial mixing mechanism, tests standard strength. | keep |
| P93 | CutMix. | Weaker pixel-removal variant; reserve. | reserve |
| P94 | Existing crop/flip augmentation. | Broader search surface. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P91 | 3 | 4 | 3 | 4 | 4 | 3 | 22 |
| P92 | 3 | 4 | 3 | 4 | 4 | 3 | 22 |
| P93 | 2 | 4 | 3 | 3 | 3 | 3 | 18 |
| P94 | 3 | 3 | 1 | 4 | 4 | 4 | 17 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P91 | `fednova_lr1875_m035_wd35e5_gc_cutmix05_feddyn1e4_feddrift2p5e5_ep5` | Current best stack but `--mixup_alpha 0`, `--cutmix_alpha 0.5` |
| 2 | P92 | `fednova_lr1875_m035_wd35e5_gc_cutmix10_feddyn1e4_feddrift2p5e5_ep5` | Current best stack but `--mixup_alpha 0`, `--cutmix_alpha 1.0` |

### Reflective Memory

- CutMix must remain local-only; do not combine it with mixup in the first batch.
- If both CutMix alphas fail, revert the optional code path and do not continue augmentation code without another literature reset.
- Outcome: CutMix failed. `cutmix_alpha=0.5` scored `0.904800`; `1.0` scored `0.897800`.
- Action: revert the default-off CutMix code path; keep local-only mixup as the surviving augmentation.

## Thirtieth Literature Loop

### Trigger

- Reason: CutMix failed and mixup remains the only surviving augmentation. The next branch should target label-skew/class-imbalance loss behavior locally, without changing data splits, evaluation, or FL metadata.
- Current best: `0.914100` with FedNova/FedDyn/FedDrift, `--gradient_centralization`, `--mixup_alpha 0.2`, `weight_decay=3.5e-4`, and `server_lr=1.875`.
- Recent symptoms: local interpolation helped; patch mixing did not. Class/label skew may still benefit from focusing hard local examples.
- Candidate width: `PARALLEL_CANDIDATES=2`; add default-off focal loss and test it with the kept mixup setting.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `federated learning non-IID label skew focal loss client local loss CIFAR arXiv` | Find local loss functions for label skew. | arXiv, paper indexes | FL-specific focal variants exist, but base focal loss is enough for a safe client-local audit. |
| `focal loss dense object detection class imbalance arXiv 1708.02002` | Confirm primary focal-loss mechanism. | arXiv | Focal loss down-weights easy examples and focuses hard examples. |
| `class-balanced loss effective number of samples CIFAR arXiv 1901.05555` | Compare a class-count weighted alternative. | arXiv | Effective-number loss needs local class-count weighting; reserve behind focal. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Lin17 | Focal Loss for Dense Object Detection / 2017 | https://arxiv.org/abs/1708.02002 | Class imbalance can be handled by down-weighting easy examples. | focal loss | keep |
| Cui19 | Class-Balanced Loss Based on Effective Number of Samples / 2019 | https://arxiv.org/abs/1901.05555 | Class-count imbalance can be corrected by effective-number reweighting. | class-balanced loss | reserve |
| Zhang17 | mixup / 2017 | https://arxiv.org/abs/1710.09412 | Soft interpolation is the current surviving augmentation. | local augmentation | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Hard-example focus under label skew | Lin17 proposes reshaping CE to focus hard examples. | Mixup improved, but optimizer and CutMix retunes failed. | Focal loss is local and does not need client metadata. | `client.py`, `job.py`. |
| C2 | Class-count weighting is heavier | Cui19 uses class counts/effective samples. | Site class distributions are skewed, but count weighting may overfit local priors. | Reserve until focal is tested. | Future client-local class weights. |
| C3 | Mixup compatibility | Mixup uses two hard labels with a convex loss. | Best row uses `mixup_alpha=0.2`. | Apply focal loss per mixed label component. | Existing mixup loss path. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P95 | Mild focal loss with mixup. | Lin17; Zhang17 | Add default-off `--focal_gamma`; candidate `--mixup_alpha 0.2 --focal_gamma 1.0`. | Focus hard examples while preserving the current mixup gain. | Score below `0.914100`. | Medium-low. |
| P96 | Standard focal loss with mixup. | Lin17; Zhang17 | Same code; candidate `--mixup_alpha 0.2 --focal_gamma 2.0`. | Test common focal strength. | Score below best or unstable training. | Medium-low. |
| P97 | Effective-number class-balanced loss. | Cui19 | Add local class-count weighting. | Directly handle local class imbalance. | More local-prior bias and code surface. | reserve |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P95 | Label smoothing / mixup. | Different loss focusing mechanism; compatible with mixup. | keep |
| P96 | Label smoothing / mixup. | Stronger focal setting, same local code. | keep |
| P97 | FedLC/classifier calibration. | Similar class-prior family; reserve. | reserve |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P95 | 2 | 4 | 3 | 3 | 4 | 3 | 19 |
| P96 | 2 | 4 | 3 | 3 | 4 | 3 | 19 |
| P97 | 3 | 3 | 2 | 4 | 3 | 3 | 18 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P95 | `fednova_lr1875_m035_wd35e5_gc_mixup02_focal1_feddyn1e4_feddrift2p5e5_ep5` | Current best plus `--focal_gamma 1.0` |
| 2 | P96 | `fednova_lr1875_m035_wd35e5_gc_mixup02_focal2_feddyn1e4_feddrift2p5e5_ep5` | Current best plus `--focal_gamma 2.0` |

### Reflective Memory

- If focal loss fails, revert the optional code path and revisit class-balanced/effective-number loss only with a separate literature loop.
- Outcome: focal loss failed. `focal_gamma=1.0` scored `0.906000`; `2.0` scored `0.896800`.
- Action: revert the optional focal code path; keep plain cross-entropy with local-only mixup.

## Thirty-first Literature Loop

### Trigger

- Reason: focal loss failed, but class imbalance under label skew remains plausible. Use a separate loop for effective-number class-balanced loss as planned in the focal-loop reflective memory.
- Current best: `0.914100` with FedNova/FedDyn/FedDrift, `--gradient_centralization`, `--mixup_alpha 0.2`, plain cross-entropy, `weight_decay=3.5e-4`, and `server_lr=1.875`.
- Feasibility check: `CIFAR10_Idx` exposes each client's local `targets`, so class weights can be computed client-side after dataset creation without modifying `data/*`, sending metadata, or changing evaluation.
- Candidate width: `PARALLEL_CANDIDATES=2`; add default-off effective-number class weighting and test two betas with the kept mixup setting.

### Search Queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| `class-balanced loss effective number of samples CIFAR arXiv 1901.05555` | Primary method for local class-count weighting. | arXiv | Effective-number weights were designed for long-tailed CIFAR/ImageNet-like classification. |
| `federated learning non-IID label skew class balanced loss local client class counts` | Check FL compatibility. | arXiv, paper indexes | Local class weighting is protocol-safe but may overfit client priors. |
| `mixup class balanced loss label skew CIFAR` | Check interaction risk with mixup. | paper indexes | Mixup uses two labels; weighted CE can still apply per mixed component. |

### Candidate Papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Cui19 | Class-Balanced Loss Based on Effective Number of Samples / 2019 | https://arxiv.org/abs/1901.05555 | Long-tailed class counts can be corrected by effective-number reweighting. | class-balanced loss | keep |
| Lin17 | Focal Loss / 2017 | https://arxiv.org/abs/1708.02002 | Hard-example focusing failed here but motivates imbalance-sensitive losses. | focal loss | context |
| Zhang17 | mixup / 2017 | https://arxiv.org/abs/1710.09412 | Current surviving augmentation uses weighted CE over two labels. | local augmentation | context |

### Challenge Cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local label skew | Cui19 supports class-count based weighting for imbalanced classification. | FedLC/focal failed, but mixup improved and label skew remains in the data split. | Class weights can be computed from local `train_dataset.targets`. | `client.py`, `job.py`. |
| C2 | Local prior overfit risk | Client-local class weighting may bias updates toward locally rare labels. | Many client-loss changes have regressed. | Test only two conservative betas and default off. | CLI `--class_balance_beta`. |
| C3 | Mixup compatibility | Weighted CE can apply to each mixed label component. | Best row uses `mixup_alpha=0.2`. | Reuse existing mixup loss path with weighted criterion. | Client-local loss only. |

### Proposal Cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P98 | Mild effective-number weighting. | Cui19; Zhang17 | Add default-off `--class_balance_beta`; candidate `--mixup_alpha 0.2 --class_balance_beta 0.99`. | Reweight local CE without extreme rare-class amplification. | Score below `0.914100`. | Medium-low. |
| P99 | Stronger effective-number weighting. | Cui19; Zhang17 | Same code; candidate `--class_balance_beta 0.999`. | Test common stronger weighting. | Score below best or unstable local loss. | Medium-low. |
| P100 | Per-round dynamic class weights. | Cui19 | Recompute weights every round. | No need; local data fixed. | unnecessary complexity. | reject |

### Duplicate and Null Filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P98 | FedLC/focal. | Different count-weighted CE mechanism; default off. | keep |
| P99 | FedLC/focal. | Stronger count-weighted CE mechanism. | keep |
| P100 | P98/P99. | Same data and no benefit. | reject |

### Proposal Scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P98 | 2 | 4 | 3 | 4 | 3 | 3 | 19 |
| P99 | 2 | 4 | 3 | 4 | 3 | 3 | 19 |
| P100 | 1 | 4 | 1 | 3 | 1 | 3 | 12 |

### QWBE-style Next-Candidate Batch Plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P98 | `fednova_lr1875_m035_wd35e5_gc_mixup02_cb99_feddyn1e4_feddrift2p5e5_ep5` | Current best plus `--class_balance_beta 0.99` |
| 2 | P99 | `fednova_lr1875_m035_wd35e5_gc_mixup02_cb999_feddyn1e4_feddrift2p5e5_ep5` | Current best plus `--class_balance_beta 0.999` |

### Reflective Memory

- If effective-number class weighting fails, revert the optional code path and stop local loss-code mutations until the next literature reset.
- Outcome: effective-number class-balanced loss failed. `class_balance_beta=0.99` scored `0.906000`; `0.999` crashed with NVFlare abort/score-extraction failure.
- Action: revert the optional class-balanced code path and stop local loss-code mutations until the next literature reset.

## Post Thirty-first Continuation Note

- Plateau watchdog after the class-balanced batch recommended `continue`, not `literature`.
- CLI-only local-compute audit under the kept mixup stack did not improve: `aggregation_epochs=4` scored `0.905400`; `6` scored `0.911300`.
- CLI-only server-momentum retune under the kept mixup stack also did not improve: `server_momentum=0.40` scored `0.911000`; `0.30` scored `0.910300`.
- CLI-only client-LR sweep found a near miss: `lr=0.055` scored `0.914500`, below the `0.000500` material-improvement threshold over the kept `0.914100`; `0.045` scored `0.910700`.
- Narrow client-LR sweep did not confirm the near miss: `lr=0.0525` scored `0.912400`; `0.0575` scored `0.910600`.
- FedDyn-alpha retune under mixup did not improve: `feddyn_alpha=5e-5` scored `0.910200`; `2e-4` scored `0.909800`.
- Keep the current best `aggregation_epochs=5`, `server_momentum=0.35`, default client LR, and `feddyn_alpha=1e-4`; avoid more client-loss code until a later literature reset.
