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
