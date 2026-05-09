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

# Literature loop 2026-05-09 local sharpness minimization

## Trigger

- Reason: watchdog is `recommendation=continue`, but the ledger scan found no clear non-duplicate scalar/local-compute axis after Cutout missed and was removed.
- Current best: 0.918600, `moderate_cnn`, FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `cosine_lr_eta_min_factor=0.00015`, `fedproxloss_mu=3e-5`, `zero_mean_gradients`, `class_balanced_loss_beta=0.90`, `aggregation_epochs=7`.
- Recent symptoms from `results.tsv`: Cutout sizes 8/12/10/14 scored 0.915600/0.917500/0.918400/0.917000; local epochs 5/6/8/9/11, exact steps, scheduler, FedProx, class-balanced beta, LR/momentum, clipping, Nesterov, architecture variants, mixup, and smoothing are null.
- Confirmed null/worse ideas to avoid: scalar optimizer/local-compute jitter, target-mixing regularizers, local occlusion, client clipping, Nesterov, FedNova, SCAFFOLD, FedAdam, median aggregation, focal/LDAM/FedLC/FedRS/logit-prior/self-distillation.
- Candidate width: run width 1 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`, because width-2 NVFlare runs previously exposed communication failures.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| FedSAM federated learning sharpness-aware minimization non-IID CIFAR arXiv | find a non-duplicate FL-local optimizer for non-IID sharp minima | web search, ICML/PMLR | FedSAM directly targets ERM local optimizers falling into sharp valleys under non-IID FL. |
| Sharpness-Aware Minimization for Efficiently Improving Generalization ICLR 2021 arXiv Foret | primary SAM source and CIFAR generalization evidence | OpenReview, Google Research | SAM minimizes both loss value and sharpness with two local forward/backward passes. |
| federated learning stochastic weight averaging local weight averaging non-IID CIFAR | reserve flat-minima alternative with lower runtime cost | web search, arXiv | SWA is simple and low overhead but less FL-specific than FedSAM. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://proceedings.mlr.press/v162/qu22a.html | non-IID FL local ERM can drive sharp valleys and local-client deviation | FedSAM local optimizer | keep |
| Foret21 | Sharpness-Aware Minimization for Efficiently Improving Generalization / 2021 | https://openreview.net/forum?id=6Tm1mposlrM | conventional loss minimization can generalize poorly; flat neighborhoods improve robustness | SAM | keep |
| Izmailov18 | Averaging Weights Leads to Wider Optima and Better Generalization / 2018 | https://arxiv.org/abs/1803.05407 | averaging SGD trajectory points finds flatter solutions with low overhead | local SWA | reserve |
| Qu22 MoFedSAM | momentum bridge from local to global SAM | https://proceedings.mlr.press/v162/qu22a.html | local/global model mismatch under SAM | server-coupled momentum variant | reject now: active FedAvgM already provides server momentum; avoid broader protocol/state change |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Active local ERM may be landing in sharp client minima | Qu22 argues local ERM under distribution shift can increase deviation and sharp global valleys | many local regularizers and scalar sweeps miss the 0.918600 mark | swap local optimizer step while preserving DIFF upload and metadata | `client.py`, `job.py` |
| C2 | Generalization, not training budget, is the likely limit | Foret21 reports SAM generalization gains on CIFAR-style benchmarks | more local epochs/steps underfit or regress | add two-backward SAM only when `--sam_rho > 0` | `client.py` |
| C3 | A cheaper flat-minima fallback exists | Izmailov18 supports SWA with minimal compute | SAM may exceed runtime or underfit with dropout/FedProx | reserve local weight averaging if SAM fails due runtime/cost | `client.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | FedSAM light radius | Qu22; Foret21 | add default-off `--sam_rho`; run active stack with `--sam_rho 0.02` | flatten local objectives without changing aggregation | timeout or score below 0.916 | low-medium: two backward passes but client-local |
| P2 | FedSAM standard radius | Qu22; Foret21 | active stack with `--sam_rho 0.05` | stronger sharpness penalty if 0.02 is too weak | underfit, timeout, or worse than P1 | low-medium |
| P3 | Local SWA | Izmailov18 | future default-off local weight averaging near end of each round | cheap flat-solution bias | no gain after SAM or code complexity not justified | low |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1/P2 | no existing SAM rows or `sam_rho` arg | different from clipping, FedProx, and zero-mean gradients because it perturbs weights before the optimizer step | select |
| P3 | related flat-minima family | less FL-specific than FedSAM; could average dropout-heavy local trajectories poorly | reserve |

## Proposal scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 SAM rho 0.02 | 3 | 4 | 3 | 5 | 5 | 4 | 24 |
| P2 SAM rho 0.05 | 3 | 4 | 3 | 5 | 4 | 4 | 23 |
| P3 local SWA | 2 | 5 | 3 | 4 | 4 | 1 | 24 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_sam002_w1` | active best stack plus `--sam_rho 0.02` |
| 2 | P2 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_sam005_w1` | active best stack plus `--sam_rho 0.05` |
| 3 | P3 reserve |  | no launch unless SAM shows signal or fails only on cost |

## Reflective memory

- Keep: SAM is the lowest-risk non-duplicate optimizer mechanism left because it is local-only and source-backed for non-IID FL.
- Discard: MoFedSAM/server-coupled variants for now because the active stack already uses FedAvgM and protocol/state risk is higher.
- Do not retry: scalar jitter, Cutout, mixup, label smoothing, Nesterov, clipping, and architecture variants under the current active stack.
- Sources to carry forward: Qu22 FedSAM; Foret21 SAM; Izmailov18 SWA.
- Validation: default-off `sam_rho` branch passed `make validate`, `make smoke`, and a no-ledger `--sam_rho 0.02` smoke.

---

# Literature loop 2026-05-09 local occlusion augmentation

## Trigger

- Reason: watchdog is still `recommendation=continue`, but after Nesterov, architecture variants, client clipping, label smoothing, mixup, FedNova, and scalar optimizer/loss sweeps, no clear non-duplicate safe local axis remains.
- Current best: 0.918600, `moderate_cnn`, FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `cosine_lr_eta_min_factor=0.00015`, `fedproxloss_mu=3e-5`, `zero_mean_gradients`, `class_balanced_loss_beta=0.90`, `aggregation_epochs=7`.
- Recent symptoms from `results.tsv`: mixup 0.912200, label smoothing 0.912600, client grad clipping 0.897200/0.915200, architecture variants 0.911600/0.911800, Nesterov 0.908600.
- Confirmed null/worse ideas to avoid: standalone mixup and label smoothing, client/server update clipping, Nesterov, registered architecture variants, FedNova, SCAFFOLD, FedAdam, median aggregation, focal/LDAM/FedLC/FedRS/logit-prior/self-distillation, and scalar FedAvgM/FedProx/local-compute jitter.
- Candidate width: run width 1 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`, because width-2 NVFlare runs previously exposed communication failures.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| Cutout regularization convolutional neural networks CIFAR arXiv 1708.04552 | find a local augmentation distinct from failed mixup/label smoothing | arXiv, web search | Cutout targets overfitting with square masking and reports CIFAR gains. |
| CutMix regularization strong classifiers CIFAR arXiv 1905.04899 | compare patch reuse versus patch deletion after mixup failed | arXiv, web search | CutMix is stronger but label-mixing repeats the failed mixup-style target path. |
| Random Erasing Data Augmentation arXiv 1708.04896 image classification | check a close alternative to Cutout with random rectangles | arXiv, web search | Random erasing supports occlusion robustness with no protocol change. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| DeVries17 | Improved Regularization of Convolutional Neural Networks with Cutout / 2017 | https://arxiv.org/abs/1708.04552 | CNN overfitting on image classification; regularization beyond crop/flip | square occlusion augmentation | keep |
| Zhong17 | Random Erasing Data Augmentation / 2017 | https://arxiv.org/abs/1708.04896 | overfitting and occlusion sensitivity in image classifiers | random rectangle erasing | keep as supporting variant |
| Yun19 | CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features / 2019 | https://arxiv.org/abs/1905.04899 | regional dropout loses information; patch replacement can preserve pixels | patch replacement with label mixing | keep as context, reserve due label mixing |
| Zhang18 | mixup: Beyond Empirical Risk Minimization / 2018 | https://arxiv.org/abs/1710.09412 | vicinal regularization and memorization | convex sample/label mixing | reject: local mixup already scored 0.912200 |
| Muller19 | When Does Label Smoothing Help? / 2019 | https://arxiv.org/abs/1906.02629 | overconfidence and calibration | target smoothing | reject: smoothing already scored 0.912600 |
| Li21 | FedBN / 2021 | https://arxiv.org/abs/2102.07623 | feature-shift non-IID | local normalization | reject: architecture/protocol comparability issue in this campaign |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Overfitting despite standard crop/flip | DeVries17 shows square input masking complements existing augmentation and improves CIFAR/SVHN robustness | scalar optimizer/loss and architecture searches no longer approach the 0.918600 high-water mark | add local masking after the existing data transform without editing `data/*` | `client.py`, `job.py` |
| C2 | Occlusion/shortcut sensitivity under class skew | Zhong17 frames random erasing as generating occluded examples that reduce overfitting | class-balanced loss helps but local clients may still learn class-specific shortcuts | per-batch cutout can regularize client updates while preserving DIFF uploads | `client.py`, `job.py` |
| C3 | Patch mixing may be too close to failed mixup | Yun19 supports patch replacement, but label mixing repeats a path that failed locally | mixup `alpha=0.2` scored 0.912200 | prefer label-preserving cutout before implementing CutMix | duplicate/null filter |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Local Cutout, small mask | DeVries17; Zhong17 | add `--cutout_size`; run active stack with `--cutout_size 8` | regularize local image shortcuts while preserving labels | score <= 0.916100 near-miss or clear underfitting | low, client-local tensor masking only |
| P2 | Local Cutout, medium mask | DeVries17 | same code with `--cutout_size 12` | stronger augmentation if size 8 is too weak | score below P1 or severe underfitting | low |
| P3 | Random-erasing probability/area variant | Zhong17 | future `--random_erasing_prob` if Cutout is near-best | more diverse occlusion shapes | no gain over Cutout | low-medium, more knobs |
| P4 | CutMix | Yun19 | future `--cutmix_alpha` only if Cutout helps | preserve pixels while using patch-level mixing | repeats mixup underfit | medium, label mixing complexity |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | no existing `cutout_size` arg or row | different from mixup/label smoothing because labels stay hard | select |
| P2 | P1 variant | may underfit if masking too large | select paired severity check |
| P3 | P1 family | defer until Cutout shows signal | reserve |
| P4 | mixup-style label mixing | mixup already scored 0.912200 | reject for next batch |

## Proposal scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 Cutout size 8 | 3 | 5 | 5 | 4 | 5 | 1 | 29 |
| P2 Cutout size 12 | 3 | 5 | 5 | 4 | 4 | 1 | 28 |
| P3 Random erasing | 2 | 5 | 3 | 4 | 4 | 1 | 24 |
| P4 CutMix | 2 | 4 | 2 | 5 | 3 | 2 | 20 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_cutout8_w1` | active best stack plus `--cutout_size 8` |
| 2 | P2 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_cutout12_w1` | active best stack plus `--cutout_size 12` |
| 3 | reserve |  | no launch unless Cutout is near-best |
| 4 | reserve |  | keep empty at width 1 to avoid NVFlare contention |

## Reflective memory

- Keep: local occlusion augmentation is distinct from the failed target-mixing regularizers and does not require data or protocol edits.
- Discard: CutMix for now because label mixing overlaps with failed mixup and is more complex.
- Do not retry: standalone mixup, label smoothing, Nesterov, client/server clipping, and architecture variants on this active stack.
- Sources to carry forward: DeVries17 Cutout, Zhong17 Random Erasing, Yun19 CutMix.

## Batch outcome

- First-pass Cutout missed the active best: `cutout_size=8` scored 0.915600 and `cutout_size=12` scored 0.917500.
- Both rows were finalized as `discard`; the size-12 near miss justifies a narrow 10/14 bracket before keeping or reverting the Cutout code.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the Cutout literature reset.
- Follow-up Cutout missed as well: `cutout_size=10` scored 0.918400 and `cutout_size=14` scored 0.917000.
- Both follow-up rows were finalized as `discard`; the default-off Cutout knob was removed from `client.py`, `job.py`, and `mutation_schema.yaml`.
- Post-removal `PYTHON=.venv/bin/python make validate` and `make smoke` passed.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the Cutout literature reset.

---

# Literature loop 2026-05-09 local vicinal regularization

## Trigger

- Reason: watchdog `recommendation=literature` after 32 scored non-crash class-balanced FedZMG candidates since the row 309 high-water mark.
- Current best: 0.918600, FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `cosine_lr_eta_min_factor=0.00015`, `fedproxloss_mu=3e-5`, `zero_mean_gradients`, `class_balanced_loss_beta=0.90`, `aggregation_epochs=7`.
- Recent symptoms from `results.tsv`: scalar beta, FedProx, client LR/momentum, server LR/momentum, scheduler floor, local-compute, exact-step, and FedNova branches all regressed; the best recent near-miss is client `lr=0.04375` at 0.916100.
- Confirmed null/worse ideas to avoid: FedNova, SCAFFOLD, FedAdam, median aggregation, focal loss, LDAM, FedLC/FedRS/logit-prior variants, self-distillation, update clipping, SAM, current-stack FedProx brackets, local-compute endpoints, and scalar FedAvgM jitter.
- Candidate width: run width 1 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`, because width-2 NVFlare runs previously exposed communication failures.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| federated learning non-IID overfitting label smoothing mixup CIFAR-10 paper | plateau now looks like local overfitting/generalization rather than optimizer scaling | web search, arXiv | mixup and label smoothing are client-local and do not touch FL protocol. |
| mixup Beyond Empirical Risk Minimization arXiv 1710.09412 label smoothing calibration neural networks paper | find primary regularization papers with CIFAR/image evidence | arXiv, web search | Zhang18 mixup and Muller19 label smoothing are primary sources. |
| federated learning batch normalization non-IID FedBN paper arXiv | check whether normalization-locality is relevant after many FedAvgM scalar nulls | arXiv, web search | FedBN is relevant but would require architecture/protocol care outside this optimizer campaign. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zhang18 | mixup: Beyond Empirical Risk Minimization / 2018 | https://arxiv.org/abs/1710.09412 | neural nets memorize and overfit; mixup improves CIFAR/ImageNet generalization by training on convex combinations | vicinal data augmentation | keep |
| Yoon21 | FedMix: Approximation of Mixup under Mean Augmented Federated Learning / 2021 | https://arxiv.org/abs/2107.00233 | heterogeneity degrades FL as non-IID severity increases; mixup-style augmentation can help FL benchmarks | federated mixup | keep as support, reject mean-sharing mechanics |
| Muller19 | When Does Label Smoothing Help? / 2019 | https://arxiv.org/abs/1906.02629 | over-confident classifiers can generalize/calibrate poorly; soft targets improve generalization and calibration | target regularization | keep |
| Szegedy16 | Rethinking the Inception Architecture for Computer Vision / 2016 | https://arxiv.org/abs/1512.00567 | large image classifiers benefit from aggressive regularization including label smoothing | label smoothing in vision | keep as secondary |
| Li21 | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization / 2021 | https://arxiv.org/abs/2102.07623 | non-IID feature shift can make averaging BN parameters harmful | local normalization | reject for next batch: fixed `model_arch` and no protocol change |
| Zhang21 | Understanding Clipping for Federated Learning / 2021 | https://arxiv.org/abs/2106.13673 | update clipping can help under heterogeneity | clipping | reject: update clipping already nulled on this stack |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local overconfidence after class reweighting | Muller19 shows smoothed labels reduce overconfidence and can improve generalization/calibration | class-balanced beta helped once, then all beta and LR/momentum refinements regressed | add a small soft-target regularizer without changing class weights or server logic | `client.py`, `job.py` |
| C2 | Sparse local support from non-IID class skew | Zhang18 and Yoon21 support mixup-style interpolation as a low-overhead regularizer; FedMix targets non-IID FL but mean sharing is out of scope | site splits are label-skewed and local-compute endpoints overfit/regressed | local-only mixup can smooth per-client decision boundaries while preserving DIFF uploads | `client.py`, `job.py` |
| C3 | Feature/normalization shift remains plausible but too invasive | Li21 shows local BN can help feature-shift non-IID FL | optimizer/loss jitter plateau suggests a representational issue may remain | architecture/BN locality is not safe inside the fixed `moderate_cnn` optimizer campaign | reject or future architecture subcampaign |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Add small label smoothing | Muller19; Szegedy16 | add `--label_smoothing`; run active stack with `--label_smoothing 0.05` | reduce overconfidence from local class-balanced training | score <= recent 0.916100 near-miss or clear underfitting | low, local loss only |
| P2 | Add local mixup | Zhang18; Yoon21 | add `--mixup_alpha`; run active stack with `--mixup_alpha 0.2` | smooth local decision boundaries under class skew | score <= active stack by >0.002 or slower without gain | low, local batch transform only |
| P3 | Combine light mixup and light smoothing | Zhang18; Muller19 | run `--mixup_alpha 0.2 --label_smoothing 0.025` only if P1/P2 are near-best | regularize both inputs and targets without changing server | worse than both single mechanisms | low-medium, two regularizers may underfit |
| P4 | Local BN / registered normalized architecture | Li21 | future architecture subcampaign using registered `moderate_cnn_norm` only if human labels new budget | mitigate feature-shift style drift | not comparable in current optimizer ledger | rejected now: fixed `model_arch` budget |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | no existing `label_smoothing` arg or row | not the same as class-balanced beta; source targets overconfidence | select |
| P2 | no existing `mixup_alpha` arg or row | local augmentation differs from loss reweighting and FedMix mean sharing | select |
| P3 | P1/P2 combination | could duplicate failure if both singles underfit | reserve |
| P4 | architecture/normalization branch | violates fixed optimizer-campaign `model_arch` comparability | reject |

## Proposal scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 label smoothing 0.05 | 3 | 5 | 5 | 4 | 5 | 1 | 29 |
| P2 mixup alpha 0.2 | 4 | 5 | 4 | 5 | 5 | 2 | 30 |
| P3 mixup plus smoothing | 3 | 5 | 3 | 4 | 4 | 2 | 25 |
| P4 local BN architecture | 3 | 2 | 2 | 5 | 4 | 2 | 19 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P2 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_mixup02_w1` | active best stack plus `--mixup_alpha 0.2` |
| 2 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_ls005_w1` | active best stack plus `--label_smoothing 0.05` |
| 3 | P3 reserve | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_mixup02_ls0025_w1` | run only if one single-mechanism result is near-best |
| 4 | reserve |  | keep empty at width 1 to avoid NVFlare contention |

## Reflective memory

- Keep: source-backed client regularization is the next branch because optimizer/loss scalar jitter has plateaued.
- Discard: FedMix mean-sharing and FedBN/local-BN mechanics for this campaign because they would alter data/protocol or architecture comparability.
- Do not retry: scalar class-balanced beta, FedProx, scheduler, client/server LR/momentum, and local-compute jitter unless a new mechanism first changes the failure mode.
- Sources to carry forward: Zhang18 mixup, Yoon21 FedMix, Muller19 label smoothing, Szegedy16 label smoothing, Li21 FedBN.

## Batch outcome

- Local mixup `alpha=0.2` scored 0.912200 and label smoothing `0.05` scored 0.912600.
- Both were discarded, and the default-off `mixup_alpha` / `label_smoothing` code was removed after review because neither source-backed regularizer approached the 0.918600 high-water mark.
- Do not retry standalone mixup or label smoothing on this class-balanced FedZMG stack without a different mechanism or architecture subcampaign.

---

# Literature loop 2026-05-09 FedNova normalized aggregation

## Trigger

- Reason: watchdog `recommendation=literature` after 32 scored non-crash candidates since the class-balanced beta `0.90` improvement at row 309.
- Current best: 0.918600, FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `cosine_lr_eta_min_factor=0.00015`, `fedproxloss_mu=3e-5`, `zero_mean_gradients`, `class_balanced_loss_beta=0.90`, `aggregation_epochs=7`.
- Recent symptoms from `results.tsv`: beta, FedProx, LR, momentum, weight decay, scheduler floor, local-compute, FedAvg, median, SCAFFOLD, and upper server-LR sweeps all regressed; `server_lr=2.6` crashed before a comparable score.
- Confirmed null/worse ideas to avoid: current-stack FedProx brackets, SCAFFOLD, FedAdam, median aggregation, focal loss, LDAM, FedLC/FedRS/logit-prior variants, self-distillation, update clipping, SAM, and scalar beta/LR/momentum jitter.
- Candidate width: run width 1 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`, because recent width-2 NVFlare runs exposed communication failures.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| federated learning local steps objective inconsistency FedNova non IID | recent epoch and exact-step sweeps show local-compute sensitivity and split sizes differ strongly by site | web search, arXiv, NeurIPS | Wang20 FedNova directly targets objective inconsistency from heterogeneous local updates. |
| federated learning non-IID client drift control variates SCAFFOLD | compare FedNova against already-implemented drift correction family and avoid repeat nulls | web search, arXiv | SCAFFOLD is relevant but already scored 0.906600 on the current class-balanced stack. |
| momentum benefits non-IID federated learning FedAvg SCAFFOLD | current best depends on server momentum; need source support for a momentum variant if pure FedNova under-scales | web search, arXiv, ICLR/OpenReview | Cheng24 supports momentum as a heterogeneity aid, but candidate should stay a small FedNova variant. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Wang20 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | variable local steps and heterogeneous data bias naive averaging | FedNova normalized averaging | keep |
| Cheng24 | Momentum Benefits Non-IID Federated Learning Simply and Provably / 2024 | https://arxiv.org/abs/2306.16504 | FedAvg/SCAFFOLD convergence under data heterogeneity | momentum variants | keep as FedNova momentum variant support |
| Karimireddy20 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning / 2020 | https://arxiv.org/abs/1910.06378 | client drift under heterogeneous data | control variates | reject for next batch: current-stack SCAFFOLD scored 0.906600 |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | FedAvg tuning/convergence under heterogeneity | FedOpt/FedAdam | reject: FedAdam was repeatedly poor or crashed |
| Zantalis26 | FedZMG: Efficient Client-Side Optimization in Federated Learning / 2026 | https://arxiv.org/abs/2602.18384 | client drift from biased local gradients | zero-mean gradients | keep only as active-stack context; already in best |
| Cui19 | Class-Balanced Loss Based on Effective Number of Samples / 2019 | https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html | class imbalance | effective-number local loss | keep only as active-stack context; beta refinements now null |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Objective inconsistency from unequal local work | Wang20 argues naive averaging can converge to a mismatched objective when local update counts vary and proposes normalized averaging | site split sizes range from 2013 to 8302 samples; local-compute sweeps and exact-step variants regressed | current weighted aggregation uses `NUM_STEPS_CURRENT_ROUND` as weight but does not normalize each client DIFF by its local trajectory length | `custom_aggregators.py`, `job.py` |
| C2 | Momentum helps heterogeneity but scalar momentum jitter is exhausted | Cheng24 supports momentum as a simple non-IID convergence aid | current high-water depends on FedAvgM, but scalar server/client momentum brackets all missed | add momentum only as a FedNova variant, not another FedAvgM jitter run | `custom_aggregators.py`, CLI |
| C3 | Drift-control/adaptive-server repeats are confirmed nulls | Karimireddy20 and Reddi21 motivate SCAFFOLD/FedOpt, but local evidence matters | current-stack SCAFFOLD 0.906600, median 0.884800, FedAdam poor/crashy | reject these despite paper relevance; choose a distinct normalized-aggregation mechanism | ledger filter, `templates/literature_loop.md` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | FedNova normalized DIFF aggregation | Wang20 FedNova | add `--aggregator fednova`; run active stack with `--server_lr 1.0 --server_momentum 0.0` | reduce bias from unequal client local steps while preserving all DIFF keys | score <= 0.918600 or severe under-scaling | low, server-local aggregation over existing DIFFs and `NUM_STEPS_CURRENT_ROUND` |
| P2 | FedNova with server LR rescale | Wang20 FedNova | same as P1 but `--server_lr 1.8 --server_momentum 0.0` | compensate if normalized updates are too small relative to active FedAvgM stack | score <= pure FedNova or instability | low, same aggregator with scalar multiplier |
| P3 | FedNova with momentum | Wang20 FedNova; Cheng24 momentum | `--aggregator fednova --server_lr 1.0 --server_momentum 0.475` | combine normalized local-work correction with the momentum property that helped this campaign | score <= P1/P2 or oscillation | low-medium, persistent server velocity but no protocol change |
| P4 | Revisit SCAFFOLD/FedAdam | Karimireddy20; Reddi21 | CLI-only aggregation swap | none expected after nulls | any score below current best confirms reject | rejected before scoring; duplicate null conflict |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | no exact duplicate; `fednova` not previously available | differs from weighted/FedAvg/FedAvgM because it normalizes each DIFF by local steps | select |
| P2 | P1 variant | scalar server-LR jitter was null for FedAvgM, but this rescales a new normalized aggregator | select after P1 if P1 under-scales or as paired variant |
| P3 | P1 variant plus momentum | scalar momentum jitter was null, but momentum is applied after step normalization | reserve unless P1/P2 near best |
| P4 | current-stack SCAFFOLD/FedAdam/median rows | SCAFFOLD 0.906600, median 0.884800, FedAdam poor/crashy | reject |

## Proposal scoring

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 pure FedNova | 4 | 5 | 4 | 5 | 5 | 1 | 31 |
| P2 FedNova lr rescale | 3 | 5 | 4 | 4 | 4 | 1 | 27 |
| P3 FedNova momentum | 3 | 4 | 3 | 4 | 4 | 1 | 24 |
| P4 SCAFFOLD/FedAdam repeat | 1 | 4 | 5 | 5 | 1 | 1 | 20 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fednova_lr10_m0_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_w1` | active best client stack plus `--aggregator fednova --server_lr 1.0 --server_momentum 0.0` |
| 2 | P2 | `fednova_lr18_m0_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_w1` | active best client stack plus `--aggregator fednova --server_lr 1.8 --server_momentum 0.0` |
| 3 | P3 reserve | `fednova_lr10_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_w1` | run only if P1/P2 are close enough to justify momentum |
| 4 | reserve |  | keep empty at width 1 to avoid NVFlare contention |

## Reflective memory

- Keep: FedNova is the next source-backed branch because it targets unequal client local-work normalization, a different failure mode from loss reweighting and FedAvgM scalar tuning.
- Discard: paper-relevant but locally nulled SCAFFOLD/FedAdam/median repeats.
- Do not retry: class-balanced beta, LDAM, focal, FedProx, local-compute, scheduler floor, and scalar FedAvgM jitter without a new implementation-level mechanism.
- Sources to carry forward: Wang20 FedNova, Cheng24 momentum, Zantalis26 FedZMG, Cui19 class-balanced loss.

## Batch outcome

- Pure FedNova scored 0.900900; server-LR-rescaled FedNova scored 0.899000.
- Both were discarded, and the reserve momentum variant was not run because normalized aggregation was far below the high-water mark.
- FedNova code was removed after review; keep Wang20 as a null result for this stack unless a future branch changes the implementation substantially.

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

---

# Literature loop 2026-05-07 plateau after row 141

## Trigger

- Reason: watchdog recommendation=literature after 32 scored non-crash candidates since the material improvement to 0.903400 at row 109.
- Current best: 0.903400, FedAvgM `server_lr=1.8`, `server_momentum=0.45`, client `lr=0.04`, `momentum=0.925`, `weight_decay=5e-4`, `aggregation_epochs=7`, `cosine_lr_eta_min_factor=0.005`.
- Recent symptoms from `results.tsv`: scheduler floor/off, server/client momentum, client LR, weight decay, FedProx, exact local steps, FedAvg, and robust median sweeps all regressed.
- Confirmed null/worse ideas to avoid: no scheduler, eta floor 0.0001/0.001/0.0025/0.0075/0.01, server momentum 0.4375/0.445/0.455/0.4625, FedProx 1e-6/3e-6/3e-5/1e-4, FedAdam conservative/aggressive, tuned SCAFFOLD, robust median.
- Candidate width: 2 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0` after earlier width-4 contention.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| federated learning update clipping non-IID convergence client update clipping | update norms vary on the best run; soft clipping may suppress outlier client DIFFs without median's harsh coordinate-wise aggregation | web search, arXiv | Zhang21 directly studies clipped client updates in FL. |
| adaptive self-distillation minimizing client drift heterogeneous federated learning | current plateau looks like client drift/local overfitting after many optimizer retunes | web search, arXiv/OpenReview | ASD uses the global model as a teacher to constrain client drift. |
| federated sharpness-aware minimization non-IID client drift | generalization/flatness is an alternative to drift regularization | web search, arXiv | FedSAM is relevant but doubles local backward cost and needs more code. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zhang21 | Understanding Clipping for Federated Learning / 2021 | https://arxiv.org/abs/2106.13673 | client updates can have large or uneven norms under heterogeneity | client update clipping | keep |
| Yashwanth24 | Adaptive Self-Distillation for Minimizing Client Drift in Heterogeneous Federated Learning / 2024 | https://arxiv.org/abs/2305.19600 | local client objectives drift from global behavior | global-model self-distillation | keep |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | non-IID clients overfit sharper local minima | sharpness-aware local training | reserve: higher runtime and code complexity |
| Gao22 | FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction / 2022 | https://arxiv.org/abs/2203.11751 | local drift accumulates across rounds | drift correction state | reject for now: larger stateful client objective change |
| Li20 | Federated Optimization in Heterogeneous Networks / 2020 | https://arxiv.org/abs/1812.06127 | statistical heterogeneity destabilizes local training | FedProx proximal loss | reject: this campaign already bracketed current-stack FedProx nulls |
| Karimireddy20 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning / 2020 | https://arxiv.org/abs/1910.06378 | client drift from heterogeneous data | control variates | reject: tuned SCAFFOLD was clearly below best |
| Reddi21 | Adaptive Federated Optimization / 2021 | https://arxiv.org/abs/2003.00295 | server optimizer sensitivity in heterogeneous FL | FedOpt/FedAdam | reject: FedAvgM is best; FedAdam branch failed badly |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Uneven client update norms | Zhang21 motivates clipping client updates before aggregation | best-run diff-norm telemetry spans p50=28.99 to max=62.54; median aggregation failed but softer norm control is untested | apply L2 clipping to each client DIFF before FedAvgM momentum without changing keys or params type | `custom_aggregators.py`, `job.py` |
| C2 | Client drift from local objectives | Yashwanth24 targets heterogeneous FL drift using self-distillation | FedProx and SCAFFOLD did not help, but output-level global anchoring is untested | use the received global model as a frozen teacher during local training | `client.py`, `job.py` |
| C3 | Local overfitting and sharp minima | FedSAM-style work targets flatter local solutions under non-IID data | 8 epochs and exact local steps regressed, suggesting local compute can overfit or destabilize | reserve for later because SAM doubles local training work near the timeout cap | `client.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | L2-clip each client DIFF before FedAvgM aggregation | Zhang21 | add `--update_clip_norm`; run best stack with `--update_clip_norm 45` | damp top-quartile client update spikes while preserving weighted FedAvgM direction | score <= 0.903400 or clipped run behaves like underfit median/FedAvg | low, server-local DIFF preprocessing |
| P2 | Client-side global-model self-distillation | Yashwanth24 | add `--global_distill_alpha 0.05 --global_distill_temperature 2.0` to best stack | constrain local drift without proximal weight-space penalty | score <= 0.903400 or runtime exceeds cap | low, client-local loss using existing received model |
| P3 | FedSAM local sharpness-aware update | Qu22 | add SAM two-step SGD around best stack | improve generalization under non-IID | runtime doubles and score does not exceed best | medium, code complexity and runtime |
| P4 | FedDC-style drift correction | Gao22 | add persistent client drift correction state | correct repeated client drift across rounds | requires broader stateful objective tuning | medium-high, larger algorithm change |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | not median; clipping is global-norm soft suppression | robust median scored 0.764500, but no soft DIFF clipping tested | select |
| P2 | not FedProx; it regularizes logits against the global teacher | FedProx/SCAFFOLD nulls make this lower confidence but still distinct | select |
| P3 | no direct duplicate | 8 local epochs and exact-step variants regressed; runtime risk high | reserve |
| P4 | no direct duplicate | larger code/state change than needed for first post-plateau batch | reject for this batch |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 update clipping | 4 | 5 | 4 | 4 | 4 | 1 | 29 |
| P2 global self-distillation | 3 | 5 | 3 | 4 | 5 | 2 | 26 |
| P3 FedSAM | 3 | 4 | 2 | 4 | 5 | 4 | 21 |
| P4 FedDC-style drift correction | 3 | 3 | 1 | 4 | 5 | 3 | 19 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m045_epochs7_clientlr004_cm0925_wd5e4_eta0005_clip45` | best stack plus `--update_clip_norm 45` |
| 2 | P2 | `fedavgm_lr18_m045_epochs7_clientlr004_cm0925_wd5e4_eta0005_gdistill005_t2` | best stack plus `--global_distill_alpha 0.05 --global_distill_temperature 2.0` |

## Reflective memory

- Keep: update clipping and global self-distillation are the first source-backed code changes after the current-stack CLI plateau.
- Discard: more FedProx/SCAFFOLD/FedAdam retries without new implementation evidence.
- Do not retry: median aggregation as a robust outlier fix under this budget.
- Sources to carry forward: Zhang21 update clipping, Yashwanth24 adaptive self-distillation, Qu22/FedSAM reserve.

### Batch outcome

- Update clipping with norm 45 tied the best at 0.903400 but added aggregation complexity, so it was discarded rather than kept.
- Global self-distillation with alpha 0.05 and temperature 2.0 scored 0.901000, below the current best.
- The default-off code knobs were reverted after review; carry the paper hypotheses forward only as null results unless a stronger threshold/adaptive implementation is justified by new evidence.
- Follow-up FedProx under the lower scheduler floor found a new best: `--fedproxloss_mu 3e-5 --cosine_lr_eta_min_factor 0.0001` scored 0.904100, while `mu=1e-6` scored 0.900200.
- Server momentum refinement on that stack improved again: `--server_momentum 0.475` scored 0.906100 and becomes the active best; `0.425` scored 0.904900 but is a non-survivor below the new best.

---

# Literature Loop 2026-05-08 Label-Skew Plateau

## Trigger

- Reason: watchdog `recommendation=literature` after 32 scored candidates without material improvement after the FedAvgM/FedProx lower-floor stack.
- Current best: active kept stack is 0.906100 with FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.04`, `momentum=0.925`, `weight_decay=5e-4`, `aggregation_epochs=7`, `cosine_lr_eta_min_factor=0.0001`, `fedproxloss_mu=3e-5`. A costlier exact-step row scored 0.906400 but was discarded as below the material threshold.
- Recent symptoms from `results.tsv`: server momentum/lr micro sweeps, FedProx mu brackets, weight decay, client lr, exact local steps, scheduler floor/off, FedAdam, SCAFFOLD, median, weighted/default FedAvg all missed.
- Confirmed null/worse ideas to avoid: current-stack FedAdam, SCAFFOLD, median/default/weighted FedAvg, scheduler floor/off variants, FedProx mu 2e-5/5e-5, exact steps 640/720/832 and repeat 768 without a cost justification.
- Candidate width: 2 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0` due prior contention at wider batches.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| FedLC federated learning label distribution skew logits calibration arXiv 2022 | current Dirichlet split and FedProx/SCAFFOLD misses suggest local classifier bias under label skew | arXiv, ResearchGate mirror, dblp | FedLC is client-local and can compose with FedProx/FedAvgM without protocol changes. |
| FedRS restricted softmax label distribution non IID federated learning KDD 2021 | missing or rare local classes can corrupt classifier head updates | ACM/KDD listing, author PDF, dblp | FedRS is client-local and targets missing classes. |
| model contrastive / dynamic regularization / sharpness-aware federated learning non-IID CIFAR10 | broader client-drift/generalization alternatives after scalar retunes stalled | CVF, arXiv, NeurIPS, ResearchGate mirror | MOON/FedDyn/FedSAM are relevant but higher-risk or higher-cost here. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zhang22 FedLC | Federated Learning with Label Distribution Skew via Logits Calibration / 2022 | https://arxiv.org/abs/2209.00189 | majority/minority/missing local labels bias standard CE and worsen drift | client logit calibration | keep |
| Li21 FedRS | FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data / 2021 | https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf | missing local classes receive only indirect pushing in softmax classifier updates | restricted softmax | keep |
| Muller19 LS | When Does Label Smoothing Help? / 2019 | https://papers.neurips.cc/paper/8717-when-does-label-smoothing-help | overconfident local CE can overfit skewed clients | label smoothing | keep as simple reserve |
| Acar21 FedDyn | Federated Learning Based on Dynamic Regularization / 2021 | https://arxiv.org/abs/2111.04263 | local optima inconsistent with global empirical loss | dynamic regularization | reject for now: stateful broad objective change |
| Li21 MOON | Model-Contrastive Federated Learning / 2021 | https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html | non-IID image FL needs representation-level correction | contrastive model loss | reject for now: requires feature path/previous model state |
| Wang20 FedNova | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://papers.nips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html | objective inconsistency from variable local updates | normalized averaging | reject: all clients use fixed equal local compute in this budget |
| Qu22 FedSAM | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | local ERM finds sharper/non-generalizing minima under shift | SAM local optimizer | reserve only: likely doubles runtime |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Classifier-head bias from label skew | FedLC reports standard CE is unsuitable when labels are majority/minority/missing and calibrates logits by local class occurrence | FedProx/SCAFFOLD and optimizer retunes plateau near 0.906, suggesting drift control alone is insufficient | CIFAR-10 Dirichlet alpha 0.5 creates label-prior skew per client | `client.py`, `job.py` default-off loss arg |
| C2 | Missing-class proxy corruption | FedRS shows missing-class proxies receive only pushing forces and restricts their local updates with alpha in [0,1] | median/default/weighted/simple FedAvg are much worse, implying local update quality matters before aggregation | local client subsets may have absent or near-absent CIFAR classes | `client.py`, `job.py` default-off loss arg |
| C3 | Overconfident local CE | label smoothing literature shows soft targets can improve generalization/calibration | scheduler-off overfits badly and high local compute has narrow wins, so local CE regularization is plausible | cheap client-only regularizer with no protocol surface | `client.py`, `job.py` default-off arg |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | FedLC-style logit calibration using local class counts | Zhang22 FedLC | add `--fedlc_tau`; first candidate `--fedlc_tau 0.5` on active FedAvgM/FedProx stack | reduce biased local updates for minority/missing classes | score <= 0.906100 or instability | low: client-local loss transform |
| P2 | FedRS restricted softmax for missing local classes | Li21 FedRS | add `--fedrs_alpha`; first candidate `--fedrs_alpha 0.5` on active stack | reduce missing-class classifier drift | score <= 0.906100 or no classes missing so no effect | low: client-local logits scaling |
| P3 | Label smoothing | Muller19 LS | add `--label_smoothing`; reserve candidate `--label_smoothing 0.02` | reduce overconfident local CE under skew | score <= active best | low: built-in CE smoothing |
| P4 | FedSAM local optimizer | Qu22 FedSAM | future `--sam_rho` default-off if runtime budget allows | flatter local minima/generalization | timeout or no material gain | medium: double backward cost |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | not duplicate of FedProx or prior self-distillation | no FedLC/FedRS rows exist | select |
| P2 | not duplicate of median/default/weighted aggregation | no restricted-softmax row exists | select |
| P3 | not duplicate | lower priority than FL-specific label-skew losses | reserve |
| P4 | FedSAM already source-listed but not implemented | runtime likely exceeds safe cost at current 12m baseline | reserve |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 FedLC | 4 | 5 | 4 | 5 | 5 | 1 | 31 |
| P2 FedRS | 3 | 5 | 4 | 4 | 5 | 1 | 28 |
| P3 label smoothing | 2 | 5 | 5 | 4 | 4 | 1 | 26 |
| P4 FedSAM | 3 | 4 | 2 | 4 | 4 | 4 | 20 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr004_cm0925_wd5e4_eta00001_mu3e5_fedlc05` | active stack plus `--fedlc_tau 0.5` |
| 2 | P2 | `fedavgm_lr18_m0475_epochs7_clientlr004_cm0925_wd5e4_eta00001_mu3e5_fedrs05` | active stack plus `--fedrs_alpha 0.5` |

## Reflective memory

- Keep: FedLC/FedRS are the first classifier-calibration branch for this campaign; judge against active kept 0.906100 and material threshold 0.0005.
- Discard: more scalar FedAvgM/FedProx/scheduler jitter until label-skew branch is tested.
- Do not retry: prior FedAdam/SCAFFOLD/median/default/weighted FedAvg, scheduler floor/off, and exact-step variants unless a new source-backed implementation reason appears.
- Sources to carry forward: Zhang22 FedLC, Li21 FedRS, Muller19 label smoothing, Qu22 FedSAM as runtime-expensive reserve.

### Batch outcome

- FedLC `--fedlc_tau 0.5` scored 0.899200, below the active kept 0.906100.
- FedRS `--fedrs_alpha 0.5` scored 0.901200, below the active kept 0.906100.
- Label smoothing reserve scored 0.903400 at `0.02` and 0.904400 at `0.05`, still below the active kept stack.
- The default-off client loss knobs were removed after review because the whole classifier-calibration branch missed and would add unsupported surface area.
- Treat FedLC/FedRS/label-smoothing as null results for this budget unless a new source-backed implementation variant is materially different.

---

# Literature Loop 2026-05-08 Step-Normalization Plateau

## Trigger

- Reason: watchdog `recommendation=literature` after 32 scored candidates since the label-skew literature reset; scheduler, FedProx, server momentum, client momentum, weight decay, and exact-step brackets all missed.
- Current best: active kept stack is 0.906100 with FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.04`, `momentum=0.925`, `weight_decay=5e-4`, `aggregation_epochs=7`, `cosine_lr_eta_min_factor=0.0001`, `fedproxloss_mu=3e-5`. The ledger high-water exact-step row is 0.906400 but was discarded as not material and costlier.
- Recent symptoms from `results.tsv`: `eta_min_factor=0.000125` reached 0.905900 but tighter scheduler, FedProx, server-momentum, and exact-step follow-ups regressed; label-skew calibration and architecture variants also missed.
- Confirmed null/worse ideas to avoid: FedLC/FedRS/label smoothing, FedAdam, SCAFFOLD, median/default/weighted FedAvg, current-stack scheduler floor/off variants, FedProx micro-brackets, exact-step repeats near 768 without a new mechanism.
- Candidate width: 2 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| FedZMG efficient client-side optimization federated learning zero mean gradients non-IID | look for a cheap client-side drift correction after local loss/optimizer jitters stalled | arXiv, web search | FedZMG is parameter-free and maps to a local gradient projection. |
| FedNova objective inconsistency heterogeneous federated optimization variable local steps | exact-step and epoch modes differ; Dirichlet splits can vary local step counts | arXiv, Princeton page, web search | FedNova-style normalization can be expressed inside DIFF aggregation. |
| FedSAM sharpness aware minimization federated learning non-IID CIFAR runtime | reserve a stronger flatness idea if cheap mechanisms fail | arXiv, ResearchGate mirror | Good evidence, but double-backward cost is risky near the timeout cap. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Zantalis26 FedZMG | FedZMG: Efficient Client-Side Optimization in Federated Learning / 2026 | https://arxiv.org/abs/2602.18384 | client drift and gradient bias under non-IID data | zero-mean gradient projection | keep |
| Wang20 FedNova | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / 2020 | https://arxiv.org/abs/2007.07481 | objective inconsistency from variable local update counts | normalized averaging | keep |
| Qu22 FedSAM | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | local ERM can converge to sharp, less generalizable minima under distribution shift | SAM local optimizer | reserve: likely doubles local training cost |
| Acar21 FedDyn | Federated Learning Based on Dynamic Regularization / 2021 | https://arxiv.org/abs/2111.04263 | local optima inconsistent with global empirical loss | dynamic regularization | reject: broader stateful objective change |
| Krouka25 DRDM | Distributionally Robust Federated Learning with Client Drift Minimization / 2025 | https://arxiv.org/abs/2505.15371 | worst-client performance and drift | DRO plus dynamic regularization | reject: changes objective and fairness target beyond current single-score budget |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Biased local gradients | FedZMG projects gradients to a zero-mean space to reduce client-drift variance without extra communication | FedProx, FedRS/FedLC, and scalar optimizer retunes all missed, suggesting a different local gradient regularizer is needed | client-only projection before `optimizer.step()` preserves FLModel fields | `client.py`, `job.py` |
| C2 | Objective inconsistency from varying local steps | FedNova targets bias from clients performing different numbers of local updates | epoch-based training uses `NUM_STEPS_CURRENT_ROUND` from local batch counts; exact-step variants were sensitive and costlier | normalize each DIFF by local steps before server momentum | `custom_aggregators.py`, `job.py` |
| C3 | Flatness/generalization under non-IID | FedSAM reports local SAM improves global generalization under non-IID distribution shift | high local compute and scheduler changes have narrow, noisy gains | reserve for lower-width or reduced-local-compute trial if cheap ideas fail | `client.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | FedZMG-style zero-mean gradients | Zantalis26 FedZMG | add `--zero_mean_gradients`; run active FedAvgM/FedProx stack | reduce client-specific gradient bias with negligible runtime cost | score <= 0.906100 or instability | low: client-local gradient transform |
| P2 | FedNova-style step-normalized FedAvgM | Wang20 FedNova; Reddi21 FedOpt | add `--aggregator fednovam`; run active client stack with `server_lr=1.8`, `server_momentum=0.475` | reduce local-step objective bias while retaining successful server momentum | score <= 0.906100 or behaves like weaker FedAvg | low-medium: new server-local DIFF normalization |
| P3 | FedSAM local optimizer | Qu22 FedSAM | future `--sam_rho` default-off | flatter local minima and better non-IID generalization | timeout or no material gain | medium: two backward passes |
| P4 | FedDyn/DRDM dynamic regularization | Acar21; Krouka25 | not selected | align local/global objectives | requires broader stateful objective and tuning | medium-high |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | not FedProx/FedLC/self-distillation; operates directly on gradients | no zero-mean gradient rows exist | select |
| P2 | not weighted FedAvg or median; normalizes DIFFs by local step count before momentum | pure FedNova was previously rejected for equal-step concern, but this campaign has variable epoch batch counts and exact-step sensitivity | select |
| P3 | FedSAM was reserved before | runtime risk high at active 7 epochs | reserve |
| P4 | no direct duplicate | too broad for first post-plateau batch | reject |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 FedZMG zero-mean gradients | 3 | 5 | 5 | 3 | 5 | 1 | 28 |
| P2 FedNovaM step-normalized momentum | 3 | 4 | 3 | 5 | 5 | 1 | 26 |
| P3 FedSAM | 3 | 4 | 2 | 5 | 4 | 4 | 21 |
| P4 dynamic regularization | 3 | 3 | 1 | 4 | 4 | 3 | 18 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr004_cm0925_wd5e4_eta00001_mu3e5_zmg` | active stack plus `--zero_mean_gradients` |
| 2 | P2 | `fednovam_lr18_m0475_epochs7_clientlr004_cm0925_wd5e4_eta00001_mu3e5` | active client stack plus `--aggregator fednovam --server_lr 1.8 --server_momentum 0.475` |

## Reflective memory

- Keep: zero-mean gradients and step-normalized momentum are the cheapest source-backed mechanisms not already falsified by this ledger.
- Discard: more scalar scheduler/FedProx/server-momentum jitter until a new mechanism creates a better stack.
- Do not retry: FedLC/FedRS/label smoothing, FedAdam, SCAFFOLD, median/default/weighted FedAvg under this budget.
- Sources to carry forward: Zantalis26 FedZMG, Wang20 FedNova, Qu22 FedSAM reserve.

### Batch outcome

- FedZMG zero-mean gradients scored 0.913700 and is the new kept stack.
- FedNova-style step-normalized FedAvgM scored 0.899100, far below the kept stack.
- The FedNovaM default-off aggregator was removed after review; carry Wang20 FedNova as a null result for this budget unless local-step heterogeneity becomes explicit.
- Keep the `--zero_mean_gradients` client knob and continue sweeps around the FedZMG/FedAvgM/FedProx stack.

---

# Literature Loop 2026-05-08 FedZMG Generalization Plateau

## Trigger

- Reason: watchdog `recommendation=literature` after 32 scored candidates since the FedZMG material improvement at row 235.
- Current kept stack: 0.916700 with FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `aggregation_epochs=7`, `cosine_lr_eta_min_factor=0.0001`, `fedproxloss_mu=3e-5`, and `--zero_mean_gradients`.
- Raw high-water: `eta_min_factor=0.00015` scored 0.916900, but was below the material keep threshold and subsequent scheduler brackets regressed.
- Recent symptoms from `results.tsv`: FedProx, scheduler floor, client/server LR, client/server momentum, weight decay, local epochs, and exact-step variants all missed around the FedZMG stack.
- Confirmed null/worse ideas to avoid: FedNovaM, FedLC/FedRS/label smoothing, prior update clipping and global self-distillation, FedAdam, SCAFFOLD, median/default/weighted FedAvg, more scalar jitter near already tested values.
- Candidate width: 2 on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`.
- Ledger event: timer started with `scripts/log_literature_review.py --start`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| federated mixup label skew non-IID CIFAR10 client augmentation | local label-skew regularization after classifier calibration failed | web search, arXiv | Mixup-style augmentation has FL label-skew evidence and can be client-local. |
| federated sharpness aware minimization non-IID FedSAM runtime CIFAR | current high-epoch local ERM may overfit sharp local minima | web search, arXiv | FedSAM is repeatedly reserved; reducing epochs can keep runtime inside cap. |
| federated stochastic weight averaging highly heterogeneous FedSWA FedSAM generalization | search for newer flat-minima alternatives to expensive SAM | web search, arXiv | FedSWA suggests flat-minimum averaging, but server-state design is broader than one batch. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Yoon21 FedMix | FedMix: Approximation of Mixup under Mean Augmented Federated Learning / 2021 | https://arxiv.org/abs/2107.00233 | non-IID degradation from heterogeneous local data | mixup-style FL augmentation | keep as source support; do not exchange averaged data |
| Sang24 MixNoise | Balancing Label Imbalance in Federated Environments Using Only Mixup and Artificially-Labeled Noise / 2024 | https://arxiv.org/abs/2409.13235 | label-skewed client distributions in CIFAR-10 | mixup plus pseudo-image balancing | keep for local-mixup rationale; reject noise generator |
| Zhang17 Mixup | mixup: Beyond Empirical Risk Minimization / 2017 | https://arxiv.org/abs/1710.09412 | memorization and brittle decision boundaries | convex input/label interpolation | keep as implementation basis |
| Qu22 FedSAM | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://arxiv.org/abs/2206.02618 | ERM local training can find sharp, non-generalizing minima under distribution shift | local SAM optimizer | keep with reduced epochs for runtime |
| Liu25 FedSWA | FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging / 2025 | https://arxiv.org/abs/2507.20016 | FedSAM can struggle under high heterogeneity; flatter minima help | stochastic weight averaging | reserve; server/global averaging state needs more design |
| Lewy22 StatMix | StatMix: Data augmentation method that relies on image statistics in federated learning / 2022 | https://arxiv.org/abs/2207.04103 | FL image augmentation can improve CIFAR accuracy | statistic-based augmentation | reject: statistic exchange/data transform beyond current safe surface |
| Bao24 BOBA | BOBA: Byzantine-Robust Federated Learning with Label Skewness / 2024 | https://proceedings.mlr.press/v238/bao24a.html | robust aggregation under label skew has selection bias | robust two-stage aggregation | reject: Byzantine target and stronger protocol assumptions |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local label-skew overfitting | FedMix and Sang24 both target heterogeneous/label-skewed FL with mixup-style augmentation; Zhang17 gives the local interpolation loss | FedLC/FedRS/label smoothing failed, but no vicinal image interpolation has been tested | client-local mixup can smooth local decision boundaries without sharing data or changing FLModel fields | `client.py`, `job.py` |
| C2 | Sharp local minima after high local compute | Qu22 argues ERM local optimizers in non-IID FL can push the global model toward sharp valleys; Liu25 also frames flatness as a high-heterogeneity issue | seven-epoch FedZMG is best, but further local compute and scheduler retunes regress, suggesting a generalization ceiling | SAM can regularize local updates; using four epochs offsets the two-backward cost | `client.py`, `job.py` |
| C3 | Server-side flat model averaging | Liu25 proposes SWA-style global averaging to improve heterogeneous FL generalization | many scalar FedAvgM retunes miss after the FedZMG jump | potential future server-local state, but needs careful compatibility with final global-model scoring | `custom_aggregators.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Client-local mixup | Yoon21 FedMix; Sang24 MixNoise; Zhang17 Mixup | add default-off `--mixup_alpha`; run active stack with `--mixup_alpha 0.2` | reduce label-skew overfitting and smooth local classifier boundaries with little runtime cost | score <= 0.916700 or underfitting relative to active stack | low: client-local batch transform and mixed CE loss |
| P2 | Reduced-epoch FedSAM | Qu22 FedSAM | add default-off `--sam_rho`; run active stack with `--aggregation_epochs 4 --sam_rho 0.03` | trade some local epochs for flatness-aware updates while staying under timeout | timeout or score below active stack | medium: two backward passes but no protocol change |
| P3 | FedSWA-style global averaging | Liu25 FedSWA | future custom aggregator maintaining SWA of global states late in training | flatter final global model after FedZMG plateau | evaluation uses wrong final checkpoint or score does not improve | medium: server state and final-model semantics require care |
| P4 | StatMix/noise balancing | Lewy22 StatMix; Sang24 MixNoise | not selected | add label-balanced pseudo-images/statistics | requires data-statistic exchange, generated noise, or data-pipeline changes | high for this harness |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | not FedLC/FedRS/label smoothing; changes inputs and soft targets, not logits only | no mixup rows exist | select |
| P2 | FedSAM has been reserved but never run; reduced epochs addresses timeout risk | no SAM rows exist | select |
| P3 | no direct duplicate | bigger server-state change; defer until client-local batch is scored | reserve |
| P4 | data augmentation family overlaps P1 | requires unsafe data/statistic generation surface | reject |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 local mixup | 3 | 5 | 4 | 4 | 5 | 1 | 28 |
| P2 reduced-epoch FedSAM | 3 | 4 | 3 | 5 | 5 | 3 | 24 |
| P3 FedSWA-style averaging | 3 | 3 | 2 | 4 | 5 | 1 | 22 |
| P4 StatMix/noise balancing | 2 | 2 | 1 | 3 | 4 | 2 | 14 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta00001_mu3e5_zmg_mixup02` | active FedZMG stack plus `--mixup_alpha 0.2` |
| 2 | P2 | `fedavgm_lr18_m0475_epochs4_clientlr0045_cm0925_wd5e4_eta00001_mu3e5_zmg_sam003` | active FedZMG stack with `--aggregation_epochs 4 --sam_rho 0.03` |

## Reflective memory

- Keep: mixup is the cheapest non-duplicate label-skew augmentation left after logit-only calibration failed.
- Keep with caution: FedSAM gets one reduced-epoch runtime-controlled attempt; do not run full seven-epoch SAM unless the reduced candidate is promising and comfortably inside timeout.
- Reserve: FedSWA-style final averaging only after client-local mixup/SAM results because final-model semantics must stay comparable.

### Batch outcome

- Local mixup with `--mixup_alpha 0.2` scored 0.914800, below the active kept 0.916700 stack and below the raw high-water 0.916900.
- Reduced-epoch FedSAM with `--aggregation_epochs 4 --sam_rho 0.03` scored 0.910700 and stayed within runtime, but did not recover the accuracy lost from reducing local epochs.
- The default-off `--mixup_alpha` and `--sam_rho` knobs were removed after review because both source-backed mechanisms missed and would add non-surviving client surface area.
- Do not retry local mixup or reduced-epoch SAM for this stack unless a materially different source-backed implementation is selected.

---

# Literature Loop 2026-05-08 Local Class-Imbalance Losses

## Working memory

- Watchdog trigger: `recommendation=literature` after 32 scored candidates since the prior literature reset.
- Active kept stack: FedAvgM/FedProx/FedZMG, `aggregation_epochs=7`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `cosine_lr_eta_min_factor=0.0001`, `fedproxloss_mu=3e-5`, `server_lr=1.8`, `server_momentum=0.475`, score 0.916700.
- Raw high-water: same stack with `cosine_lr_eta_min_factor=0.00015`, score 0.916900, not material enough to keep as a code or config branch.
- Confirmed null/worse ideas to avoid: FedLC/FedRS/label smoothing, mixup, reduced-epoch SAM, update clipping, FedNovaM, FedAdam, SCAFFOLD, median/default/weighted FedAvg, and more local scalar jitter near tested values.
- Candidate width: `PARALLEL_CANDIDATES=4` on one H100, pinned with `CUDA_VISIBLE_DEVICES=0`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| federated learning label skew class imbalance focal loss CIFAR non-IID paper | find objective changes for skewed local labels after logit calibration failed | web search, arXiv | Fed-Focal directly targets FL class imbalance with focal-style CE reshaping. |
| federated learning class balanced loss label distribution skew non-IID paper | explore reweighting instead of logit shifting | web search, AAAI, CVF | FL imbalance literature supports the failure mode; CVPR effective-number loss is implementation-simple. |
| federated learning long-tailed class imbalance LDAM focal loss label skew | search margin/reweighting alternatives for local objective changes | web search, arXiv | LDAM is relevant but adds margin and schedule complexity; reserve unless simpler losses miss. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Wang21 RatioLoss | Addressing Class Imbalance in Federated Learning / 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/17219 | client-side class imbalance and non-IID data can damage the shared model | FL imbalance loss | keep as challenge evidence; reject direct ratio-loss implementation because the paper uses a monitoring scheme beyond this client-only surface |
| Sarkar20 FedFocal | Fed-Focal Loss for imbalanced data classification in Federated Learning / 2020 | https://arxiv.org/abs/2011.06283 | class imbalance causes variable FL training performance | focal-style local loss | keep |
| Cui19 CBLoss | Class-Balanced Loss Based on Effective Number of Samples / 2019 | https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html | long-tailed class counts bias CE toward dominant classes | effective-number reweighting | keep |
| Lin17 Focal | Focal Loss for Dense Object Detection / 2017 | https://arxiv.org/abs/1708.02002 | many easy examples dominate CE gradients | focal loss | keep as implementation basis |
| Cao19 LDAM | Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss / 2019 | https://arxiv.org/abs/1906.07413 | rare classes need larger margins and delayed reweighting | margin loss plus schedule | reserve |
| Zhang22 FedLC | Federated Learning with Label Distribution Skew via Logits Calibration / 2022 | https://proceedings.mlr.press/v162/zhang22p.html | label skew biases local classifiers | logit calibration | reject: already tested and removed |
| Li21 FedRS | FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data / 2021 | https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf | missing local classes corrupt softmax updates | restricted softmax | reject: already tested and removed |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Local class-imbalance gradients | Wang21 and Sarkar20 identify class imbalance as a direct FL training failure mode | FedZMG improved update geometry but 32 follow-ups failed to beat 0.916900 | Dirichlet alpha 0.5 creates skewed local class counts each round | `client.py`, `job.py` default-off local loss args |
| C2 | Easy-majority examples dominate local CE | Lin17 and Fed-Focal motivate downweighting well-classified examples | higher local compute and scheduler retunes regress, consistent with local ERM overfitting | focal scaling changes only local loss weighting and keeps optimizer/DIFF flow intact | `client.py`, `job.py` |
| C3 | Rare local classes need stronger loss weight | Cui19 effective-number loss reweights by diminishing sample benefit rather than raw inverse frequency | FedLC/FedRS/logit-only fixes missed, so a weight-space CE change is a non-duplicate mechanism | per-client `train_dataset.targets` are already local and require no metadata exchange | `client.py`, `job.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Local multiclass focal loss | Sarkar20 Fed-Focal; Lin17 Focal | add default-off `--focal_loss_gamma`; run gamma `1.0` and `2.0` on active FedZMG stack | reduce easy-majority dominance and sharpen hard examples without changing batches | score <= 0.916900 or unstable loss | low: client-local loss scalar |
| P2 | Effective-number class-balanced CE | Cui19 CBLoss; Wang21 imbalance evidence | add default-off `--class_balanced_loss_beta`; run beta `0.99` on active stack | boost rare local class gradients while avoiding raw inverse-count extremes | score <= 0.916900 or obvious rare-class overfit | low: client-local class weights from local targets |
| P3 | Class-balanced focal loss | Cui19 CBLoss; Sarkar20 Fed-Focal | combine `--class_balanced_loss_beta 0.99 --focal_loss_gamma 1.0` | jointly boost rare classes and downweight easy examples | score <= isolated P1/P2 or training instability | low-medium: two loss transforms combined |
| P4 | LDAM local margin | Cao19 LDAM | future `--ldam_max_margin` plus optional delayed reweighting | improve rare-class margin after CE/focal misses | no gain from simpler P1/P2/P3 or schedule ambiguity | medium: more math and schedule state |
| P5 | Ratio-loss style local reweighting | Wang21 RatioLoss | not selected; direct paper method needs monitoring/inference beyond current surface | FL-specific imbalance mitigation | requires server/client composition monitoring or a materially different local approximation | medium-high |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | not FedLC/FedRS/label smoothing; changes per-example loss weight, not logits or targets | no focal rows exist | select |
| P2 | not raw logit calibration; uses local effective-number CE weights | no class-balanced rows exist | select |
| P3 | composition of two selected local loss terms | only run one conservative combo after isolated candidates | select |
| P4 | related class-imbalance family | more complex than necessary before focal/CB evidence | reserve |
| P5 | FL imbalance objective | direct implementation would exceed client-only safe surface | reject |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- |
| P1 focal loss | 3 | 5 | 5 | 4 | 5 | 1 | 29 |
| P2 class-balanced CE | 3 | 5 | 5 | 4 | 5 | 1 | 29 |
| P3 class-balanced focal | 3 | 4 | 4 | 4 | 5 | 1 | 26 |
| P4 LDAM | 3 | 4 | 2 | 4 | 5 | 1 | 24 |
| P5 ratio loss | 2 | 3 | 2 | 4 | 5 | 1 | 20 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_focal1` | active high-floor FedZMG stack plus `--focal_loss_gamma 1.0` |
| 2 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_focal2` | active high-floor FedZMG stack plus `--focal_loss_gamma 2.0` |
| 3 | P2 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb099` | active high-floor FedZMG stack plus `--class_balanced_loss_beta 0.99` |
| 4 | P3 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb099_focal1` | active high-floor FedZMG stack plus `--class_balanced_loss_beta 0.99 --focal_loss_gamma 1.0` |

## Reflective memory

- Keep the loss knobs only if one candidate materially beats the 0.916900 raw high-water or is close enough to justify a narrow source-backed follow-up.
- If all four miss, remove the default-off loss knobs and mark focal/class-balanced loss as null for this FedZMG stack.
- Reserve LDAM only as a materially different class-imbalance branch; do not retry FedLC/FedRS/label smoothing, mixup, reduced-epoch SAM, or clipping.

### Batch outcome

- Four-wide launch was too aggressive for this stack: focal gamma `1.0`, focal gamma `2.0`, and class-balanced focal beta `0.99` gamma `1.0` were still in round 19 when `RUN_TIMEOUT_SECONDS=1200` killed them.
- Class-balanced CE beta `0.99` failed earlier with `Diff norm is NaN or Inf: nan` at round 4.
- The watchdog reset on this literature row and reports `recommendation=continue`; rerun the focal-only candidates at `PARALLEL_CANDIDATES=2` before deciding whether focal loss is a null result.
- Do not retry beta `0.99` class-balanced weighting without a lower-beta stability bracket.

### Focal retry outcome

- Width-2 focal retry completed inside the 1200-second cap.
- `--focal_loss_gamma 1.0` scored 0.910100.
- `--focal_loss_gamma 2.0` scored 0.903400.
- Both were marked `discard`; focal loss is a null result for this FedZMG stack unless a different paper-backed objective changes the loss shape materially.
- Lower-beta class-balanced bracket completed inside the cap: `--class_balanced_loss_beta 0.90` scored 0.918600 and was marked `keep`; `0.95` scored 0.913600 and was marked `discard`.
- The watchdog reset on the beta `0.90` material improvement and reports `recommendation=continue`.
- Remove the null focal-loss knob from `client.py` and `job.py`; keep only `--class_balanced_loss_beta` as the surviving source-backed loss surface.
- Refinement around beta `0.90` missed: beta `0.875` scored 0.915300 and beta `0.925` scored 0.914800, both discarded.
- Carry forward beta `0.90` as the active class-balanced setting.

---

# Literature Loop 2026-05-09 LDAM Margin Loss

## Working memory

- Watchdog trigger: `recommendation=literature` after 32 scored candidates since the beta `0.90` material improvement at row 309.
- Active kept stack: FedAvgM/FedProx/FedZMG/class-balanced CE, `aggregation_epochs=7`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `cosine_lr_eta_min_factor=0.00015`, `fedproxloss_mu=3e-5`, `server_lr=1.8`, `server_momentum=0.475`, `class_balanced_loss_beta=0.90`, score 0.918600.
- Recent symptoms: beta refinements, client/server LR, local compute, scheduler floors, FedProx, momentum, weight decay, exact local steps, FedAvg/median/SCAFFOLD aggregation, and component ablations all missed.
- Crash note: width-2 scheduler-floor sweep hit NVFlare communication timeouts; width-1 retries completed, so use sequential width 1 for the next source-backed batch.
- Candidate width: `PARALLEL_CANDIDATES=1`, pinned with `CUDA_VISIBLE_DEVICES=0`.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| federated learning label distribution skew LDAM loss class imbalance CIFAR non-IID | find a margin-level class-imbalance branch after effective-number CE plateaued | web search, NeurIPS/arXiv | LDAM is distinct from class weighting and focal loss. |
| balanced softmax long-tailed visual recognition federated label skew | check whether class-prior softmax corrections are a safe non-duplicate | web search, NeurIPS | Balanced Softmax is simple but overlaps logit-calibration nulls. |
| model contrastive federated learning non-IID client drift image classification | look for a representation/client-drift mechanism after scalar FedProx/SCAFFOLD misses | web search, CVPR | MOON is relevant but needs feature/previous-model state and more code risk. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Cao19 LDAM | Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss / 2019 | https://papers.nips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss | rare classes need larger decision margins, not only larger CE weights | margin loss | keep |
| Ren20 Balanced Softmax | Balanced Meta-Softmax for Long-Tailed Visual Recognition / 2020 | https://papers.nips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html | softmax gradients are biased under long-tailed class priors | logit/prior adjusted CE | reserve/reject as too close to FedLC/FedRS nulls |
| Li21 MOON | Model-Contrastive Federated Learning / 2021 | https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html | local representations drift under heterogeneous data | model-contrastive regularization | reserve; more implementation risk |
| Acar21 FedDyn | Federated Learning Based on Dynamic Regularization / 2021 | https://iclr.cc/virtual/2021/oral/3503 | local-device minima are inconsistent with global minima under heterogeneity | dynamic regularization | reject: needs server-coupled per-client state outside current safe surface |
| Zhang22 FedLC | Federated Learning with Label Distribution Skew via Logits Calibration / 2022 | https://proceedings.mlr.press/v162/zhang22p.html | local softmax CE overfits under label skew | logits calibration | reject: already tested and removed |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Reweighting alone may not fix minority margins | Cao19 separates label-distribution-aware margins from reweighting and reports both can help | effective-number beta `0.90` helped, but all beta and optimizer follow-ups plateaued | local class counts already exist; margin changes only local logits | `client.py`, `job.py` |
| C2 | Logit-prior corrections are mostly exhausted | Ren20 and Zhang22 motivate prior/logit softmax corrections | FedLC/FedRS and multiple scheduler/CE variants were null | another prior-shift CE is likely duplicate unless LDAM fails | `client.py`, `job.py` |
| C3 | Drift regularizers need more state than this loop should add now | MOON/FedDyn target heterogeneous local drift | FedProx/SCAFFOLD/aggregation retunes missed, but protocol/state risk is higher | keep as reserve after a low-risk margin-loss attempt | `client.py` or explicit protocol mode only |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | LDAM true-class margin | Cao19 LDAM; Cui19 CBLoss | add default-off `--ldam_max_margin`; run margins `0.25` and `0.50` with beta `0.90` active stack | improve rare-class decision boundaries beyond CE reweighting | both margins score <= 0.918600 or destabilize | low: client-local logit adjustment |
| P2 | Balanced Softmax local-count prior | Ren20 Balanced Softmax | add CE over `logits + log(local_counts)` with clamped counts | reduce biased softmax gradients from skewed local priors | repeats FedLC/FedRS failure pattern | low-medium but duplicate risk |
| P3 | MOON-style model contrastive loss | Li21 MOON | add local previous/global representation contrastive term | reduce representation drift after FedProx/SCAFFOLD miss | runtime or feature plumbing exceeds budget; no score gain | medium: needs feature access and local state |
| P4 | FedDyn-style dynamic regularization | Acar21 FedDyn | not selected; would need per-client dynamic server/client state | align local/global optima under heterogeneity | violates safe state boundary | high |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | distinct from focal and class-balanced CE; margin not weight/logit calibration | no LDAM rows exist | select |
| P2 | close to FedLC/FedRS/logit calibration null results | prior logit calibration already removed | reserve/reject |
| P3 | not duplicate, but larger code surface than P1 | FedProx/SCAFFOLD misses reduce confidence | reserve |
| P4 | dynamic regularization family | requires forbidden server-coupled state | reject |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 LDAM | 3 | 5 | 5 | 4 | 4 | 1 | 28 |
| P2 Balanced Softmax | 2 | 4 | 4 | 4 | 2 | 1 | 21 |
| P3 MOON | 3 | 3 | 2 | 4 | 5 | 2 | 21 |
| P4 FedDyn | 3 | 1 | 1 | 4 | 5 | 2 | 16 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_ldam025_w1` | active kept stack plus `--ldam_max_margin 0.25` |
| 2 | P1 | `fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg_cb090_ldam050_w1` | active kept stack plus `--ldam_max_margin 0.50` |

## Reflective memory

- Keep: LDAM is the lowest-risk non-duplicate after effective-number CE became the surviving source-backed class-imbalance mechanism.
- Discard: do not start Balanced Softmax until LDAM scores, because FedLC/FedRS/logit-prior changes are already null for this campaign.
- Do not retry: width-2 for long seven-epoch candidates after the scheduler communication timeouts; use sequential width 1 unless runtimes become too slow.
- Sources to carry forward: Cao19 LDAM; Cui19 CBLoss; Ren20 Balanced Softmax; Li21 MOON; Acar21 FedDyn.

### Batch outcome

- LDAM width-1 candidates completed inside the 1200-second cap.
- `--ldam_max_margin 0.25` scored 0.911700.
- `--ldam_max_margin 0.50` scored 0.911300.
- Both were marked `discard`; LDAM is a null result for the active beta `0.90` FedZMG stack.
- Remove the default-off LDAM knob from `client.py`, `job.py`, and `mutation_schema.yaml`; keep `--class_balanced_loss_beta` as the only surviving class-imbalance loss surface.
- The watchdog reset on the literature row and reports `recommendation=continue` with two scored candidates since reset.
