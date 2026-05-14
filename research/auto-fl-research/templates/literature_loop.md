# Literature loop worksheet

Use this worksheet only when progress stalls or the next axis is unclear. Keep entries short and source-backed.

## Literature loop: post-r47 FedAvgM ep8 plateau

## Trigger

- Reason: `plateau_watchdog.py` printed `recommendation=literature` after 32 scored candidates since the previous literature reset.
- Current best: `0.900100`, FedAvgM, `aggregation_epochs=8`, `server_lr=1.75`, `server_momentum=0.15`, `client_momentum=0.895`, `weight_decay=4e-4`, `model_arch=moderate_cnn`, `alpha=0.5`, final eval `site-1`.
- Recent symptoms from `results.tsv`: client momentum gave a tiny new best, but refinements around client momentum, client LR, label smoothing, gradient clipping, scheduler floor, weight decay, FedProx, SCAFFOLD, FedAdam, and server-LR interactions all missed.
- Confirmed null/worse ideas to avoid for the next batch: no scheduler, median aggregation, FedAdam damped retry, tuned SCAFFOLD, FedProx `1e-3/1e-2`, label smoothing alone, gradient clipping alone, and more server-LR jitter around `1.70-1.85`.
- Candidate width: 2 on one local 80 GB H100, pinned with `CUDA_VISIBLE_DEVICES=0`.
- Ledger event: started with `scripts/log_literature_review.py --start`; finish with `--finish` before launching the next batch.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| mixup CIFAR-10 overfitting regularization non-IID federated learning | Recent local regularizers did not improve; need a stronger client-local regularizer that preserves protocol. | arXiv, OpenReview, FedMix arXiv | Mixup is a direct tensor/label-space change inside `client.py`. |
| federated learning mixup non-IID CIFAR-10 augmentation | Need FL-specific evidence for mixing under heterogeneous client data. | arXiv, paper indexes | FedMix motivates mixup-like augmentation for non-IID FL, but mean sharing is not needed for a local-only pilot. |
| adaptive federated optimization FedYogi FedAdagrad non-IID visual classification | Prior FedAdam was bad; check whether other FedOpt variants are worth code work. | ICLR/OpenReview PDF, arXiv mirrors | FedYogi/FedAdagrad are source-backed but require aggregator extension after a failed FedAdam family branch. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Hsu19 | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification / 2019 | https://research.google/pubs/measuring-the-effects-of-non-identical-data-distribution-for-federated-visual-classification/ | Non-IID CIFAR-style visual FL is sensitive to client drift and server momentum. | FedAvgM context | Keep as plateau diagnosis. |
| Zhang18 | mixup: Beyond Empirical Risk Minimization / ICLR 2018 | https://arxiv.org/abs/1710.09412 | Large visual models can overfit/memorize; interpolation regularizes between examples and labels. | Mixup | Keep; implemented locally without protocol changes. |
| Yoon21 | FedMix: Approximation of Mixup under Mean Augmented Federated Learning / 2021 | https://arxiv.org/abs/2107.00233 | Standard FL degrades as heterogeneity increases; Mixup-inspired augmentation helps non-IID FL. | Federated Mixup | Keep as FL-specific support; use local Mixup only to avoid mean sharing. |
| Yun19 | CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features / ICCV 2019 | https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf | Image classifiers can overfit to discriminative regions; patch mixing regularizes with informative pixels. | CutMix | Reserve; more code and harder to tune than Mixup. |
| Cubuk20 | RandAugment: Practical Automated Data Augmentation with a Reduced Search Space / NeurIPS 2020 | https://papers.nips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html | Stronger image augmentation can improve CIFAR generalization. | RandAugment | Reject for this loop: would touch shared data transforms, which the task notes discourage. |
| Reddi21 | Adaptive Federated Optimization / ICLR 2021 | https://openreview.net/pdf?id=LkFG3lB13U5 | Server adaptivity can improve heterogeneous FL, including FedAdagrad/FedYogi variants. | FedOpt | Reserve; FedAdam already failed badly and new variants need aggregator work. |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Long local ep8 training may be overfitting local class-skewed subsets despite FedAvgM. | Zhang18 shows input/label interpolation regularizes CIFAR training; Yoon21 connects Mixup-like augmentation to non-IID FL. | Label smoothing and gradient clipping missed, but they were weaker regularizers than mixing examples. | Can add Mixup entirely in `client.py` with a job arg and no protocol change. | `tasks/cifar10/client.py`, `tasks/cifar10/job.py` |
| C2 | More image augmentation may help, but data-transform edits have higher harness risk. | Yun19 and Cubuk20 show augmentation gains on CIFAR/ImageNet. | Existing crop/flip is strong but not exhaustive. | Shared `data/*` is discouraged for mutation; client tensor mixing is safer. | Reserve only; avoid `data/*` edits. |
| C3 | FedAvgM plateau may need a different FedOpt rule, but adaptive Adam already underperformed. | Reddi21 includes FedAdagrad/FedYogi as adaptive server variants. | FedAdam damped scored `0.808600`; FedAvgM remains dominant. | Aggregator extension is possible but lower priority than a client-local regularizer. | `tasks/shared/custom_aggregators.py`, `tasks/cifar10/job.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Add opt-in local Mixup and run mild alpha values. | Zhang18, Yoon21 | Add `--mixup_alpha`; run current best stack with `--mixup_alpha 0.2` and `0.4`. | Reduce overconfident local fits and improve generalization under non-IID ep8 training. | Scores stay below `0.900100`, or training becomes unstable. | Low: client-local tensor/label mixing only. |
| P2 | Add CutMix tensor-space augmentation. | Yun19 | Add `--cutmix_alpha` in client loop and run one mild alpha. | Stronger spatial regularization than Mixup. | Worse than Mixup or runtime/implementation complexity increases. | Medium: more code and bbox sampling edge cases. |
| P3 | Add FedYogi/FedAdagrad server optimizer variants. | Reddi21 | Extend `FedOptAggregator` choices and run damped FedYogi. | Adaptive server denominator may escape FedAvgM plateau more safely than FedAdam. | Repeats FedAdam underperformance or crashes. | Medium: aggregator code change. |
| P4 | RandAugment stronger image transforms. | Cubuk20 | Add transform policy to CIFAR data pipeline. | Improve visual generalization beyond crop/flip. | Requires `data/*` mutation or hurts comparability. | High for this loop; reject. |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | Not duplicate: label smoothing did not mix inputs or pair labels. | Compatible with existing crop/flip and default evaluation. | Launch two alpha variants. |
| P2 | Not duplicate. | More complex than P1; reserve until Mixup falsified. | Reserve. |
| P3 | Related to failed FedAdam, but FedYogi/Adagrad are different Reddi21 variants. | FedAdam branch is weak here. | Reserve, do not launch before Mixup. |
| P4 | Not duplicate. | Violates preferred edit surface by touching shared data transforms. | Reject. |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 4 | 5 | 4 | 5 | 4 | 3 | 28 |
| P3 | 3 | 3 | 3 | 4 | 4 | 3 | 20 |
| P2 | 3 | 3 | 2 | 4 | 4 | 3 | 19 |
| P4 | 3 | 1 | 2 | 4 | 3 | 3 | 14 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P1 | `r48_lit_mixup02_cm0895_wd4e4_sm015` | Current best FedAvgM ep8 stack plus `--mixup_alpha 0.2`. |
| 2 | P1 variant | `r48_lit_mixup04_cm0895_wd4e4_sm015` | Current best FedAvgM ep8 stack plus `--mixup_alpha 0.4`. |
| 3 | P3 reserve | reserve | Damped FedYogi/FedAdagrad only if Mixup misses and watchdog later permits another literature branch. |

## Reflective memory

- Local jitter around the FedAvgM ep8 optimizer stack is exhausted for now.
- Label smoothing and gradient clipping alone did not fix the plateau.
- Prefer client-local Mixup before deeper aggregator work because it is source-backed, opt-in, and avoids `data/*` or protocol edits.

---

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

## Literature loop: post-r118 local-step label-smoothing plateau

## Trigger

- Reason: `plateau_watchdog.py` printed `recommendation=literature` after 32 scored candidates since row 244, the last material improvement.
- Current best: `0.909600`, FedAvgM, `local_train_steps=990`, `server_lr=1.75`, `server_momentum=0.15`, `client_momentum=0.890`, `weight_decay=3.5e-4`, `mixup_alpha=0.03`, `label_smoothing=0.01`, `cosine_lr_eta_min_factor=0.001`, `model_arch=moderate_cnn`, final eval `site-1`.
- Recent symptoms from `results.tsv`: local-step, label-smoothing, weight-decay, mixup, client-momentum, server-momentum, server-LR, gradient-clipping, and scheduler-floor jitter all missed; FedProx at `5e-7` and `1e-6` timed out at round 19 under the current local-step budget.
- Confirmed null/worse ideas to avoid next: more close jitter around label smoothing `0.008-0.012`, mixup `0.029-0.031`, weight decay `3.4e-4-3.6e-4`, client momentum `0.889-0.891`, server momentum `0.1525-0.155`, gradient clipping `6.5/8.5`, and FedProx on `local_train_steps=990`.
- Candidate width: 2 on one local 80 GB H100, pinned with `CUDA_VISIBLE_DEVICES=0`.
- Ledger event: started with `scripts/log_literature_review.py --start`; finish with `--finish` before launching the next batch.

## Search queries

| query | rationale | source(s) searched | notes |
| --- | --- | --- | --- |
| SCAFFOLD client drift non-IID CIFAR federated learning local steps | Plateau follows long exact local training; need a source-backed drift correction that preserves DIFF uploads. | arXiv, PMLR | SCAFFOLD directly targets client drift and is already implemented as an explicit protocol mode. |
| Group Normalization federated learning feature shift non-IID CIFAR | Current CNN lacks normalization; a registered buffer-free normalization variant may improve local stability without batch-stat buffers. | arXiv, CVF, FedBN arXiv | GroupNorm avoids batch-stat dependence; FedBN motivates normalization choices under FL feature shift. |
| sharpness aware minimization federated learning non-IID CIFAR FedSAM | Regularization jitter is exhausted; check whether flat-minima local optimization is worth code work. | arXiv, PMLR, OpenReview | FedSAM is relevant but needs client optimizer changes and likely doubles local compute. |

## Candidate papers

| ref | title / year | url | challenge | method family | keep/reject |
| --- | --- | --- | --- | --- | --- |
| Hsu19 | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification / 2019 | https://research.google/pubs/measuring-the-effects-of-non-identical-data-distribution-for-federated-visual-classification/ | Non-IID CIFAR-style visual FL is sensitive to heterogeneity and server optimization. | FedAvgM context | Keep as diagnosis. |
| Karimireddy20 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning / 2020 | https://proceedings.mlr.press/v119/karimireddy20a.html | FedAvg suffers client drift on heterogeneous data; control variates reduce drift. | SCAFFOLD | Keep; implemented and profile-supported. |
| Wu18 | Group Normalization / 2018 | https://arxiv.org/abs/1803.08494 | Normalization without batch-stat dependence can stabilize visual models when batch statistics are unreliable. | GroupNorm architecture | Keep; registered variant exists. |
| Li21 | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization / 2021 | https://arxiv.org/abs/2102.07623 | Feature shift in FL interacts with normalization; local normalization can help non-IID feature distributions. | Normalization in FL | Keep as FL-specific normalization motivation; avoid adding BN buffers. |
| Foret21 | Sharpness-Aware Minimization for Efficiently Improving Generalization / 2021 | https://arxiv.org/abs/2010.01412 | Flat-minima optimization improves image-model generalization on CIFAR-style tasks. | SAM | Keep as reserve. |
| Qu22 | Generalized Federated Learning via Sharpness Aware Minimization / 2022 | https://proceedings.mlr.press/v162/qu22a.html | ERM local optimizers can fall into sharp valleys under FL distribution shift. | FedSAM | Reserve; source-backed but needs code and more runtime. |

## Challenge cards

| id | challenge | paper evidence | `results.tsv` symptom | harness relevance | allowed surface |
| --- | --- | --- | --- | --- | --- |
| C1 | Long local training still likely induces client drift even after FedAvgM tuning. | Karimireddy20 targets client drift under heterogeneous data; Hsu19 supports FedAvgM sensitivity in visual FL. | FedAvgM is best, but 32 local jitter candidates failed to improve and FedProx timed out. | Existing `--aggregator scaffold` can test control variates without changing model keys or `ParamsType.DIFF`. | `tasks/cifar10/client.py`, `tasks/shared/custom_aggregators.py`, `--aggregator scaffold` |
| C2 | The current CNN may lack enough activation stabilization for high local-step client training. | Wu18 shows GroupNorm is independent of batch statistics; Li21 shows normalization choices matter under FL feature shift. | Fine optimizer/regularizer sweeps near the best stack missed, suggesting representation stability rather than scalar knob tuning may be limiting. | Registered `moderate_cnn_norm` is buffer-free and under the parameter cap, but must be labeled as an architecture subcampaign. | `tasks/cifar10/model.py`, `--model_arch moderate_cnn_norm` |
| C3 | Generalization may require flat-minima optimization rather than more scalar regularizer jitter. | Foret21 and Qu22 support SAM/FedSAM for generalization and FL distribution shift. | Label smoothing, mixup, gradient clipping, and scheduler jitter did not move past `0.909600`. | SAM can be client-local and DIFF-preserving, but requires code and likely extra runtime. | Reserve for `tasks/cifar10/client.py`, `tasks/cifar10/job.py` |

## Proposal cards

| id | mechanism | source refs | exact change / args | expected effect | falsifier | contract risk |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Run SCAFFOLD on the current best local-step and regularization stack. | Karimireddy20, Hsu19 | `--aggregator scaffold --local_train_steps 990 --weight_decay 3.5e-4 --momentum 0.890 --mixup_alpha 0.03 --label_smoothing 0.01 --cosine_lr_eta_min_factor 0.001`. | Reduce drift that FedAvgM scalar tuning could not remove. | Crashes, times out, or scores far below `0.909600`. | Medium: explicit supported metadata mode, no parameter-key change. |
| P2 | Launch a labeled architecture subcampaign with the registered GroupNorm model. | Wu18, Li21 | `--model_arch moderate_cnn_norm` with the current best FedAvgM stack and parameter cap. | Stabilize activations during long local client training without adding BN buffers. | Scores below `0.907000` or runtime/shape issues appear. | Medium: architecture change is registered but must stay labeled. |
| P3 | Add opt-in FedSAM client optimizer. | Foret21, Qu22 | Add `--sam_rho`; first candidate would use a reduced local-step cap if full 990 exceeds timeout. | Favor flatter minima after local regularizer jitter stalls. | Runtime doubles into timeout or score does not beat simple FedAvgM. | Medium-high: code change plus higher compute. |
| P4 | Retry FedOpt/FedYogi at current local-step stack. | Reddi21 | `--aggregator fedyogi` or `fedadam` with damped LR/tau. | Adaptive server denominator may escape plateau. | Prior damped FedAdam/FedYogi rows remain far below best; likely repeats. | Medium; reject for this batch. |

## Duplicate and null filter

| proposal | duplicate of | null/worse conflict | decision |
| --- | --- | --- | --- |
| P1 | Prior SCAFFOLD used default/older stacks, not current `local_train_steps=990 + label_smoothing + mixup`. | SCAFFOLD `0.888900` under old ep8 stack is weak but not duplicate. | Launch. |
| P2 | No architecture candidates exist in `results.tsv`. | Starts a labeled architecture subcampaign; do not mix unlabeled with optimizer-only rows. | Launch with architecture label. |
| P3 | Not duplicate. | Requires code and extra runtime; reserve after no-code source-backed candidates. | Reserve. |
| P4 | Damped FedAdam/FedYogi already scored around `0.81` or crashed. | Strong null conflict. | Reject. |

## Proposal scoring

Score each axis from 1-5. Total = `2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost`.

| proposal | expected gain | contract safety | simplicity | evidence | novelty | runtime cost | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 4 | 4 | 5 | 5 | 3 | 3 | 26 |
| P2 | 4 | 4 | 5 | 4 | 5 | 3 | 27 |
| P3 | 4 | 3 | 2 | 5 | 5 | 5 | 21 |
| P4 | 2 | 3 | 4 | 3 | 1 | 3 | 15 |

## QWBE-style next-candidate batch plan

| slot | proposal | candidate name | args / code variant |
| --- | --- | --- | --- |
| 1 | P2 | `r119_lit_arch_gn_lts990_ls001_mix003_wd35e5_cm0890_eta001_sm015` | `--n_clients 8 --num_rounds 20 --aggregation_epochs 8 --local_train_steps 990 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn_norm --max_model_params 5000000 --final_eval_clients site-1 --aggregator fedavgm --server_lr 1.75 --server_momentum 0.15 --weight_decay 3.5e-4 --momentum 0.890 --mixup_alpha 0.03 --label_smoothing 0.01 --cosine_lr_eta_min_factor 0.001` |
| 2 | P1 | `r119_lit_scaffold_lts990_ls001_mix003_wd35e5_cm0890_eta001` | `--n_clients 8 --num_rounds 20 --aggregation_epochs 8 --local_train_steps 990 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1 --aggregator scaffold --weight_decay 3.5e-4 --momentum 0.890 --mixup_alpha 0.03 --label_smoothing 0.01 --cosine_lr_eta_min_factor 0.001` |
| 3 | P3 | reserve | Do not implement until this no-code source-backed batch is reviewed. |
| 4 | P4 | rejected | Do not launch. |

## Reflective memory

- Keep target: `0.909600` FedAvgM with `local_train_steps=990`, `label_smoothing=0.01`, `mixup_alpha=0.03`, `weight_decay=3.5e-4`, `client_momentum=0.890`, and `server_momentum=0.15`.
- Local jitter around that recipe is exhausted for now; only revisit after a source-backed change creates a new interaction surface.
- FedProx at current local-step budget is a timeout risk; avoid it unless local compute is explicitly reduced and labeled.
- Architecture candidates must be labeled as architecture subcampaign rows because `model_arch` changes the model key/shape schema by design.
