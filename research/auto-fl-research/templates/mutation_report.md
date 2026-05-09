# Mutation report

## Hypothesis

Successful runs are appended to `results.tsv` as `candidate`, but the previous instructions did not force agents to rewrite reviewed rows to `keep` or `discard` after run analysis. This leaves long campaigns with hundreds of stale candidates and progress plots with no kept markers. Run review needs an explicit ledger-finalization step and a helper script.

## Files changed

- `README.md`
- `program.md`
- `scripts/finalize_batch_status.py`
- `scripts/summarize_results.py`
- `skills/autofl-nvflare/SKILL.md`
- `skills/autofl-nvflare/references/provenance.md`
- `skills/autofl-nvflare/references/runbook.md`
- `templates/mutation_report.md`

## Commands run

- `make validate`
- `make smoke`

## Observed outcome

- Current local `results.tsv` has 469 `candidate` rows, 25 `crash` rows, and 0 `keep` rows, confirming the prompt gap.
- `program.md`, README, the autofl skill, and the runbook now state that `candidate` means unreviewed and that every completed run or batch must update statuses before the next candidate batch.
- Added `scripts/finalize_batch_status.py` to promote the best reviewed candidate to `keep` and demote reviewed non-survivors to `discard`.
- `scripts/summarize_results.py` now reminds agents to finalize statuses after reviewing candidate runs.
- The README and skill provenance now acknowledge the Camyla-inspired literature-loop / QWBE-style proposal workflow.
- No local `results.tsv` rows were modified by this harness change.

## Literature basis

None. This is ledger hygiene and prompt hardening.

## Run analysis

Not run. This change does not affect training behavior.

## Contract check

- No FL client loop, aggregation, model, data split, scoring behavior, or run script behavior changed.
- Validation status recorded in this report after checks complete.

## Rollback risk

Low. The change adds a standalone ledger helper and tightens instructions. It does not change candidate execution or score extraction.

## Next mutation

Use `scripts/finalize_batch_status.py` after every completed run or batch. For stale ledgers, run it once with `--all-candidates --keep-best --discard-others` after confirming the intended cleanup policy.

---

# Literature Loop 2026-05-07

## Hypothesis

The optimizer-only plateau near 0.899 is likely driven by non-IID client drift and server optimizer sensitivity. Source-backed mechanisms that directly regularize local drift or stabilize adaptive server updates may produce a better next batch than further tiny LR/momentum jitter.

## Sources

- Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020, arXiv:1812.06127, https://arxiv.org/abs/1812.06127. FedProx adds a proximal local term for statistical/system heterogeneity.
- Reddi et al., "Adaptive Federated Optimization", ICLR 2021, arXiv:2003.00295, https://arxiv.org/abs/2003.00295. FedAdam/FedYogi-style server optimizers target heterogeneous FL convergence.
- Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020, arXiv:1910.06378, https://arxiv.org/abs/1910.06378. SCAFFOLD corrects client drift with control variates.
- Zhang et al., "Understanding Clipping for Federated Learning", arXiv:2106.13673, https://arxiv.org/abs/2106.13673. Update clipping is a reserve idea for heterogeneity-stable FedAvg variants.

## Ledger Memory

- Current best: 0.898900 with FedAvgM `server_lr=1.8`, `server_momentum=0.35`, client `lr=0.04`, client `momentum=0.925`, `weight_decay=5e-4`, `aggregation_epochs=7`.
- Plateau evidence: watchdog reached 33/32 scored candidates since last material reset; local retunes of server LR/momentum, client LR/weight decay/momentum, epochs, and exact local steps mostly regressed.
- Source-backed near miss: FedProx `mu=1e-5` on the previous best stack scored 0.898300 before the current client/server momentum improvements.

## Selected Batch

- P1: current best stack plus `--fedproxloss_mu 1e-5`, description tagged `[src: Li20 FedProx arXiv:1812.06127]`.
- P2: conservative FedAdam with current best client/local settings, `--server_lr 0.2 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 0.1`, description tagged `[src: Reddi21 FedOpt arXiv:2003.00295]`.

## Contract Check

- No code changes are required for the selected batch.
- Both candidates preserve DIFF uploads, `NUM_STEPS_CURRENT_ROUND`, the existing receive/send loop, fixed evaluation, and the same `moderate_cnn` architecture/cap.
- SCAFFOLD and clipping remain reserve proposals; neither is launched in this first literature batch.

## Literature Batch Outcome

- FedProx current-stack `mu=1e-5` scored 0.897600, below the 0.898900 best despite the source-backed drift hypothesis.
- Conservative FedAdam `server_lr=0.2,tau=0.1` avoided the prior crash but scored 0.807800, so the FedAdam branch should not be retried without a stronger implementation-level reason.
- Next source-backed reserve is tuned SCAFFOLD, clearly labeled as the implemented opt-in control-variate protocol mode.
- Tuned SCAFFOLD lr 0.04 and lr 0.02 scored 0.886800 and 0.884000, respectively. This rules out SCAFFOLD as a useful CLI-only recovery branch for the current budget.

---

# Literature Loop 2026-05-07 Row 141 Plateau

## Hypothesis

The current FedAvgM stack has likely exhausted scalar optimizer retuning. Two source-backed mechanisms remain compatible with the FLModel contract and target different failure modes: soft clipping of large client DIFFs before server momentum, and client-local self-distillation against the received global model to reduce drift.

## Sources

- Zhang et al., "Understanding Clipping for Federated Learning", 2021, arXiv:2106.13673, https://arxiv.org/abs/2106.13673. The selected proposal clips each client's full model update norm before aggregation.
- Yashwanth et al., "Adaptive Self-Distillation for Minimizing Client Drift in Heterogeneous Federated Learning", 2024, arXiv:2305.19600, https://arxiv.org/abs/2305.19600. The selected proposal uses the received global model as a frozen teacher during local training.
- Qu et al., "Generalized Federated Learning via Sharpness Aware Minimization", 2022, arXiv:2206.02618, https://arxiv.org/abs/2206.02618. FedSAM is a reserve idea because it likely doubles local training cost.
- Gao et al., "FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction", 2022, arXiv:2203.11751, https://arxiv.org/abs/2203.11751. FedDC-style drift correction is reserved because it is a broader stateful objective change.

## Files changed

- `client.py`
- `custom_aggregators.py`
- `job.py`
- `mutation_schema.yaml`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Commands run

- `PYTHON=.venv/bin/python make validate`
- `PYTHON=.venv/bin/python make smoke`

## Contract check

- The client still receives full model params, loads them with `strict=True`, computes `compute_model_diff`, sends `ParamsType.DIFF`, and preserves `NUM_STEPS_CURRENT_ROUND`.
- `--update_clip_norm` is default-off and only rescales each already-received client DIFF inside FedOpt-style aggregation.
- `--global_distill_alpha` is default-off and adds a client-local KL term using the existing received global model; it introduces no server-coupled metadata.
- Fixed budget fields remain unchanged for the selected candidates.

## Selected batch

- P1: best stack plus `--update_clip_norm 45`, description tagged `[src: Zhang21 clipping arXiv:2106.13673]`.
- P2: best stack plus `--global_distill_alpha 0.05 --global_distill_temperature 2.0`, description tagged `[src: Yashwanth24 ASD arXiv:2305.19600]`.

## Reflective memory

- Do not retry current-stack FedProx, SCAFFOLD, FedAdam, median aggregation, or scheduler floor/off variants without a stronger implementation-level reason.
- Update clipping at norm 45 tied the best at 0.903400 but added complexity, so it was discarded and the default-off aggregation knob was reverted.
- Global self-distillation at alpha 0.05, temperature 2.0 scored 0.901000 and the default-off client knob was reverted.
- FedProx `mu=3e-5` combined with the lower scheduler floor `eta_min_factor=0.0001` scored a new best of 0.904100; `mu=1e-6` under the same floor scored 0.900200. Carry this stack forward as the active best.
- Server momentum `0.475` on the FedProx/lower-floor stack scored a new best of 0.906100; `0.425` also beat the prior best at 0.904900 but is not the survivor. Carry `server_momentum=0.475` forward.
- Reserve FedSAM only if the runtime budget remains healthy; otherwise return to contract-safe aggregation changes.

---

# Literature Loop 2026-05-08 Label-Skew Calibration

## Hypothesis

The active FedAvgM/FedProx stack has exhausted scalar optimizer, scheduler, and local-compute retunes. The remaining failure mode appears to be local classifier bias under Dirichlet label skew: local CE updates overfit majority classes and damage rare or missing class proxies before aggregation. A client-local classifier-calibration branch is contract-safe and directly targets that symptom.

## Sources

- Zhang et al., "Federated Learning with Label Distribution Skew via Logits Calibration", ICML 2022, arXiv:2209.00189, https://arxiv.org/abs/2209.00189. FedLC calibrates logits by local class occurrence to reduce biased local updates for majority, minority, and missing classes.
- Li and Zhan, "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data", KDD 2021, https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf. FedRS scales missing-class logits during local softmax so missing-class classifier weights are not pushed as strongly without positive examples.
- Muller, Kornblith, and Hinton, "When Does Label Smoothing Help?", NeurIPS 2019, https://papers.neurips.cc/paper/8717-when-does-label-smoothing-help. Label smoothing is a simple reserve regularizer for overconfident local CE.
- Qu et al., "Generalized Federated Learning via Sharpness Aware Minimization", 2022, arXiv:2206.02618, https://arxiv.org/abs/2206.02618. FedSAM remains a reserve due likely double-backward runtime cost.

## Files changed

- `client.py`
- `job.py`
- `mutation_schema.yaml`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Contract check

- The client still receives full model params, loads them with `strict=True`, computes `compute_model_diff`, sends `ParamsType.DIFF`, and preserves `NUM_STEPS_CURRENT_ROUND`.
- New knobs are default-off and only transform the client-local CE logits or smoothing target: `--fedlc_tau`, `--fedrs_alpha`, and `--label_smoothing`.
- No data, evaluation, architecture, dependency, or server-client metadata change is introduced.

## Selected batch

- P1: active FedAvgM/FedProx stack plus `--fedlc_tau 0.5`, description tagged `[src: Zhang22 FedLC arXiv:2209.00189]`.
- P2: active FedAvgM/FedProx stack plus `--fedrs_alpha 0.5`, description tagged `[src: Li21 FedRS KDD:10.1145/3447548.3467254]`.

## Reflective memory

- Treat label-skew calibration as a new branch; do not mix FedLC and FedRS in the same candidate until each has an isolated result.
- If both miss, try the simple label smoothing reserve before returning to higher-cost FedSAM.
- Do not resume scalar FedAvgM/FedProx jitter unless a classifier-calibration candidate creates a new active stack.

## Literature Batch Outcome

- FedLC `--fedlc_tau 0.5` scored 0.899200.
- FedRS `--fedrs_alpha 0.5` scored 0.901200.
- Label smoothing scored 0.903400 at `0.02` and 0.904400 at `0.05`.
- None beat the active kept 0.906100 FedAvgM/FedProx lower-floor stack, so the default-off loss knobs were removed after review.
- Do not retry FedLC, FedRS, or label smoothing under the current active stack unless a new paper-backed implementation differs materially from these simple local-logit variants.

---

# Literature Loop 2026-05-08 Step-Normalization Plateau

## Hypothesis

The active FedAvgM/FedProx stack has exhausted scalar retunes. The next viable mechanisms should change the update geometry without changing the FL protocol: a cheap client-side zero-mean gradient projection for non-IID drift, and a FedNova-style step-normalized FedAvgM aggregator for objective inconsistency from variable local step counts.

## Sources

- Zantalis, Zervas, and Koulouras, "FedZMG: Efficient Client-Side Optimization in Federated Learning", 2026, arXiv:2602.18384, https://arxiv.org/abs/2602.18384. FedZMG projects local gradients to a zero-mean space to reduce non-IID client-drift variance without extra communication.
- Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization", 2020, arXiv:2007.07481, https://arxiv.org/abs/2007.07481. FedNova normalizes local updates to reduce objective inconsistency when clients perform different numbers of local updates.
- Qu et al., "Generalized Federated Learning via Sharpness Aware Minimization", 2022, arXiv:2206.02618, https://arxiv.org/abs/2206.02618. FedSAM remains a reserve idea because its two-backward local optimizer is likely too costly near the current runtime cap.
- Acar et al., "Federated Learning Based on Dynamic Regularization", 2021, arXiv:2111.04263, https://arxiv.org/abs/2111.04263. Dynamic regularization is rejected for this batch because it is a broader stateful objective change.

## Files changed

- `client.py`
- `custom_aggregators.py`
- `job.py`
- `mutation_schema.yaml`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Contract check

- The client still receives full model params, loads them with `strict=True`, computes `compute_model_diff`, sends `ParamsType.DIFF`, and preserves `NUM_STEPS_CURRENT_ROUND`.
- `--zero_mean_gradients` is default-off and only projects existing local gradients before `optimizer.step()`.
- `--aggregator fednovam` is default-off and only normalizes received DIFFs by `NUM_STEPS_CURRENT_ROUND` before applying server momentum; it does not add metadata or change parameter keys.
- Fixed data, communication, architecture, and evaluation budget fields remain unchanged for the selected candidates.

## Selected batch

- P1: active FedAvgM/FedProx stack plus `--zero_mean_gradients`, description tagged `[src: Zantalis26 FedZMG arXiv:2602.18384]`.
- P2: active client/FedProx stack plus `--aggregator fednovam --server_lr 1.8 --server_momentum 0.475`, description tagged `[src: Wang20 FedNova arXiv:2007.07481; Reddi21 FedOpt arXiv:2003.00295]`.

## Reflective memory

- Do not resume scalar FedAvgM/FedProx/scheduler jitter until these source-backed update-geometry candidates are scored.
- Keep FedSAM as a reserve only if runtime room is available or if a reduced-local-compute variant is explicitly justified.
- Treat FedLC/FedRS/label smoothing, FedAdam, SCAFFOLD, median/default/weighted FedAvg, and current-stack exact-step repeats as null under this budget.

## Literature Batch Outcome

- FedZMG zero-mean gradients on the active FedAvgM/FedProx stack scored 0.913700 and reset the watchdog as a material improvement.
- FedNova-style step-normalized FedAvgM scored 0.899100 and was discarded.
- The FedNovaM aggregator surface was removed after review; `--zero_mean_gradients` remains as the kept source-backed client-local mutation.
- Next sweeps should start from FedAvgM `server_lr=1.8`, `server_momentum=0.475`, client `lr=0.045`, `momentum=0.925`, `weight_decay=5e-4`, `aggregation_epochs=7`, `cosine_lr_eta_min_factor=0.0001`, `fedproxloss_mu=3e-5`, and `--zero_mean_gradients`.

---

# Literature Loop 2026-05-08 FedZMG Generalization Plateau

## Hypothesis

The FedZMG stack has reached a local generalization ceiling: scalar optimizer, scheduler, FedProx, local-compute, and weight-decay retunes all missed after the 0.916700 material improvement. The next source-backed candidates should change the local training objective while preserving the DIFF protocol: client-local mixup for label-skew overfitting and reduced-epoch SAM for flatter local updates.

## Sources

- Yoon et al., "FedMix: Approximation of Mixup under Mean Augmented Federated Learning", ICLR 2021, arXiv:2107.00233, https://arxiv.org/abs/2107.00233. FedMix motivates mixup-style augmentation for heterogeneous FL, though this harness keeps it client-local and does not exchange averaged data.
- Sang, Rabbani, and Huang, "Balancing Label Imbalance in Federated Environments Using Only Mixup and Artificially-Labeled Noise", 2024, arXiv:2409.13235, https://arxiv.org/abs/2409.13235. The paper targets label-skewed FL on CIFAR-10 with mixup/noise augmentation; only the local mixup part is compatible here.
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018, arXiv:1710.09412, https://arxiv.org/abs/1710.09412. Mixup is the implementation basis for convex input/label interpolation.
- Qu et al., "Generalized Federated Learning via Sharpness Aware Minimization", ICML 2022, arXiv:2206.02618, https://arxiv.org/abs/2206.02618. FedSAM motivates a local SAM optimizer for non-IID generalization; this batch uses reduced epochs to stay within runtime.
- Liu et al., "FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging", ICML 2025, arXiv:2507.20016, https://arxiv.org/abs/2507.20016. FedSWA is reserved because server-side final-model averaging needs more careful semantics.

## Files changed

- `client.py`
- `job.py`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Contract check

- The client still receives full model params, loads them with `strict=True`, computes `compute_model_diff`, sends `ParamsType.DIFF`, and preserves `NUM_STEPS_CURRENT_ROUND`.
- `--mixup_alpha` is default-off and only changes local batch inputs plus the CE target mix.
- `--sam_rho` is default-off and only performs a local two-pass SAM perturb/restore before the same `optimizer.step()`.
- No data files, evaluation, architecture, dependencies, server-client metadata, or parameter keys changed.

## Selected batch

- P1: active FedZMG/FedAvgM/FedProx stack plus `--mixup_alpha 0.2`, description tagged `[src: Yoon21 FedMix arXiv:2107.00233; Sang24 mixup-noise arXiv:2409.13235; Zhang17 mixup arXiv:1710.09412]`.
- P2: active FedZMG/FedAvgM/FedProx stack with `--aggregation_epochs 4 --sam_rho 0.03`, description tagged `[src: Qu22 FedSAM arXiv:2206.02618]`.

## Validation

- `PYTHON=.venv/bin/python make validate` passed.
- `PYTHON=.venv/bin/python make smoke` passed.
- No-ledger targeted smoke with `--mixup_alpha 0.2 --sam_rho 0.03` passed.

## Reflective memory

- Do not retry logit-only FedLC/FedRS/label smoothing for the current stack; mixup is a different input-level augmentation branch.
- Do not run full seven-epoch SAM unless reduced-epoch SAM is promising and runtime remains comfortably below 1200 seconds.
- Treat FedSWA-style averaging as a future server-side proposal only after client-local objective changes are scored.

## Literature Batch Outcome

- Local mixup with `--mixup_alpha 0.2` scored 0.914800, below the active kept 0.916700 stack and below the raw 0.916900 high-water.
- Reduced-epoch FedSAM with `--aggregation_epochs 4 --sam_rho 0.03` scored 0.910700.
- Both rows were marked `discard`; the plateau watchdog reset on the literature row and now reports `recommendation=continue` with two scored candidates since reset.
- The default-off mixup and SAM client/job knobs were removed after review because neither mechanism survived the batch.

---

# FedZMG Update Clipping Retry

## Hypothesis

Update clipping was only tested on the pre-FedZMG stack, where norm 45 tied the then-best score but added non-surviving aggregator surface. The current FedZMG stack changes client update geometry, so soft server-side DIFF clipping is a materially different retry that may suppress remaining outlier client updates without switching to harsh coordinate-wise median aggregation.

## Source

- Zhang et al., "Understanding Clipping for Federated Learning", 2021, arXiv:2106.13673, https://arxiv.org/abs/2106.13673. Use only the contract-safe idea of clipping already-received client updates before aggregation.

## Files changed

- `custom_aggregators.py`
- `job.py`
- `templates/mutation_report.md`

## Contract check

- Clients still send the same `ParamsType.DIFF` payloads and `NUM_STEPS_CURRENT_ROUND`; clipping is server-local preprocessing inside the existing FedOpt-style aggregators.
- `--update_clip_norm` is default-off and preserves parameter keys and shapes.

## Selected batch

- P1: kept FedZMG stack plus `--update_clip_norm 45`, description tagged `[src: Zhang21 clipping arXiv:2106.13673]`.
- P2: kept FedZMG stack plus `--update_clip_norm 60`, description tagged `[src: Zhang21 clipping arXiv:2106.13673]`.

## Validation

- `PYTHON=.venv/bin/python make validate` passed.
- `PYTHON=.venv/bin/python make smoke` passed.
- No-ledger targeted smoke with `--aggregator fedavgm --update_clip_norm 45` passed.

## Batch Outcome

- `--update_clip_norm 45` scored 0.916700.
- `--update_clip_norm 60` scored 0.916700.
- Both matched the kept material stack but did not beat the raw 0.916900 high-water and added aggregator complexity, so both rows were marked `discard`.
- The default-off clipping aggregator knob was removed after review.

---

# Literature Loop 2026-05-08 Local Class-Imbalance Losses

## Hypothesis

FedZMG improved update geometry, but the plateau after 32 non-improving candidates suggests the remaining failure may be the local CE objective under Dirichlet label skew. A client-only loss reweighting branch can target rare-class and easy-example imbalance without adding metadata, changing model keys, or altering the DIFF contract.

## Sources

- Wang et al., "Addressing Class Imbalance in Federated Learning", AAAI 2021, https://ojs.aaai.org/index.php/AAAI/article/view/17219. The paper identifies class imbalance and non-IID data as a direct FL failure mode and motivates loss-level mitigation.
- Sarkar, Narang, and Rai, "Fed-Focal Loss for imbalanced data classification in Federated Learning", 2020, arXiv:2011.06283, https://arxiv.org/abs/2011.06283. Fed-Focal supports focal-style local CE reshaping for imbalanced FL.
- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html. This supplies the effective-number class weighting formula.
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017, arXiv:1708.02002, https://arxiv.org/abs/1708.02002. This is the focal-loss implementation basis.
- Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019, arXiv:1906.07413, https://arxiv.org/abs/1906.07413. LDAM is reserved as a materially different margin-loss branch if simpler reweighting misses.

## Files changed

- `client.py`
- `job.py`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Contract check

- The client still receives full model params, loads them with `strict=True`, computes `compute_model_diff`, sends `ParamsType.DIFF`, and preserves `NUM_STEPS_CURRENT_ROUND`.
- `--class_balanced_loss_beta` is default-off and only constructs local class weights from `train_dataset.targets` already present on the client.
- The focal-loss knob was removed after scored null results; the surviving code surface is class-balanced CE only.
- No data files, model architecture, evaluation path, dependency set, server-client metadata, or parameter keys changed.

## Selected batch

- P1: active high-floor FedZMG stack plus `--focal_loss_gamma 1.0`, description tagged `[src: Sarkar20 Fed-Focal arXiv:2011.06283; Lin17 Focal arXiv:1708.02002]`.
- P2: active high-floor FedZMG stack plus `--focal_loss_gamma 2.0`, same source tags.
- P3: active high-floor FedZMG stack plus `--class_balanced_loss_beta 0.99`, description tagged `[src: Cui19 CBLoss CVPR; Wang21 RatioLoss AAAI]`.
- P4: active high-floor FedZMG stack plus `--class_balanced_loss_beta 0.99 --focal_loss_gamma 1.0`, description tagged `[src: Cui19 CBLoss CVPR; Sarkar20 Fed-Focal arXiv:2011.06283]`.

## Validation

- `PYTHON=.venv/bin/python make validate` passed.
- No-ledger targeted smoke with `--focal_loss_gamma 1.0 --class_balanced_loss_beta 0.99` passed before focal cleanup.
- Cleanup `PYTHON=.venv/bin/python make validate` passed after focal removal.
- No-ledger targeted smoke with `--class_balanced_loss_beta 0.90` passed after focal removal.

## Reflective memory

- Do not retry FedLC/FedRS/label smoothing, mixup, reduced-epoch SAM, update clipping, or scalar FedAvgM/FedProx jitter for this stack unless a materially different paper-backed mechanism is selected.
- Keep LDAM only as a distinct class-imbalance reserve if class-balanced CE follow-ups plateau.

## Batch Outcome

- Four concurrent source-backed candidates exceeded the one-H100 runtime envelope for this seven-epoch stack. The focal gamma `1.0`, focal gamma `2.0`, and class-balanced focal beta `0.99` gamma `1.0` runs were killed at 1200 seconds near round 19.
- Class-balanced CE beta `0.99` failed at round 4 with `Diff norm is NaN or Inf: nan`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` after the crash batch.
- Width-2 focal retry completed: gamma `1.0` scored 0.910100 and gamma `2.0` scored 0.903400. Both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` after finalizing the focal retry.
- Lower-beta class-balanced bracket completed at width 2: beta `0.90` scored 0.918600 and was marked `keep`; beta `0.95` scored 0.913600 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reset on the material improvement at row 309 and reported `recommendation=continue`.
- The focal-loss knob was removed after its null result; keep `--class_balanced_loss_beta` as the surviving source-backed mutation.
- Narrow beta refinement around the kept value missed: beta `0.875` scored 0.915300 and beta `0.925` scored 0.914800, so both were marked `discard`.
- Keep beta `0.90` as the active class-balanced setting for follow-up optimizer or scheduler sweeps.
- Client-LR retune around beta `0.90` missed: `lr=0.04375` scored 0.916100 and `lr=0.04625` scored 0.911800, both below the kept 0.918600 stack.
- Server-LR retune around beta `0.90` missed: `server_lr=1.75` scored 0.910400 and `server_lr=1.85` scored 0.913400, both discarded.
- Local-compute retune around beta `0.90` missed: `aggregation_epochs=6` scored 0.914300 and `aggregation_epochs=8` scored 0.910000, both discarded.
- Scheduler-floor retune around beta `0.90` missed: `cosine_lr_eta_min_factor=0.000125` scored 0.912700 and `0.000175` scored 0.913300, both discarded.
- FedProx retune around beta `0.90` missed: `mu=2.5e-5` scored 0.913100 and `mu=3.5e-5` scored 0.914100, both discarded.
- Server-momentum retune around beta `0.90` missed: `server_momentum=0.45` scored 0.914500 and `0.50` scored 0.912200, both discarded.
- Client-momentum retune around beta `0.90` missed: `momentum=0.9125` scored 0.914700 and `0.9375` scored 0.910700, both discarded.
- Weight-decay retune around beta `0.90` missed: `weight_decay=2.5e-4` scored 0.910800 and `7.5e-4` scored 0.908900, both discarded.
- Exact-step local training around beta `0.90` missed: `local_train_steps=640` scored 0.909000 and `768` scored 0.913000, both discarded. One 768 launch hit a pre-training pycache race and was retried with an isolated `PYTHONPYCACHEPREFIX`.
- FedAvg comparison around beta `0.90` missed badly: built-in FedAvg scored 0.898100 and weighted FedAvg scored 0.898000, both discarded.
- Tight beta refinement around the kept class-balanced setting missed: beta `0.895` scored 0.913000 and beta `0.905` scored 0.911400, both discarded.
- Component ablations around beta `0.90` missed: removing FedProx scored 0.914900, and removing zero-mean gradients scored 0.901500, both discarded. Keep both `--fedproxloss_mu 3e-5` and `--zero_mean_gradients` in the active stack.
- Lower local-compute sweep around beta `0.90` missed: `aggregation_epochs=4` scored 0.907700 and `aggregation_epochs=5` scored 0.910400, both discarded. Keep `aggregation_epochs=7` as the active local-compute setting.
- Aggregation-mode audit around beta `0.90` missed: median aggregation scored 0.884800 and SCAFFOLD metadata mode scored 0.906600, both discarded.
- Broader scheduler-floor width-2 sweep crashed due a shared NVFlare communication failure: eta `0.0005` and eta `0.001` both timed out at 1200 seconds with target-unreachable / failed-download errors. Retry this scheduler axis at width 1 before treating it as a scored model result.
- Width-1 scheduler-floor retry completed and missed: eta `0.0005` scored 0.913300 and eta `0.001` scored 0.915800, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with 32 scored candidates since the row 309 material improvement.

---

# Literature Loop 2026-05-09 LDAM Margin Loss

## Hypothesis

Class-balanced CE beta `0.90` was the only class-imbalance mechanism to materially improve the FedZMG stack, but 32 follow-ups failed to move past 0.918600. LDAM adds a per-class decision margin from local class counts, which is distinct from both effective-number CE weights and the focal/logit-calibration null results.

## Sources

- Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019, https://papers.nips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss. This supplies the LDAM margin formula and motivates margin changes as separate from reweighting.
- Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020, https://papers.nips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html. Considered as a prior-corrected CE reserve, but too close to earlier logit-calibration nulls for the next batch.
- Li et al., "Model-Contrastive Federated Learning", CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html. Kept as a higher-risk reserve for representation drift.
- Acar et al., "Federated Learning Based on Dynamic Regularization", ICLR 2021, https://iclr.cc/virtual/2021/oral/3503. Rejected for now because FedDyn-style state exceeds the current safe protocol surface.

## Files changed

- `client.py`
- `job.py`
- `mutation_schema.yaml`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Contract check

- LDAM is default-off via `--ldam_max_margin 0.0`.
- LDAM only adjusts local logits before local CE and uses local `train_dataset.targets`; it does not add metadata, change model keys, or alter the DIFF upload path.
- The client still preserves `compute_model_diff`, `ParamsType.DIFF`, `NUM_STEPS_CURRENT_ROUND`, strict model loading, and the evaluation branch.

## Validation

- `PYTHON=.venv/bin/python make validate` passed.
- No-ledger targeted smoke with `--class_balanced_loss_beta 0.90 --ldam_max_margin 0.5` passed.

## Selected batch

- P1a: active kept beta `0.90` stack plus `--ldam_max_margin 0.25`.
- P1b: active kept beta `0.90` stack plus `--ldam_max_margin 0.50`.
- Run sequentially at width 1 because the previous width-2 scheduler batch exposed NVFlare communication contention.

## Reflective memory

- Treat Balanced Softmax/logit-prior corrections as reserve only; FedLC and FedRS already missed.
- Treat MOON or other representation-drift regularizers as higher-risk reserves because they require model feature plumbing or local state beyond this lightweight LDAM branch.

## Batch Outcome

- LDAM width-1 candidates completed inside `RUN_TIMEOUT_SECONDS=1200`.
- `--ldam_max_margin 0.25` scored 0.911700.
- `--ldam_max_margin 0.50` scored 0.911300.
- Both were marked `discard`; remove the LDAM knob and do not retry LDAM on this beta `0.90` stack without a materially different paper-backed schedule or representation change.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the literature reset.
- Exact-step continuation after LDAM removal missed: `local_train_steps=896` scored 0.910100 and `1000` scored 0.914700, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the literature reset.
- Broad server-LR continuation around beta `0.90` missed: `server_lr=1.6` scored 0.914400 and `2.0` scored 0.914100, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with six scored candidates since the literature reset.
- Broad server-momentum continuation around beta `0.90` missed: `server_momentum=0.40` scored 0.915100 and `0.55` scored 0.911900, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eight scored candidates since the literature reset.
- Broad client-momentum continuation around beta `0.90` missed badly: `momentum=0.875` scored 0.909300 and `0.95` scored 0.909100, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with ten scored candidates since the literature reset.
- Broad weight-decay endpoints around beta `0.90` missed badly: `weight_decay=0.0` scored 0.867900 and `1e-3` scored 0.905100, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twelve scored candidates since the literature reset.
- Broad client-LR continuation around beta `0.90` missed: `lr=0.04` scored 0.912700 and `0.05` scored 0.914400, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fourteen scored candidates since the literature reset.
- Broad FedProx continuation around beta `0.90` missed: `mu=1e-5` scored 0.911200 and `1e-4` scored 0.913700, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with sixteen scored candidates since the literature reset.
- Scheduler-floor continuation around beta `0.90` missed: `cosine_lr_eta_min_factor=0.00005` scored 0.914000 and `0.00025` scored 0.913400, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eighteen scored candidates since the literature reset.
- Lower class-balanced beta bracket missed: beta `0.85` scored 0.913400 and beta `0.80` scored 0.909600, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty scored candidates since the literature reset.
- Upper exact-step local-compute continuation missed: `local_train_steps=1152` scored 0.913200 and `1280` scored 0.914900, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-two scored candidates since the literature reset.
- Lower exact-step local-compute continuation missed: `local_train_steps=512` scored 0.913600 and `384` scored 0.908100, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-four scored candidates since the literature reset.
- Upper epoch-based local-compute continuation missed: `aggregation_epochs=9` scored 0.911700 and `10` scored 0.911400, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-six scored candidates since the literature reset.
- Upper scheduler-floor continuation missed: `cosine_lr_eta_min_factor=0.002` scored 0.911700 and `0.005` scored 0.912100, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-eight scored candidates since the literature reset.
- Server-LR edge continuation missed: `server_lr=1.4` scored 0.911500 and `2.2` scored 0.915700, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty scored candidates since the literature reset.
- Upper server-LR extension missed: `server_lr=2.4` scored 0.909700 and was discarded; `server_lr=2.6` crashed before a comparable cross-site score, so do not extend this upper edge further.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty-one scored candidates since the literature reset.
- Class-balanced beta midpoint missed: `class_balanced_loss_beta=0.9125` scored 0.910400 and was discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with thirty-two scored candidates since the literature reset; stop local jitter and run the literature loop before the next candidates.

---

# Literature Loop 2026-05-09 FedNova Normalized Aggregation

## Hypothesis

The active beta `0.90` FedZMG stack has exhausted scalar optimizer, scheduler, local-compute, and class-imbalance sweeps. The cached Dirichlet split is highly uneven across clients, so naive step-weighted DIFF aggregation may still carry objective inconsistency from unequal local trajectories. A FedNova-style server aggregator can normalize each client DIFF by `NUM_STEPS_CURRENT_ROUND`, rescale by the weighted mean local steps, and preserve the existing FLModel DIFF contract.

## Sources

- Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization", NeurIPS 2020, arXiv:2007.07481, https://arxiv.org/abs/2007.07481. This motivates normalized averaging to remove objective inconsistency from heterogeneous local update counts.
- Cheng et al., "Momentum Benefits Non-IID Federated Learning Simply and Provably", ICLR 2024, arXiv:2306.16504, https://arxiv.org/abs/2306.16504. This supports keeping a momentum variant in reserve if pure normalized aggregation is near-best.
- Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020, arXiv:1910.06378, https://arxiv.org/abs/1910.06378. Rejected for the next batch because current-stack SCAFFOLD already scored 0.906600.
- Reddi et al., "Adaptive Federated Optimization", ICLR 2021, arXiv:2003.00295, https://arxiv.org/abs/2003.00295. Rejected for the next batch because FedAdam variants were poor or crashy in this harness.

## Files changed

- `custom_aggregators.py`
- `job.py`
- `mutation_schema.yaml`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Contract check

- `fednova` is a server-side aggregator only; clients still receive full params, load with `strict=True`, compute `compute_model_diff`, send `ParamsType.DIFF`, and preserve `NUM_STEPS_CURRENT_ROUND`.
- The new aggregator uses the existing DIFF keys and existing local-step metadata; it adds no client metadata, no model keys, no dependencies, no data changes, and no evaluation changes.
- When all clients take the same number of local steps, the normalized update reduces to the existing weighted FedAvg DIFF.

## Validation

- `PYTHON=.venv/bin/python make validate` passed.
- `PYTHON=.venv/bin/python make smoke` passed.
- No-ledger one-round FedNova smoke passed with `--aggregator fednova --server_lr 1.0 --server_momentum 0.0`.

## Selected batch

- P1: active class-balanced FedZMG client stack plus `--aggregator fednova --server_lr 1.0 --server_momentum 0.0`, description tagged `[src: Wang20 FedNova NeurIPS]`.
- P2: active class-balanced FedZMG client stack plus `--aggregator fednova --server_lr 1.8 --server_momentum 0.0`, same source tag.
- Reserve: `--aggregator fednova --server_lr 1.0 --server_momentum 0.475`, tagged `[src: Wang20 FedNova NeurIPS; Cheng24 Momentum ICLR]`, only if P1/P2 are close enough to justify the momentum variant.

## Batch Outcome

- P1 scored 0.900900 and P2 scored 0.899000; both were marked `discard`.
- The FedNova branch underfit badly relative to the 0.918600 high-water mark, so the reserve momentum variant was not launched.
- The default-off `fednova` aggregator code and schema choice were removed after review; do not retry FedNova on this stack without a materially different implementation or evidence.
- Post-removal `PYTHON=.venv/bin/python make validate` and `make smoke` passed.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the literature reset.
- Post-literature server-LR continuation missed: `server_lr=2.1` scored 0.913300 and was discarded; `server_lr=2.3` crashed before final cross-site scoring, matching the high-LR instability pattern.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with three scored candidates since the literature reset.
- Exact-step continuation under the post-literature stack missed: `local_train_steps=928` scored 0.912300 and `960` scored 0.911700, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with five scored candidates since the literature reset.
- Server-momentum continuation missed: `server_momentum=0.425` scored 0.913600 and `0.525` scored 0.907800, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seven scored candidates since the literature reset.
- Weight-decay continuation missed: `weight_decay=4e-4` scored 0.914000 and `6e-4` scored 0.911000, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nine scored candidates since the literature reset.
- Client-LR continuation missed: `lr=0.0425` scored 0.911700 and `0.0475` scored 0.914900, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eleven scored candidates since the literature reset.
- Lower class-balanced beta midpoint continuation missed: beta `0.8925` scored 0.914200 and beta `0.8875` scored 0.913500, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirteen scored candidates since the literature reset.
- Tight client-momentum continuation missed: `momentum=0.921875` scored 0.911300 and `0.928125` scored 0.911500, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fifteen scored candidates since the literature reset.
- Aggregation-epoch endpoint continuation missed: `aggregation_epochs=3` scored 0.901900 and `11` scored 0.915500, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seventeen scored candidates since the literature reset.
- Exact-step local-compute endpoint continuation missed: `local_train_steps=256` scored 0.895300 and `1408` scored 0.912500, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nineteen scored candidates since the literature reset.
