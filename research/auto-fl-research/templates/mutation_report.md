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
- Scheduler-floor continuation missed: `cosine_lr_eta_min_factor=0.000075` scored 0.914400 and `0.0001` scored 0.913700, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-one scored candidates since the literature reset.
- FedProx `mu` continuation missed: `fedproxloss_mu=5e-5` scored 0.914100 and `7.5e-5` scored 0.912100, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-three scored candidates since the literature reset.
- Server-learning-rate continuation missed: `server_lr=1.7` scored 0.915500 and `1.9` scored 0.914400, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-five scored candidates since the literature reset.
- Server-momentum continuation missed: `server_momentum=0.4625` scored 0.910700 and `0.4875` scored 0.913900, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-seven scored candidates since the literature reset.
- Class-balanced beta upper-midpoint continuation missed: `class_balanced_loss_beta=0.9025` and `0.9075` both scored 0.914900, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-nine scored candidates since the literature reset.
- Client-learning-rate midpoint continuation missed: `lr=0.04375` scored 0.916100 and `0.04625` scored 0.911800, both discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty-one scored candidates since the literature reset.
- Scheduler-floor threshold probe missed: `cosine_lr_eta_min_factor=0.0002` scored 0.911400 and was discarded.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with thirty-two scored candidates since the literature reset.

## Literature Basis: Local Vicinal Regularization

- Trigger: watchdog plateau at thirty-two scored candidates without a material improvement after the class-balanced FedZMG high-water mark.
- Sources: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018, https://arxiv.org/abs/1710.09412); Yoon et al., "FedMix: Approximation of Mixup under Mean Augmented Federated Learning" (ICLR 2021, https://arxiv.org/abs/2107.00233); Muller et al., "When Does Label Smoothing Help?" (NeurIPS 2019, https://arxiv.org/abs/1906.02629); Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (CVPR 2016, https://arxiv.org/abs/1512.00567); Li et al., "FedBN" (ICLR 2021, https://arxiv.org/abs/2102.07623).
- Hypothesis: the active stack may now be limited by local overconfidence or class-skew overfitting rather than server optimizer scaling; local-only mixup and label smoothing should regularize client training without altering DIFF uploads, `NUM_STEPS_CURRENT_ROUND`, model keys, data splits, or cross-site evaluation.
- Selected next candidates: active best stack plus `--mixup_alpha 0.2`, and active best stack plus `--label_smoothing 0.05`. Reserve: combined light mixup/smoothing only if either single mechanism is near-best.
- Rejected source-backed ideas: FedMix mean-sharing and FedBN/local-BN mechanics are not selected because they would change data exchange, protocol semantics, or fixed `model_arch` comparability in this optimizer campaign.

## Batch Outcome

- Local mixup `alpha=0.2` scored 0.912200; label smoothing `0.05` scored 0.912600.
- Both source-backed local regularization candidates were marked `discard`.
- The default-off mixup and label-smoothing code/schema additions were removed after review because the mechanism underfit badly relative to the 0.918600 high-water mark.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the literature reset.

## Hypothesis: Client Gradient Clipping

- Source basis: Zhang et al., "Understanding Clipping for Federated Learning" (2021, https://arxiv.org/abs/2106.13673) supports clipping as a heterogeneity-control mechanism; prior nulls were server update clipping, not per-step client gradient clipping.
- Proposed change: add a default-off `--grad_clip_norm` client knob and test norms `1.0` and `5.0` on the active class-balanced FedZMG stack.
- Expected effect: suppress rare high-norm local optimizer steps from label-skewed clients without changing DIFF uploads, model keys, or `NUM_STEPS_CURRENT_ROUND`.

## Batch Outcome

- Client gradient clipping missed: `grad_clip_norm=1.0` scored 0.897200 and `5.0` scored 0.915200, both discarded.
- The default-off clipping code/schema additions were removed after review because even the looser norm remained below the 0.918600 high-water mark.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the literature reset.

## Architecture Subcampaign Outcome

- Registered architecture audit under the active optimizer stack missed: `moderate_cnn_norm` scored 0.911600 and `moderate_cnn_small_head` scored 0.911800.
- Both were marked `discard`; the original `moderate_cnn` remains the active architecture for this run.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with six scored candidates since the literature reset.

## Hypothesis: Nesterov Client Momentum

- Proposed change: add a default-off `--nesterov` client optimizer toggle and test it under the active class-balanced FedZMG stack.
- Expected effect: use the same momentum magnitude but look ahead in the local SGD update, which may help the current high-momentum client optimizer without changing the federated protocol.

## Batch Outcome

- Client Nesterov momentum scored 0.908600 and was marked `discard`.
- The default-off Nesterov code/schema additions were removed after review.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seven scored candidates since the literature reset.

## Literature Basis: Local Occlusion Augmentation

- Trigger: no clear non-duplicate safe local axis remained after the post-literature regularization, clipping, architecture, and optimizer-toggle nulls.
- Sources: DeVries and Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout" (2017, https://arxiv.org/abs/1708.04552); Zhong et al., "Random Erasing Data Augmentation" (2017, https://arxiv.org/abs/1708.04896); Yun et al., "CutMix" (2019, https://arxiv.org/abs/1905.04899).
- Hypothesis: label-preserving local occlusion may regularize class-skewed clients differently from failed target-mixing methods, while preserving the FL contract and avoiding `data/*` edits.
- Selected next candidates: active best stack plus `--cutout_size 8`, and active best stack plus `--cutout_size 12`. CutMix is rejected for now because local mixup already underfit.

## Batch Outcome

- First-pass Cutout missed the 0.918600 high-water mark: `cutout_size=8` scored 0.915600 and `cutout_size=12` scored 0.917500; both were marked `discard`.
- `cutout_size=12` is the closest post-literature result, so a narrow bracket at sizes 10 and 14 follows before deciding whether to keep or remove the default-off Cutout code.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the Cutout literature reset.
- Follow-up Cutout also missed: `cutout_size=10` scored 0.918400 and `cutout_size=14` scored 0.917000; both were marked `discard`.
- The default-off Cutout code/schema additions were removed after review because no mask size beat the 0.918600 high-water mark.
- Post-removal `PYTHON=.venv/bin/python make validate` and `make smoke` passed.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the Cutout literature reset.

## Literature Basis: Local Sharpness Minimization

- Trigger: watchdog still recommends `continue`, but a ledger scan found no clear non-duplicate scalar/local-compute axis after the Cutout branch missed and was removed.
- Sources: Qu et al., "Generalized Federated Learning via Sharpness Aware Minimization" (ICML 2022, https://proceedings.mlr.press/v162/qu22a.html); Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021, https://openreview.net/forum?id=6Tm1mposlrM); Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (2018, https://arxiv.org/abs/1803.05407).
- Hypothesis: client-local ERM on label-skewed sites is landing in sharp local minima; a default-off FedSAM-style `--sam_rho` knob can favor flatter local updates while preserving DIFF uploads, `NUM_STEPS_CURRENT_ROUND`, fixed model keys, data splits, and cross-site evaluation.
- Selected next candidates: active best stack plus `--sam_rho 0.02`, and active best stack plus `--sam_rho 0.05`; local SWA remains a reserve if SAM fails only on cost or shows a near-best signal.
- Validation: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger `--sam_rho 0.02` smoke passed before launching full candidates.

## Batch Outcome

- FedSAM `sam_rho=0.02` scored 0.914800 and was marked `discard`.
- FedSAM `sam_rho=0.05` scored 0.923900 and was marked `keep`, becoming the new active high-water mark.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` and reset on material improvement at row 427.
- SAM radius bracket missed: `sam_rho=0.04` scored 0.917800 and `sam_rho=0.06` scored 0.920700; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the SAM improvement.
- SAM local-compute bracket missed: `aggregation_epochs=6` scored 0.917900 and `aggregation_epochs=8` scored 0.918900 under `sam_rho=0.05`; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the SAM improvement.
- SAM client-LR bracket missed: `lr=0.04` scored 0.918800 and `lr=0.05` scored 0.922200; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with six scored candidates since the SAM improvement.
- SAM server-LR bracket missed: `server_lr=1.7` scored 0.920200 and `server_lr=1.9` scored 0.919700; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eight scored candidates since the SAM improvement.
- SAM server-momentum bracket missed: `server_momentum=0.45` scored 0.918200 and `server_momentum=0.5` scored 0.919700; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with ten scored candidates since the SAM improvement.
- SAM FedProx bracket missed: `fedproxloss_mu=1e-5` scored 0.921700 and `fedproxloss_mu=5e-5` scored 0.919300; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twelve scored candidates since the SAM improvement.
- SAM class-balanced beta bracket missed: `class_balanced_loss_beta=0.875` scored 0.923000 and `0.925` scored 0.918500; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fourteen scored candidates since the SAM improvement.
- SAM scheduler-floor bracket missed: `cosine_lr_eta_min_factor=0.0001` scored 0.922200 and `0.0002` scored 0.920400; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with sixteen scored candidates since the SAM improvement.
- SAM weight-decay bracket missed: `weight_decay=4e-4` scored 0.920100 and `6e-4` scored 0.918300; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eighteen scored candidates since the SAM improvement.
- SAM client-momentum bracket missed: `momentum=0.9125` scored 0.921700 and `0.9375` scored 0.919900; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty scored candidates since the SAM improvement.
- SAM zero-mean-gradient ablation scored 0.907400 and was marked `discard`; keep `--zero_mean_gradients` in the active SAM stack.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-one scored candidates since the SAM improvement.
- SAM class-balanced-loss ablation scored 0.918600 and was marked `discard`; keep `class_balanced_loss_beta=0.90` in the active SAM stack.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-two scored candidates since the SAM improvement.
- Tight SAM radius bracket missed: `sam_rho=0.0475` scored 0.917600 and `0.0525` scored 0.920900; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-four scored candidates since the SAM improvement.
- SAM exact-step bracket missed: `local_train_steps=704` scored 0.919200 and `768` scored 0.920500; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-six scored candidates since the SAM improvement.
- SAM aggregation-family check missed: plain FedAvg scored 0.901200 and median aggregation scored 0.883700; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-eight scored candidates since the SAM improvement.
- Upper SAM radius check missed: `sam_rho=0.075` scored 0.922900 and `0.1` scored 0.920600; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty scored candidates since the SAM improvement.
- Final pre-watchdog SAM checks missed: scheduler-off scored 0.763200 and SCAFFOLD mode scored 0.912100; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with thirty-two scored candidates since the SAM improvement.

## Literature Basis: Local Weight Averaging

- Trigger: watchdog plateau after the row 427 FedSAM improvement and thirty-two scored SAM follow-ups.
- Sources: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI 2018, https://arxiv.org/abs/1803.05407); Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019, https://papers.nips.cc/paper/9155-lookahead-optimizer-k-steps-forward-1-step-back); Foret et al., "Sharpness-Aware Minimization" (ICLR 2021, https://openreview.net/forum?id=6Tm1mposlrM).
- Hypothesis: after SAM improved the local optimizer, averaging late local epoch endpoints may further bias each client DIFF toward a flatter local solution without changing model keys, communication budget, data splits, or evaluation.
- Selected next candidates: active SAM stack plus `--local_swa_start_frac 0.5`, and active SAM stack plus `--local_swa_start_frac 0.75`.
- Validation: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger `--local_swa_start_frac 0.5` smoke passed before launching full candidates.

## Batch Outcome

- Local SWA missed: `local_swa_start_frac=0.5` scored 0.919300 and `0.75` scored 0.919900; both were marked `discard`.
- The default-off local SWA code/schema additions were removed after review because neither averaging window approached the 0.923900 high-water mark.
- Post-removal `PYTHON=.venv/bin/python make validate` and `make smoke` passed.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the local-SWA literature reset.
- SAM architecture subcampaign missed: `moderate_cnn_norm` scored 0.916000 and `moderate_cnn_small_head` scored 0.917800; both were marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the local-SWA literature reset.
- Conservative FedAdam under the kept FedSAM stack (`server_lr=0.1`, `fedopt_tau=0.01`) crashed with NaN local losses and `Diff norm is NaN or Inf`; keep the active FedAvgM server optimizer.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue`; the FedAdam crash did not add a scored candidate after the local-SWA literature reset.
- Client-LR fine bracket missed: `lr=0.0475` scored 0.917800 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with five scored candidates since the local-SWA literature reset.
- Class-balanced beta fine bracket missed: `class_balanced_loss_beta=0.8875` scored 0.919800 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with six scored candidates since the local-SWA literature reset.
- Exact-step local-compute check missed: `local_train_steps=896` scored 0.919200 and was marked `discard`; the epoch-based `aggregation_epochs=7` active stack remains better and faster.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seven scored candidates since the local-SWA literature reset.
- SAM radius fine bracket missed: `sam_rho=0.07` scored 0.919900 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eight scored candidates since the local-SWA literature reset.
- FedProx fine bracket missed: `fedproxloss_mu=2e-5` scored 0.916800 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nine scored candidates since the local-SWA literature reset.
- Client-momentum fine bracket missed: `momentum=0.91875` scored 0.920600 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with ten scored candidates since the local-SWA literature reset.
- Scheduler-floor fine bracket missed: `cosine_lr_eta_min_factor=0.000125` scored 0.918500 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eleven scored candidates since the local-SWA literature reset.
- Server-LR fine bracket missed: `server_lr=1.75` scored 0.917700 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twelve scored candidates since the local-SWA literature reset.
- Class-balanced beta lower fine bracket missed: `class_balanced_loss_beta=0.8625` scored 0.920100 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirteen scored candidates since the local-SWA literature reset.
- Client-LR high-side fine bracket missed: `lr=0.0525` scored 0.917400 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fourteen scored candidates since the local-SWA literature reset.
- Weight-decay fine bracket missed: `weight_decay=4.5e-4` scored 0.920300 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fifteen scored candidates since the local-SWA literature reset.
- Server-momentum fine bracket missed: `server_momentum=0.4875` scored 0.918100 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with sixteen scored candidates since the local-SWA literature reset.
- Server-LR upper fine bracket missed: `server_lr=1.85` scored 0.920700 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seventeen scored candidates since the local-SWA literature reset.
- Exact-step midpoint missed: `local_train_steps=832` scored 0.919500 and was marked `discard`; keep epoch-based `aggregation_epochs=7`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eighteen scored candidates since the local-SWA literature reset.
- Lower epoch-compute check missed: `aggregation_epochs=5` scored 0.919600 and was marked `discard`; epoch 7 remains the active local-compute setting despite the higher runtime.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nineteen scored candidates since the local-SWA literature reset.
- FedProx upper fine bracket missed: `fedproxloss_mu=4e-5` scored 0.921200 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty scored candidates since the local-SWA literature reset.
- Weight-decay upper fine bracket missed: `weight_decay=5.5e-4` scored 0.920200 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-one scored candidates since the local-SWA literature reset.
- Scheduler-floor upper fine bracket missed: `cosine_lr_eta_min_factor=0.000175` scored 0.920800 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-two scored candidates since the local-SWA literature reset.
- SAM radius upper fine bracket missed: `sam_rho=0.08` scored 0.921500 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-three scored candidates since the local-SWA literature reset.
- Server-LR narrow fine bracket missed: `server_lr=1.825` scored 0.919700 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-four scored candidates since the local-SWA literature reset.
- Client-momentum upper fine bracket missed: `momentum=0.93125` scored 0.920000 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-five scored candidates since the local-SWA literature reset.
- Server-momentum lower fine bracket missed: `server_momentum=0.4625` scored 0.919600 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-six scored candidates since the local-SWA literature reset.
- Exact-step lower check missed: `local_train_steps=640` scored 0.918700 and was marked `discard`; exact-step training is consistently below epoch-based `aggregation_epochs=7` under the kept SAM stack.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-seven scored candidates since the local-SWA literature reset.
- Lower epoch-compute bound missed: `aggregation_epochs=4` scored 0.912800 and was marked `discard`.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-eight scored candidates since the local-SWA literature reset.
- SAM radius upper shoulder check missed: `sam_rho=0.085` scored 0.922200 and was marked `discard`; the previous `sam_rho=0.05` high-water stack remains active.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-nine scored candidates since the local-SWA literature reset.
- Class-balanced beta upper shoulder missed: `class_balanced_loss_beta=0.95` scored 0.918800 and was marked `discard`; beta values above the active `0.90` have not improved the kept stack.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty scored candidates since the local-SWA literature reset.
- FedProx upper shoulder missed: `fedproxloss_mu=6e-5` scored 0.920100 and was marked `discard`; the active `3e-5` remains the best proximal setting under the kept FedSAM stack.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty-one scored candidates since the local-SWA literature reset.
- Client learning-rate lower shoulder missed: `lr=0.04375` scored 0.921800 and was marked `discard`; the active `0.045` remains better.
- `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with thirty-two scored candidates since the local-SWA literature reset, so the next batch should come from the literature loop rather than routine scalar jittering.

## Literature Reset: Aligned FedAvgM

The post-SAM plateau exhausted scalar jitter and the active aggregation-family checks already rejected plain FedAvg, median, SCAFFOLD, and FedAdam. The selected source-backed branch is a default-off `aligned_fedavgm` aggregator: compute the normal step-weighted round mean DIFF, score each client by cosine alignment with that direction, apply a conservative alignment floor, and then feed the aligned mean into the existing FedAvgM server-momentum update.

Sources:
- Rahil et al., "FedSCAM (Federated Sharpness-Aware Minimization with Clustered Aggregation and Modulation)", arXiv:2601.00853, https://arxiv.org/abs/2601.00853. Motivation: SAM under client heterogeneity can benefit from aggregation that prioritizes updates aligned with the global optimization direction.
- Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", ICML 2018, https://proceedings.mlr.press/v80/yin18a.html. Motivation: coordinate-wise robust aggregation families reduce sensitivity to outlier updates.
- Sun et al., "Dynamic Regularized Sharpness Aware Minimization in Federated Learning", ICML 2023, arXiv:2305.11584, https://arxiv.org/abs/2305.11584. Reserve idea: dynamic client regularization if server-only alignment misses.

Validation before launch: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger `--aggregator aligned_fedavgm` smoke passed.

Aligned FedAvgM missed: the full candidate scored 0.918300 and was marked `discard`, so the default-off `aligned_fedavgm` implementation was removed from `custom_aggregators.py`, `job.py`, and `mutation_schema.yaml`. Post-removal `PYTHON=.venv/bin/python make validate` and `make smoke` passed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with one scored candidate since the literature reset.

Reserve P2 is now active: a default-off `--fedproxloss_mu_schedule cosine_decay` knob was added to test Sun23-style dynamic client regularization without changing metadata or aggregation. Validation before launch: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger dynamic FedProx smoke passed.

Dynamic FedProx scheduling missed: `fedproxloss_mu=1e-4` with cosine decay scored 0.919000 and was marked `discard`. The default-off schedule implementation was removed from `client.py`, `job.py`, and `mutation_schema.yaml`; post-removal `PYTHON=.venv/bin/python make validate` and `make smoke` passed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the literature reset.

Near-miss combination missed: pairing `class_balanced_loss_beta=0.875` with `sam_rho=0.075` scored 0.920500 and was marked `discard`; the separately strong near-misses did not combine constructively. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with three scored candidates since the literature reset.

Beta plus scheduler-floor combination missed: pairing `class_balanced_loss_beta=0.875` with `cosine_lr_eta_min_factor=0.0001` scored 0.920800 and was marked `discard`; keep the active beta `0.90` and eta floor `0.00015`. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the literature reset.

Beta plus client-LR combination missed: pairing `class_balanced_loss_beta=0.875` with `lr=0.05` scored 0.919300 and was marked `discard`; the higher client LR did not rescue the lower beta under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with five scored candidates since the literature reset.

Client-LR plus scheduler-floor combination missed: pairing `lr=0.05` with `cosine_lr_eta_min_factor=0.0001` scored 0.921800 and was marked `discard`; the lower eta floor helped relative to the lower-beta LR combo but did not beat the active `lr=0.045`, eta `0.00015` stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with six scored candidates since the literature reset.

SAM-radius plus scheduler-floor combination missed: pairing `sam_rho=0.075` with `cosine_lr_eta_min_factor=0.0001` scored 0.920100 and was marked `discard`; the lower eta floor did not make the larger SAM radius competitive with the active `sam_rho=0.05`, eta `0.00015` stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seven scored candidates since the literature reset.

Local-compute epoch bracket missed: `aggregation_epochs=6` scored 0.917900 and was marked `discard`; the active `aggregation_epochs=7` stack remains materially better despite the extra runtime. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eight scored candidates since the literature reset.

Local-compute upper epoch bracket missed: `aggregation_epochs=8` scored 0.918900 and was marked `discard`; adding local epochs beyond the active `aggregation_epochs=7` increased runtime to 16.2m without improving accuracy. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nine scored candidates since the literature reset.

Class-balanced beta fine check missed: `class_balanced_loss_beta=0.9125` scored 0.917800 and was marked `discard`; the active beta `0.90` remains better under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with ten scored candidates since the literature reset.

Client learning-rate midpoint missed: `lr=0.044375` scored 0.921800 and was marked `discard`; it matched the lower-shoulder score and did not improve on the active `lr=0.045`. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eleven scored candidates since the literature reset.

Class-balanced beta midpoint missed: `class_balanced_loss_beta=0.8875` scored 0.919800 and was marked `discard`; the active beta `0.90` remains the best point in the lower-beta bracket. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twelve scored candidates since the literature reset.

SAM-radius lower bracket missed: `sam_rho=0.04` scored 0.917800 and was marked `discard`; reducing the radius below the active `0.05` hurt the kept FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirteen scored candidates since the literature reset.

Client learning-rate upper tight bracket missed: `lr=0.04625` scored 0.920900 and was marked `discard`; the active `lr=0.045` remains better than both nearby lower and upper checks. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fourteen scored candidates since the literature reset.

FedProx tight upper bracket missed: `fedproxloss_mu=3.5e-5` scored 0.919300 and was marked `discard`; the active `3e-5` remains better than the nearby upper proximal settings. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fifteen scored candidates since the literature reset.

Class-balanced beta upper tight bracket missed: `class_balanced_loss_beta=0.903125` scored 0.921400 and was marked `discard`; this improves over the wider upper check but remains below the active `0.90` beta stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with sixteen scored candidates since the literature reset.

Class-balanced beta lower-side bracket missed: `class_balanced_loss_beta=0.86875` scored 0.923200 and was marked `discard`; it is the strongest post-reset near miss but remains below the active `0.90` beta high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seventeen scored candidates since the literature reset.

Class-balanced beta lower-midpoint bracket missed: `class_balanced_loss_beta=0.871875` scored 0.921500 and was marked `discard`; the lower beta shoulder is noisy and did not interpolate above the `0.86875` near miss. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eighteen scored candidates since the literature reset.

SAM-radius midpoint missed: `sam_rho=0.07` scored 0.919900 and was marked `discard`; the earlier `0.075` near miss did not imply a smooth improvement between active `0.05` and the larger-radius shoulder. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nineteen scored candidates since the literature reset.

FedProx lower tight bracket missed: `fedproxloss_mu=2.5e-5` scored 0.921300 and was marked `discard`; the active `3e-5` remains better than nearby lower and upper proximal settings. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty scored candidates since the literature reset.

Class-balanced beta lower tight bracket missed: `class_balanced_loss_beta=0.865625` scored 0.921000 and was marked `discard`; the earlier `0.86875` near miss remains the best lower-beta check, while the active `0.90` beta high-water still holds. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-one scored candidates since the literature reset.

Server learning-rate midpoint missed: `server_lr=1.775` scored 0.920100 and was marked `discard`; the active FedAvgM server step size `1.8` remains better than the nearby lower-side bracket. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-two scored candidates since the literature reset.

Client learning-rate midpoint missed: `lr=0.04875` scored 0.921500 and was marked `discard`; it improved over the weak `0.0475` check but did not match the earlier `0.05` near-miss or the active `0.045` high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-three scored candidates since the literature reset.

Class-balanced beta active-lower bracket missed: `class_balanced_loss_beta=0.896875` scored 0.919600 and was marked `discard`; the tight lower-side check near active `0.90` fell below both the active high-water and the wider lower-beta near misses. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-four scored candidates since the literature reset.

Weight-decay midpoint missed: `weight_decay=4.75e-4` scored 0.920700 and was marked `discard`; the active `5e-4` remains better than both lower-side regularization checks and the prior upper bracket. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-five scored candidates since the literature reset.

Class-balanced beta lower near-miss bracket missed: `class_balanced_loss_beta=0.8703125` scored 0.919600 and was marked `discard`; the lower-beta shoulder remains noisy and below both the `0.86875` near miss and the active `0.90` high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-six scored candidates since the literature reset.

Server learning-rate upper bracket missed: `server_lr=1.875` scored 0.920600 and was marked `discard`; the active FedAvgM server step size `1.8` remains better than both nearby lower and upper-side checks under the kept FedSAM/class-balanced stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-seven scored candidates since the literature reset.

Class-balanced beta lower interpolation missed: `class_balanced_loss_beta=0.8671875` scored 0.921300 and was marked `discard`; it did not reproduce the stronger `0.86875` near miss and remains below the active `0.90` beta high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-eight scored candidates since the literature reset.

Client learning-rate upper interpolation missed: `lr=0.050625` scored 0.916200 and was marked `discard`; the high client-LR shoulder degrades quickly above the `0.05` near miss, so keep the active `lr=0.045` stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-nine scored candidates since the literature reset.

SAM-radius upper interpolation missed: `sam_rho=0.055` scored 0.916000 and was marked `discard`; the active `sam_rho=0.05` remains better than the tight upper interpolation and wider upper checks. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty scored candidates since the literature reset.

Server learning-rate tight interpolation missed: `server_lr=1.8125` scored 0.917500 and was marked `discard`; the active FedAvgM server step size `1.8` remains better than tight nearby upper-side checks. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty-one scored candidates since the literature reset.

Weight-decay upper interpolation missed: `weight_decay=5.125e-4` scored 0.917100 and was marked `discard`; the active `5e-4` regularization remains better than nearby upper and lower checks under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with thirty-two scored candidates since the literature reset.

## Literature Reset: Trajectory SAM

The post-FedSAM high-water plateau has now exhausted scalar radius, FedAvgM, FedProx, class-balanced beta, scheduler, weight-decay, and local-compute jitter. The selected source-backed branch is FedLESAM-style trajectory SAM: keep a client-local copy of the previously received global model and use the normalized difference from the current global model as the SAM perturbation direction. This changes only the client perturbation source; the NVFlare receive/send loop, strict state load, DIFF upload, and `NUM_STEPS_CURRENT_ROUND` metadata stay intact.

Sources:
- Fan et al., "Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization", ICML 2024, https://openreview.net/forum?id=6axTFAlzRV. Motivation: local SAM perturbations can disagree with the global loss landscape under heterogeneous FL; the paper estimates global perturbations from consecutive received global models and uses one backward pass.
- Li et al., "FedWMSAM: Fast and Flat Federated Learning via Weighted Momentum and Sharpness-Aware Minimization", NeurIPS 2025, https://openreview.net/forum?id=75JiIa0fU1. Motivation: FedAvgM plus SAM can suffer local-global curvature misalignment and momentum-echo oscillation; full momentum-guided perturbation is reserved because it would need server-to-client momentum state.
- Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks", ICML 2021, https://proceedings.mlr.press/v139/kwon21b.html. Reserve idea: adaptive perturbation scaling if the FL-specific trajectory branch is falsified.
- Qu et al., "Generalized Federated Learning via Sharpness Aware Minimization", ICML 2022, https://proceedings.mlr.press/v162/qu22a.html. Context: the active high-water stack already benefits from local FedSAM at `sam_rho=0.05`.

Proposal selected: add default-off `--sam_global_trajectory` and run the active best stack with `--sam_rho 0.05 --sam_global_trajectory`. If it is stable but misses, reserve a trajectory radius variant at `sam_rho=0.075`; otherwise fall back to late-SAM or ASAM only after this FL-specific perturbation branch is falsified.

Validation before launch: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger two-round `--sam_global_trajectory` smoke passed. The targeted smoke confirmed the default-off flag parsed on clients, used the local SAM fallback on round 0, and completed after a round with previous global weights available.

Trajectory SAM missed: `--sam_global_trajectory` with `sam_rho=0.05` scored 0.915400 in 510 seconds and was marked `discard`, well below both the local-SAM high-water and recent near misses. The source-backed trajectory branch was therefore falsified for this stack; remove the default-off knob and keep the simpler local FedSAM implementation. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with one scored candidate since the trajectory-SAM literature reset.

Reserve P3 promoted after the FedLESAM miss: add default-off `--sam_rho_schedule late_cosine`, which ramps the existing local SAM radius from zero to the configured `--sam_rho` over communication rounds. This tests the FedWMSAM phase-mismatch hypothesis without adding server momentum state. Candidate planned: active stack with `--sam_rho 0.10 --sam_rho_schedule late_cosine`, so the average radius is near the active `0.05` while late rounds receive stronger flatness pressure. Validation before launch: `make validate`, `make smoke`, and a no-ledger two-round late-cosine smoke passed; the targeted smoke logged `effective_sam_rho=0.000000` at round 0 and the configured max at round 1.

Late-cosine SAM scheduling missed: `sam_rho=0.10` with `sam_rho_schedule=late_cosine` scored 0.920000 in 846 seconds and was marked `discard`. It stayed stable but did not beat the active constant `sam_rho=0.05` stack or the prior constant `0.075` near miss, so the default-off schedule knob was removed from `client.py`, `job.py`, and `mutation_schema.yaml`. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with two scored candidates since the trajectory-SAM literature reset.

Reserve P4 promoted after FedLESAM and late-SAM scheduling both missed: add default-off `--sam_adaptive_scale` with `--sam_adaptive_eta 0.01`, an ASAM-style parameter-magnitude perturbation scale. Candidate planned: active stack with `--sam_rho 0.10 --sam_adaptive_scale --sam_adaptive_eta 0.01`, testing whether scale-invariant perturbations can use a stronger radius without the constant-SAM `0.10` regression. Validation before launch: `make validate`, `make smoke`, and a no-ledger two-round ASAM smoke passed; the targeted smoke logged `sam_adaptive_scale=True` on both clients.

ASAM adaptive scaling missed: `sam_rho=0.10 --sam_adaptive_scale --sam_adaptive_eta 0.01` scored 0.915500 in 906 seconds and was marked `discard`. It underperformed the active constant `sam_rho=0.05` stack and did not improve on the prior late-cosine miss, so the default-off ASAM parser, forwarding, schema, and perturbation scaling code were removed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with three scored candidates since the trajectory-SAM literature reset.

Weighted aggregation audit missed: replacing FedAvgM with step-weighted FedAvg under the kept FedSAM client stack scored 0.901700 in 858 seconds and was marked `discard`. This confirms the server momentum branch remains essential after adding SAM and class-balanced loss; do not spend further routine sweeps on non-momentum FedAvg variants for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with four scored candidates since the trajectory-SAM literature reset.

Near-miss shoulder combination missed: `lr=0.05` with `class_balanced_loss_beta=0.86875` under the kept FedSAM stack scored 0.917500 in 861 seconds and was marked `discard`. The two strongest single-axis shoulders did not compose; keep the active `lr=0.045` and `class_balanced_loss_beta=0.90` pairing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with five scored candidates since the trajectory-SAM literature reset.

Class-balanced beta tight interpolation missed: `class_balanced_loss_beta=0.86953125` scored 0.920800 in 864 seconds and was marked `discard`. It did not recover the `0.86875` near miss and remains below the active beta `0.90`, so the lower-beta shoulder is not worth further routine interpolation. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with six scored candidates since the trajectory-SAM literature reset.

Client learning-rate tight upper interpolation missed: `lr=0.049375` scored 0.919800 in 864 seconds and was marked `discard`. This is below both the earlier `lr=0.05` near miss and the active `lr=0.045` high-water, so the upper client-LR shoulder remains noisy rather than smoothly improving. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seven scored candidates since the trajectory-SAM literature reset.

Server momentum tight lower interpolation missed: `server_momentum=0.46875` scored 0.922400 in 861 seconds and was marked `discard`. This is the strongest post-trajectory-reset local interpolation so far but remains below the active `0.475` high-water, so the FedAvgM server-momentum ridge still peaks at the kept setting. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eight scored candidates since the trajectory-SAM literature reset.

Server momentum lower-mid interpolation missed: `server_momentum=0.471875` scored 0.920000 in 861 seconds and was marked `discard`. It failed to interpolate between the `0.46875` near miss and active `0.475` high-water, so further routine tightening on this momentum edge is not justified. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nine scored candidates since the trajectory-SAM literature reset.

Server momentum upper interpolation missed: `server_momentum=0.478125` scored 0.921700 in 861 seconds and was marked `discard`. The active `0.475` remains better than nearby lower and upper momentum checks under the kept FedSAM/FedAvgM stack, so the server-momentum ridge is now locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with ten scored candidates since the trajectory-SAM literature reset.

Lower-beta/server-momentum combination missed: pairing `class_balanced_loss_beta=0.86875` with `server_momentum=0.46875` scored 0.920700 in 864 seconds and was marked `discard`. The strongest lower-beta shoulder did not benefit from the lower FedAvgM momentum near miss, so keep the active `class_balanced_loss_beta=0.90` and `server_momentum=0.475` pairing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eleven scored candidates since the trajectory-SAM literature reset.

SAM-radius tight lower bracket missed: `sam_rho=0.045` scored 0.921000 in 864 seconds and was marked `discard`. The active local SAM radius `0.05` remains better than nearby lower checks (`0.045`, `0.04`) and the upper-side checks, so the radius is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twelve scored candidates since the trajectory-SAM literature reset.

Class-balanced beta mid-lower interpolation missed: `class_balanced_loss_beta=0.88125` scored 0.916300 in 865 seconds and was marked `discard`. The active `0.90` beta remains better than the lower-beta shoulder and its interpolations under the kept FedSAM/FedAvgM stack, so do not spend more routine sweeps on class-balanced beta tightening. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirteen scored candidates since the trajectory-SAM literature reset.

Weight-decay tight lower interpolation missed: `weight_decay=4.875e-4` scored 0.920000 in 864 seconds and was marked `discard`. The active `5e-4` regularization remains better than both lower-side checks (`4.75e-4`, `4.875e-4`) and the upper-side `5.125e-4` check under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fourteen scored candidates since the trajectory-SAM literature reset.

Weight-decay tight upper interpolation missed: `weight_decay=5.0625e-4` scored 0.919400 in 859 seconds and was marked `discard`. The active `5e-4` setting is now locally bracketed on both sides (`4.875e-4`, `5.0625e-4`, and the wider checks), so further routine weight-decay tightening is not justified. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with fifteen scored candidates since the trajectory-SAM literature reset.

Client-momentum tight lower interpolation missed: `momentum=0.9234375` scored 0.922300 in 863 seconds and was marked `discard`. It is the best recent client-momentum interpolation but remains below the active `0.925` high-water stack, so only a symmetric upper-side check is worth considering before treating client momentum as locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with sixteen scored candidates since the trajectory-SAM literature reset.

Client-momentum tight upper interpolation missed: `momentum=0.9265625` scored 0.918700 in 861 seconds and was marked `discard`. Together with the lower-side `0.9234375` miss and wider momentum checks, this locally brackets the active `0.925` setting under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with seventeen scored candidates since the trajectory-SAM literature reset.

FedProx lower-mu interpolation missed: `fedproxloss_mu=2.75e-5` scored 0.921100 in 864 seconds and was marked `discard`. It did not improve over the active `3e-5` setting, and the prior `2.5e-5` and upper-side checks also remain below the high-water stack, so FedProx strength is locally bracketed enough for routine search. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with eighteen scored candidates since the trajectory-SAM literature reset.

Server-learning-rate tight lower interpolation missed: `server_lr=1.7875` scored 0.919200 in 858 seconds and was marked `discard`. It underperformed the active `1.8` high-water and the prior lower-side `1.775` check, so the useful FedAvgM server-LR ridge still centers on the kept value. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with nineteen scored candidates since the trajectory-SAM literature reset.

Server-learning-rate tight upper interpolation missed: `server_lr=1.80625` scored 0.917200 in 858 seconds and was marked `discard`. Combined with the lower-side `1.7875` miss and prior wider checks, this brackets the active `1.8` server learning rate under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty scored candidates since the trajectory-SAM literature reset.

Exact local-step compute revisit missed: `local_train_steps=928` scored 0.917900 in 1003 seconds and was marked `discard`. It stayed within `RUN_TIMEOUT_SECONDS=1200` but did not improve over the epoch-based `aggregation_epochs=7` high-water, so exact-step local compute is not a promising follow-up unless a source-backed reason appears. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-one scored candidates since the trajectory-SAM literature reset.

Large-batch local-noise reduction missed: `batch_size=128` scored 0.914100 in 510 seconds and was marked `discard`. The run was much cheaper than the active `batch_size=64` high-water but lost too much accuracy, so larger batches are a cost-only trade-off for this kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-two scored candidates since the trajectory-SAM literature reset.

Lower server/client momentum pairing missed: `server_momentum=0.46875` with client `momentum=0.9234375` scored 0.916900 in 858 seconds and was marked `discard`. The two strongest momentum-side near misses did not compose, so keep the active `server_momentum=0.475` and client `momentum=0.925` under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-three scored candidates since the trajectory-SAM literature reset.

Higher-SAM/lower-server-momentum pairing missed: `sam_rho=0.075` with `server_momentum=0.46875` scored 0.922500 in 864 seconds and was marked `discard`. This is a strong near miss but still below the kept `sam_rho=0.05`, `server_momentum=0.475` high-water, so lower FedAvgM inertia does not rescue the higher local-SAM radius. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-four scored candidates since the trajectory-SAM literature reset.

Higher-SAM/lower-client-momentum pairing missed: `sam_rho=0.075` with client `momentum=0.9234375` scored 0.918900 in 861 seconds and was marked `discard`. Lower local optimizer momentum did not stabilize the higher SAM radius and underperformed both single-axis near misses, so keep the active `sam_rho=0.05` and client `momentum=0.925` pairing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-five scored candidates since the trajectory-SAM literature reset.

Lower-server-momentum/stronger-client-LR pairing missed: `server_momentum=0.46875` with client `lr=0.05` scored 0.919200 in 858 seconds and was marked `discard`. Lower server inertia did not rescue the more aggressive client step, so keep the active `server_momentum=0.475` and `lr=0.045` under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-six scored candidates since the trajectory-SAM literature reset.

Lower-server-momentum/lower-cosine-tail pairing missed: `server_momentum=0.46875` with `cosine_lr_eta_min_factor=0.0001` scored 0.918500 in 858 seconds and was marked `discard`. The colder late local schedule did not help the lower FedAvgM momentum shoulder and underperformed the active `eta_min_factor=0.00015`, so keep the current server-momentum and cosine-tail pairing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-seven scored candidates since the trajectory-SAM literature reset.

Lower-server-momentum/lower-FedProx pairing missed: `server_momentum=0.46875` with `fedproxloss_mu=1e-5` scored 0.922400 in 864 seconds and was marked `discard`. This matched the single-axis lower-server-momentum shoulder but did not improve it, so reducing client proximal regularization does not explain the remaining gap to the active `server_momentum=0.475`, `fedproxloss_mu=3e-5` high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-eight scored candidates since the trajectory-SAM literature reset.

Higher-SAM/lower-FedProx pairing missed: `sam_rho=0.075` with `fedproxloss_mu=1e-5` scored 0.920900 in 858 seconds and was marked `discard`. Lower proximal regularization did not stabilize or improve the higher local-SAM radius, so keep the active `sam_rho=0.05`, `fedproxloss_mu=3e-5` pairing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with twenty-nine scored candidates since the trajectory-SAM literature reset.

Stronger-client-LR/lower-FedProx pairing missed: client `lr=0.05` with `fedproxloss_mu=1e-5` scored 0.920200 in 858 seconds and was marked `discard`. Lower proximal regularization did not rescue the more aggressive client learning rate, so keep the active `lr=0.045`, `fedproxloss_mu=3e-5` pairing under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty scored candidates since the trajectory-SAM literature reset.

Lower-server-momentum/beta-0.875 pairing missed: `server_momentum=0.46875` with `class_balanced_loss_beta=0.875` scored 0.919500 in 860 seconds and was marked `discard`. The lower beta shoulder did not compose with reduced FedAvgM server inertia, so keep the active `server_momentum=0.475`, `class_balanced_loss_beta=0.90` pairing under the kept FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with thirty-one scored candidates since the trajectory-SAM literature reset.

Lower-server-momentum/slightly-lower-FedProx pairing missed: `server_momentum=0.46875` with `fedproxloss_mu=2.5e-5` scored 0.922400 in 861 seconds and was marked `discard`. This again matched the lower-server-momentum shoulder but did not improve it, so the local scalar search around FedAvgM inertia and FedProx strength is exhausted for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with thirty-two scored candidates since the trajectory-SAM literature reset.

## Literature Reset: Client Forgetting Controls

The post-trajectory-reset plateau has exhausted scalar and near-miss pairings around the kept FedSAM/FedAvgM stack. The selected source-backed branch is Federated Not-True Distillation: add a default-off client-local KL term against the received global model, but mask the true class so the teacher preserves global off-label knowledge without directly fighting local cross-entropy.

Sources:
- Lee et al., "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning", NeurIPS 2022, https://arxiv.org/abs/2106.03097. Motivation: local training under non-IID data induces forgetting outside each client's distribution; FedNTD preserves the global perspective on not-true classes without extra communication.
- Song et al., "FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning", 2024, https://arxiv.org/abs/2404.09210. Support: imbalanced local data can bias local models and cause local forgetting, especially for underrepresented classes.
- Yan et al., "Rethinking Client Drift in Federated Learning: A Logit Perspective", 2023, https://arxiv.org/abs/2308.10162. Support: local/global logit differences can grow during training under non-IID data, motivating a targeted logit-level alignment term.
- Li et al., "Model-Contrastive Federated Learning", CVPR 2021, https://arxiv.org/abs/2103.16257. Reserve idea: representation-level alignment if masked logit distillation misses.
- Acar et al., "Federated Learning Based on Dynamic Regularization", ICLR 2021, https://arxiv.org/abs/2111.04263. Reserve idea: dynamic client regularization if forgetting controls miss cleanly.

Proposal selected: add default-off `--fedntd_beta` and `--fedntd_temperature` in `client.py`/`job.py`, record the bounds in `mutation_schema.yaml`, and run the active best stack with `--fedntd_beta 0.05 --fedntd_temperature 2.0`. This differs from the earlier failed full global distillation branch because it masks the true class and only preserves non-true global knowledge.

Validation before launch: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger two-round FedNTD smoke passed. The targeted smoke confirmed the default-off flags were forwarded to both clients and completed cross-site evaluation without writing to `results.tsv`.

Reviewed candidate: `fedntd_beta=0.05`, `fedntd_temperature=2.0` under the kept FedSAM/FedAvgM stack scored 0.920300 in 966s and was marked `discard`. The run was stable but missed the 0.923900 high-water, consistent with over-regularization from the not-true teacher signal. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=1`, so the reserved lower-weight FedNTD candidate is the next source-backed follow-up before removing or abandoning this branch.

Reviewed follow-up: `fedntd_beta=0.02`, `fedntd_temperature=2.0` scored 0.920000 in 966s and was marked `discard`. The lower teacher weight did not improve on P1 or the 0.923900 high-water, so FedNTD-style masked distillation is treated as falsified for this stack. The default-off FedNTD CLI, client loss, and schema bounds were removed before continuing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=2`.

Reserve promoted: add default-off `--feddyn_alpha` for a FedDyn-lite local dynamic regularizer from Acar21. The implementation keeps client-side correction state only, normalizes the dynamic term by parameter count, and preserves the existing DIFF send path and metadata. Validation before launch: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger two-round FedDyn smoke passed; the targeted smoke confirmed `feddyn_alpha=0.01` was forwarded and `feddyn_state_norm` updated on both clients. Candidate planned: active FedSAM/FedAvgM stack plus `--feddyn_alpha 0.01`.

Reviewed reserve: FedDyn-lite `feddyn_alpha=0.01` reached final evaluation with `site-1` accuracy 0.921500, but `run_iteration.sh` recorded it as `crash` because runtime exceeded `RUN_TIMEOUT_SECONDS=1200` immediately after evaluation. The observed score was below the 0.923900 high-water and the overhead violated the budget, so the default-off FedDyn-lite client/job/schema code was removed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=2`; the crash row does not advance the scored plateau counter.

Reserve promoted: add `fedyogi`, a conservative FedOpt/Yogi server aggregator from Reddi21, with the same DIFF payload and client contract as existing FedAdam/FedAvgM. The update uses Yogi's signed second-moment correction instead of Adam's EMA. Validation before launch: `PYTHON=.venv/bin/python make validate`, `make smoke`, and a no-ledger two-round FedYogi smoke passed; the targeted smoke confirmed `Using FedYogiAggregator (server_lr=0.2, beta1=0.9, beta2=0.99, tau=0.1)`. Candidate planned: active client stack with `--aggregator fedyogi --server_lr 0.2 --fedopt_tau 0.1`.

Reviewed reserve: FedYogi `server_lr=0.2`, `fedopt_tau=0.1` under the active FedSAM stack scored 0.857200 in 864s and was marked `discard`. The Yogi second-moment variant avoided FedAdam's NaN crash but collapsed score quality, so the adaptive FedOpt family remains rejected for this campaign. The `fedyogi` aggregator/job/schema support was removed before continuing. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=3`.

Local continuation: the non-duplicate near-miss pair `class_balanced_loss_beta=0.86875` plus `sam_rho=0.075` scored 0.922200 in 861s and was marked `discard`. This underperformed both single-axis near misses (`0.923200` beta-only and `0.922900` SAM-only), so do not retry this pair or adjacent `0.875 + 0.075` without a new mechanism. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=4`.

Local continuation: the single-axis SAM interpolation `sam_rho=0.0775` scored 0.921900 in 864s and was marked `discard`. It underperformed the existing `sam_rho=0.075` shoulder at 0.922900 and `sam_rho=0.085` at 0.922200, so the upper SAM shoulder is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=5`.

Local continuation: the single-axis class-balanced interpolation `class_balanced_loss_beta=0.868359375` scored 0.919400 in 865s and was marked `discard`. This is much worse than the existing `0.86875` near-miss at 0.923200, so the lower side of the class-balanced beta peak is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=6`.

Local continuation: the single-axis client momentum interpolation `momentum=0.92421875` scored 0.919400 in 868s and was marked `discard`. It underperformed both the kept `0.925` setting and the earlier `0.9234375` near-miss, so this tight lower shoulder is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=7`.

Local continuation: the single-axis server momentum interpolation `server_momentum=0.4765625` scored 0.922400 in 858s and was marked `discard`. It matched the lower-side `0.46875` score and remained below the 0.923900 high-water, so the immediate upper side of the server-momentum peak is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=8`.

Local continuation: the single-axis server learning-rate interpolation `server_lr=1.796875` scored 0.922100 in 865s and was marked `discard`. It remained below the 0.923900 high-water and below the active `server_lr=1.8` setting, so the immediate lower side of the server-lr peak is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=9`.

Local continuation: the single-axis class-balanced interpolation `class_balanced_loss_beta=0.89375` scored 0.923000 in 861s and was marked `discard`. It tied the earlier `0.875` shoulder but stayed below the 0.923900 high-water, so the upper side between `0.8875` and the kept `0.90` remains a near-miss bracket rather than a replacement. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=10`.

Local continuation: the single-axis server-momentum interpolation `server_momentum=0.4734375` scored 0.921600 in 861s and was marked `discard`. It underperformed both the kept `0.475` setting and the earlier lower `0.46875` near-miss, so the lower side around the FedAvgM momentum peak is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=11`.

Local continuation: the single-axis server learning-rate interpolation `server_lr=1.8015625` scored 0.920800 in 860s and was marked `discard`. It underperformed the kept `1.8` setting and the prior lower-side `1.796875` miss, so the immediate upper side of the server-lr peak is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=12`.

Local continuation: the single-axis SAM interpolation `sam_rho=0.0725` scored 0.925200 in 867s and was marked `keep`, improving the high-water from 0.923900. This replaces `sam_rho=0.05` as the active radius while preserving the same FedAvgM, class-balanced, FedProx, zero-mean, and cosine settings. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=0` after the material-improvement reset.

Local continuation: the immediate upper SAM neighbor `sam_rho=0.07375` scored 0.921400 in 861s and was marked `discard`. It fell below both the new kept `0.0725` high-water and the prior `0.075` shoulder, so the upper side of the new SAM peak is locally bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=1`.

Local continuation: the immediate lower SAM neighbor `sam_rho=0.07125` scored 0.920300 in 864s and was marked `discard`. It underperformed the new kept `0.0725` high-water and the earlier lower-side `0.07` miss, so both sides around the new SAM peak are bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=2`.

Local continuation: the lower class-balanced coefficient `class_balanced_loss_beta=0.89375` under the new kept `sam_rho=0.0725` stack scored 0.921700 in 864s and was marked `discard`. It failed to reproduce the earlier lower-beta near miss once paired with the higher SAM radius, so the active `class_balanced_loss_beta=0.90` remains preferred for the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=3`.

Local continuation: the upper class-balanced coefficient `class_balanced_loss_beta=0.903125` under the new kept `sam_rho=0.0725` stack scored 0.921500 in 864s and was marked `discard`. With the lower `0.89375` check also missing, the class-balanced beta axis is bracketed around the active `0.90` value for this high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=4`.

Local continuation: the stronger FedProx coefficient `fedproxloss_mu=3.25e-5` under the new kept `sam_rho=0.0725` stack scored 0.920200 in 864s and was marked `discard`. The higher proximal strength did not stabilize the larger SAM radius and instead regressed well below the high-water, so the active `fedproxloss_mu=3e-5` remains preferred on the upper side. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=5`.

Local continuation: the weaker FedProx coefficient `fedproxloss_mu=2.75e-5` under the new kept `sam_rho=0.0725` stack scored 0.920500 in 864s and was marked `discard`. Together with the `3.25e-5` miss, this brackets the active `fedproxloss_mu=3e-5` setting around the kept SAM high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=6`.

Local continuation: the upper FedAvgM server momentum `server_momentum=0.4765625` under the new kept `sam_rho=0.0725` stack scored 0.918400 in 864s and was marked `discard`. The higher server inertia regressed sharply, so the active `server_momentum=0.475` remains preferred on the upper side for this high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=7`.

Local continuation: the lower FedAvgM server momentum `server_momentum=0.4734375` under the new kept `sam_rho=0.0725` stack scored 0.919500 in 861s and was marked `discard`. Together with the upper `0.4765625` miss, this brackets the active `server_momentum=0.475` value for the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=8`.

Local continuation: the lower FedAvgM server learning rate `server_lr=1.796875` under the new kept `sam_rho=0.0725` stack scored 0.922000 in 861s and was marked `discard`. It underperformed the active `server_lr=1.8` high-water, so the immediate lower side of the server-lr axis is locally bracketed for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=9`.

Local continuation: the upper FedAvgM server learning rate `server_lr=1.8015625` under the new kept `sam_rho=0.0725` stack scored 0.920600 in 863s and was marked `discard`. Together with the lower `1.796875` miss, this brackets the active `server_lr=1.8` setting around the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=10`.

Local continuation: the lower client learning rate `lr=0.044375` under the new kept `sam_rho=0.0725` stack scored 0.918700 in 866s and was marked `discard`. The lower client step size regressed sharply, so the active `lr=0.045` remains preferred on the lower side for this high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=11`.

Local continuation: the upper client learning rate `lr=0.045625` under the new kept `sam_rho=0.0725` stack scored 0.920300 in 864s and was marked `discard`. Together with the lower `0.044375` miss, this brackets the active `lr=0.045` client step size around the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=12`.

Local continuation: the lower client momentum `momentum=0.9234375` under the new kept `sam_rho=0.0725` stack scored 0.921200 in 867s and was marked `discard`. It failed to beat the active `momentum=0.925` setting, so the lower side of the client-momentum axis remains worse for this high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=13`.

Local continuation: the upper client momentum `momentum=0.9265625` under the new kept `sam_rho=0.0725` stack scored 0.920700 in 858s and was marked `discard`. Together with the lower `0.9234375` miss, this brackets the active `momentum=0.925` client inertia for the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=14`.

Local continuation: the lower cosine floor `cosine_lr_eta_min_factor=0.0001` under the new kept `sam_rho=0.0725` stack scored 0.922500 in 860s and was marked `discard`. It improved on the recent momentum misses but stayed below the active `0.00015` high-water, so the lower side of the cosine-floor axis remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=15`.

Local continuation: the upper cosine floor `cosine_lr_eta_min_factor=0.0002` under the new kept `sam_rho=0.0725` stack scored 0.920000 in 858s and was marked `discard`. Together with the lower `0.0001` miss, this brackets the active `0.00015` cosine floor around the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=16`.

Local continuation: the lower weight decay `weight_decay=4.75e-4` under the new kept `sam_rho=0.0725` stack scored 0.922300 in 864s and was marked `discard`. It stayed below the active `5e-4` high-water, so the lower side of the weight-decay axis remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=17`.

Local continuation: the upper weight decay `weight_decay=5.25e-4` under the new kept `sam_rho=0.0725` stack scored 0.921000 in 865s and was marked `discard`. Together with the lower `4.75e-4` miss, this brackets the active `5e-4` weight decay for the current high-water stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=18`.

Local continuation: the lower local-compute budget `aggregation_epochs=6` under the new kept `sam_rho=0.0725` stack scored 0.919700 in 746s and was marked `discard`. It stayed below the active `aggregation_epochs=7` high-water despite the shorter runtime, so the lower epoch budget remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=19`.

Local continuation: the upper local-compute budget `aggregation_epochs=8` under the new kept `sam_rho=0.0725` stack scored 0.922000 in 966s and was marked `discard`. It improved over the lower `aggregation_epochs=6` miss but remained below the active `aggregation_epochs=7` high-water, so both sides of the epoch-count axis are bracketed for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=20`.

Local continuation: the exact local-compute representation `local_train_steps=704` under the new kept `sam_rho=0.0725` stack scored 0.920000 in 780s and was marked `discard`. Matching the approximate seven-epoch step budget with exact local steps did not preserve the epoch-based high-water, so the active `local_train_steps=0` epoch-mode training remains preferred at this compute level. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=21`.

Local continuation: the upper exact local-compute check `local_train_steps=768` under the new kept `sam_rho=0.0725` stack scored 0.921800 in 851s and was marked `discard`. It recovered part of the gap from the 704-step exact run but still trailed the epoch-based `aggregation_epochs=7` high-water, so exact-step training remains inferior for this stack so far. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=22`.

Local continuation: the upper exact-step continuation `local_train_steps=832` under the new kept `sam_rho=0.0725` stack scored 0.919500 in 906s and was marked `discard`. The score dropped from the 768-step check while runtime increased, so the exact-step local-compute sweep is bracketed below epoch-mode training for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=23`.

Local continuation: the combined lower cosine floor and lower weight decay candidate `cosine_lr_eta_min_factor=0.0001`, `weight_decay=4.75e-4` under the new kept `sam_rho=0.0725` stack scored 0.921400 in 864s and was marked `discard`. Combining the two best recent near-miss directions regressed below their individual checks, so this interaction does not improve on the active `0.00015` cosine floor and `5e-4` weight decay. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=24`.

Local continuation: the combined lower cosine floor and lower server learning rate candidate `cosine_lr_eta_min_factor=0.0001`, `server_lr=1.796875` under the new kept `sam_rho=0.0725` stack scored 0.919000 in 864s and was marked `discard`. Pairing the lower scheduler floor with the best recent lower server-lr near miss regressed below both individual checks, so this interaction is not useful for the current FedSAM/FedAvgM stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=25`.

Local continuation: the combined lower weight decay and lower server learning rate candidate `weight_decay=4.75e-4`, `server_lr=1.796875` under the new kept `sam_rho=0.0725` stack scored 0.922800 in 861s and was marked `discard`. This was the best recent interaction check but still remained below the active 0.925200 high-water and below the 0.925700 material-improvement threshold, so the active `weight_decay=5e-4` and `server_lr=1.8` pair remains preferred. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=26`.

Local continuation: the `aggregation_epochs=8` follow-up on the lower server learning rate and lower weight decay near-miss stack scored 0.919700 in 978s and was marked `discard`. Extra epoch-based local compute did not rescue the 0.922800 near miss and instead regressed while consuming more runtime, so `aggregation_epochs=7` remains the better local-compute budget for this interaction. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=27`.

Local continuation: the lower class-balanced loss beta follow-up `class_balanced_loss_beta=0.89375` on the lower server learning rate and lower weight decay near-miss stack scored 0.921600 in 866s and was marked `discard`. Reducing the class-balancing strength did not improve the 0.922800 interaction near miss, so the active `class_balanced_loss_beta=0.90` remains preferred for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=28`.

Local continuation: the upper cosine floor follow-up `cosine_lr_eta_min_factor=0.0002` on the lower server learning rate and lower weight decay near-miss stack scored 0.920000 in 865s and was marked `discard`. Raising the late-round scheduler floor did not improve the 0.922800 interaction near miss and matched the earlier upper-floor miss on the active high-water stack, so the active `0.00015` floor remains preferred. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=29`.

Local continuation: the upper SAM-radius follow-up `sam_rho=0.07375` on the lower server learning rate and lower weight decay near-miss stack scored 0.918600 in 864s and was marked `discard`. Increasing the SAM radius on this already-regularized interaction sharply regressed, so the active `sam_rho=0.0725` remains preferred and the upper SAM side is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=30`.

Local continuation: the lower SAM-radius follow-up `sam_rho=0.07125` on the lower server learning rate and lower weight decay near-miss stack scored 0.920900 in 867s and was marked `discard`. Together with the upper `sam_rho=0.07375` miss, this brackets the active `sam_rho=0.0725` setting for this interaction stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=31`.

Local continuation: the lower FedProx follow-up `fedproxloss_mu=2.75e-5` on the lower server learning rate and lower weight decay near-miss stack scored 0.921200 in 867s and was marked `discard`. Reducing the proximal term did not improve the 0.922800 interaction near miss and stayed below the active high-water stack, so `fedproxloss_mu=3e-5` remains preferred here. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=literature` with `scored_since_reset=32`, so local hyperparameter jittering stops until a source-backed candidate is selected.

Literature loop: the 2026-05-11 plateau review selected a gradient-norm-penalty SAM blend as the next low-risk source-backed branch. Zhao22 GNP (ICML 2022, https://proceedings.mlr.press/v162/zhao22i.html) frames SAM as the `alpha=1` endpoint of a vanilla/SAM gradient blend; Sun23 FedSpeed (ICLR 2023, https://openreview.net/forum?id=bZjxxYURKT) uses a blended perturbation-gradient idea for heterogeneous FL local overfitting; Xu24 FedGAM (Mathematics 2024, https://www.mdpi.com/2227-7390/12/17/2644) supports first-order flatness as a client-drift/generalization mechanism. The selected candidate will add default-off `--sam_blend` and run the active 0.925200 FedSAM/FedAvgM stack with `--sam_blend 0.5`, avoiding another scalar `sam_rho` or FedProx jitter.

Implementation note: added default-off `--sam_blend` to blend the first vanilla gradient with the perturbed SAM gradient when `sam_rho > 0`; the default `1.0` preserves existing pure-SAM behavior. `make validate`, `make smoke`, and a no-ledger `--sam_rho 0.01 --sam_blend 0.5` smoke all passed before the full source-backed candidate launch.

Literature candidate outcome: the P1 gradient-norm SAM blend `--sam_blend 0.5` on the active FedSAM/FedAvgM stack scored 0.922100 in 879s and was marked `discard`. The implementation was stable but the lower blend weakened the high-water pure-SAM behavior, so the P2 reserve `--sam_blend 0.75` is promoted next before removing the default-off blend code.

Literature candidate outcome: the P2 gradient-norm SAM blend `--sam_blend 0.75` on the active FedSAM/FedAvgM stack scored 0.919200 in 879s and was marked `discard`. Together with the P1 0.5 miss, this falsifies vanilla/SAM gradient blending for the current high-water stack; the default-off `sam_blend` implementation was removed from `client.py`, `job.py`, and `mutation_schema.yaml`. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=2`.

Implementation note: promoted the P3 FedExP reserve from the 2026-05-11 literature loop. Added default-off `--aggregator fedexp` as a server-only adaptive extrapolation step over already-received weighted DIFFs, using `--server_lr` as the maximum step cap and `--fedexp_epsilon` as the denominator stabilizer. This preserves the client receive/train/diff/send contract and changes only `custom_aggregators.py`, `job.py`, and `mutation_schema.yaml`.

Literature candidate outcome: the P3 FedExP cap-3.0 candidate crashed after 167s, with NaN losses across clients by round 3 and no comparable cross-site score. The ledger row is marked `crash`; `scripts/plateau_watchdog.py results.tsv` returned `recommendation=continue` with `scored_since_reset=2`. Because the failure matches excessive extrapolation rather than a protocol break, run one conservative cap-1.8 reserve before removing the default-off `fedexp` branch.

Literature candidate outcome: the FedExP cap-1.8 reserve completed stably with score 0.917200 in 858s and was marked `discard`. This is well below the 0.925200 high-water, so FedExP is falsified for the active stack; the default-off `fedexp` implementation was removed from `custom_aggregators.py`, `job.py`, and `mutation_schema.yaml`. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=3`.

Local continuation: the `aggregation_epochs=9` higher local-compute extension on the active FedSAM/FedAvgM high-water stack scored 0.920000 in 1089s and was marked `discard`. The extra epoch budget stayed under `RUN_TIMEOUT_SECONDS=1200` but regressed below both `aggregation_epochs=8` and the kept `aggregation_epochs=7` setting, so the local epoch-count axis remains bracketed around 7 for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=4`.

Local continuation: the tight upper SAM interpolation `sam_rho=0.0728125` on the active FedSAM/FedAvgM high-water stack scored 0.921800 in 865s and was marked `discard`. It underperformed the kept `sam_rho=0.0725` point and did not recover the wider upper-side `0.07375` miss, so the upper side of the SAM-radius peak remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=5`.

Local continuation: the tight lower SAM interpolation `sam_rho=0.0721875` on the active FedSAM/FedAvgM high-water stack scored 0.923500 in 863s and was marked `discard`. This was the best post-literature local miss and suggests the useful SAM radius is slightly asymmetric on the lower side, but it still stayed below the kept `sam_rho=0.0725` high-water and below the 0.925700 material-improvement threshold. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=6`.

Local continuation: the lower-mid SAM interpolation `sam_rho=0.07234375` between the kept `0.0725` radius and the `0.0721875` near miss scored 0.919200 in 864s and was marked `discard`. The sharp regression means this immediate SAM fine bracket is noisy and not worth further routine midpoint refinement without a new mechanism. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=7`.

Next local axis: switch from immediate SAM-radius refinement to a tight FedAvgM server learning-rate interpolation under the active high-water stack. Existing new-stack checks at `server_lr=1.796875` and `1.8015625` missed, but the lower side was less harmful; the next non-duplicate candidate will test `server_lr=1.7984375` while keeping `sam_rho=0.0725` and all other active settings fixed.

Local continuation: the lower-mid FedAvgM server learning rate `server_lr=1.7984375` under the active FedSAM/FedAvgM high-water stack scored 0.921400 in 859s and was marked `discard`. It underperformed the active `1.8` high-water and did not improve on the wider lower-side `1.796875` check, so the lower side of the server-lr ridge remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=8`.

Local continuation: the tight lower FedProx coefficient `fedproxloss_mu=2.875e-5` under the active FedSAM/FedAvgM high-water stack scored 0.922900 in 855s and was marked `discard`. It recovered slightly from the wider `2.75e-5` and `3.25e-5` FedProx misses but still stayed below the kept `3e-5` high-water and below the 0.925700 material-improvement threshold, so the proximal coefficient remains bracketed around the active value. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=9`.

Local continuation: the tight upper FedProx coefficient `fedproxloss_mu=3.125e-5` under the active FedSAM/FedAvgM high-water stack scored 0.919000 in 863s and was marked `discard`. This regressed below both the active `3e-5` point and the tight lower `2.875e-5` check, so the upper proximal side is not useful for the current high-water stack and the FedProx axis is now tightly bracketed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=10`.

Local continuation: the tight upper class-balanced coefficient `class_balanced_loss_beta=0.9015625` under the active FedSAM/FedAvgM high-water stack scored 0.920100 in 858s and was marked `discard`. It underperformed the active `0.90` setting and the wider `0.903125` upper check, so the upper side of the class-balanced-loss axis remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=11`.

Local continuation: the tight upper FedAvgM server learning rate `server_lr=1.80078125` under the active FedSAM/FedAvgM high-water stack scored 0.921900 in 864s and was marked `discard`. It underperformed the active `1.8` high-water and did not improve on the wider upper `1.8015625` check, so the upper side of the server-lr ridge remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=12`.

Local continuation: the tight lower class-balanced coefficient `class_balanced_loss_beta=0.896875` under the active FedSAM/FedAvgM high-water stack scored 0.918500 in 862s and was marked `discard`. It regressed below both the active `0.90` setting and the wider lower `0.89375` check, so the lower side of the class-balanced-loss axis is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=13`.

Local continuation: the tight lower weight decay `weight_decay=4.875e-4` under the active FedSAM/FedAvgM high-water stack scored 0.922300 in 857s and was marked `discard`. It matched the wider lower `4.75e-4` check but stayed below the active `5e-4` high-water and below the 0.925700 material-improvement threshold, so the lower side of the weight-decay axis remains worse for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=14`.

Local continuation: the tight lower cosine floor `cosine_lr_eta_min_factor=0.000125` under the active FedSAM/FedAvgM high-water stack scored 0.921000 in 858s and was marked `discard`. It underperformed both the active `0.00015` floor and the wider lower `0.0001` check, so the lower scheduler-floor side is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=15`.

Local continuation: the tight upper weight decay `weight_decay=5.125e-4` under the active FedSAM/FedAvgM high-water stack scored 0.918300 in 860s and was marked `discard`. It regressed below both the active `5e-4` point and the wider upper `5.25e-4` check, so the upper weight-decay side is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=16`.

Local continuation: the tight upper cosine floor `cosine_lr_eta_min_factor=0.000175` under the active FedSAM/FedAvgM high-water stack scored 0.924300 in 864s and was marked `discard`. This is the strongest post-literature near miss so far, but it stayed below the kept `0.00015` floor and below the 0.925700 material-improvement threshold; keep the active floor while treating the upper scheduler side as a close shoulder. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=17`.

Local continuation: the very tight upper cosine floor `cosine_lr_eta_min_factor=0.0001625` under the active FedSAM/FedAvgM high-water stack scored 0.925800 in 858s and was marked `keep`. This is a material improvement over the previous 0.925200 high-water, so the active scheduler floor moves from `0.00015` to `0.0001625` while keeping the rest of the stack fixed. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=0` after resetting on the material improvement.

Local continuation: the post-improvement lower cosine floor `cosine_lr_eta_min_factor=0.00015625` under the active FedSAM/FedAvgM high-water stack scored 0.920400 in 865s and was marked `discard`. Dropping halfway back toward the old `0.00015` floor sharply regressed, so the new `0.0001625` floor remains the active setting and the lower side should not get more routine refinement. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=1`.

Local continuation: the post-improvement upper cosine floor `cosine_lr_eta_min_factor=0.00016875` under the active FedSAM/FedAvgM high-water stack scored 0.918400 in 864s and was marked `discard`. The upper midpoint regressed even more than the lower `0.00015625` check, so the new kept `0.0001625` scheduler floor is a narrow peak rather than a broad shoulder; stop routine cosine-floor midpoint refinement for now. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=2`.

Local continuation: the post-improvement lower FedAvgM server momentum `server_momentum=0.47421875` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.923800 in 861s and was marked `discard`. This is a respectable near miss but still below the kept `server_momentum=0.475` setting and below the 0.926300 material-improvement threshold, so the lower server-momentum side does not improve the current stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=3`.

Local continuation: the post-improvement upper FedAvgM server momentum `server_momentum=0.47578125` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.923100 in 863s and was marked `discard`. Together with the lower `0.47421875` miss, this brackets the active `server_momentum=0.475` setting under the new scheduler floor; move off server momentum for the next non-duplicate local axis. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=4`.

Local continuation: the post-improvement lower client momentum `momentum=0.92421875` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921300 in 865s and was marked `discard`. The lower client-momentum interpolation regressed below the active `momentum=0.925` setting and did not recover the old-stack lower-momentum miss, so the lower side of client momentum is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=5`.

Local continuation: the post-improvement upper client momentum `momentum=0.92578125` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.925700 in 861s and was marked `discard`. This is only 0.000100 below the active 0.925800 high-water but still below the 0.926300 material-improvement threshold, so client momentum is now tightly bracketed around `0.925` and should not receive more routine midpoint refinement for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=6`.

Local continuation: the post-improvement upper client learning rate `lr=0.04515625` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921900 in 861s and was marked `discard`. Raising the client learning rate from the active `0.045` point regressed well below both the high-water and the near-miss upper client-momentum check, so the upper client-lr side is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=7`.

Local continuation: the post-improvement lower client learning rate `lr=0.04484375` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921900 in 864s and was marked `discard`. Lowering the client learning rate matched the upper-side miss and stayed far below the active `0.045` high-water setting, so the client-lr axis is bracketed around the active value for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=8`.

Local continuation: the post-improvement tight lower FedAvgM server learning rate `server_lr=1.799609375` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921300 in 857s and was marked `discard`. Even this small reduction from the active `1.8` server step regressed well below the high-water, so the lower server-lr side remains unhelpful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=9`.

Local continuation: the post-improvement tight upper FedAvgM server learning rate `server_lr=1.800390625` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.923300 in 864s and was marked `discard`. This recovered some accuracy versus the lower-side check but still remained below the active `1.8` high-water and below the 0.926300 material-improvement threshold, so the server-lr axis is now tightly bracketed around the active value for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=10`.

Local continuation: the post-improvement very tight upper client momentum `momentum=0.925390625` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.920600 in 927s and was marked `discard`. The midpoint between the active `0.925` setting and the prior `0.92578125` near miss regressed sharply, so the upper client-momentum shoulder appears noisy rather than a useful ridge. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=11`.

Local continuation: the post-improvement lower SAM radius `sam_rho=0.0721875` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921700 in 864s and was marked `discard`. Retesting the old-floor lower-SAM near miss with the improved scheduler floor did not recover the earlier 0.923500 shoulder and stayed well below the kept `sam_rho=0.0725` high-water, so the lower SAM-radius side remains unhelpful under the new floor. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=12`.

Local continuation: the post-improvement lower FedProx coefficient `fedproxloss_mu=2.875e-5` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921200 in 864s and was marked `discard`. The old-floor lower-FedProx near miss did not transfer to the improved scheduler floor and regressed below both the active `3e-5` setting and the old 0.922900 shoulder, so the lower proximal side remains bracketed away from the current high-water. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=13`.

Local continuation: the post-improvement lower weight decay `weight_decay=4.875e-4` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.922100 in 864s and was marked `discard`. This did not improve on the old-floor 0.922300 lower-weight-decay shoulder and stayed far below the active `5e-4` high-water, so the lower regularization side does not transfer under the new scheduler floor. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=14`.

Local continuation: the post-improvement lower class-balanced coefficient `class_balanced_loss_beta=0.89375` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921800 in 862s and was marked `discard`. The lower-beta shoulder again failed to transfer to the active SAM radius and scheduler floor, so the current `class_balanced_loss_beta=0.90` remains bracketed as the better local-loss setting. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=15`.

Local continuation: the post-improvement upper SAM radius `sam_rho=0.0728125` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921000 in 861s and was marked `discard`. Together with the lower `0.0721875` miss, this brackets the active `sam_rho=0.0725` radius under the improved scheduler floor; the upper SAM-radius side should not receive more routine refinement. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=16`.

Local continuation: the post-improvement upper FedProx coefficient `fedproxloss_mu=3.125e-5` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.922300 in 864s and was marked `discard`. It improved over the old-floor upper-proximal miss but remained below the active `3e-5` high-water and below the 0.926300 material-improvement threshold, so FedProx is now bracketed on both sides under the current scheduler floor. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=17`.

Local continuation: the post-improvement upper weight decay `weight_decay=5.125e-4` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921900 in 863s and was marked `discard`. This stayed far below the active `5e-4` high-water and did not improve on the old-floor upper-weight-decay miss, so weight decay is now bracketed on both sides under the current scheduler floor. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=18`.

Local continuation: the post-improvement upper class-balanced coefficient `class_balanced_loss_beta=0.903125` retest under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921900 in 858s and was marked `discard`. This matched the upper-weight-decay miss and stayed well below the active `0.90` loss setting, so the class-balanced-loss coefficient is bracketed on both sides under the current scheduler floor. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=19`.

Local continuation: the post-improvement upper local-compute neighbor `aggregation_epochs=8` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.920900 in 975s and was marked `discard`. The higher epoch budget stayed within `RUN_TIMEOUT_SECONDS=1200` but regressed below the active 7-epoch setting and below the older 8-epoch SAM shoulder, so the upper local-compute side does not improve this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=20`.

Local continuation: the post-improvement lower local-compute neighbor `aggregation_epochs=6` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.919800 in 752s and was marked `discard`. The shorter epoch budget saved runtime but regressed below both the active 7-epoch setting and the upper 8-epoch check, so local compute is bracketed around `aggregation_epochs=7` for the current stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=21`.

Local continuation: the post-improvement exact-step local-compute check `local_train_steps=768` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.921800 in 843s and was marked `discard`. It matched the old-floor exact-step result but stayed below the active epoch-based 7-epoch high-water, so the exact-step representation does not improve this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=22`.

Local continuation: the post-improvement lower exact-step local-compute neighbor `local_train_steps=704` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.920900 in 782s and was marked `discard`. Lowering exact steps saved runtime but regressed below the `768`-step check and the active epoch-based 7-epoch high-water, so the lower exact-step side is not useful for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=23`.

Local continuation: the post-improvement upper exact-step local-compute neighbor `local_train_steps=832` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.919000 in 909s and was marked `discard`. Raising exact steps regressed below both the `768`-step and `704`-step checks while remaining below the active epoch-based 7-epoch high-water, so exact-step local compute is bracketed as worse than the epoch-based setting for this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=24`.

Local continuation: the post-improvement wider upper exact-step local-compute neighbor `local_train_steps=896` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.923200 in 974s and was marked `discard`. It recovered versus the poor `832`-step result but still stayed below the active epoch-based 7-epoch high-water and below the 0.926300 material-improvement threshold, so exact-step local compute remains a non-kept alternative under this stack. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=25`.

Local continuation: the post-improvement wider upper exact-step local-compute probe `local_train_steps=960` under the new `cosine_lr_eta_min_factor=0.0001625` high-water stack scored 0.922400 in 1035s and was marked `discard`. It stayed within `RUN_TIMEOUT_SECONDS=1200` but regressed below the 896-step shoulder and remained below the active epoch-based 7-epoch high-water, so the high-step exact local-compute extension does not justify its extra runtime. `scripts/plateau_watchdog.py results.tsv` reported `recommendation=continue` with `scored_since_reset=26`.
