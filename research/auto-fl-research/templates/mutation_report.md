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
