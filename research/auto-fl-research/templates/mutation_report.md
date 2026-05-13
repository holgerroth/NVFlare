# Mutation report

## Mutation: CIFAR-10 Mixup knob

## Hypothesis

Source-backed client-local Mixup may improve the FedAvgM ep8 plateau by regularizing long local training on class-skewed client subsets, while preserving DIFF uploads, server aggregation contracts, fixed data/evaluation fields, and the current model architecture.

## Files changed

- `tasks/cifar10/client.py`
- `tasks/cifar10/job.py`
- `tasks/cifar10/mutation_schema.yaml`
- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Commands run

- `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make validate`
- `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make smoke`

## Observed outcome

- Added a dormant `--mixup_alpha` argument with default `0.0`.
- `job.py` forwards the argument to every client.
- `client.py` applies input and label Mixup inside both local-step and epoch-based training loops only when `mixup_alpha > 0`.
- The literature loop selected `mixup_alpha=0.2` and `0.4` as the next source-backed batch.
- Validation passed static contract checks and compiled Python sources.
- Smoke passed a one-round CIFAR-10 simulation with `RUN_ITERATION_LOG_RESULTS=0`.

## Literature basis

- Zhang et al., 2018, "mixup: Beyond Empirical Risk Minimization", ICLR/arXiv: https://arxiv.org/abs/1710.09412
- Yoon et al., 2021, "FedMix: Approximation of Mixup under Mean Augmented Federated Learning", arXiv: https://arxiv.org/abs/2107.00233
- Hsu, Qi, and Brown, 2019, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification", Google Research/arXiv: https://research.google/pubs/measuring-the-effects-of-non-identical-data-distribution-for-federated-visual-classification/

## Run analysis

Candidate runs are pending. The selected batch should compare `--mixup_alpha 0.2` and `0.4` against the current `0.900100` FedAvgM ep8 best.

## Contract check

- The default value preserves existing behavior.
- The FL loop, strict state loading, DIFF upload, `NUM_STEPS_CURRENT_ROUND`, evaluation branch, and fixed model architecture remain unchanged.
- Mixup is client-local and does not alter data splits, validation data, aggregation metadata, or parameter keys.

## Rollback risk

Low. The new behavior is opt-in, validation/smoke passed, and there are no dependency changes.

## Next mutation

Finish the literature event, then run the two Mixup candidates.

---

## Mutation: CIFAR-10 gradient clipping knob

## Hypothesis

Long local training under non-IID CIFAR-10 can produce client updates with high variance. Optional gradient-norm clipping may stabilize the kept FedAvgM ep8 stack without changing DIFF uploads, aggregation metadata, model architecture, or evaluation.

## Files changed

- `tasks/cifar10/client.py`
- `tasks/cifar10/job.py`
- `tasks/cifar10/mutation_schema.yaml`
- `templates/mutation_report.md`

## Commands run

- `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make validate`
- `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make smoke`

## Observed outcome

- Added a dormant `--max_grad_norm` argument with default `0.0`.
- `job.py` forwards the argument to every client.
- `client.py` clips gradients after `loss.backward()` and before `optimizer.step()` only when the value is positive.
- Validation passed static contract checks and compiled Python sources.
- Smoke passed a one-round CIFAR-10 simulation with `RUN_ITERATION_LOG_RESULTS=0`.

## Literature basis

None in this local loop. The active task profile lists gradient clipping as a typical safe client-side mutation idea.

## Run analysis

Candidate runs are pending. The first sweep should use conservative values such as `1.0` and `5.0` on the current best FedAvgM ep8 stack.

## Contract check

- The default value preserves existing behavior.
- The FL loop, strict state loading, DIFF upload, `NUM_STEPS_CURRENT_ROUND`, evaluation branch, and fixed model architecture remain unchanged.

## Rollback risk

Low. The new behavior is opt-in and validation/smoke passed.

## Next mutation

Run a two-point gradient-clipping sweep under the current best optimizer stack.

---

## Mutation: CIFAR-10 label smoothing knob

## Hypothesis

Mild label smoothing may improve the current FedAvgM ep8 plateau by reducing overconfident local fits during long client training, while preserving the DIFF upload contract and all fixed communication/data/evaluation budget fields.

## Files changed

- `tasks/cifar10/client.py`
- `tasks/cifar10/job.py`
- `tasks/cifar10/mutation_schema.yaml`
- `templates/mutation_report.md`

## Commands run

- `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make validate`
- `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 make smoke`

## Observed outcome

- Added a dormant `--label_smoothing` training argument with default `0.0`.
- `job.py` forwards the argument to every client, and `client.py` applies it through `nn.CrossEntropyLoss(label_smoothing=...)`.
- Validation passed static contract checks and compiled Python sources.
- Smoke passed a one-round CIFAR-10 simulation with `RUN_ITERATION_LOG_RESULTS=0`.

## Literature basis

None in this local loop. The active task profile lists label smoothing as a typical safe client-side mutation idea.

## Run analysis

Candidate runs are pending. The first sweep should use small values such as `0.02` and `0.05` on the current best FedAvgM ep8 stack.

## Contract check

- The default value preserves existing behavior.
- The FL loop, strict state loading, DIFF upload, `NUM_STEPS_CURRENT_ROUND`, evaluation branch, and fixed model architecture remain unchanged.

## Rollback risk

Low. The new argument is opt-in and validation/smoke passed.

## Next mutation

Run a two-point label-smoothing sweep under the current best optimizer stack.

---

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

## Literature loop: CIFAR-10 FedAvgM plateau

## Hypothesis

The best optimizer-only stack has saturated around FedAvgM with long local training and weight decay. The next useful branch should test paper-backed drift correction or adaptive server optimization rather than more narrow jitter around server LR, server momentum, scheduler floors, or client LR.

## Files changed

- `templates/literature_loop.md`
- `templates/mutation_report.md`

## Commands run

- `.venv/bin/python scripts/plateau_watchdog.py results.tsv`
- `.venv/bin/python scripts/log_literature_review.py --start --description "plateau after scheduler floor sweep: 33 scored candidates since best FedAvgM ep8 regularized stack"`
- Web literature search across PMLR, MLSys, ICLR/OpenReview-linked pages, NeurIPS proceedings, Google Research, and arXiv mirrors.

## Observed outcome

- Watchdog reported `recommendation=literature` after 33 scored candidates since the last material improvement.
- Recent local sweeps did not improve on `0.899900`: FedProx light/medium, exact local steps, client LR/momentum, weight decay refinements, robust aggregation, server LR/momentum micro sweeps, and scheduler floors all missed or tied.
- Selected the next batch from `templates/literature_loop.md`: tuned SCAFFOLD under the best ep8/weight-decay stack, plus a damped FedAdam retry with low server LR and larger tau.
- Post-run outcome: tuned SCAFFOLD scored `0.888900` and damped FedAdam scored `0.808600`, so both literature-selected candidates were discarded.

## Literature basis

- McMahan et al., 2017, "Communication-Efficient Learning of Deep Networks from Decentralized Data", PMLR: https://proceedings.mlr.press/v54/mcmahan17a.html
- Hsu, Qi, and Brown, 2019, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification", Google Research/arXiv 1909.06335: https://research.google/pubs/measuring-the-effects-of-non-identical-data-distribution-for-federated-visual-classification/
- Karimireddy et al., 2020, "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", PMLR: https://proceedings.mlr.press/v119/karimireddy20a.html
- Reddi et al., 2021, "Adaptive Federated Optimization", ICLR: https://iclr.cc/virtual/2021/poster/2691
- Li et al., 2020, "Federated Optimization in Heterogeneous Networks", MLSys: https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
- Wang et al., 2020, "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization", NeurIPS: https://papers.nips.cc/paper_files/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html

## Run analysis

SCAFFOLD is justified because the plateau still looks like non-IID client drift after long local training. FedAdam is justified because adaptive server optimizers are source-backed for heterogeneous FL, but the prior FedAdam crash means the retry must be damped with `server_lr=0.1` and `fedopt_tau=1e-2`.

The r32 results falsified both selected candidates under the current fixed budget. SCAFFOLD remained stable but underperformed the FedAvgM best by `0.011000`. The damped FedAdam retry avoided the earlier crash but scored far below the plateau, so adaptive Adam-style server updates should not be retried without a stronger source-backed change.

## Contract check

- No code or protocol change is introduced by the selected next batch.
- Both selected candidates preserve the fixed CIFAR-10 budget fields and use implemented aggregator modes.
- SCAFFOLD uses the existing profile-supported control-variate metadata path.

## Rollback risk

Low for the worksheet/report edits. Candidate runtime risk is medium for FedAdam because an aggressive FedAdam setting previously produced NaNs.

## Next mutation

With the watchdog reset to `recommendation=continue`, continue with non-duplicate local axes around the retained FedAvgM best rather than starting another literature loop immediately.
