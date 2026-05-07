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
