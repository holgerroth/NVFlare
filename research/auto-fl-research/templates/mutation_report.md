# Mutation report

## Hypothesis

The initial campaign should establish which already-available algorithm family is strongest under the fixed H100 budget before any code mutation or open-ended hyperparameter tuning. Server momentum through the existing FedAvgM aggregator may improve the final cross-site score while preserving DIFF uploads, strict model loading, and the current evaluation contract.

## Files changed

- `results.tsv`
- `templates/mutation_report.md`

## Commands run

- `test -x "$PYTHON"`
- `"$PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)"`
- `bash scripts/init_run.sh h100-algo-calibration-20260506`
- `git branch --show-current`
- `PYTHON=.venv/bin/python make validate`
- `PYTHON=.venv/bin/python make smoke`
- Baseline: weighted FedAvg, 8 clients, 20 rounds, 4 aggregation epochs, cross-site eval on `site-1`
- Calibration batch 1: built-in FedAvg, explicit FedAvg, FedProx `mu=1e-5`, FedProx `mu=1e-4`
- Calibration batch 2: FedAvgM `server_lr=1.0, momentum=0.6`, FedAvgM `server_lr=2.0, momentum=0.4`, FedAdam, SCAFFOLD

## Observed outcome

- Baseline weighted FedAvg scored `0.849800`.
- Explicit FedAvg scored `0.851200` and was kept after batch 1.
- FedProx `mu=1e-5` scored `0.849100`; FedProx `mu=1e-4` scored `0.848200`; both were discarded.
- FedAvgM `server_lr=2.0, momentum=0.4` scored `0.856200`, the current best calibration result.
- SCAFFOLD scored `0.854800`, below the best FedAvgM row but above baseline.
- FedAvgM `server_lr=1.0, momentum=0.6` scored `0.848500`.
- FedAdam crashed early with `ValueError: Diff norm is NaN or Inf: nan`.

## Literature basis

- FedProx: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. "Federated Optimization in Heterogeneous Networks." MLSys 2020; arXiv:1812.06127.
- FedOpt / FedAvgM / FedAdam: Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneczny, Sanjiv Kumar, H. Brendan McMahan. "Adaptive Federated Optimization." ICLR 2021; arXiv:2003.00295.
- SCAFFOLD: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020; arXiv:1910.06378.

## Run analysis

The calibration result favors the existing FedAvgM path, especially higher server learning rate with lower momentum. FedAdam is currently unsafe at the tested server learning rate because it produced NaN model diffs. The next same-budget sweep should stay in the FedAvgM family and vary one narrow server-optimizer axis at a time.

## Contract check

- No source code or FL protocol fields were changed.
- All completed candidates used `--cross_site_eval`, `--num_rounds 20`, `--model_arch moderate_cnn`, `--max_model_params 5000000`, and `--final_eval_clients site-1`.
- DIFF upload, `NUM_STEPS_CURRENT_ROUND`, and strict state-dict loading remain governed by the existing validated code.

## Rollback risk

Low. The campaign has only added ledger/report data and tested existing CLI-selectable algorithms. No kept code mutation exists yet.

## Next mutation

Finalize the second calibration batch, run the plateau watchdog, then launch a narrow FedAvgM sweep around `server_lr=2.0` and `server_momentum=0.4` under the same fixed budget.
