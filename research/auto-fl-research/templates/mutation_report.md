# Mutation report

## Hypothesis

The initial campaign should establish which already-available algorithm family is strongest under the fixed H100 budget before any code mutation or open-ended hyperparameter tuning. Server momentum through the existing FedAvgM aggregator may improve the final cross-site score while preserving DIFF uploads, strict model loading, and the current evaluation contract.

## Files changed

- `results.tsv`
- `client.py`
- `job.py`
- `mutation_schema.yaml`
- `templates/literature_loop.md`
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
- FedAvgM LR sweep at fixed `server_momentum=0.4`: `server_lr` in `{1.5, 1.75, 2.25, 2.5}`
- FedAvgM momentum probe at fixed `server_lr=1.5`: `server_momentum=0.2`
- FedAvgM lower-momentum sweep at fixed `server_lr=1.5`: `server_momentum` in `{0.0, 0.1}`
- FedAvgM close-neighbor momentum sweep at fixed `server_lr=1.5`: `server_momentum` in `{0.15, 0.25}`
- FedLC mutation validation: `PYTHON=.venv/bin/python make validate`
- FedLC mutation smoke: `PYTHON=.venv/bin/python make smoke`

## Observed outcome

- Baseline weighted FedAvg scored `0.849800`.
- Explicit FedAvg scored `0.851200` and was kept after batch 1.
- FedProx `mu=1e-5` scored `0.849100`; FedProx `mu=1e-4` scored `0.848200`; both were discarded.
- FedAvgM `server_lr=2.0, momentum=0.4` scored `0.856200`, the current best calibration result.
- SCAFFOLD scored `0.854800`, below the best FedAvgM row but above baseline.
- FedAvgM `server_lr=1.0, momentum=0.6` scored `0.848500`.
- FedAdam crashed early with `ValueError: Diff norm is NaN or Inf: nan`.
- FedAvgM `server_lr=1.5, momentum=0.4` scored `0.858900`, the current best result.
- FedAvgM `server_lr=2.25, momentum=0.4` scored `0.855300`; `server_lr=2.5, momentum=0.4` scored `0.852000`.
- FedAvgM `server_lr=1.75, momentum=0.4` crashed during validation with a host shared-memory allocation failure, so later batches should reduce candidate width from 4 to 2.
- FedAvgM `server_lr=1.5, momentum=0.2` scored `0.864700`, the current best result.
- The paired `server_momentum=0.6` run failed before training due a parallel bytecode-cache race in validation, so future concurrent launches should isolate `PYTHONPYCACHEPREFIX` per candidate.
- FedAvgM `server_lr=1.5, momentum=0.0` scored `0.862700`; `momentum=0.1` scored `0.860900`. Both were discarded because they did not improve over `momentum=0.2`.
- FedAvgM `server_lr=1.5, momentum=0.15` scored `0.863400`; `momentum=0.25` scored `0.862200`. Both were discarded.
- Literature loop was triggered after two consecutive non-improving same-budget batches. The next selected candidates are FedAvgM+FedProx `mu=1e-3` and a safer FedAdam retry with lower server learning rate and larger `tau`.
- FedAvgM+FedProx `mu=1e-3` scored `0.860100`; safer FedAdam `server_lr=0.1, tau=1e-2` scored `0.744300`. Both were discarded.
- FedLC `tau=1.0` scored `0.864000`; FedLC `tau=0.5` scored `0.863600`. Both were below the current best, so the optional FedLC code path was reverted rather than kept.
- Exact `local_train_steps=400` scored `0.857600`; `local_train_steps=300` scored `0.847500`. Both were discarded.
- Architecture subcampaign rows also underperformed: `moderate_cnn_small_head` scored `0.860900`; `moderate_cnn_norm` scored `0.835400`.

## Literature basis

- FedProx: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. "Federated Optimization in Heterogeneous Networks." MLSys 2020; arXiv:1812.06127.
- FedOpt / FedAvgM / FedAdam: Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneczny, Sanjiv Kumar, H. Brendan McMahan. "Adaptive Federated Optimization." ICLR 2021; arXiv:2003.00295.
- SCAFFOLD: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020; arXiv:1910.06378.
- FedLC: Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu. "Federated Learning with Label Distribution Skew via Logits Calibration." ICML 2022; arXiv:2209.00189.
- FedNova: Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS 2020; arXiv:2007.07481.
- Momentum analysis: Ziheng Cheng, Xinmeng Huang, Pengfei Wu, Kun Yuan. "Momentum Benefits Non-IID Federated Learning Simply and Provably." arXiv:2306.16504.

## Run analysis

The calibration result favors the existing FedAvgM path with the original `moderate_cnn` architecture. The first LR sweep suggests the best region is below `server_lr=2.0` at fixed `server_momentum=0.4`, and the first momentum probe improved further by lowering momentum to `0.2`. Two consecutive same-budget momentum batches failed to improve after that point, and the literature candidates did not improve. FedLC came close but did not justify a new client loss path. FedAdam is currently unsafe or ineffective at tested settings. Exact local-step training and registered architecture variants regressed. The shared-memory validation crash at candidate width 4 is a resource-contention signal, so subsequent batches should use `PARALLEL_CANDIDATES=2`. Parallel run launches should also set unique pycache prefixes to avoid validator races.

## Contract check

- No FL protocol fields were changed.
- All completed candidates used `--cross_site_eval`, `--num_rounds 20`, `--model_arch moderate_cnn`, `--max_model_params 5000000`, and `--final_eval_clients site-1`.
- DIFF upload, `NUM_STEPS_CURRENT_ROUND`, and strict state-dict loading remain governed by the existing validated code.

## Rollback risk

Low. The campaign has only added ledger/report data and tested existing CLI-selectable algorithms. No kept code mutation exists yet.

## Next mutation

Return to the original architecture and run a client learning-rate sweep around the current best FedAvgM stack: `--lr 0.03` and `0.07` with all fixed budget fields unchanged.
