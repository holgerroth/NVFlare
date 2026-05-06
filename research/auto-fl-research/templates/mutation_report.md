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
- Label-smoothing mutation validation: `PYTHON=.venv/bin/python make validate`
- Label-smoothing mutation smoke: `PYTHON=.venv/bin/python make smoke`

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
- Client learning-rate sweep underperformed: `--lr 0.03` scored `0.862700`; `--lr 0.07` scored `0.850200`.
- Weight decay improved the current best: `--weight_decay 5e-4` scored `0.873900`; `1e-4` scored `0.867300`.
- Narrowed weight decay improved again: `--weight_decay 3e-4` scored `0.878900`; `7e-4` scored `0.874100`.
- Weight-decay neighbors did not improve: `4e-4` scored `0.875400`; `2e-4` scored `0.867200`.
- Scheduler floor sweep did not help: `cosine_lr_eta_min_factor=0.03` scored `0.866200`; `0.001` crashed after an NVFlare simulator child-process timeout.
- Client momentum sweep underperformed: `--momentum 0.8` scored `0.863700`; `0.95` scored `0.799000`.
- Server learning-rate revisit with weight decay did not improve: `server_lr=1.75` scored `0.876200`; `1.25` scored `0.864600`.
- Server momentum retune kept a tiny score edge: `server_momentum=0.3` scored `0.879000`; `0.1` scored `0.869900`.
- Server momentum neighbor sweep kept another tiny edge: `server_momentum=0.35` scored `0.879200`; `0.25` scored `0.878900`.
- Higher server momentum found a material improvement: `server_momentum=0.4` scored `0.881400`; `0.45` scored `0.877900`.
- Momentum neighbors regressed: `server_momentum=0.425` scored `0.874200`; `0.375` scored `0.873300`.
- Weight decay retune under `server_momentum=0.4` regressed: `4e-4` scored `0.876300`; `2e-4` scored `0.869200`.
- Server learning-rate revisit under `server_momentum=0.4` also regressed: `server_lr=1.75` scored `0.879400`; `1.25` scored `0.873600`.
- Second literature loop selected client-local label smoothing as the next low-risk source-backed mutation before heavier SAM-style optimizer changes.
- Added optional `--label_smoothing` forwarding to PyTorch `CrossEntropyLoss`; default `0.0` preserves the previous loss.
- Label smoothing did not improve: `0.05` scored `0.881200`; `0.1` scored `0.879900`. The optional label-smoothing code path was reverted.
- SCAFFOLD and median source-backed probes underperformed: SCAFFOLD with `weight_decay=3e-4` scored `0.859000`; median scored `0.747000`.
- Scheduler toggles underperformed: no scheduler scored `0.831600`; `cosine_lr_eta_min_factor=0.1` scored `0.866000`.
- Third literature loop selected FedZMG-style gradient centralization as the next low-risk client-local mutation.
- Added optional `--gradient_centralization` to project eligible weight gradients to zero mean before local SGD steps; default off preserves prior behavior.
- Gradient centralization materially improved the campaign: with `weight_decay=3e-4` it scored `0.900700`; with `weight_decay=1e-4` it scored `0.890900`.
- Gradient centralization weight-decay narrowing improved further: `weight_decay=4e-4` scored `0.902600`; `2e-4` scored `0.896300`.
- Further narrowing found a new best: gradient centralization with `weight_decay=3.5e-4` scored `0.903400`; `5e-4` scored `0.901800`.
- Extra weight-decay neighbors regressed: `3.25e-4` scored `0.899600`; `3.75e-4` scored `0.899800`.
- Server-momentum retune improved the best stack: `server_momentum=0.35` scored `0.904600`; `0.45` scored `0.902600`.
- Server-momentum neighbors under the new best regressed: `0.30` and `0.375` both scored `0.899700`.
- Weight-decay retune under `server_momentum=0.35` also regressed: `3e-4` scored `0.900800`; `4e-4` scored `0.900900`.
- Fourth literature loop selected an epoch-based local-compute sweep before more optimizer jitter or SAM-style code.
- Literature-selected local compute improved the best: `aggregation_epochs=5` scored `0.906500`; `3` scored `0.895500`.
- Upward local-compute narrowing did not improve: `aggregation_epochs=6` scored `0.904600`; `7` scored `0.906300`.
- Epoch-5 server-momentum retune did not improve: `server_momentum=0.40` scored `0.905600`; `0.30` scored `0.904700`.
- Fifth literature loop selected a narrow client learning-rate sweep under the epoch-5 best before server-LR retuning or SAM code.
- Client learning-rate probes under the epoch-5 best regressed: `lr=0.06` scored `0.901400`; `0.04` scored `0.900600`.
- Server learning-rate reserve improved the best: `server_lr=1.75` scored `0.907300`; `1.25` scored `0.898900`.
- Server learning-rate narrowing improved further: `server_lr=1.875` scored `0.909900`; `1.625` scored `0.907100`.
- Higher server learning rates regressed: `server_lr=2.0` scored `0.908800`; `2.125` scored `0.908200`.
- Tight server learning-rate neighbors narrowly missed: `server_lr=1.8125` scored `0.909800`; `1.9375` scored `0.909600`.
- Sixth literature loop selected a default-off SAM/FedSAM client-local optimizer mutation after server-LR tuning reached a local peak.
- Added optional `--sam_rho` for client-local SAM perturb-and-second-gradient steps; default `0.0` preserves prior behavior.

## Literature basis

- FedProx: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. "Federated Optimization in Heterogeneous Networks." MLSys 2020; arXiv:1812.06127.
- FedAvg local epochs: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017; arXiv:1602.05629.
- FedOpt / FedAvgM / FedAdam: Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneczny, Sanjiv Kumar, H. Brendan McMahan. "Adaptive Federated Optimization." ICLR 2021; arXiv:2003.00295.
- SCAFFOLD: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020; arXiv:1910.06378.
- FedLC: Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu. "Federated Learning with Label Distribution Skew via Logits Calibration." ICML 2022; arXiv:2209.00189.
- FedNova: Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS 2020; arXiv:2007.07481.
- Momentum analysis: Ziheng Cheng, Xinmeng Huang, Pengfei Wu, Kun Yuan. "Momentum Benefits Non-IID Federated Learning Simply and Provably." arXiv:2306.16504.
- Label smoothing in FL: Yeji Cho, Junghyun Kim. "FedENLC: An End-to-End Noisy Label Correction Framework in Federated Learning." Mathematics 2026; doi:10.3390/math14020290.
- Federated domain generalization with label smoothing: Milad Soltany, Farhad Pourpanah, Mahdiyar Molahasani Majdabadi, Michael Greenspan, Ali Etemad. "Federated Domain Generalization with Label Smoothing and Balanced Decentralized Training." arXiv:2412.11408.
- Label smoothing implementation: PyTorch `torch.nn.CrossEntropyLoss(label_smoothing=...)` documentation.
- FedZMG: Fotios Zantalis, Evangelos Zervas, Grigorios Koulouras. "FedZMG: Efficient Client-Side Optimization in Federated Learning." arXiv:2602.18384.
- SAM: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. "Sharpness-Aware Minimization for Efficiently Improving Generalization." arXiv:2010.01412.
- FedSAM: Zhe Qu, Xingyu Li, Rui Duan, Yao Liu, Bo Tang, Zhuo Lu. "Generalized Federated Learning via Sharpness Aware Minimization." arXiv:2206.02618.
- Auto-tuned clients: Junhyung Lyle Kim, Mohammad Taha Toghani, Cesar A. Uribe, Anastasios Kyrillidis. "Adaptive Federated Learning with Auto-Tuned Clients." arXiv:2306.11201.
- FedCM: Jing Xu, Sen Wang, Liwei Wang, Andrew Chi-Chih Yao. "FedCM: Federated Learning with Client-level Momentum." arXiv:2106.10874.

## Run analysis

The calibration result favors the existing FedAvgM path with the original `moderate_cnn` architecture. The best stack is now FedAvgM `server_lr=1.875`, `server_momentum=0.35`, default client LR, epoch-based local training with `aggregation_epochs=5`, `weight_decay=3.5e-4`, and enabled gradient centralization. FedLC and label smoothing came close but did not justify new client loss paths. FedAdam is currently unsafe or ineffective at tested settings. Exact local-step training and registered architecture variants regressed. The shared-memory validation crash at candidate width 4 is a resource-contention signal, so subsequent batches should use `PARALLEL_CANDIDATES=2`. Parallel run launches should also set unique pycache prefixes to avoid validator races.

## Contract check

- No FL protocol fields were changed.
- Gradient centralization is client-local and does not alter FLModel params, metadata, aggregation keys, or evaluation.
- All completed candidates used `--cross_site_eval`, `--num_rounds 20`, `--model_arch moderate_cnn`, `--max_model_params 5000000`, and `--final_eval_clients site-1`.
- DIFF upload, `NUM_STEPS_CURRENT_ROUND`, and strict state-dict loading remain governed by the existing validated code.

## Rollback risk

Low to medium. Gradient centralization is kept behind `--gradient_centralization`. The new SAM path is also default-off behind `--sam_rho`, but it adds an extra backward pass when enabled; revert it if the SAM candidates fail or time out.

## Next mutation

Implement optional local SAM behind `--sam_rho`, then test `--sam_rho 0.01` and `0.02` under the current `server_lr=1.875`, `aggregation_epochs=5` best stack.
