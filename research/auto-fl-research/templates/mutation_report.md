# Mutation report

## Hypothesis

The initial campaign should establish which already-available algorithm family is strongest under the fixed H100 budget before any code mutation or open-ended hyperparameter tuning. Server momentum through the existing FedAvgM aggregator may improve the final cross-site score while preserving DIFF uploads, strict model loading, and the current evaluation contract.

## Files changed

- `results.tsv`
- `client.py`
- `custom_aggregators.py`
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
- SAM mutation validation: `PYTHON=.venv/bin/python make validate`
- SAM mutation smoke: `PYTHON=.venv/bin/python make smoke`
- SAM candidate batch: `--sam_rho 0.01` and `0.02` under the current epoch-5 FedAvgM/GC best stack
- SAM batch finalization: `scripts/finalize_batch_status.py results.tsv --last 2 --keep-best --discard-others`
- Post-SAM watchdog: `scripts/plateau_watchdog.py results.tsv`
- SAM rollback validation: `PYTHON=.venv/bin/python make validate`
- Sixth-loop reserve weight-decay batch: `weight_decay=3e-4` and `4e-4` under FedAvgM `server_lr=1.875`, `server_momentum=0.35`, `aggregation_epochs=5`, and gradient centralization
- Sixth-loop reserve server-momentum batch: `server_momentum=0.30` and `0.40` under FedAvgM `server_lr=1.875`, `aggregation_epochs=5`, `weight_decay=3.5e-4`, and gradient centralization
- FedNova mutation validation: `PYTHON=.venv/bin/python make validate`
- FedNova mutation smoke: `PYTHON=.venv/bin/python make smoke`

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
- SAM/FedSAM candidates underperformed and were slower: `sam_rho=0.01` scored `0.907600` in 830 seconds; `sam_rho=0.02` scored `0.907500` in 832 seconds. The optional SAM code path was reverted.
- Weight-decay reserve under `server_lr=1.875` did not improve: `weight_decay=3e-4` scored `0.907500`; `4e-4` scored `0.903900`.
- Server-momentum reserve under `server_lr=1.875` did not improve: `server_momentum=0.40` scored `0.908800`; `0.30` scored `0.906600`.
- Seventh literature loop selected FedNova-style normalized aggregation as the next source-backed code mutation.
- Added optional `--aggregator fednova`, which normalizes client DIFFs by `NUM_STEPS_CURRENT_ROUND` and supports optional server momentum without changing client uploads.
- FedNova with current server settings improved the best to `0.910300`; pure normalized FedNova scored `0.894300` and was discarded.
- FedNova server-LR neighbors did not improve: `server_lr=2.0` scored `0.909300`; `1.75` scored `0.908700`.
- FedNova server-momentum neighbors did not improve: `server_momentum=0.40` scored `0.910000`; `0.30` scored `0.909100`.
- FedNova weight-decay retune did not improve: `weight_decay=4e-4` scored `0.905800`; `3e-4` scored `0.905700`.
- Exact local-step retuning under FedNova did not improve: `local_train_steps=600` scored `0.910000`; `500` scored `0.906100`.
- FedYogi and FedAdagrad conservative adaptive-server probes both crashed in round 1 with NaN client diffs; the optional adaptive-server code was reverted.
- Eighth literature loop selected default-off FedNova median-norm update clipping as the next source-backed server-side mutation.
- FedNova median-norm clipping tied but did not improve: factors `1.5` and `2.0` both scored `0.910300`; the optional clipping code was reverted.
- Ninth literature loop selected a CLI-only client learning-rate retune under the kept FedNova stack.
- FedNova client-LR retune regressed: `lr=0.045` scored `0.906600`; `lr=0.055` scored `0.906100`.
- Tenth literature loop selected a labeled registered-architecture subcampaign under the kept FedNova stack.
- Registered architecture subcampaign underperformed: `moderate_cnn_small_head` scored `0.909200`; `moderate_cnn_norm` scored `0.905300`.
- Eleventh literature loop selected a CLI-only client momentum retune under the kept FedNova stack.
- FedNova client-momentum retune underperformed: `momentum=0.925` scored `0.909200`; `0.875` scored `0.906000`.
- Twelfth literature loop selected a CLI-only FedProx retune under the kept FedNova stack before attempting FedDyn/FedDC/FedCM-style drift-correction code.
- FedProx under the kept FedNova stack did not improve: `mu=1e-5` scored `0.907300`; `mu=1e-4` scored `0.909900`. Both were discarded.
- FedNova local-compute epoch audit did not improve: `aggregation_epochs=4` scored `0.905500`; `aggregation_epochs=6` scored `0.908200`. Both were discarded.
- Thirteenth literature loop selected a default-preserving FedNova aggregation-weight exponent before attempting FedDyn/FedDC/FedCM-style stateful drift correction.
- Added optional `--fednova_weight_power`; `1.0` preserves the current step-weighted FedNova behavior, `0.5` partially flattens client weighting, and `0.0` gives uniform client weighting after local-step normalization.
- FedNova weight-power candidates underperformed: uniform weighting scored `0.904000`; square-root weighting scored `0.904800`. The optional weight-power code path was reverted.
- Added optional `--feddyn_alpha` client-side dynamic regularization, default off, with persistent per-client correction state held inside the existing client process and no new FLModel fields.
- FedDyn-style dynamic regularization improved the best: `alpha=1e-4` scored `0.910900`; `alpha=5e-4` scored `0.907300`. The default-off code path is kept.
- FedDyn alpha neighbors did not improve: `alpha=5e-5` scored `0.906600`; `alpha=2e-4` scored `0.909600`.
- FedDyn-enabled server-LR neighbors did not improve: `server_lr=1.9375` scored `0.910500`; `server_lr=1.8125` scored `0.908300`.
- Fourteenth literature loop selected a FedDyn-enabled server-momentum retune before more local regularization or FedDC/FedRed-style code.
- FedDyn-enabled server momentum neighbors did not improve: `server_momentum=0.30` scored `0.910800`; `server_momentum=0.40` scored `0.910400`.
- FedDyn-enabled weight-decay retune did not improve: `weight_decay=4e-4` scored `0.910400`; `weight_decay=3e-4` scored `0.907700`.
- Fifteenth literature loop selected a FedDyn-enabled epoch-count audit because pre-FedDyn local-compute rows may not transfer after the local objective changed.
- FedDyn-enabled epoch-count audit did not improve: `aggregation_epochs=4` scored `0.908800`; `aggregation_epochs=6` scored `0.909500`.
- FedDyn-enabled exact local-step audit crashed: `local_train_steps=500` and `600` both hit `RUN_TIMEOUT_SECONDS=1200` with NVFlare target-unreachable/get-task failures.
- Sixteenth literature loop selected a width-1 exact-step reliability audit before adding FedDC/FedRed-style drift-correction code.
- The width-1 exact-step audit completed successfully but did not improve: `local_train_steps=600` scored `0.909300`, so exact local steps are now a discarded axis under the current FedNova/FedDyn stack.
- Added optional `--feddrift_mu` / `--feddrift_beta` client-local EMA drift correction inspired by FedDC/FedRed residual drift correction. The default `feddrift_mu=0.0` preserves existing behavior and adds no FLModel params, metadata, or aggregation keys.
- FedDrift improved the best score: `feddrift_mu=5e-5, beta=0.9` scored `0.911400`; `mu=1e-4, beta=0.9` scored `0.910300` and was discarded.
- FedDrift narrowing improved again: `feddrift_mu=2.5e-5, beta=0.9` scored `0.913200`; `7.5e-5` regressed to `0.905300`.
- FedDrift lower-side mu neighbors did not improve: `1.25e-5` and `3.75e-5` both scored `0.909700`.
- FedDrift beta neighbors did not improve: `beta=0.8` scored `0.908900`; `0.95` scored `0.908100`.
- Seventeenth literature loop selected a FedDrift-enabled server-LR retune; it did not improve. `server_lr=1.9375` scored `0.909300`; `1.8125` scored `0.908800`.
- The reserved FedDrift-enabled server-momentum retune also did not improve: `server_momentum=0.40` scored `0.910200`; `0.30` scored `0.909500`.
- Eighteenth literature loop selected a FedDrift-enabled weight-decay retune; it did not improve. `weight_decay=3.75e-4` scored `0.909700`; `3.25e-4` scored `0.906800`.
- The reserved FedDrift-enabled FedDyn-alpha interaction also did not improve: `feddyn_alpha=2e-4` scored `0.911900`; `5e-5` scored `0.909800`.
- Nineteenth literature loop selected a FedDrift-enabled client learning-rate retune; it did not improve. `lr=0.055` scored `0.911500`; `0.045` scored `0.906900`.
- The reserved FedDrift-enabled epoch-count audit also did not improve: `aggregation_epochs=6` scored `0.911500`; `4` scored `0.905000`.
- The reserved FedDrift EMA-state clipping variant did not improve: `clip_norm=2.0` scored `0.910900`; `1.0` scored `0.906500`. The optional clipping code path was reverted.
- Twentieth literature loop selected a FedDrift-enabled cosine scheduler-floor sweep; it did not improve. `eta_min_factor=0.003` scored `0.910900`; `0.03` scored `0.906100`.
- Twenty-first literature loop selected a labeled registered-architecture subcampaign; it did not improve. `moderate_cnn_small_head` scored `0.912100`; `moderate_cnn_norm` scored `0.904600`.
- Twenty-second literature loop selected a very-light FedProx interaction; it did not improve. `mu=1e-6` scored `0.908500`; `5e-6` scored `0.906000`.
- Twenty-third literature loop selected a default-off local AdamW optimizer mutation; it failed badly. `lr=0.0005` scored `0.252000`; `0.001` scored `0.100000`. The optional AdamW code path was reverted.
- Twenty-fourth literature loop selected default-off client gradient clipping; it did not improve. `clip_norm=5.0` scored `0.908700`; `1.0` scored `0.898400`. The optional clipping code path was reverted.
- Twenty-fifth literature loop selected a small-head architecture weight-decay retune; it did not improve. `weight_decay=4.5e-4` scored `0.910300`; `2.5e-4` scored `0.907600`.
- Twenty-sixth literature loop selected a small-head architecture server-LR retune; it did not improve. `server_lr=1.8125` scored `0.908500`; `1.9375` scored `0.908400`.
- Twenty-seventh literature loop selected a current-stack client momentum audit after closing the small-head branch.
- Current-stack client momentum retune did not improve: `momentum=0.85` scored `0.906100`; `0.95` scored `0.905800`.
- Twenty-eighth literature loop selected default-off local-only mixup after rejecting FedMix/MAFL averaged-data exchange as a protocol change.
- Local-only mixup improved the best: `mixup_alpha=0.2` scored `0.914100`; `0.1` scored `0.913700`.
- Mixup alpha narrowing did not improve: `mixup_alpha=0.15` scored `0.911800`; `0.3` scored `0.906100`.
- Mixup-enabled weight-decay interaction did not improve: `weight_decay=3.0e-4` scored `0.912100`; `4.0e-4` scored `0.906600`.
- Mixup-enabled server-LR interaction did not improve: `server_lr=1.9375` scored `0.913200`; `1.8125` scored `0.910000`.
- Twenty-ninth literature loop selected default-off local-only CutMix after mixup improved but post-mixup retunes failed.
- CutMix failed and was reverted: `cutmix_alpha=0.5` scored `0.904800`; `1.0` scored `0.897800`.
- Thirtieth literature loop selected default-off focal loss with the kept mixup setting.
- Focal loss failed and was reverted: `focal_gamma=1.0` scored `0.906000`; `2.0` scored `0.896800`.
- Thirty-first literature loop selected default-off effective-number class-balanced loss with the kept mixup setting.
- Effective-number class-balanced loss failed and was reverted: `class_balance_beta=0.99` scored `0.906000`; `0.999` crashed with NVFlare abort/score-extraction failure.
- Post-class-balanced local-compute audit under the mixup best also did not improve: `aggregation_epochs=4` scored `0.905400`; `6` scored `0.911300`. Keep `aggregation_epochs=5`.

## Literature basis

- FedProx: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. "Federated Optimization in Heterogeneous Networks." MLSys 2020; arXiv:1812.06127.
- FedAvg local epochs: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017; arXiv:1602.05629.
- FedOpt / FedAvgM / FedAdam: Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneczny, Sanjiv Kumar, H. Brendan McMahan. "Adaptive Federated Optimization." ICLR 2021; arXiv:2003.00295.
- SCAFFOLD: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020; arXiv:1910.06378.
- FedLC: Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu. "Federated Learning with Label Distribution Skew via Logits Calibration." ICML 2022; arXiv:2209.00189.
- FedNova: Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS 2020; arXiv:2007.07481.
- FedBN / normalization in FL: Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." ICLR 2021.
- FedRed / DANE drift correction: Xiaowen Jiang, Anton Rodomanov, Sebastian U. Stich. "Federated Optimization with Doubly Regularized Drift Correction." arXiv:2404.08447.
- FedDC drift correction: Liang Gao, Huazhu Fu, Li Li, Yingwen Chen, Ming Xu, Cheng-Zhong Xu. "FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction." CVPR 2022; arXiv:2203.11751.
- FedDyn dynamic regularization: Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, Venkatesh Saligrama. "Federated Learning Based on Dynamic Regularization." ICLR 2021.
- FedLAW / weighted aggregation: Zexi Li, Tao Lin, Xinyi Shang, Chao Wu. "Revisiting Weighted Aggregation in Federated Learning with Neural Networks." ICML 2023; arXiv:2302.10911.
- Adaptive clipping: Galen Andrew, Om Thakkar, Brendan McMahan, Swaroop Ramaswamy. "Differentially Private Learning with Adaptive Clipping." NeurIPS 2021.
- Momentum analysis: Ziheng Cheng, Xinmeng Huang, Pengfei Wu, Kun Yuan. "Momentum Benefits Non-IID Federated Learning Simply and Provably." arXiv:2306.16504.
- Label smoothing in FL: Yeji Cho, Junghyun Kim. "FedENLC: An End-to-End Noisy Label Correction Framework in Federated Learning." Mathematics 2026; doi:10.3390/math14020290.
- Federated domain generalization with label smoothing: Milad Soltany, Farhad Pourpanah, Mahdiyar Molahasani Majdabadi, Michael Greenspan, Ali Etemad. "Federated Domain Generalization with Label Smoothing and Balanced Decentralized Training." arXiv:2412.11408.
- Label smoothing implementation: PyTorch `torch.nn.CrossEntropyLoss(label_smoothing=...)` documentation.
- FedZMG: Fotios Zantalis, Evangelos Zervas, Grigorios Koulouras. "FedZMG: Efficient Client-Side Optimization in Federated Learning." arXiv:2602.18384.
- SAM: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. "Sharpness-Aware Minimization for Efficiently Improving Generalization." arXiv:2010.01412.
- FedSAM: Zhe Qu, Xingyu Li, Rui Duan, Yao Liu, Bo Tang, Zhuo Lu. "Generalized Federated Learning via Sharpness Aware Minimization." arXiv:2206.02618.
- Auto-tuned clients: Junhyung Lyle Kim, Mohammad Taha Toghani, Cesar A. Uribe, Anastasios Kyrillidis. "Adaptive Federated Learning with Auto-Tuned Clients." arXiv:2306.11201.
- FedCM: Jing Xu, Sen Wang, Liwei Wang, Andrew Chi-Chih Yao. "FedCM: Federated Learning with Client-level Momentum." arXiv:2106.10874.
- FedMix: Tehrim Yoon, Sumin Shin, Sung Ju Hwang, Eunho Yang. "FedMix: Approximation of Mixup under Mean Augmented Federated Learning." ICLR 2021; arXiv:2107.00233.
- CCVR: Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, Jiashi Feng. "No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data." arXiv:2106.05001.
- Mixup: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz. "mixup: Beyond Empirical Risk Minimization." ICLR 2018; arXiv:1710.09412.
- CutMix: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV 2019; arXiv:1905.04899.
- Cutout: Terrance DeVries, Graham W. Taylor. "Improved Regularization of Convolutional Neural Networks with Cutout." arXiv:1708.04552.
- RandAugment: Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le. "RandAugment: Practical automated data augmentation with a reduced search space." arXiv:1909.13719.
- Focal loss: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar. "Focal Loss for Dense Object Detection." ICCV 2017; arXiv:1708.02002.
- Class-balanced loss: Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie. "Class-Balanced Loss Based on Effective Number of Samples." CVPR 2019; arXiv:1901.05555.

## Run analysis

The calibration result now favors FedNova-style normalized DIFF aggregation plus small client-local drift corrections and local-only mixup with the original `moderate_cnn` architecture. The best stack is `--aggregator fednova`, `server_lr=1.875`, `server_momentum=0.35`, default client LR and momentum, epoch-based local training with `aggregation_epochs=5`, `weight_decay=3.5e-4`, `--gradient_centralization`, `--feddyn_alpha 1e-4`, `--feddrift_mu 2.5e-5 --feddrift_beta 0.9`, and `--mixup_alpha 0.2`. FedLC, label smoothing, SAM/FedSAM, FedProx, weight-power flattening, exact local steps, FedDrift state clipping, scheduler-floor retunes, local-loss mutations after mixup, and registered architecture variants did not improve. FedAdam-style adaptive server variants are currently unsafe or ineffective at tested settings. Exact local-step training is operationally unreliable at width 2 and lower-scoring at width 1, so stop that axis under this stack. Client-LR, client-momentum, scheduler-floor, epoch-count, and FedProx neighbors around the FedDrift best stack did not improve, and the post-mixup epoch-count audit also regressed, so keep default client LR, default client momentum, default cosine floor, `aggregation_epochs=5`, and `fedproxloss_mu=0`. The small-head architecture came close but did not beat the current best, and its weight-decay and server-LR retunes regressed, so keep `moderate_cnn` and close the small-head branch unless a later literature loop provides a stronger reason. The shared-memory validation crash at candidate width 4 is a resource-contention signal, so subsequent batches should use `PARALLEL_CANDIDATES=2` for epoch-based runs. Parallel run launches should also set unique pycache prefixes to avoid validator races.

## Contract check

- No FL protocol fields were changed.
- Gradient centralization is client-local and does not alter FLModel params, metadata, aggregation keys, or evaluation.
- FedDyn-style dynamic regularization is client-local state inside the existing client loop and adds no FLModel params, metadata, aggregation keys, or evaluation changes.
- FedDrift EMA correction is client-local state inside the existing client loop and adds no FLModel params, metadata, aggregation keys, or evaluation changes.
- Local-only mixup is client-local loss/data interpolation inside each batch and adds no FLModel params, metadata, aggregation keys, shared data, or evaluation changes.
- FedNova is server-local and reuses existing DIFF params plus `NUM_STEPS_CURRENT_ROUND`; it adds no client metadata.
- All completed candidates used `--cross_site_eval`, `--num_rounds 20`, `--max_model_params 5000000`, and `--final_eval_clients site-1`; architecture-subcampaign rows were explicitly labeled when using `--model_arch moderate_cnn_small_head`.
- DIFF upload, `NUM_STEPS_CURRENT_ROUND`, and strict state-dict loading remain governed by the existing validated code.

## Rollback risk

Low to medium. The kept code mutations are optional gradient centralization behind `--gradient_centralization`, optional FedNova aggregation behind `--aggregator fednova`, optional FedDyn-style client regularization behind `--feddyn_alpha`, optional FedDrift EMA correction behind `--feddrift_mu`, and optional local-only mixup behind `--mixup_alpha`; default behavior remains unchanged and validation has passed. The unsuccessful optional SAM, FedNova weight-power, FedDrift state-clipping, local AdamW, client gradient-clipping, CutMix, focal-loss, and class-balanced-loss code paths have been reverted.

## Next mutation

Class-balanced loss failed and was reverted, and the post-mixup epoch-count audit regressed. Avoid more client-loss code until another literature reset; next CLI-only continuation should retune server momentum around `0.35` under the kept mixup stack.
