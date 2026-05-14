# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-cifar10-algocal-20260512` at `a12d741f3`
- **Rows analyzed:** 260 total, 258 scored
- **Best score:** 0.909600 at experiment `#244`
- **Baseline:** 0.850900 at experiment `#0`
- **Lift:** +0.058700 absolute, 6.9% relative
- **Runtime cost:** 49.88h aggregate; 11.5m average over 260 timed candidates
- **Agent model/effort:** Agent model/effort telemetry unavailable in this Codex runtime; no real model or effort output was pasted into the re...
- **Agent/tooling cost:** Agent cost telemetry unavailable in this Codex runtime; no real cost output was pasted into the reporting prompt. Exp...
- **Best status:** `keep`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #244 |
| score | 0.909600 |
| delta vs baseline | +0.058700 |
| relative lift | 6.9% |
| status | keep |
| commit | bd769ed1a |
| runtime | 14.1m |
| target | tasks/cifar10/client.py |
| description | FedAvgM ep8 local_train_steps=990 label_smoothing=0.01 mixup_alpha=0.03 eta_min_factor=0.001 weight_decay=3.5e-4 client_momentum=0.890 server_momentum=0.15 |
| artifact | /tmp/nvflare/simulation/r101_lts990_ls001_mix003_wd35e5_cm0890_eta001_sm015 |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --aggregation_epochs 8 --local_train_steps 990 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1 --aggregator fedavgm --server_lr 1.75 --server_momentum 0.15 --weight_decay 3.5e-4 --momentum 0.890 --mixup_alpha 0.03 --label_smoothing 0.01 --cosine_lr_eta_min_factor 0.001 --name r101_lts990_ls001_mix003_wd35e5_cm0890_eta001_sm015
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.850900 | baseline | weighted baseline default H100 budget | configuration change |  |
| #2 | 0.855100 | +0.003700 | calibration step 0 builtin FedAvg audit | configuration change |  |
| #11 | 0.858900 | +0.002700 | FedAvgM sweep server_lr=1.5 momentum=0.4 | server momentum, server diff amplification |  |
| #15 | 0.862700 | +0.003800 | FedAvgM sweep server_lr=1.5 momentum=0.0 | server momentum, server diff amplification |  |
| #17 | 0.867200 | +0.002500 | FedAvgM refine server_lr=1.65 momentum=0.2 | server momentum, server diff amplification |  |
| #37 | 0.877700 | +0.009200 | FedAvgM local compute aggregation_epochs=8 lr=1.75 momentum=0.2 | server momentum, server diff amplification |  |
| #56 | 0.891900 | +0.012800 | FedAvgM ep8 retry width2 weight_decay=5e-4 lr=1.75 momentum=0.2 | server momentum, server diff amplification, weight decay |  |
| #58 | 0.895700 | +0.003800 | FedAvgM ep8 refine weight_decay=3e-4 lr=1.75 momentum=0.2 | server momentum, server diff amplification, weight decay |  |
| #61 | 0.899100 | +0.003400 | FedAvgM ep8 refine weight_decay=4e-4 lr=1.75 momentum=0.2 | server momentum, server diff amplification, weight decay |  |
| #244 | 0.909600 | +0.001900 | FedAvgM ep8 local_train_steps=990 label_smoothing=0.01 mixup_alpha=0.... | server momentum, server diff amplification, label smoothing, weight decay |  |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 49.88h |
| average runtime per timed candidate | 11.5m |
| timed candidates | 260 |
| candidate rows | 0 |
| kept rows | 22 |
| crash rows | 13 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
Agent model/effort telemetry unavailable in this Codex runtime; no real model or effort output was pasted into the reporting prompt.
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
Agent cost telemetry unavailable in this Codex runtime; no real cost output was pasted into the reporting prompt. Experiment runtime cost is reported from results.tsv.
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | calibration step 6 FedAdam server_lr=1.0 beta1=0.9 beta2=0.99... |
| 1 | FedAvgM refine server_lr=1.65 momentum=0.3 |
| 1 | FedAvgM refine retry server_lr=1.65 momentum=0.1 |
| 1 | FedAvgM refine retry server_lr=1.65 momentum=0.15 |
| 1 | FedAvgM refine retry server_lr=1.65 momentum=0.25 |
| 1 | FedAvgM refine retry server_lr=1.65 momentum=0.3 |
| 1 | FedAvgM ep8 weight_decay=1e-5 lr=1.75 momentum=0.2 |
| 1 | FedAvgM ep8 weight_decay=1e-3 lr=1.75 momentum=0.2 |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Karimireddy20 SCAFFOLD | 1 | 0.888900 | SCAFFOLD ep8 weight_decay=4e-4 [src: Karimireddy20 SCAFFOLD] | SCAFFOLD control variates, weight decay | helped |
| Li20 FedProx | 4 | 0.892900 | FedAvgM ep8 FedProx mu=1e-5 client_momentum=0.895 server_momentum=0.1... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Reddi21 FedOpt | 3 | 0.811600 | FedYogi damped ep8 server_lr=0.1 tau=1e-2 client_momentum=0.895 weigh... | server diff amplification, weight decay | not confirmed |
| Zhang18 mixup; Yoon21 FedMix | 4 | 0.896500 | FedAvgM ep8 Mixup alpha=0.05 client_momentum=0.895 server_momentum=0.... | server momentum, server diff amplification, weight decay | helped |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

| experiment | score | description | mechanism |
| --- | --- | --- | --- |
| #5 | 0.000000 | calibration step 6 FedAdam server_lr=1.0 beta1=0.9 beta2=0.99 tau=1e-3 | server diff amplification |
| #21 | 0.000000 | FedAvgM refine server_lr=1.65 momentum=0.3 | server momentum, server diff amplification |
| #22 | 0.000000 | FedAvgM refine retry server_lr=1.65 momentum=0.1 | server momentum, server diff amplification |
| #23 | 0.000000 | FedAvgM refine retry server_lr=1.65 momentum=0.15 | server momentum, server diff amplification |
| #24 | 0.000000 | FedAvgM refine retry server_lr=1.65 momentum=0.25 | server momentum, server diff amplification |
| #25 | 0.000000 | FedAvgM refine retry server_lr=1.65 momentum=0.3 | server momentum, server diff amplification |
| #50 | 0.000000 | FedAvgM ep8 weight_decay=1e-5 lr=1.75 momentum=0.2 | server momentum, server diff amplification, weight decay |
| #51 | 0.000000 | FedAvgM ep8 weight_decay=1e-3 lr=1.75 momentum=0.2 | server momentum, server diff amplification, weight decay |

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| crash | 13 |
| discard | 223 |
| keep | 22 |
| literature | 2 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #244 | 0.909600 | 14.1m | keep | FedAvgM ep8 local_train_steps=990 label_smoothing=0.01 mixup_alpha=0.03 eta_min_facto... |
| 2 | #253 | 0.908400 | 13.5m | discard | FedAvgM ep8 local_train_steps=990 label_smoothing=0.01 mixup_alpha=0.03 eta_min_facto... |
| 3 | #245 | 0.908000 | 14.1m | discard | FedAvgM ep8 local_train_steps=990 label_smoothing=0.02 mixup_alpha=0.03 eta_min_facto... |
| 4 | #224 | 0.907700 | 13.6m | keep | FedAvgM ep8 local_train_steps=990 mixup_alpha=0.03 eta_min_factor=0.001 weight_decay=... |
| 5 | #232 | 0.907100 | 13.5m | discard | FedAvgM ep8 local_train_steps=990 mixup_alpha=0.03 eta_min_factor=0.001 weight_decay=... |
| 6 | #247 | 0.906800 | 14.2m | discard | FedAvgM ep8 local_train_steps=990 label_smoothing=0.015 mixup_alpha=0.03 eta_min_fact... |
| 7 | #256 | 0.906700 | 14.0m | discard | FedAvgM ep8 local_train_steps=990 label_smoothing=0.01 mixup_alpha=0.03 eta_min_facto... |
| 8 | #223 | 0.906400 | 13.3m | keep | FedAvgM ep8 local_train_steps=1000 mixup_alpha=0.03 eta_min_factor=0.001 weight_decay... |
| 9 | #249 | 0.906400 | 14.3m | discard | FedAvgM ep8 local_train_steps=995 label_smoothing=0.01 mixup_alpha=0.03 eta_min_facto... |
| 10 | #226 | 0.905700 | 12.9m | discard | FedAvgM ep8 local_train_steps=992 mixup_alpha=0.03 eta_min_factor=0.001 weight_decay=... |

