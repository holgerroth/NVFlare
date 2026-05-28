# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-continue-20260527` at `13c00a019`
- **Rows analyzed:** 157 total, 155 scored
- **Best score:** 0.859900 at experiment `#46`
- **Baseline:** 0.000000 at experiment `#1`
- **Lift:** +0.859900 absolute, 0.0% relative
- **Runtime cost:** 9.15h aggregate; 3.5m average over 156 timed candidates
- **Agent model/effort:** Agent model/effort telemetry unavailable in this Claude Code runtime; /model and /effort are interactive and were not...
- **Agent/tooling cost:** No agent/tooling cost telemetry was provided. The report still includes experiment runtime cost from results.tsv.
- **Best status:** `candidate`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #46 |
| score | 0.859900 |
| delta vs baseline | +0.859900 |
| relative lift | 0.0% |
| status | candidate |
| commit | 16895ffcf |
| runtime | 3.6m |
| target | tasks/cifar10/client.py |
| description | momentum=0.93 lr=0.001 seed=202 |
| artifact | /tmp/nvflare/simulation/autofl_cifar10_weighted_alpha0.5_seed202 |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --alpha 0.5 --seed 202 --batch_size 64 --eval_batch_size 1024 --model_arch moderate_cnn --max_model_params 5000000 --aggregation_epochs 4 --final_eval_clients 1
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.846000 | baseline | weighted aggregator | configuration change |  |
| #26 | 0.856900 | +0.010900 | aggregator=scaffold | SCAFFOLD control variates |  |
| #46 | 0.859900 | +0.003000 | momentum=0.93 lr=0.001 seed=202 | configuration change |  |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 9.15h |
| average runtime per timed candidate | 3.5m |
| timed candidates | 156 |
| candidate rows | 131 |
| kept rows | 1 |
| crash rows | 21 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
Agent model/effort telemetry unavailable in this Claude Code runtime; /model and /effort are interactive and were not provided to the reporting agent.
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
No agent/tooling cost telemetry was provided. The report still includes experiment runtime cost from results.tsv.
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | baseline weighted |
| 1 | builtin FedAvg audit |
| 1 | explicit FedAvg audit |
| 1 | FedProx light |
| 1 | explicit FedAvg with eval |
| 1 | explicit FedAvg alpha 0.5 sweep 5 |
| 1 | explicit FedAvg alpha 0.5 sweep 6 |
| 1 | explicit FedAvg batch_size 128 seed 123 |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Li20 FedProx arXiv:1812.06127 | 1 |  | Literature review: investigated adaptive momentum and regularization ... | FedProx/client drift regularization | not confirmed |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

No scored rows fell below the baseline.

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| candidate | 131 |
| crash | 21 |
| discard | 2 |
| keep | 1 |
| literature | 2 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #46 | 0.859900 | 3.6m | candidate | momentum=0.93 lr=0.001 seed=202 |
| 2 | #48 | 0.857600 | 3.6m | candidate | momentum=0.94 lr=0.001 seed=789 |
| 3 | #150 | 0.857600 | 4.8m | candidate | fedavgm server_lr=0.9 mom=0.3 seed=16 |
| 4 | #76 | 0.857200 | 3.6m | candidate | aggregator=fedavgm momentum=0.93 lr=0.0011 seed=7474 |
| 5 | #87 | 0.857100 | 3.5m | candidate | aggregator=fedavgm momentum=0.96 lr=0.0014 seed=9696 |
| 6 | #26 | 0.856900 | 4.5m | candidate | aggregator=scaffold |
| 7 | #134 | 0.856900 | 5.8m | candidate | SCAFFOLD metadata mode seed=0 |
| 8 | #102 | 0.856700 | 4.5m | candidate | aggregator=fedavg momentum=0.94 lr=0.0012 weight_decay=0.0002 seed=14212 |
| 9 | #55 | 0.856600 | 3.5m | candidate | momentum=0.85 lr=0.0012 seed=2630 |
| 10 | #104 | 0.855700 | 4.3m | candidate | aggregator=fedavg momentum=0.88 lr=0.001 seed=14612 |

