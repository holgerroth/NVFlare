# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-baseline-20260504` at `77ed0bba8`
- **Rows analyzed:** 387 total, 384 scored
- **Best score:** 0.913000 at experiment `#231`
- **Baseline:** 0.847200 at experiment `#0`
- **Lift:** +0.065800 absolute, 7.8% relative
- **Runtime cost:** 74.24h aggregate; 11.5m average over 387 timed candidates
- **Agent model/effort:** Agent: Claude Code
- **Agent/tooling cost:** Session
- **Best status:** `keep`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #231 |
| score | 0.913000 |
| delta vs baseline | +0.065800 |
| relative lift | 7.8% |
| status | keep |
| commit | a69dea78f |
| runtime | 11.8m |
| target | client.py |
| description | keep + lr=0.052 |
| artifact | /tmp/nvflare/simulation/keep_lr052 |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --aggregation_epochs 8 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --aggregator fedavgm --server_lr 2.0 --server_momentum 0.27 --weight_decay 2e-4 --label_smoothing 0.10 --mixup_alpha 0.32 --nesterov --momentum 0.92 --final_eval_clients site-1 --lr 0.052 --name keep_lr052
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.847200 | baseline | baseline weighted FedAvg | configuration change |  |
| #6 | 0.852000 | +0.003800 | explicit FedAvg audit [src: McMahan17 FedAvg arXiv:1602.05629] | configuration change | McMahan17 FedAvg arXiv:1602.05629 |
| #11 | 0.856200 | +0.004200 | FedAvgM lr=2.0 momentum=0.4 [src: Hsu19 FedAvgM arXiv:1909.06335] | server momentum, server diff amplification | Hsu19 FedAvgM arXiv:1909.06335 |
| #17 | 0.862200 | +0.003200 | FedAvgM lr=2.25 m=0.3 [src: Hsu19 FedAvgM arXiv:1909.06335] | server momentum, server diff amplification | Hsu19 FedAvgM arXiv:1909.06335 |
| #25 | 0.872200 | +0.007900 | FedAvgM lr=2.0 m=0.1 wd=5e-4 [src: Hsu19 FedAvgM] | server momentum, server diff amplification, weight decay | Hsu19 FedAvgM |
| #38 | 0.878400 | +0.003800 | FedAvgM stack + aggregation_epochs=5 | server momentum, server diff amplification, weight decay |  |
| #62 | 0.893000 | +0.009400 | kept stack + label_smoothing=0.1 [src: Szegedy16 arXiv:1512.00567] | server momentum, server diff amplification, label smoothing, weight decay | Szegedy16 arXiv:1512.00567 |
| #101 | 0.901700 | +0.003300 | ae=6 + mixup α=0.25 | server momentum, server diff amplification, label smoothing, weight decay |  |
| #145 | 0.909800 | +0.003200 | ae=8 nesterov + client mom=0.92 | server momentum, server diff amplification, label smoothing, weight decay |  |
| #231 | 0.913000 | +0.001500 | keep + lr=0.052 | server momentum, server diff amplification, label smoothing, weight decay |  |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 74.24h |
| average runtime per timed candidate | 11.5m |
| timed candidates | 387 |
| candidate rows | 0 |
| kept rows | 21 |
| crash rows | 20 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
Agent: Claude Code
Model: Opus 4.7 (1M context)
Effort: Max
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
Session
Total cost:            $332.48
Total duration (API):  1h 47m 2s
Total duration (wall): 1d 17h 28m
Total code changes:    285 lines added, 89 lines removed
Usage by model:
    claude-haiku-4-5:  1.2k input, 21 output, 0 cache read, 0 cache write ($0.0013)
     claude-opus-4-7:  42.6k input, 382.4k output, 183.8m cache read, 36.9m cache write ($332.48)
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | FedProx mu=1e-4 [src: Li20 FedProx arXiv:1812.06127] |
| 1 | builtin FedAvg audit [src: McMahan17 FedAvg arXiv:1602.05629] |
| 1 | FedProx mu=1e-5 [src: Li20 FedProx arXiv:1812.06127] |
| 1 | explicit FedAvg audit [src: McMahan17 FedAvg arXiv:1602.05629] |
| 1 | FedAdam slr=1.0 b1=0.9 b2=0.99 tau=1e-3 [src: Reddi21 FedOpt ... |
| 1 | FedAvgM lr=2.0 m=0.2 [src: Hsu19 FedAvgM arXiv:1909.06335] |
| 1 | FedAvgM wd=1e-3 |
| 1 | FedAvgM/ae5 + no_lr_scheduler |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Hsu19 FedAvgM | 2 | 0.874600 | FedAvgM lr=2.0 m=0.1 wd=1e-4 [src: Hsu19 FedAvgM] | server momentum, server diff amplification, weight decay | helped |
| Hsu19 FedAvgM arXiv:1909.06335 | 14 | 0.864300 | FedAvgM lr=2.0 m=0.1 [src: Hsu19 FedAvgM arXiv:1909.06335] | server momentum, server diff amplification | helped |
| Hsu19;Li20 | 1 | 0.862600 | FedAvgM lr=2.0 m=0.1 + FedProx mu=1e-5 [src: Hsu19;Li20] | server momentum, server diff amplification, FedProx/client drift regularization | helped |
| Karimireddy20 SCAFFOLD arXiv:1910.06378 | 1 | 0.854800 | SCAFFOLD metadata mode [src: Karimireddy20 SCAFFOLD arXiv:1910.06378] | SCAFFOLD control variates | helped |
| Li20 | 3 | 0.877500 | ae5/wd2e4/m02 + FedProx mu=5e-5 [src: Li20] | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127 | 4 | 0.850300 | FedProx mu=1e-4 [src: Li20 FedProx arXiv:1812.06127] | FedProx/client drift regularization | helped |
| Li21 MOON | 5 | 0.895000 | ae=6 stack + MOON μ=1.0 [src: Li21 MOON] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Li21 MOON CVPR arXiv:2103.16257 | 1 |  | literature cycle 3: implemented MOON contrastive [src: Li21 MOON CVPR... | configuration change | not confirmed |
| Li21 MOON arXiv:2103.16257 | 2 | 0.895000 | ae=6 stack + MOON μ=0.5 [src: Li21 MOON arXiv:2103.16257] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Loshchilov19 arXiv:1711.05101 | 1 | 0.821200 | kept stack + AdamW lr=1e-3 wd=1e-2 [src: Loshchilov19 arXiv:1711.05101] | server momentum, server diff amplification, label smoothing, weight decay | not confirmed |
| McMahan17 FedAvg arXiv:1602.05629 | 4 | 0.852000 | explicit FedAvg audit [src: McMahan17 FedAvg arXiv:1602.05629] | configuration change | helped |
| Mueller19 | 1 | 0.891600 | kept stack + LS=0.075 [src: Mueller19] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Mueller19 arXiv:1906.02629 | 1 | 0.888800 | kept stack + label_smoothing=0.05 [src: Mueller19 arXiv:1906.02629] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Pascanu13 | 1 | 0.891100 | LS=0.1 keep + grad_clip=3.0 (loose) [src: Pascanu13] | server momentum, server diff amplification, gradient clipping, label smoothing, weight decay | helped |
| Pascanu13 arXiv:1211.5063 | 1 | 0.883900 | kept stack + grad_clip_max_norm=1.0 [src: Pascanu13 arXiv:1211.5063] | server momentum, server diff amplification, gradient clipping, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295 | 1 | 0.000000 | FedAdam slr=1.0 b1=0.9 b2=0.99 tau=1e-3 [src: Reddi21 FedOpt arXiv:20... | server diff amplification | not confirmed |
| Szegedy16 | 3 | 0.891300 | kept stack + LS=0.12 [src: Szegedy16] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Szegedy16 LS arXiv:1512.00567; Mueller19 LS NeurIPS arXiv:1906.02629; Pascanu13 clip arXiv:1211.5063 | 1 |  | literature review: plateau at 0.8836; selected P1/P2/P3/P4 (label smo... | gradient clipping, label smoothing | not confirmed |
| Szegedy16 arXiv:1512.00567 | 1 | 0.893000 | kept stack + label_smoothing=0.1 [src: Szegedy16 arXiv:1512.00567] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Szegedy16;Pascanu13 | 1 | 0.883400 | kept stack + LS=0.1 + clip=1.0 [src: Szegedy16;Pascanu13] | server momentum, server diff amplification, gradient clipping, label smoothing, weight decay | helped |
| Yin18 arXiv:1803.01498 | 1 | 0.812000 | LS=0.1 stack + aggregator=median [src: Yin18 arXiv:1803.01498] | label smoothing, weight decay | not confirmed |
| Zhang17 | 4 | 0.896100 | kept stack + mixup α=0.3 retry [src: Zhang17] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Zhang17 arXiv:1710.09412 | 2 | 0.895900 | kept stack + mixup α=0.2 [src: Zhang17 arXiv:1710.09412] | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Zhang17 mixup arXiv:1710.09412; Loshchilov19 AdamW arXiv:1711.05101; Yin18 median arXiv:1803.01498 | 1 |  | literature review cycle 2: selected mixup α=0.2/0.4, AdamW lr=1e-3, m... | configuration change | not confirmed |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

| experiment | score | description | mechanism |
| --- | --- | --- | --- |
| #1 | 0.000000 | FedProx mu=1e-4 [src: Li20 FedProx arXiv:1812.06127] | FedProx/client drift regularization |
| #2 | 0.000000 | builtin FedAvg audit [src: McMahan17 FedAvg arXiv:1602.05629] | configuration change |
| #3 | 0.000000 | FedProx mu=1e-5 [src: Li20 FedProx arXiv:1812.06127] | FedProx/client drift regularization |
| #4 | 0.000000 | explicit FedAvg audit [src: McMahan17 FedAvg arXiv:1602.05629] | configuration change |
| #9 | 0.000000 | FedAdam slr=1.0 b1=0.9 b2=0.99 tau=1e-3 [src: Reddi21 FedOpt arXiv:20... | server diff amplification |
| #20 | 0.000000 | FedAvgM lr=2.0 m=0.2 [src: Hsu19 FedAvgM arXiv:1909.06335] | server momentum, server diff amplification |
| #28 | 0.000000 | FedAvgM wd=1e-3 | server momentum, server diff amplification, weight decay |
| #40 | 0.000000 | FedAvgM/ae5 + no_lr_scheduler | server momentum, server diff amplification, weight decay |

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| crash | 20 |
| discard | 343 |
| keep | 21 |
| literature | 3 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #231 | 0.913000 | 11.8m | keep | keep + lr=0.052 |
| 2 | #247 | 0.913000 | 12.2m | discard | lr=0.052 keep determinism rerun |
| 3 | #251 | 0.913000 | 12.2m | discard | keep lr=0.052 + smom=0.27 (vary cmom) |
| 4 | #275 | 0.913000 | 12.2m | discard | lr=0.052 + wd=2.0e-4 (control) |
| 5 | #279 | 0.913000 | 11.6m | discard | keep determinism rerun #4 |
| 6 | #300 | 0.913000 | 11.4m | discard | lr=0.052 keep determinism rerun #5 |
| 7 | #316 | 0.913000 | 11.6m | discard | lr=0.052 keep determinism rerun #6 |
| 8 | #340 | 0.913000 | 11.6m | discard | keep determinism #7 |
| 9 | #362 | 0.913000 | 11.9m | discard | lr=0.052 + grad_clip=100 (effectively off) |
| 10 | #363 | 0.913000 | 11.9m | discard | lr=0.052 + grad_clip=25 (very loose) |

