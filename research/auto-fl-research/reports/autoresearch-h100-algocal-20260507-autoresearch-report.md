# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-algocal-20260507` at `3e3d8dc9e`
- **Rows analyzed:** 279 total, 274 scored
- **Best score:** 0.916900 at experiment `#252`
- **Baseline:** 0.852100 at experiment `#0`
- **Lift:** +0.064800 absolute, 7.6% relative
- **Runtime cost:** 49.56h aggregate; 10.7m average over 279 timed candidates
- **Agent model/effort:** Agent model output, optional: GPT-5.5 Agent effort output, optional: xhigh
- **Agent/tooling cost:** Agent cost telemetry unavailable; no agent cost output was pasted. Experiment runtime cost is reported from results.tsv.
- **Best status:** `discard`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #252 |
| score | 0.916900 |
| delta vs baseline | +0.064800 |
| relative lift | 7.6% |
| status | discard |
| commit | f6c812245 |
| runtime | 12.7m |
| target | client.py |
| description | Scheduler eta-min factor 0.00015 tight check around FedZMG floor near-miss [src: Zantalis26 FedZMG arXiv:2602.18384] |
| artifact | /tmp/nvflare/simulation/fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --aggregation_epochs 7 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --aggregator fedavgm --server_lr 1.8 --server_momentum 0.475 --lr 0.045 --momentum 0.925 --weight_decay 5e-4 --cosine_lr_eta_min_factor 0.00015 --fedproxloss_mu 3e-5 --zero_mean_gradients --final_eval_clients site-1 --name fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta000015_mu3e5_zmg
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.852100 | baseline | baseline weighted default budget | configuration change |  |
| #9 | 0.858900 | +0.002700 | FedAvgM server_lr sweep lr 1.5 momentum 0.4 | server momentum, server diff amplification |  |
| #12 | 0.863300 | +0.004400 | FedAvgM server_lr sweep lr 1.75 momentum 0.4 | server momentum, server diff amplification |  |
| #24 | 0.866000 | +0.002700 | FedAvgM local compute sweep aggregation_epochs 6 lr 1.75 momentum 0.4 | server momentum, server diff amplification |  |
| #42 | 0.875800 | +0.004400 | Client lr sweep lr 0.04 with FedAvgM 1.80 momentum 0.40 aggregation_e... | server momentum, server diff amplification |  |
| #47 | 0.898500 | +0.022700 | Client weight_decay sweep 5e-4 lr 0.04 with FedAvgM 1.80 momentum 0.4... | server momentum, server diff amplification, weight decay |  |
| #109 | 0.903400 | +0.003100 | Server momentum retune 0.45 with eta_min_factor 0.005 current best stack | server momentum, server diff amplification, weight decay |  |
| #227 | 0.913700 | +0.007300 | FedZMG zero-mean gradients on active FedAvgM FedProx stack [src: Zant... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |
| #235 | 0.916700 | +0.002900 | Client lr 0.045 around FedZMG new best stack [src: Zantalis26 FedZMG ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |
| #252 | 0.916900 | +0.000200 | Scheduler eta-min factor 0.00015 tight check around FedZMG floor near... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 49.56h |
| average runtime per timed candidate | 10.7m |
| timed candidates | 279 |
| candidate rows | 0 |
| kept rows | 20 |
| crash rows | 4 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
Agent model output, optional: GPT-5.5  Agent effort output, optional: xhigh
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
Agent cost telemetry unavailable; no agent cost output was pasted. Experiment runtime cost is reported from results.tsv.
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | algo calibration FedAdam |
| 1 | FedAvgM narrow server_lr sweep lr 1.8 momentum 0.4 |
| 1 | FedAvgM narrow server_lr sweep lr 1.9 momentum 0.4 |
| 1 | FedAvgM narrow server_lr sweep lr 1.6 momentum 0.4 |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Karimireddy20 SCAFFOLD arXiv:1910.06378 | 2 | 0.886800 | Tuned SCAFFOLD protocol mode lr 0.04 momentum 0.925 wd 5e-4 [src: Kar... | SCAFFOLD control variates, weight decay | helped |
| Karimireddy20 SCAFFOLD; Li20 FedProx | 1 | 0.888300 | SCAFFOLD metadata mode with active FedProx lower-floor client stack [... | FedProx/client drift regularization, SCAFFOLD control variates, weight decay | helped |
| Li20 FedProx | 62 | 0.906400 | Exact local_train_steps 768 around FedProx lower-floor server_momentu... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127 | 1 | 0.897600 | FedProx mu 1e-5 on current best FedAvgM stack [src: Li20 FedProx arXi... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127; Zantalis26 FedZMG arXiv:2602.18384 | 6 | 0.914700 | FedProx mu 4e-5 tight bracket around kept FedZMG stack [src: Li20 Fed... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li21 FedRS KDD:10.1145/3447548.3467254 | 1 | 0.901200 | FedRS restricted softmax alpha 0.5 on active FedAvgM FedProx stack [s... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Muller19 label smoothing NeurIPS | 2 | 0.904400 | Label smoothing 0.05 on active FedAvgM FedProx stack [src: Muller19 l... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Qu22 FedSAM arXiv:2206.02618 | 1 | 0.910700 | Reduced-epoch FedSAM rho 0.03 on kept FedZMG stack [src: Qu22 FedSAM ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Reddi21 FedOpt | 2 | 0.750900 | Conservative FedAdam server_lr 0.1 tau 0.1 with active FedProx client... | server diff amplification, FedProx/client drift regularization, weight decay | not confirmed |
| Reddi21 FedOpt arXiv:2003.00295 | 1 | 0.807800 | Conservative FedAdam server optimizer on current client stack [src: R... | server diff amplification, weight decay | not confirmed |
| Reddi21 FedOpt; Li20 FedProx | 2 | 0.904700 | Server momentum 0.4825 with eta_min_factor 0.000125 near-miss FedAvgM... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Reddi21 FedOpt arXiv:2003.00295 | 1 | 0.899100 | FedNova-style step-normalized FedAvgM on active FedProx stack [src: W... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Yashwanth24 ASD arXiv:2305.19600 | 1 | 0.901000 | Global self-distillation alpha 0.05 T2 with current FedAvgM best stac... | server momentum, server diff amplification, weight decay | helped |
| Yoon21 FedMix arXiv:2107.00233; Sang24 mixup-noise arXiv:2409.13235; Zhang17 mixup arXiv:1710.09412 | 1 | 0.914800 | Local mixup alpha 0.2 on kept FedZMG stack [src: Yoon21 FedMix arXiv:... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zantalis26 FedZMG arXiv:2602.18384 | 41 | 0.916900 | Scheduler eta-min factor 0.00015 tight check around FedZMG floor near... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zhang21 clipping arXiv:2106.13673 | 3 | 0.916700 | Update clipping norm 45 on kept FedZMG stack [src: Zhang21 clipping a... | server momentum, server diff amplification, FedProx/client drift regularization, gradient clipping, weight decay | helped |
| Zhang22 FedLC arXiv:2209.00189 | 1 | 0.899200 | FedLC logit calibration tau 0.5 on active FedAvgM FedProx stack [src:... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

| experiment | score | description | mechanism |
| --- | --- | --- | --- |
| #5 | 0.000000 | algo calibration FedAdam | server diff amplification |
| #17 | 0.000000 | FedAvgM narrow server_lr sweep lr 1.8 momentum 0.4 | server momentum, server diff amplification |
| #18 | 0.000000 | FedAvgM narrow server_lr sweep lr 1.9 momentum 0.4 | server momentum, server diff amplification |
| #19 | 0.000000 | FedAvgM narrow server_lr sweep lr 1.6 momentum 0.4 | server momentum, server diff amplification |
| #185 | 0.710600 | Conservative FedAdam server_lr 0.05 tau 0.1 with active FedProx clien... | server diff amplification, FedProx/client drift regularization, weight decay |
| #140 | 0.742500 | Scheduler off with server momentum 0.45 current best stack | server momentum, server diff amplification, weight decay |
| #186 | 0.750900 | Conservative FedAdam server_lr 0.1 tau 0.1 with active FedProx client... | server diff amplification, FedProx/client drift regularization, weight decay |
| #132 | 0.764500 | Aggregator axis robust median with current best client schedule | weight decay |

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| crash | 4 |
| discard | 250 |
| keep | 20 |
| literature | 5 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #252 | 0.916900 | 12.7m | discard | Scheduler eta-min factor 0.00015 tight check around FedZMG floor near-miss [src: Zant... |
| 2 | #235 | 0.916700 | 12.5m | keep | Client lr 0.045 around FedZMG new best stack [src: Zantalis26 FedZMG arXiv:2602.18384] |
| 3 | #271 | 0.916700 | 12.7m | discard | Update clipping norm 45 on kept FedZMG stack [src: Zhang21 clipping arXiv:2106.13673] |
| 4 | #272 | 0.916700 | 12.7m | discard | Update clipping norm 60 on kept FedZMG stack [src: Zhang21 clipping arXiv:2106.13673] |
| 5 | #239 | 0.915600 | 12.8m | discard | Client lr 0.04625 tight bracket around FedZMG client-lr best [src: Zantalis26 FedZMG ... |
| 6 | #251 | 0.915100 | 12.8m | discard | Scheduler eta-min factor 0.0002 around FedZMG client-lr 0.045 stack [src: Zantalis26 ... |
| 7 | #270 | 0.914800 | 13.2m | discard | Local mixup alpha 0.2 on kept FedZMG stack [src: Yoon21 FedMix arXiv:2107.00233; Sang... |
| 8 | #277 | 0.914800 | 12.7m | discard | Client lr 0.04375 under FedZMG scheduler high-water floor [src: Zantalis26 FedZMG arX... |
| 9 | #260 | 0.914700 | 12.7m | discard | FedProx mu 4e-5 tight bracket around kept FedZMG stack [src: Li20 FedProx arXiv:1812.... |
| 10 | #253 | 0.914600 | 12.7m | discard | Scheduler eta-min factor 0.00025 tight check around FedZMG floor near-miss [src: Zant... |

