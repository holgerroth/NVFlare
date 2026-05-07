# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-algo-calibration-20260506` at `7a205902a`
- **Rows analyzed:** 248 total, 217 scored
- **Best score:** 0.916200 at experiment `#237`
- **Baseline:** 0.849800 at experiment `#0`
- **Lift:** +0.066400 absolute, 7.8% relative
- **Runtime cost:** 33.77h aggregate; 8.2m average over 248 timed candidates
- **Agent model/effort:** Agent model output:
- **Agent/tooling cost:** Agent cost telemetry unavailable; no cost output was pasted. Experiment runtime cost is reported from results.tsv.
- **Best status:** `keep`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #237 |
| score | 0.916200 |
| delta vs baseline | +0.066400 |
| relative lift | 7.8% |
| status | keep |
| commit | 18a81777f |
| runtime | 12.0m |
| target | client.py |
| description | Mixup_alpha=0.2 cosine_lr_eta_min_factor=0.003 lr=0.0475 under new scheduler-floor best stack [src: Loshchilov16 SGDR arXiv:1608.03983; Zhang17 mixup arXiv:1710.09412] |
| artifact | /tmp/nvflare/simulation/fednova_clr0475_lr1875_m035_wd35e5_gc_mixup02_eta003_feddyn1e4_feddrift2p5e5_ep5 |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --aggregation_epochs 5 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1 --aggregator fednova --server_lr 1.875 --server_momentum 0.35 --gradient_centralization --mixup_alpha 0.2 --weight_decay 3.5e-4 --feddyn_alpha 1e-4 --feddrift_mu 2.5e-5 --feddrift_beta 0.9 --cosine_lr_eta_min_factor 0.003 --lr 0.0475 --name fednova_clr0475_lr1875_m035_wd35e5_gc_mixup02_eta003_feddyn1e4_feddrift2p5e5_ep5
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.849800 | baseline | baseline weighted FedAvg; default H100 budget | configuration change |  |
| #6 | 0.854800 | +0.003600 | calibration SCAFFOLD metadata mode [src: Karimireddy20 SCAFFOLD arXiv... | SCAFFOLD control variates | Karimireddy20 SCAFFOLD arXiv:1910.06378 |
| #12 | 0.858900 | +0.002700 | FedAvgM sweep momentum=0.4 server_lr=1.5 [src: Reddi21 FedOpt arXiv:2... | server momentum, server diff amplification | Reddi21 FedOpt arXiv:2003.00295 |
| #13 | 0.864700 | +0.005800 | FedAvgM sweep server_lr=1.5 momentum=0.2 [src: Reddi21 FedOpt arXiv:2... | server momentum, server diff amplification | Reddi21 FedOpt arXiv:2003.00295 |
| #29 | 0.873900 | +0.009200 | weight_decay=5e-4 with FedAvgM best | server momentum, server diff amplification, weight decay |  |
| #32 | 0.878900 | +0.004800 | weight_decay=3e-4 with FedAvgM best | server momentum, server diff amplification, weight decay |  |
| #61 | 0.890900 | +0.009500 | gradient centralization with lighter wd=1e-4 [src: Zantalis26 FedZMG ... | server momentum, server diff amplification, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |
| #62 | 0.900700 | +0.009800 | gradient centralization with FedAvgM best [src: Zantalis26 FedZMG arX... | server momentum, server diff amplification, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |
| #87 | 0.909900 | +0.002600 | epoch5 FedAvgM server_lr=1.875 with gradient centralization [src: Red... | server momentum, server diff amplification, weight decay | Reddi21 FedOpt arXiv:2003.00295 |
| #237 | 0.916200 | +0.000600 | Mixup_alpha=0.2 cosine_lr_eta_min_factor=0.003 lr=0.0475 under new sc... | server momentum, server diff amplification, weight decay | Loshchilov16 SGDR arXiv:1608.03983; Zhang17 mixup arXiv:1710.09412 |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 33.77h |
| average runtime per timed candidate | 8.2m |
| timed candidates | 248 |
| candidate rows | 2 |
| kept rows | 24 |
| crash rows | 8 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
Agent model output:
gpt-5.5

Agent effort output:
xhigh
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
Agent cost telemetry unavailable; no cost output was pasted. Experiment runtime cost is reported from results.tsv.
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | calibration FedAdam server_lr=1.0 beta1=0.9 beta2=0.99 tau=1e... |
| 1 | FedAvgM sweep momentum=0.4 server_lr=1.75 [src: Reddi21 FedOp... |
| 1 | cosine eta_min_factor=0.001 with FedAvgM+wd best |
| 1 | FedAdagrad server optimizer beta1=0 server_lr=0.1 tau=1e-2 wi... |
| 1 | FedYogi server optimizer beta1=0 server_lr=0.1 tau=1e-2 with ... |
| 1 | FedDyn-enabled FedNova exact local_train_steps=600 audit [src... |
| 1 | FedDyn-enabled FedNova exact local_train_steps=500 audit [src... |
| 1 | Effective-number class-balanced beta=0.999 with mixup_alpha=0... |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Acar21 FedDyn ICLR; Loshchilov16 SGDR arXiv:1608.03983 | 2 | 0.915000 | Feddyn_alpha=5e-5 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under... | server momentum, server diff amplification, weight decay | helped |
| Acar21 FedDyn ICLR; Zhang17 mixup arXiv:1710.09412 | 3 | 0.911000 | Mixup_alpha=0.2 FedDyn ablation with feddyn_alpha=0 under FedDrift st... | server momentum, server diff amplification, weight decay | helped |
| Acar21 FedDyn OpenReview B7v4QMR6Z9w | 6 | 0.910900 | FedDyn-style client dynamic regularization alpha=1e-4 under FedNova b... | server momentum, server diff amplification, weight decay | helped |
| Acar21 FedDyn OpenReview B7v4QMR6Z9w; Gao22 FedDC arXiv:2203.11751 | 4 | 0.911900 | FedDrift-enabled FedNova feddyn_alpha=2e-4 interaction [src: Acar21 F... | server momentum, server diff amplification, weight decay | helped |
| Acar21 FedDyn OpenReview B7v4QMR6Z9w; McMahan17 FedAvg arXiv:1602.05629 | 2 | 0.909500 | FedDyn-enabled FedNova aggregation_epochs=6 local-compute audit [src:... | server momentum, server diff amplification, weight decay | helped |
| Acar21 FedDyn OpenReview B7v4QMR6Z9w; Wang20 FedNova arXiv:2007.07481 | 3 | 0.909300 | FedDyn-enabled FedNova exact local_train_steps=600 single-lane reliab... | server momentum, server diff amplification, weight decay | helped |
| Andrew21 adaptive clipping OpenReview RUQ1zwZR8_; Wang20 FedNova arXiv:2007.07481 | 2 | 0.910300 | FedNova median-norm update clipping factor=2.0 [src: Andrew21 adaptiv... | server momentum, server diff amplification, gradient clipping, weight decay | helped |
| Andrew21 adaptive clipping OpenReview RUQ1zwZR8_; Zantalis26 FedZMG arXiv:2602.18384 | 2 | 0.908700 | Client gradient clip_norm=5.0 under FedDrift best stack [src: Andrew2... | server momentum, server diff amplification, gradient clipping, weight decay | helped |
| Cheng23 momentum arXiv:2306.16504 | 2 | 0.906100 | Client momentum=0.85 under FedDrift best stack [src: Cheng23 momentum... | server momentum, server diff amplification, weight decay | helped |
| Cheng23 momentum arXiv:2306.16504; Acar21 FedDyn OpenReview B7v4QMR6Z9w | 2 | 0.910800 | FedDyn-enabled FedNova server_momentum=0.30 at alpha=1e-4 [src: Cheng... | server momentum, server diff amplification, weight decay | helped |
| Cheng23 momentum arXiv:2306.16504; Reddi21 FedOpt arXiv:2003.00295 | 4 | 0.910200 | FedDrift-enabled FedNova server_momentum=0.40 under drift-corrected b... | server momentum, server diff amplification, weight decay | helped |
| Cheng23 momentum arXiv:2306.16504; Xu21 FedCM arXiv:2106.10874 | 2 | 0.909200 | FedNova client momentum=0.925 narrow retune [src: Cheng23 momentum ar... | server momentum, server diff amplification, weight decay | helped |
| Cho26 FedENLC doi:10.3390/math14020290; Soltany24 arXiv:2412.11408 | 2 | 0.881200 | label_smoothing=0.05 with FedAvgM best [src: Cho26 FedENLC doi:10.339... | server momentum, server diff amplification, label smoothing, weight decay | helped |
| Cui19 class-balanced arXiv:1901.05555; Zhang17 mixup arXiv:1710.09412 | 2 | 0.906000 | Effective-number class-balanced beta=0.99 with mixup_alpha=0.2 under ... | server momentum, server diff amplification, weight decay | helped |
| Foret20 SAM arXiv:2010.01412; Qu22 FedSAM arXiv:2206.02618 | 2 | 0.907600 | epoch5 FedAvgM local SAM rho=0.01 with gradient centralization [src: ... | server momentum, server diff amplification, weight decay | helped |
| Gao22 FedDC CVPR; Jiang24 FedRed arXiv:2404.08447; Loshchilov16 SGDR arXiv:1608.03983 | 2 | 0.912800 | Feddrift_mu=3.75e-5 with lr=0.0475 cosine_lr_eta_min_factor=0.003 und... | server momentum, server diff amplification, weight decay | helped |
| Gao22 FedDC arXiv:2203.11751; Acar21 FedDyn OpenReview B7v4QMR6Z9w | 2 | 0.911500 | FedDrift-enabled FedNova client lr=0.055 under drift-corrected best s... | server momentum, server diff amplification, weight decay | helped |
| Gao22 FedDC arXiv:2203.11751; Jiang24 FedRed arXiv:2404.08447 | 15 | 0.913200 | FedDC/FedRed-inspired client EMA drift correction narrow mu=2.5e-5 be... | server momentum, server diff amplification, weight decay | helped |
| Karimireddy20 SCAFFOLD arXiv:1910.06378 | 2 | 0.859000 | SCAFFOLD with weight_decay=3e-4 [src: Karimireddy20 SCAFFOLD arXiv:19... | SCAFFOLD control variates, weight decay | helped |
| Kim24 Auto-Tuned Clients arXiv:2306.11201; Wang20 FedNova arXiv:2007.07481 | 2 | 0.910900 | FedDrift-enabled cosine eta_min_factor=0.003 under best FedNova stack... | server momentum, server diff amplification, weight decay | helped |
| Kim24 auto-tuned clients ICLR 2024 | 4 | 0.906600 | FedNova client lr=0.045 narrow retune [src: Kim24 auto-tuned clients ... | server momentum, server diff amplification, weight decay | helped |
| Kim24 auto-tuned clients arXiv:2306.11201; McMahan17 FedAvg arXiv:1602.05629 | 4 | 0.914500 | Mixup_alpha=0.2 client_lr=0.055 under FedDrift best stack [src: Kim24... | server momentum, server diff amplification, weight decay | helped |
| Li20 FedProx MLSys; Acar21 FedDyn OpenReview B7v4QMR6Z9w; Gao22 FedDC arXiv:2203.11751 | 2 | 0.908500 | FedDrift-enabled FedProx mu=1e-6 interaction under best FedNova stack... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127 | 2 | 0.849100 | calibration FedProx mu=1e-5 [src: Li20 FedProx arXiv:1812.06127] | FedProx/client drift regularization | not confirmed |
| Li20 FedProx arXiv:1812.06127; Reddi21 FedOpt arXiv:2003.00295 | 1 | 0.860100 | literature FedAvgM best + FedProx mu=1e-3 [src: Li20 FedProx arXiv:18... | server momentum, server diff amplification, FedProx/client drift regularization | helped |
| Li20 FedProx arXiv:1812.06127; Wang20 FedNova arXiv:2007.07481 | 2 | 0.909900 | FedNova FedProx mu=1e-4 under current best stack [src: Li20 FedProx a... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li21 FedBN OpenReview 6YEQUn0QICG | 1 | 0.905300 | architecture subcampaign: FedNova with registered moderate_cnn_norm G... | server momentum, server diff amplification, weight decay | helped |
| Li21 FedBN OpenReview 6YEQUn0QICG; Wang20 FedNova arXiv:2007.07481 | 1 | 0.904600 | Architecture subcampaign: moderate_cnn_norm under FedDrift best stack... | server momentum, server diff amplification, weight decay | helped |
| Lin17 focal arXiv:1708.02002; Zhang17 mixup arXiv:1710.09412 | 2 | 0.906000 | Focal gamma=1.0 with mixup_alpha=0.2 under FedDrift best stack [src: ... | server momentum, server diff amplification, weight decay | helped |
| Loshchilov16 SGDR arXiv:1608.03983; Zhang17 mixup arXiv:1710.09412 | 4 | 0.916200 | Mixup_alpha=0.2 cosine_lr_eta_min_factor=0.003 lr=0.0475 under new sc... | server momentum, server diff amplification, weight decay | helped |
| McMahan17 FedAvg arXiv:1602.05629; Wang20 FedNova arXiv:2007.07481 | 9 | 0.912100 | Architecture subcampaign: moderate_cnn_small_head under FedDrift best... | server momentum, server diff amplification, weight decay | helped |
| McMahan17 FedAvg arXiv:1602.05629; Wang20 FedNova arXiv:2007.07481; Gao22 FedDC arXiv:2203.11751 | 2 | 0.911500 | FedDrift-enabled FedNova aggregation_epochs=6 audit [src: McMahan17 F... | server momentum, server diff amplification, weight decay | helped |
| McMahan17 FedAvg arXiv:1602.05629; Zhang17 mixup arXiv:1710.09412 | 2 | 0.911300 | Mixup_alpha=0.2 aggregation_epochs=6 under FedDrift best stack [src: ... | server momentum, server diff amplification, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295 | 23 | 0.909900 | epoch5 FedAvgM server_lr=1.875 with gradient centralization [src: Red... | server momentum, server diff amplification, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295; Acar21 FedDyn OpenReview B7v4QMR6Z9w | 2 | 0.910500 | FedDyn-enabled FedNova server_lr=1.9375 neighbor at alpha=1e-4 [src: ... | server momentum, server diff amplification, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295; Cheng23 momentum arXiv:2306.16504 | 10 | 0.914600 | Server_momentum=0.40 with lr=0.0475 cosine_lr_eta_min_factor=0.003 un... | server momentum, server diff amplification, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295; Gao22 FedDC arXiv:2203.11751 | 2 | 0.909300 | FedDrift-enabled FedNova server_lr=1.9375 under drift-corrected best ... | server momentum, server diff amplification, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295; Loshchilov16 SGDR arXiv:1608.03983 | 2 | 0.914100 | Server_lr=1.9375 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under ... | server momentum, server diff amplification, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295; Wang20 FedNova arXiv:2007.07481 | 2 | 0.908500 | Architecture subcampaign: moderate_cnn_small_head server_lr=1.8125 un... | server momentum, server diff amplification, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Cheng23 momentum arXiv:2306.16504 | 2 | 0.910000 | FedNova normalized aggregation server_momentum=0.40 at server_lr=1.87... | server momentum, server diff amplification, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Flower26 FedNova baseline | 2 | 0.910300 | FedNova normalized DIFF aggregation with current FedAvgM server setti... | server momentum, server diff amplification, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Li23 FedLAW arXiv:2302.10911 | 2 | 0.904800 | FedNova sqrt step weighting via weight_power=0.5 [src: Wang20 FedNova... | server momentum, server diff amplification, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; McMahan17 FedAvg arXiv:1602.05629 | 2 | 0.910000 | FedNova exact local_train_steps=600 at server_lr=1.875 momentum=0.35 ... | server momentum, server diff amplification, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Reddi21 FedOpt arXiv:2003.00295 | 4 | 0.909300 | FedNova normalized aggregation server_lr=2.0 with momentum=0.35 [src:... | server momentum, server diff amplification, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Zantalis26 FedZMG arXiv:2602.18384 | 2 | 0.905800 | FedNova normalized aggregation weight_decay=4e-4 at server_lr=1.875 m... | server momentum, server diff amplification, weight decay | helped |
| Wang21 local adaptivity arXiv:2106.02305; Kim24 auto-tuned clients arXiv:2306.11201; Loshchilov17 AdamW arXiv:1711.05101 | 2 | 0.252000 | Local AdamW optimizer lr=0.0005 under FedDrift best stack [src: Wang2... | server momentum, server diff amplification, weight decay | not confirmed |
| Yin18 coordinate median arXiv:1803.01498 | 1 | 0.747000 | median aggregation with weight_decay=3e-4 [src: Yin18 coordinate medi... | weight decay | not confirmed |
| Yun19 CutMix arXiv:1905.04899; Zhang17 mixup arXiv:1710.09412 | 2 | 0.904800 | Local-only cutmix_alpha=0.5 under FedDrift best stack [src: Yun19 Cut... | server momentum, server diff amplification, weight decay | helped |
| Zantalis26 FedZMG arXiv:2602.18384 | 10 | 0.903400 | gradient centralization weight_decay=3.5e-4 [src: Zantalis26 FedZMG a... | server momentum, server diff amplification, weight decay | helped |
| Zantalis26 FedZMG arXiv:2602.18384; Reddi21 FedOpt arXiv:2003.00295 | 2 | 0.907500 | epoch5 FedAvgM weight_decay=3e-4 under server_lr=1.875 with gradient ... | server momentum, server diff amplification, weight decay | helped |
| Zhang17 mixup arXiv:1710.09412; Loshchilov16 SGDR arXiv:1608.03983 | 2 | 0.914200 | Weight_decay=3.25e-4 with lr=0.0475 cosine_lr_eta_min_factor=0.003 un... | server momentum, server diff amplification, weight decay | helped |
| Zhang17 mixup arXiv:1710.09412; Reddi21 FedOpt arXiv:2003.00295 | 2 | 0.913200 | Mixup_alpha=0.2 server_lr=1.9375 under FedDrift best stack [src: Zhan... | server momentum, server diff amplification, weight decay | helped |
| Zhang17 mixup arXiv:1710.09412; Wang20 FedNova arXiv:2007.07481 | 2 | 0.912100 | Mixup_alpha=0.2 weight_decay=3.0e-4 under FedDrift best stack [src: Z... | server momentum, server diff amplification, weight decay | helped |
| Zhang17 mixup arXiv:1710.09412; Yoon21 FedMix arXiv:2107.00233 | 4 | 0.914100 | Local-only mixup_alpha=0.2 under FedDrift best stack [src: Zhang17 mi... | server momentum, server diff amplification, weight decay | helped |
| Zhang22 FedLC arXiv:2209.00189; Reddi21 FedOpt arXiv:2003.00295 | 2 | 0.864000 | FedLC tau=1.0 with FedAvgM best [src: Zhang22 FedLC arXiv:2209.00189;... | server momentum, server diff amplification | helped |
| architecture-cap registered variant | 1 | 0.909200 | architecture subcampaign: FedNova with registered moderate_cnn_small_... | server momentum, server diff amplification, weight decay | helped |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

| experiment | score | description | mechanism |
| --- | --- | --- | --- |
| #5 | 0.000000 | calibration FedAdam server_lr=1.0 beta1=0.9 beta2=0.99 tau=1e-3 [src:... | server diff amplification |
| #9 | 0.000000 | FedAvgM sweep momentum=0.4 server_lr=1.75 [src: Reddi21 FedOpt arXiv:... | server momentum, server diff amplification |
| #36 | 0.000000 | cosine eta_min_factor=0.001 with FedAvgM+wd best | server momentum, server diff amplification, weight decay |
| #111 | 0.000000 | FedAdagrad server optimizer beta1=0 server_lr=0.1 tau=1e-2 with epoch... | server diff amplification, weight decay |
| #112 | 0.000000 | FedYogi server optimizer beta1=0 server_lr=0.1 tau=1e-2 with epoch5 G... | server diff amplification, weight decay |
| #147 | 0.000000 | FedDyn-enabled FedNova exact local_train_steps=600 audit [src: Acar21... | server momentum, server diff amplification, weight decay |
| #148 | 0.000000 | FedDyn-enabled FedNova exact local_train_steps=500 audit [src: Acar21... | server momentum, server diff amplification, weight decay |
| #216 | 0.000000 | Effective-number class-balanced beta=0.999 with mixup_alpha=0.2 under... | server momentum, server diff amplification, weight decay |

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| candidate | 2 |
| crash | 8 |
| discard | 183 |
| keep | 24 |
| literature | 31 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #237 | 0.916200 | 12.0m | keep | Mixup_alpha=0.2 cosine_lr_eta_min_factor=0.003 lr=0.0475 under new scheduler-floor be... |
| 2 | #234 | 0.915600 | 11.1m | keep | Mixup_alpha=0.2 cosine_lr_eta_min_factor=0.003 under FedDrift best stack [src: Loshch... |
| 3 | #245 | 0.915000 | 12.1m | discard | Feddyn_alpha=5e-5 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under new best stack ... |
| 4 | #240 | 0.914600 | 12.1m | discard | Server_momentum=0.40 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under new best sta... |
| 5 | #222 | 0.914500 | 12.0m | discard | Mixup_alpha=0.2 client_lr=0.055 under FedDrift best stack [src: Kim24 auto-tuned clie... |
| 6 | #243 | 0.914200 | 12.1m | discard | Weight_decay=3.25e-4 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under new best sta... |
| 7 | #201 | 0.914100 | 12.1m | keep | Local-only mixup_alpha=0.2 under FedDrift best stack [src: Zhang17 mixup arXiv:1710.0... |
| 8 | #238 | 0.914100 | 12.0m | discard | Server_lr=1.9375 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under new best stack [... |
| 9 | #236 | 0.914000 | 12.0m | discard | Mixup_alpha=0.2 cosine_lr_eta_min_factor=0.003 lr=0.055 under new scheduler-floor bes... |
| 10 | #239 | 0.913800 | 12.1m | discard | Server_lr=1.8125 with lr=0.0475 cosine_lr_eta_min_factor=0.003 under new best stack [... |

