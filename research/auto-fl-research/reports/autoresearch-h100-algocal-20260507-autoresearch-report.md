# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-algocal-20260507` at `516f45501`
- **Rows analyzed:** 702 total, 684 scored
- **Best score:** 0.925800 at experiment `#626`
- **Baseline:** 0.852100 at experiment `#0`
- **Lift:** +0.073700 absolute, 8.6% relative
- **Runtime cost:** 139.07h aggregate; 11.9m average over 702 timed candidates
- **Agent model/effort:** Agent model output: GPT-5.5; Agent effort output: xhigh.
- **Agent/tooling cost:** Agent cost telemetry was unavailable; no cost output was pasted. Experiment runtime cost is reported from results.tsv.
- **Best status:** `keep`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #626 |
| score | 0.925800 |
| delta vs baseline | +0.073700 |
| relative lift | 8.6% |
| status | keep |
| commit | b1db3dd29 |
| runtime | 14.3m |
| target | client.py |
| description | Very tight upper cosine floor 0.0001625 interpolation on active FedSAM/FedAvgM high-water stack |
| artifact | /tmp/nvflare/simulation/fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta00001625_mu3e5_zmg_cb090_sam00725_w1 |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --aggregation_epochs 7 --local_train_steps 0 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --aggregator fedavgm --server_lr 1.8 --server_momentum 0.475 --lr 0.045 --momentum 0.925 --weight_decay 5e-4 --cosine_lr_eta_min_factor 0.0001625 --sam_rho 0.0725 --fedproxloss_mu 3e-5 --zero_mean_gradients --class_balanced_loss_beta 0.90 --final_eval_clients site-1 --name fedavgm_lr18_m0475_epochs7_clientlr0045_cm0925_wd5e4_eta00001625_mu3e5_zmg_cb090_sam00725_w1
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.852100 | baseline | baseline weighted default budget | configuration change |  |
| #9 | 0.858900 | +0.002700 | FedAvgM server_lr sweep lr 1.5 momentum 0.4 | server momentum, server diff amplification |  |
| #12 | 0.863300 | +0.004400 | FedAvgM server_lr sweep lr 1.75 momentum 0.4 | server momentum, server diff amplification |  |
| #42 | 0.875800 | +0.004400 | Client lr sweep lr 0.04 with FedAvgM 1.80 momentum 0.40 aggregation_e... | server momentum, server diff amplification |  |
| #47 | 0.898500 | +0.022700 | Client weight_decay sweep 5e-4 lr 0.04 with FedAvgM 1.80 momentum 0.4... | server momentum, server diff amplification, weight decay |  |
| #109 | 0.903400 | +0.003100 | Server momentum retune 0.45 with eta_min_factor 0.005 current best stack | server momentum, server diff amplification, weight decay |  |
| #227 | 0.913700 | +0.007300 | FedZMG zero-mean gradients on active FedAvgM FedProx stack [src: Zant... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |
| #235 | 0.916700 | +0.002900 | Client lr 0.045 around FedZMG new best stack [src: Zantalis26 FedZMG ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | Zantalis26 FedZMG arXiv:2602.18384 |
| #427 | 0.923900 | +0.005300 | active class-balanced FedZMG stack plus FedSAM rho 0.05 [src: Qu22 Fe... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | Qu22 FedSAM ICML; Foret21 SAM ICLR |
| #626 | 0.925800 | +0.000600 | Very tight upper cosine floor 0.0001625 interpolation on active FedSA... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay |  |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 139.07h |
| average runtime per timed candidate | 11.9m |
| timed candidates | 702 |
| candidate rows | 0 |
| kept rows | 24 |
| crash rows | 17 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
Agent model output: GPT-5.5; Agent effort output: xhigh.
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
Agent cost telemetry was unavailable; no cost output was pasted. Experiment runtime cost is reported from results.tsv.
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | algo calibration FedAdam |
| 1 | FedAvgM narrow server_lr sweep lr 1.8 momentum 0.4 |
| 1 | FedAvgM narrow server_lr sweep lr 1.9 momentum 0.4 |
| 1 | FedAvgM narrow server_lr sweep lr 1.6 momentum 0.4 |
| 1 | Class-balanced effective-number beta 0.99 on FedZMG high-floo... |
| 1 | Focal loss gamma 1.0 on FedZMG high-floor stack [src: Sarkar2... |
| 1 | Focal loss gamma 2.0 on FedZMG high-floor stack [src: Sarkar2... |
| 1 | Class-balanced focal beta 0.99 gamma 1.0 on FedZMG high-floor... |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Acar21 FedDyn | 1 | 0.000000 | FedDyn-lite client dynamic regularizer alpha 0.01 under kept FedSAM s... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | not confirmed |
| Cao19 LDAM NeurIPS; Cui19 CBLoss CVPR | 2 | 0.911700 | LDAM max_margin 0.25 with class-balanced beta 0.90 FedZMG stack [src:... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Cui19 CBLoss | 12 | 0.923200 | Class-balanced beta 0.86875 lower-side bracket under kept FedSAM/FedA... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Cui19 CBLoss CVPR; Sarkar20 Fed-Focal arXiv:2011.06283 | 1 | 0.000000 | Class-balanced focal beta 0.99 gamma 1.0 on FedZMG high-floor stack [... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | not confirmed |
| Cui19 CBLoss CVPR; Wang21 RatioLoss AAAI | 92 | 0.918600 | Class-balanced effective-number beta 0.90 stability bracket on FedZMG... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Cui19 CBLoss; Qu22 FedSAM | 1 | 0.922200 | Near-miss pair class-balanced beta 0.86875 with SAM rho 0.075 under k... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Cui19 CBLoss; Qu22 FedSAM ICML | 9 | 0.923000 | Class-balanced beta 0.875 under kept FedSAM rho 0.05 stack [src: Cui1... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Cui19 CVPR | 1 | 0.918800 | Class-balanced beta 0.95 upper shoulder under kept FedSAM stack [src:... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Cui19 class-balanced loss | 2 | 0.921700 | Class-balanced beta 0.89375 lower neighbor under new kept SAM 0.0725 ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| DeVries17 Cutout arXiv:1708.04552 | 3 | 0.918400 | active class-balanced FedZMG stack plus Cutout size 10 [src: DeVries1... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| DeVries17 Cutout arXiv:1708.04552; Zhong17 RandomErasing arXiv:1708.04896 | 1 | 0.915600 | Local Cutout size 8 under class-balanced FedZMG stack [src: DeVries17... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Fan24 FedLESAM ICML | 1 | 0.915400 | FedLESAM trajectory perturbation replacing local SAM direction [src: ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Izmailov18 SWA arXiv:1803.05407 | 2 | 0.919900 | Local SWA start_frac 0.75 under kept FedSAM rho 0.05 stack [src: Izma... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Jhunjhunwala23 FedExP | 2 | 0.917200 | FedExP adaptive server step cap 1.8 on active FedSAM stack after cap3... | server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Karimireddy20 SCAFFOLD arXiv:1910.06378 | 2 | 0.886800 | Tuned SCAFFOLD protocol mode lr 0.04 momentum 0.925 wd 5e-4 [src: Kar... | SCAFFOLD control variates, weight decay | helped |
| Karimireddy20 SCAFFOLD arXiv:1910.06378; Cui19 CBLoss CVPR | 1 | 0.906600 | SCAFFOLD metadata mode with class-balanced beta 0.90 FedZMG client st... | FedProx/client drift regularization, SCAFFOLD control variates, weight decay | helped |
| Karimireddy20 SCAFFOLD; Li20 FedProx | 1 | 0.888300 | SCAFFOLD metadata mode with active FedProx lower-floor client stack [... | FedProx/client drift regularization, SCAFFOLD control variates, weight decay | helped |
| Karimireddy20 SCAFFOLD; Qu22 FedSAM ICML | 1 | 0.912100 | SCAFFOLD mode under kept FedSAM rho 0.05 client stack [src: Karimired... | server momentum, server diff amplification, FedProx/client drift regularization, SCAFFOLD control variates, weight decay | helped |
| Kwon21 ASAM ICML | 1 | 0.915500 | ASAM adaptive perturbation scaling rho 0.10 under kept FedAvgM stack ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Lee22 FedNTD | 1 | 0.920000 | FedNTD masked global distillation beta 0.02 T2 under kept FedSAM stac... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Lee22 FedNTD; Song24 FedDistill; Yan23 FedCSD | 1 | 0.920300 | FedNTD masked global distillation beta 0.05 T2 under kept FedSAM stac... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx | 64 | 0.920500 | FedProx mu 2.75e-5 lower neighbor under new kept SAM 0.0725 FedSAM/Fe... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx MLSys | 1 | 0.920100 | FedProx mu 6e-5 upper shoulder under kept FedSAM stack [src: Li20 Fed... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127 | 1 | 0.897600 | FedProx mu 1e-5 on current best FedAvgM stack [src: Li20 FedProx arXi... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127; Cui19 CBLoss CVPR | 2 | 0.914100 | FedProx mu 3.5e-5 around class-balanced beta 0.90 FedZMG stack [src: ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx arXiv:1812.06127; Zantalis26 FedZMG arXiv:2602.18384 | 8 | 0.915600 | FedProx mu 3.25e-5 fine bracket under FedZMG scheduler high-water flo... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx; Cui19 CBLoss CVPR; Wang21 RatioLoss AAAI | 4 | 0.914100 | FedProx mu 5e-5 continuation under class-balanced beta 0.90 FedZMG st... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li20 FedProx; Qu22 FedSAM ICML | 7 | 0.921700 | FedProx mu 1e-5 under kept FedSAM rho 0.05 stack [src: Li20 FedProx; ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li21 FedRS KDD:10.1145/3447548.3467254 | 1 | 0.901200 | FedRS restricted softmax alpha 0.5 on active FedAvgM FedProx stack [s... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Li25 FedWMSAM | 1 | 0.920000 | Late-cosine SAM rho schedule max 0.10 under kept FedAvgM stack [src: ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Liang21 R-Drop NeurIPS | 1 | 0.000000 | R-Drop alpha 0.1 on active FedSAM/FedAvgM high-water stack [src: Lian... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | not confirmed |
| Muller19 LabelSmoothing arXiv:1906.02629; Szegedy16 Inception arXiv:1512.00567 | 1 | 0.912600 | Label smoothing 0.05 under class-balanced FedZMG stack [src: Muller19... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Muller19 label smoothing NeurIPS | 2 | 0.904400 | Label smoothing 0.05 on active FedAvgM FedProx stack [src: Muller19 l... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Qu22 FedSAM | 6 | 0.925200 | SAM radius 0.0725 tight upper-shoulder interpolation under kept FedSA... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Qu22 FedSAM ICML | 54 | 0.922900 | SAM radius 0.075 upper-bound check under kept stack [src: Qu22 FedSAM... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Qu22 FedSAM ICML; Foret21 SAM ICLR | 13 | 0.923900 | active class-balanced FedZMG stack plus FedSAM rho 0.05 [src: Qu22 Fe... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Qu22 FedSAM ICML; Li20 FedProx | 1 | 0.920900 | Higher SAM radius with lower FedProx mu near-miss pairing under kept ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Qu22 FedSAM arXiv:2206.02618 | 1 | 0.910700 | Reduced-epoch FedSAM rho 0.03 on kept FedZMG stack [src: Qu22 FedSAM ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Rahil25 FedSCAM arXiv:2601.00853; Qu22 FedSAM ICML | 1 | 0.918300 | Aligned FedAvgM update-alignment aggregation under kept FedSAM stack ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Reddi21 FedOpt | 16 | 0.922400 | Server momentum 0.46875 tight lower interpolation under kept FedSAM/F... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Reddi21 FedOpt arXiv:2003.00295 | 1 | 0.807800 | Conservative FedAdam server optimizer on current client stack [src: R... | server diff amplification, weight decay | not confirmed |
| Reddi21 FedOpt; Cui19 CBLoss | 2 | 0.920700 | Server momentum 0.46875 on lower-beta near-miss FedSAM stack [src: Re... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Reddi21 FedOpt; Li20 FedProx | 4 | 0.922400 | Lower server momentum with lower FedProx mu near-miss pairing under k... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Reddi21 FedOpt; Qu22 FedSAM ICML | 4 | 0.922500 | Lower server momentum with higher SAM radius near-miss pairing under ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Sarkar20 Fed-Focal arXiv:2011.06283; Lin17 Focal arXiv:1708.02002 | 4 | 0.910100 | Focal loss gamma 1.0 retry at width 2 on FedZMG high-floor stack [src... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Shi23 FedDecorr arXiv:2210.00226 | 1 | 0.920900 | FedDecorr coef 0.1 on active FedSAM/FedAvgM high-water stack [src: Sh... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Son24 FedUV CVPR | 2 | 0.921700 | FedUV classifier variance coef 1.25 on active FedSAM/FedAvgM stack [s... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Sun23 FedSMOO arXiv:2305.11584; Li20 FedProx MLSys; Qu22 FedSAM ICML | 1 | 0.919000 | Dynamic cosine-decay FedProx mu 1e-4 under kept FedSAM stack [src: Su... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Wang20 FedNova NeurIPS | 2 | 0.900900 | FedNova normalized aggregation on class-balanced FedZMG stack [src: W... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Wang20 FedNova arXiv:2007.07481; Reddi21 FedOpt arXiv:2003.00295 | 1 | 0.899100 | FedNova-style step-normalized FedAvgM on active FedProx stack [src: W... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Yashwanth24 ASD arXiv:2305.19600 | 1 | 0.901000 | Global self-distillation alpha 0.05 T2 with current FedAvgM best stac... | server momentum, server diff amplification, weight decay | helped |
| Yoon21 FedMix arXiv:2107.00233; Sang24 mixup-noise arXiv:2409.13235; Zhang17 mixup arXiv:1710.09412 | 1 | 0.914800 | Local mixup alpha 0.2 on kept FedZMG stack [src: Yoon21 FedMix arXiv:... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zantalis26 FedZMG arXiv:2602.18384 | 61 | 0.916900 | Scheduler eta-min factor 0.00015 tight check around FedZMG floor near... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zhang18 mixup arXiv:1710.09412; Yoon21 FedMix arXiv:2107.00233 | 1 | 0.912200 | Local mixup alpha 0.2 under class-balanced FedZMG stack [src: Zhang18... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zhang19 Lookahead arXiv:1907.08610 | 1 | 0.916700 | Lookahead k=5 alpha=0.5 on active FedSAM/FedAvgM high-water stack [sr... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zhang21 clipping arXiv:2106.13673 | 5 | 0.916700 | Update clipping norm 45 on kept FedZMG stack [src: Zhang21 clipping a... | server momentum, server diff amplification, FedProx/client drift regularization, gradient clipping, weight decay | helped |
| Zhang22 FedLC arXiv:2209.00189 | 1 | 0.899200 | FedLC logit calibration tau 0.5 on active FedAvgM FedProx stack [src:... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |
| Zhao22 GNP; Sun23 FedSpeed | 2 | 0.922100 | Gradient-norm SAM blend 0.5 on active FedSAM/FedAvgM high-water stack... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay | helped |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

| experiment | score | description | mechanism |
| --- | --- | --- | --- |
| #5 | 0.000000 | algo calibration FedAdam | server diff amplification |
| #17 | 0.000000 | FedAvgM narrow server_lr sweep lr 1.8 momentum 0.4 | server momentum, server diff amplification |
| #18 | 0.000000 | FedAvgM narrow server_lr sweep lr 1.9 momentum 0.4 | server momentum, server diff amplification |
| #19 | 0.000000 | FedAvgM narrow server_lr sweep lr 1.6 momentum 0.4 | server momentum, server diff amplification |
| #302 | 0.000000 | Class-balanced effective-number beta 0.99 on FedZMG high-floor stack ... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay |
| #303 | 0.000000 | Focal loss gamma 1.0 on FedZMG high-floor stack [src: Sarkar20 Fed-Fo... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay |
| #304 | 0.000000 | Focal loss gamma 2.0 on FedZMG high-floor stack [src: Sarkar20 Fed-Fo... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay |
| #305 | 0.000000 | Class-balanced focal beta 0.99 gamma 1.0 on FedZMG high-floor stack [... | server momentum, server diff amplification, FedProx/client drift regularization, weight decay |

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| crash | 17 |
| discard | 643 |
| keep | 24 |
| literature | 18 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #626 | 0.925800 | 14.3m | keep | Very tight upper cosine floor 0.0001625 interpolation on active FedSAM/FedAvgM high-w... |
| 2 | #632 | 0.925700 | 14.3m | discard | Post-improvement upper client momentum 0.92578125 interpolation on active cosine-floo... |
| 3 | #574 | 0.925200 | 14.4m | keep | SAM radius 0.0725 tight upper-shoulder interpolation under kept FedSAM/FedAvgM stack ... |
| 4 | #658 | 0.924700 | 17.9m | discard | Post-improvement schema-max exact local_train_steps 1000 under active cosine-floor hi... |
| 5 | #625 | 0.924300 | 14.4m | discard | Tight upper cosine floor 0.000175 interpolation on active FedSAM/FedAvgM high-water s... |
| 6 | #656 | 0.924100 | 14.4m | discard | Post-improvement near-miss combo server_lr 1.800390625 lower server_momentum 0.474218... |
| 7 | #427 | 0.923900 | 14.3m | keep | active class-balanced FedZMG stack plus FedSAM rho 0.05 [src: Qu22 FedSAM ICML; Foret... |
| 8 | #629 | 0.923800 | 14.3m | discard | Post-improvement lower FedAvgM server momentum 0.47421875 interpolation on active cos... |
| 9 | #672 | 0.923800 | 13.6m | discard | Noisy subcampaign deterministic-off training on active FedSAM/FedAvgM high-water stack |
| 10 | #700 | 0.923600 | 14.3m | discard | Client weight power 0.5 FedAvgM on active FedSAM/FedAvgM high-water stack |

