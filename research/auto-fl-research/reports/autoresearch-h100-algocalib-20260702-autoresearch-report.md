# Auto-FL NVFlare Autoresearch Campaign Report

## Executive Summary

- **Branch:** `autoresearch/h100-algocalib-20260702` at `43e79f620`
- **Rows analyzed:** 307 total, 300 scored
- **Best score:** 0.916800 at experiment `#302`
- **Baseline:** 0.846000 at experiment `#0`
- **Lift:** +0.070800 absolute, 8.4% relative
- **Runtime cost:** 99.19h aggregate; 19.4m average over 307 timed candidates
- **Agent model/effort:** From Claude Code session-start slash-command output (captured in the session transcript; no output was pasted into th...
- **Agent/tooling cost:** Agent cost telemetry unavailable in this Claude Code runtime; /cost is interactive and was not provided to the report...
- **Best status:** `keep`; treat as needing reproduction unless independently repeated or marked `keep`.

- **Progress plot:** `progress.png`

## Progress Plot

![Auto-FL progress](../progress.png)

## Best Candidate

| field | value |
| --- | --- |
| experiment | #302 |
| score | 0.916800 |
| delta vs baseline | +0.070800 |
| relative lift | 8.4% |
| status | keep |
| commit | 6c213bbac |
| runtime | 14.8m |
| target | tasks/cifar10/client.py |
| description | server momentum 0.322 micro rotation (solo) |
| artifact | /tmp/nvflare/simulation/rot3_smom322 |
| contract mode | Strict DIFF contract |

### Exact Budget / Args

```text
--n_clients 8 --num_rounds 20 --aggregation_epochs 4 --local_train_steps 950 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --final_eval_clients site-1 --aggregator fedavgm --server_lr 1.8 --lr 0.06 --weight_decay 2.8e-4 --cosine_lr_eta_min_factor 0.0001 --fedproxloss_mu 1e-4 --label_smoothing 0.049 --mixup_alpha 0.1 --rdrop_alpha 0.3 --server_momentum 0.322 --name rot3_smom322
```

## Improvement Path

Major running-best milestones, selected by first/last and largest score jumps:

| experiment | score | jump | description | likely mechanism | source refs |
| --- | --- | --- | --- | --- | --- |
| #0 | 0.846000 | baseline | weighted FedAvg baseline, default H100 budget | configuration change |  |
| #1 | 0.851000 | +0.005000 | builtin FedAvg audit | configuration change |  |
| #12 | 0.860600 | +0.007500 | FedAvgM lr 2.0 momentum 0.2 (momentum sweep) | server momentum, server diff amplification |  |
| #25 | 0.868700 | +0.005000 | client weight_decay 1e-3 at FedAvgM lr1.5/m0.2 | server momentum, server diff amplification, weight decay |  |
| #26 | 0.873000 | +0.004300 | client weight_decay 5e-4 at FedAvgM lr1.5/m0.2 | server momentum, server diff amplification, weight decay |  |
| #40 | 0.882500 | +0.006100 | cosine eta_min factor 0.001 (scheduler sweep) | server momentum, server diff amplification, weight decay |  |
| #47 | 0.892300 | +0.009800 | aggregation_epochs 6 (local compute sweep) | server momentum, server diff amplification, weight decay |  |
| #51 | 0.895700 | +0.003100 | aggregation_epochs 8 (epochs curve, bound max) | server momentum, server diff amplification, weight decay |  |
| #98 | 0.912400 | +0.003300 | server momentum 0.3 under mixup (interaction re-check) | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay |  |
| #302 | 0.916800 | +0.000100 | server momentum 0.322 micro rotation (solo) | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay |  |

## Runtime and Reliability

| metric | value |
| --- | --- |
| total aggregate runtime | 99.19h |
| average runtime per timed candidate | 19.4m |
| timed candidates | 307 |
| candidate rows | 0 |
| kept rows | 28 |
| crash rows | 11 |

The runtime total is aggregate candidate runtime from `runtime_seconds`, not wall-clock elapsed campaign time.

## Agent / Tooling Context

### Model / Effort Settings

Agent model and effort context provided for this report:

```text
From Claude Code session-start slash-command output (captured in the session transcript; no output was pasted into the report request):
/model: "Set model to Fable 5 and saved as your default for new sessions. Managed settings pins Sonnet 5 — that applies on restart" (model ID claude-fable-5)
/effort: "Set effort level to ultracode (this session only): xhigh + dynamic workflow orchestration" (ultracode was later switched off mid-campaign; effort remained xhigh)
```

### Agent / Tooling Cost

Agent/tooling cost context provided for this report:

```text
Agent cost telemetry unavailable in this Claude Code runtime; /cost is interactive and was not provided to the reporting agent. Experiment runtime cost is reported from results.tsv.
```

### Crash / Failure Notes

| count | description |
| --- | --- |
| 1 | FedAdam [src: Reddi20 FedOpt arXiv:2003.00295] |
| 1 | SCAFFOLD metadata mode [src: Karimireddy20 arXiv:1910.06378] |
| 1 | FedAdam stabilized server_lr 0.1 [src: Reddi20 FedOpt arXiv:2... |
| 1 | FedAvgM lr 2.0 momentum 0.0 (momentum sweep tail) |
| 1 | FedAvgM lr 2.5 momentum 0.2 (server_lr sweep) |
| 1 | client lr 0.09 (lr refinement) |
| 1 | client momentum 0.95 at best stack (retry after pycompile race) |
| 1 | server_lr 2.0 at 8-epoch stack (axis extension) |

## Literature-Derived Ideas

| source ref | rows | best score | best description | mapped mechanism | outcome |
| --- | --- | --- | --- | --- | --- |
| Caldarola22 arXiv:2203.11834 | 3 | 0.912100 | local tail-SWA 250 steps variant [src: Caldarola22 arXiv:2203.11834] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| DeVries17 arXiv:1708.04552 | 2 | 0.905900 | client cutout 8px at best stack [src: DeVries17 arXiv:1708.04552] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Hsu19 arXiv:1909.06335 | 2 | 0.853100 | FedAvgM server lr 2.0 momentum 0.4 [src: Hsu19 arXiv:1909.06335] | server momentum, server diff amplification | helped |
| Izmailov18 arXiv:1803.05407 | 2 | 0.901900 | SWA tail average last 3 rounds [src: Izmailov18 arXiv:1803.05407] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Jhunjhunwala23 arXiv:2301.09604 | 3 | 0.908900 | FedExP + momentum 0.3 hybrid [src: Jhunjhunwala23 arXiv:2301.09604] | server momentum, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Karimireddy20 arXiv:1910.06378 | 2 | 0.856900 | SCAFFOLD retry after tensor-meta fix [src: Karimireddy20 arXiv:1910.0... | SCAFFOLD control variates | helped |
| Li20 FedProx arXiv:1812.06127 | 7 | 0.901400 | FedProx mu 5e-5 fine re-check at current stack [src: Li20 FedProx arX... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Li20 arXiv:1812.06127 | 4 | 0.911500 | FedProx mu 1.2e-4 fine grid at final stack [src: Li20 arXiv:1812.06127] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Liang21 arXiv:2106.14448 | 5 | 0.914800 | R-Drop alpha 0.3 at reference stack (solo) [src: Liang21 arXiv:2106.1... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Loshchilov16 arXiv:1608.03983 | 1 | 0.846300 | SGDR per-round LR restart at best stack [src: Loshchilov16 arXiv:1608... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Pu21 arXiv:2103.11619, WiMA arXiv:2310.01366 | 2 | 0.855800 | server window avg W=5 at best stack [src: Pu21 arXiv:2103.11619, WiMA... | server momentum, server diff amplification, FedProx/client drift regularization, client LR warmup retuning, label smoothing, weight decay | helped |
| Qu22 arXiv:2206.02618 | 1 | 0.881700 | FedSAM rho 0.05 at e4 (2x step cost fits cap) with mixup+ls [src: Qu2... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Reddi20 FedOpt arXiv:2003.00295 | 3 | 0.727500 | FedAdam tau 0.01 damped adaptivity [src: Reddi20 FedOpt arXiv:2003.00... | server diff amplification | not confirmed |
| Reddi20 arXiv:2003.00295, arXiv:2107.06917 | 1 |  | literature review 4: watchdog 32/32 fired; selected FedExP adaptive s... | server diff amplification | not confirmed |
| Shi22 arXiv:2210.00226 | 4 | 0.912900 | FedDecorr beta 0.1 at steps-1000 stack [src: Shi22 arXiv:2210.00226] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Sutskever13 momentum | 1 | 0.909200 | client Nesterov SGD at best stack [src: Sutskever13 momentum] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Szegedy16 arXiv:1512.00567 | 2 | 0.901700 | label smoothing 0.05 at best stack [src: Szegedy16 arXiv:1512.00567] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Wang20 arXiv:2007.07481 | 2 | 0.910700 | FedNova-normalized DIFF averaging at best stack [src: Wang20 arXiv:20... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Wang21 arXiv:2106.02305 | 1 | 0.875400 | client AdamW lr 3e-4 variant [src: Wang21 arXiv:2106.02305] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Wang21 arXiv:2106.02305, Loshchilov17 arXiv:1711.05101 | 2 | 0.854900 | client AdamW lr 1e-3 wd 1e-3 [src: Wang21 arXiv:2106.02305, Loshchilo... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Yun19 arXiv:1905.04899 | 3 | 0.912700 | CutMix 0.2 replacing mixup at adopted stack [src: Yun19 arXiv:1905.04... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Zhang18 arXiv:1710.09412 | 6 | 0.909100 | client mixup alpha 0.1 (mixup refinement) [src: Zhang18 arXiv:1710.09... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| Zhang22 arXiv:2209.00189 | 2 | 0.909200 | FedLC tau 1.0 at steps-1000 stack [src: Zhang22 arXiv:2209.00189] | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |
| arXiv:2107.06917 | 1 | 0.906200 | server_lr decay 1.75 to 1.0 across rounds at best stack [src: arXiv:2... | server momentum, server diff amplification, FedProx/client drift regularization, label smoothing, weight decay | helped |

Source refs are extracted from `[src: ...]` markers in `results.tsv` descriptions. Check `templates/mutation_report.md` or the campaign notes for full citations and URLs.

## Null, Worse, or Unstable Ideas

| experiment | score | description | mechanism |
| --- | --- | --- | --- |
| #5 | 0.000000 | FedAdam [src: Reddi20 FedOpt arXiv:2003.00295] | server diff amplification |
| #6 | 0.000000 | SCAFFOLD metadata mode [src: Karimireddy20 arXiv:1910.06378] | SCAFFOLD control variates |
| #13 | 0.000000 | FedAdam stabilized server_lr 0.1 [src: Reddi20 FedOpt arXiv:2003.00295] | server diff amplification |
| #14 | 0.000000 | FedAvgM lr 2.0 momentum 0.0 (momentum sweep tail) | server momentum, server diff amplification |
| #20 | 0.000000 | FedAvgM lr 2.5 momentum 0.2 (server_lr sweep) | server momentum, server diff amplification |
| #33 | 0.000000 | client lr 0.09 (lr refinement) | server momentum, server diff amplification, weight decay |
| #44 | 0.000000 | client momentum 0.95 at best stack (retry after pycompile race) | server momentum, server diff amplification, weight decay |
| #55 | 0.000000 | server_lr 2.0 at 8-epoch stack (axis extension) | server momentum, server diff amplification, weight decay |

## Recommendation

1. Reproduce the best candidate with multiple seeds or repeated runs before promotion.
2. Promote only changes that preserve the declared contract mode, or keep explicit protocol modes such as SCAFFOLD labeled separately.
3. Use the milestone table to focus follow-up sweeps on mechanisms that created durable running-best jumps.
4. Retire ideas that repeatedly crash, underperform the baseline, or add complexity without a repeatable score lift.

## Technical Appendix

### Status Counts

| status | rows |
| --- | --- |
| crash | 11 |
| discard | 261 |
| keep | 28 |
| literature | 7 |

### Top Scored Rows

| rank | experiment | score | runtime | status | description |
| --- | --- | --- | --- | --- | --- |
| 1 | #302 | 0.916800 | 14.8m | keep | server momentum 0.322 micro rotation (solo) |
| 2 | #277 | 0.916700 | 14.8m | keep | label smoothing 0.049 micro rotation (solo) |
| 3 | #281 | 0.916700 | 14.7m | discard | mixup 0.10 at ls-0.049 stack (solo) |
| 4 | #226 | 0.916400 | 14.7m | keep | wd 2.8e-4 at slr-1.8 stack (solo) |
| 5 | #219 | 0.916300 | 14.8m | keep | server_lr 1.8 under R-Drop (interaction re-check, solo) |
| 6 | #243 | 0.916300 | 14.7m | discard | wd 3.0e-4 micro at adopted stack (solo) |
| 7 | #221 | 0.916000 | 14.7m | discard | server_lr 1.9 under R-Drop (axis extension, solo) |
| 8 | #223 | 0.915900 | 14.8m | discard | server momentum 0.36 at slr 1.8 under R-Drop (solo) |
| 9 | #295 | 0.915700 | 14.7m | discard | wd 2.82e-4 micro rotation (solo) |
| 10 | #262 | 0.915500 | 14.7m | discard | R-Drop alpha 0.31 micro rotation (solo) |

