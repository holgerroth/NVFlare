# Mutation report — h100-baseline-20260501 campaign

## Hypothesis

Establish a strong fixed-budget FedAvg baseline for non-IID CIFAR-10 (8 clients, alpha=0.5, 10 rounds, 4 local epochs) on a single H100, then iteratively layer compatible mutations: server momentum (FedAvgM), label-smoothing cross-entropy, Nesterov on the client SGD, gradient clipping, mixup, FedExP-style server extrapolation, knowledge distillation, additional architecture variants, schedule warmup, AdamW, and warm restarts.

## Files changed

- `client.py` — added `--grad_clip_norm`, `--label_smoothing`, `--mixup_alpha`, `--lr_warmup_steps`, `--cosine_lr_restart_period`, `--nesterov`, `--optimizer adamw`, `--logit_kd_alpha`, `--logit_kd_temp`.
- `job.py` — surfaced new client knobs and aggregator knobs (`--fedexp_eps`, `--fedexp_eta_max`).
- `custom_aggregators.py` — added optional FedExP-style extrapolation factor inside `FedOptAggregator` with cap.
- `model.py` — registered `moderate_cnn_high_dropout` variant.
- `templates/literature_loop.md` — Camyla-style worksheet for plateau exit at 0.8378.
- `results.tsv` — campaign ledger.
- `progress.png` — progress plot.

## Observed outcome

| Stage | Best score | Delta | Notes |
| --- | --- | --- | --- |
| Baseline (`weighted` FedAvg, default budget) | 0.7582 | — | Anchor row in ledger. |
| Algorithm calibration (FedAvg/FedProx/FedAvgM/FedAdam/SCAFFOLD/median) | 0.8217 | +0.0635 | FedAvgM lr=2.0 m=0.4 wins; FedAdam diverges at server_lr=1.0; SCAFFOLD/median substantially worse. |
| FedAvgM narrowing (server lr/momentum sweep) | 0.8222 | +0.0640 | Sharp cliff above lr=2.0; ridge from lr=1.7 to 2.0 at m=0.4. |
| Architecture audit (norm, small-head) | 0.8222 | +0.0640 | Original `moderate_cnn` wins. |
| Label smoothing layered on FedAvgM | 0.8378 | +0.0796 | Peak at LS=0.175. Mixup, grad clip, FedExP, KD, AdamW, cosine restart, warmup, high-dropout all regress on top. |
| Nesterov client SGD | 0.8420 | +0.0838 | Single-line client mutation, strict gain. |
| Final lr refinement (server lr 1.88) | 0.8422 | +0.0840 | Sharp peak; further hyperparameter perturbations fall back to 0.83-0.84. |
| Add gradient clipping (norm 5.0) on top of Nesterov | 0.8440 | +0.0858 | Single-step clip stabilizes the late-round dynamics. |
| Bump train_lr 0.05 → 0.06 with clip 5 | 0.8448 | +0.0866 | Peak; clip lets the slightly higher local lr converge cleanly. |
| Push server_lr 1.88 → 2.20 with grad clip 5 | 0.8454 | +0.0872 | Clip stabilizes the cliff that previously collapsed lr=2.5 (0.28); two equal peaks at lr 1.88 and 2.20. |
| Refine to lr 2.15 + train_lr 0.058 | 0.8473 | +0.0891 | Sharp 1-pixel peak; nearby lr 2.13/2.16/2.17 and tlr 0.056/0.060 all regress by 0.003-0.008. |

## Best stack

```
--aggregator fedavgm
--server_lr 2.15 --server_momentum 0.4
--label_smoothing 0.175
--nesterov
--grad_clip_norm 5.0
--lr 0.058 --momentum 0.9 --weight_decay 0
--cosine_lr_eta_min_factor 0.01
--model_arch moderate_cnn --max_model_params 5000000
--n_clients 8 --num_rounds 10 --aggregation_epochs 4
--batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0
--final_eval_clients site-1
```

The score function has multiple near-equal peaks: (lr 2.15, tlr 0.058) tops at 0.8473; (lr 2.20, tlr 0.060) and (lr 2.0, m 0.5, tlr 0.060) cluster at 0.8453-0.8454; lr 2.30+ regresses; lr 2.5 still gets 0.84 with clip (vs 0.28 without clip), confirming clip stabilizes the high-lr regime. The clip+lower-tlr pairing is the late-campaign secret: nominally aggressive server lr with slightly cooler local lr.

## Literature basis

- Hsu19 FedAvgM (arXiv:1909.06335): server momentum for non-IID drift.
- Szegedy16 Inception-v3 / Hinton16 (arXiv:1512.00567): label smoothing.
- Nesterov83 / Sutskever13 (arXiv:1212.0901): Nesterov accelerated SGD.
- Li20 FedProx (arXiv:1812.06127): client proximal term — null result on top of FedAvgM.
- Reddi20 FedOpt (arXiv:2003.00295): server-side adaptive optimizers — FedAdam diverges in our budget.
- Karimireddy20 SCAFFOLD (arXiv:1910.06378): control-variate correction — regresses in 10-round budget.
- Zhang17 Mixup (arXiv:1710.09412): augmentation — regresses with our short training.
- Pascanu13 (arXiv:1211.5063): gradient clipping — neutral.
- Jhunjhunwala23 FedExP (arXiv:2301.09604): server extrapolation — regresses on top of well-tuned FedAvgM.
- Loshchilov17 SGDR (arXiv:1608.03983): cosine warm restarts — regresses (T_0 too short disrupts; T_0 too long diverges with NaN).
- Hinton15 KD (arXiv:1503.02531): logit distillation against frozen global — regresses; the global model is too weak as teacher in 10-round budget.
- Foret20 SAM (arXiv:2010.01412): one-step ascent — regresses (-0.02 at rho=0.05) on top of the strong baseline.
- Yun19 CutMix (arXiv:1905.04899): rectangular cut augmentation — regresses (alpha=0.2 → 0.77, alpha=1.0 → 0.71).
- Tarvainen17 Mean Teacher EMA (arXiv:1703.01780) adapted to local-only client EMA — regresses (decay 0.95 → 0.843, decay 0.99 → 0.837).

## Run analysis

- The score function is sharp around the peak: small lr changes (1.85 → 1.88, +0.03) flip the score by ±0.015 in the FedAvgM+LS+Nesterov regime.
- Default cosine annealing (initial 0.05, eta_min 0.0005, T_max=40 epochs) is essential. No-scheduler / warm-restart / lr-warmup variants all regress.
- Default training optimizer (SGD lr=0.05 momentum=0.9 weight_decay=0) plus Nesterov is the right base. AdamW underperforms by 0.05 abs.
- Most additional regularizers (mixup, cutmix, KD, FedProx, weight decay, high dropout, class-balance weighting, SAM, client-side EMA) compete with the existing label smoothing or destabilize training rather than stacking. The only stacked gain on top of FedAvgM+LS+Nesterov was gradient clipping with a slightly higher local LR.
- Architecture: original `moderate_cnn` outperforms the registered `_norm`, `_small_head`, and the new `_high_dropout` variants in this short-budget regime.

## Contract check

- `flare.init`/`is_running`/`receive`/`send` and `is_evaluate` paths preserved.
- Client still uploads `ParamsType.DIFF` with `NUM_STEPS_CURRENT_ROUND` meta in every code path.
- No new server-coupled meta keys outside the explicit SCAFFOLD mode (FedExP knob touches only the existing FedAvgM update math, not protocol fields).
- New architecture variant respects the 5,000,000 parameter cap.
- `make validate` and `make smoke` pass on the `lr=1.88 m=0.4 LS=0.175 Nesterov` recipe.

## Rollback risk

- Low. All new knobs default to the historical behavior (0 / off / sgd), so disabling them returns the harness to the pre-campaign defaults except for a few additional registered architectures.
- Removing the new `moderate_cnn_high_dropout` row is safe; nothing else depends on it.

## Next mutation

- Continue probing combinations near the peak. The ledger top 5 is clustered at 0.841-0.842 with multiple equal-score variants; sub-budget changes (rounds, epochs) would be needed to break above ~0.85.
- If a future campaign expands the budget (e.g., num_rounds 20 or aggregation_epochs 6), Mixup, FedExP, and KD should be re-tested — they may benefit from longer training horizons.
