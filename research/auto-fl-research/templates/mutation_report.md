# Mutation report

Campaign: `autoresearch/h100-algocalib-20260702` (CIFAR-10/H100 default budget:
8 clients, 20 rounds, 4 local epochs, batch 64, alpha 0.5, seed 0,
moderate_cnn, cross-site eval on site-1, 1200s cap).

## Hypothesis

1. The eight-step algorithm calibration will separate FedAvg-family,
   FedProx, server-optimizer (FedAvgM/FedAdam), and SCAFFOLD behavior under
   an identical fixed budget, and the best family will support narrow
   server-side sweeps.
2. The SCAFFOLD calibration crash is a persistence bug, not an algorithmic
   failure: `FLModel.meta["scaffold_c_global"]` numpy arrays get pickled into
   `FL_global_model.pt` `meta_props`, and cross-site validation reloads
   checkpoints with `torch.load(weights_only=True)` (PyTorch >= 2.6), which
   rejects pickled numpy globals and returns a `None` model learnable.

## Files changed

- `tasks/shared/custom_aggregators.py` — `_to_meta_numpy` replaced by
  `_to_meta_tensor`; `ScaffoldAggregator.aggregate_model` now emits global
  control variates as CPU torch tensors (commit `3a06951c7`).

## Commands run

- `make validate`, `make smoke` (weighted), scaffold-specific smoke
  (2 clients / 1 round / cross-site eval) after the fix.
- Batches 1-4 of same-budget candidates via `scripts/run_iteration.sh`,
  `PARALLEL_CANDIDATES=4`, `CUDA_VISIBLE_DEVICES=0`.

## Observed outcome

- Baseline weighted FedAvg: 0.8460.
- Calibration: builtin FedAvg 0.8510, explicit FedAvg 0.8470,
  FedProx mu=1e-5 0.8462, mu=1e-4 0.8486, FedAvgM lr1.0/m0.6 0.8462,
  FedAvgM lr2.0/m0.4 0.8531, FedAdam lr1.0 crash, SCAFFOLD crash.
- Momentum sweep at server_lr=2.0: m0.2 **0.8606** (current best),
  m0.3 0.8520, m0.1 0.8575, m0.5 0.8509, m0.6 0.8305, m0.0 NaN-diverged
  around round 9 (momentum smoothing is load-bearing at lr 2.0).
- SCAFFOLD after tensor-meta fix: completed cleanly at 0.8569 — fix
  validated in a full run; kept as a bug fix even though the score does not
  beat FedAvgM.
- FedAdam diverged (NaN client diff) at server_lr 1.0 and 0.1 with
  tau=1e-3: with model-sized DIFFs the update ~ diff/(|diff|+tau) is
  sign-scaled, so every element moves ~server_lr per round regardless of
  gradient scale. A damped-adaptivity audit (tau=0.01) is queued as the
  final attempt for that family.

## Literature basis

- FedProx: Li et al. 2020, arXiv:1812.06127.
- FedAvgM (server momentum): Hsu et al. 2019, arXiv:1909.06335.
- FedAdam/FedOpt and the tau adaptivity floor: Reddi et al. 2020,
  arXiv:2003.00295 (paper uses much smaller effective server steps; our
  mutation bounds floor server_lr at 0.1, so tau is the stabilizing knob).
- SCAFFOLD control variates: Karimireddy et al. 2020, arXiv:1910.06378.

## Run analysis

FedAvgM with modest server momentum is the clear leading family
(+0.0146 over baseline). Low momentum (0.2) at an over-relaxed server step
(2.0) beats both higher momentum and no momentum; zero momentum diverges.
Server_lr refinement at m=0.2 is in flight (batch 5).

## Contract check

- Client loop, DIFF uploads, `NUM_STEPS_CURRENT_ROUND`, and eval branch
  untouched. The scaffold meta change stays inside the explicitly supported
  scaffold protocol mode; client ingestion already used `torch.as_tensor`
  and accepts tensors unchanged. Static contract checks and both smokes
  pass.

## Rollback risk

Low. The aggregator fix only changes the in-flight/persisted type of
scaffold meta arrays. Reverting restores the cross-site-val crash.

## Literature loop 1 outcome (batches 28-31)

Trigger: all CLI axes resolved/re-checked at 0.9044; watchdog still
`continue` but no non-duplicate safe axis remained. Worksheet in
`templates/literature_loop.md`.

- P2 mixup (Zhang18 arXiv:1710.09412): **kept** — alpha 0.1 → 0.9091
  (new best, +0.0047 over pre-literature best). Curve 0.05/0.1/0.15/0.2/0.4 =
  0.9051/0.9091/0.9058/0.9072/0.9063. Complementary to label smoothing
  (removing ls under mixup drops to 0.9024).
- P1 fed-back server window average (Pu21 arXiv:2103.11619, WiMA
  arXiv:2310.01366): **discarded** — W=5 at best stack scored 0.8558;
  mid-training feedback of the averaged model fights tuned FedAvgM momentum.
  Tail-only averaging remains an untested variant.
- P3 FedSAM (Qu22 arXiv:2206.02618): **discarded in cap-fitting form** —
  rho 0.05 at e4 (to fit the 1200s cap) scored 0.8817; halved local epochs
  cost more than flatness gains. Do not retry SAM unless the runtime budget
  changes.

## Literature loop 2 outcome (batches 40-42)

Trigger: interaction lattice at 0.9124 fully mined. Selected cutout
(DeVries17 arXiv:1708.04552) and FedNova normalization (Wang20
arXiv:2007.07481); momentum-reset rejected on Mime evidence
(arXiv:2008.03606).

- Cutout: **discarded** in every dose (8px stacked 0.9059, 12px replacing
  mixup 0.9035, mild splits 0.9065/0.9084) — input-regularization budget is
  saturated by mixup 0.1 + label smoothing 0.05.
- FedNova: **discarded** (0.9107) — alpha 0.5 shards are near-equal, so
  normalization is close to identity here.
- SGDR per-round LR restart (Loshchilov16 arXiv:1608.03983): **discarded
  hard** (0.8463) — the single global cosine decay to a ~0 floor is
  load-bearing.

## Milestone: steps-1000 stack (batches 43-45)

`local_train_steps=1000` (~10.2 epochs, schema bound; solo runs ~800s)
beat the 8-epoch stack: **0.9135** vs 0.9124. Steps curve rises into the
bound (900 -> 0.9100). Server momentum re-check at the new compute level
confirms 0.3 (0.35 -> 0.9123, 0.25 -> 0.9118). Steps-1000 candidates must
run solo: two concurrent runs exceed the 1200s cap.

Best stack (0.9135): FedAvgM server_lr 1.75 / momentum 0.3 over DIFFs;
local_train_steps 1000; client SGD lr 0.06, momentum 0.9, wd 2.5e-4;
global cosine floor 1e-4; FedProx mu 1e-4; label smoothing 0.05;
mixup 0.1. Delta over weighted baseline: +0.0675.

## Literature loop 3 outcome (batches 51-52)

Trigger: steps-1000 lattice fully confirmed at 0.9136; only jitter axes
left. Selected FedLC logit calibration (Zhang22 arXiv:2209.00189) and
FedDecorr feature decorrelation (Shi22 arXiv:2210.00226).

- FedLC tau=1.0: **discarded** (0.9092) — label-prior correction is
  redundant with mixup + label smoothing at alpha 0.5 skew.
- FedDecorr: **discarded** — beta 0.1 -> 0.9129 (noise-adjacent),
  but no dose response (0.05 -> 0.9063, 0.3 -> 0.9093); treated as null.

Both knobs remain in client.py (default off) for future campaigns.

## Literature loop 4 outcome (batches 68-69)

Trigger: watchdog fired at 32/32 scored candidates without material
improvement after the full fine-grid confirmation at 0.9136.

- FedExP adaptive server step (Jhunjhunwala23 arXiv:2301.09604):
  **discarded** — pure (momentum 0) 0.9063, hybrid with momentum 0.3
  0.9089. The adaptive rule under-steps relative to the hand-tuned fixed
  1.75 over-relaxation at this budget.
- Server-lr decay across rounds (arXiv:2107.06917 field guide):
  **discarded** — decay to 1.0 -> 0.9062, to 1.4 -> 0.9106. Sustained
  over-relaxation beats annealing when the client lr already anneals to a
  ~0 floor.

Both knobs remain available (default off).

## Campaign state after four literature loops

Best 0.9136 (+0.0676 over 0.8460 weighted baseline): FedAvgM
server_lr 1.75 / momentum 0.3 over DIFFs; local_train_steps 1000;
client SGD lr 0.06 / momentum 0.9 / wd 3e-4; global cosine floor 1e-4;
FedProx mu 1e-4; label smoothing 0.05; mixup 0.1. All axes confirmed at
fine resolution; literature keepers: mixup (loop 1). Confirmed nulls:
FedAdam, SCAFFOLD, median, window/tail averaging, SGDR, SAM (cap-bound),
cutout, FedNova, FedLC, FedDecorr, FedExP, server-lr decay.

## Next mutation

Flat-peak combination probes; next literature loop on watchdog cadence.
