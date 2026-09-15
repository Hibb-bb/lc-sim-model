# Survey-pair results: is the worse partner the limit?

Question: when a model is trained on two surveys of the same stars, does the
lower-quality survey cap what the representation learns about the better one?
Every model is trained on exactly two surveys, either AB or AC, and all are
probed on the same held-out stars.

All numbers below come from the code aligned with LeJEPA's MINIMAL.md
(2026-09-15). Earlier results (`results/base`, `add_bad`, `gap_*`) were
produced before that change and are not comparable. They are in git history.

## Setup

Synthetic light curves from `lcsim/simulate.py`, as in the README. 8,000 training
and 2,000 test stars per seed, 5 balanced classes, and a narrow periodic
"fine" bump (height up to 0.6 × amplitude) that only a clean, dense observation
can resolve. All surveys share one band (550 nm) and differ only in noise and
cadence:

| survey | σ | cadence | mean quality q | role |
|---|---|---|---|---|
| A | 0.015 | 50% | 9.75 | best |
| B | 0.05 | 30% | 8.03 | middle |
| C | 0.15 | 10% | 5.83 | worst |
| D | 0.08 | 50% | 8.08 | evaluation only, never trained on |

Shared training settings: 1D-CNN encoder → 64-d embedding → 16-d projector
(3-layer MLP with BatchNorm). AdamW (lr 2e-3, weight decay 5e-2), batch 256,
300 epochs with a 1-epoch warmup and cosine decay, λ = 0.02 unless stated, 3 seeds. SSL
losses act on the projector output; probes read the embedding. Probes are MLPs
(256 hidden units, early stopping), fit separately for each survey on the training stars
as that survey observes them.

## Variants

In each pair the second survey is the lower-quality one (B in AB, C in AC).
"Two views" means two augmented crops per survey, taken from non-overlapping
halves of the 30-day window, so each star gives 4 views per step. An augmentation is a
random 60–100% crop, up to 30% point dropout, and added noise.

| variant | views per star | loss |
|---|---|---|
| `lejepa` | 2 per survey (4) | (1 − λ) × invariance (spread of all 4 views around their mean) + λ × SIGReg, on the projector output |
| `lejepa_noproj` | 2 per survey (4) | same as `lejepa`, but applied to the 64-d embedding directly (no projector) |
| `lejepa_pred` | 2 per survey (4) | within-survey invariance for each survey, plus a residual-MLP predictor that maps the worse survey's projection onto a stop-gradient copy of the better survey's (all 4 cross pairs), averaged 50/50; + SIGReg. The better survey is never pulled toward the worse one |
| `lejepa_pred_x` | 1 per survey (2) | predictor term only (worse → stop-grad better) + SIGReg. No within-survey term |
| `lejepa_pred_x_raw` | 1 per survey (2), unaugmented | same as `lejepa_pred_x` on the raw observations |
| `contrastive` | 2 per survey (4) | multi-positive NT-Xent (cosine, temperature 0.2): every other view of the same star is a positive, all other stars' views in the batch are negatives |
| `contrastive_x` | 1 per survey (2) | NT-Xent where the only positive is the other survey's view of the same star |
| `contrastive_x_raw` | 1 per survey (2), unaugmented | same as `contrastive_x` on the raw observations |

## Results: AB vs AC

MLP probes, mean ± std over 3 seeds, λ = 0.02. Chance class accuracy is 0.20.
Amplitude is left out because it is recovered almost perfectly by every model that
trains normally. The full per-survey tables for A–D are in `results/pairs_final/results.md`.

| variant | trained on | A: fine R² | A: class acc | A: logP R² | D (unseen): fine R² | D: class acc |
|---|---|---|---|---|---|---|
| *supervised on raw A* | – | *0.87 ± 0.01* | *1.00 ± 0.00* | *0.88 ± 0.01* | – | – |
| *supervised on raw C* | – | *0.13 ± 0.01* | *0.50 ± 0.01* | *0.37 ± 0.02* | – | – |
| *supervised on raw D* | – | – | – | – | *0.75 ± 0.01* | *0.97 ± 0.00* |
| `lejepa` | AB | 0.81 ± 0.01 | 1.00 ± 0.00 | 0.91 ± 0.00 | 0.67 ± 0.02 | 0.98 ± 0.01 |
| | AC | 0.79 ± 0.01 | 0.97 ± 0.01 | 0.88 ± 0.00 | 0.60 ± 0.05 | 0.92 ± 0.01 |
| `lejepa_noproj` | AB | 0.53 ± 0.05 | 0.98 ± 0.00 | 0.88 ± 0.01 | 0.40 ± 0.03 | 0.94 ± 0.00 |
| | AC | 0.70 ± 0.03 | 0.96 ± 0.00 | 0.87 ± 0.00 | 0.38 ± 0.02 | 0.88 ± 0.01 |
| `lejepa_pred` | AB | 0.83 ± 0.01 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.67 ± 0.02 | 0.99 ± 0.00 |
| | AC | 0.82 ± 0.01 | 0.99 ± 0.00 | 0.90 ± 0.00 | 0.63 ± 0.02 | 0.95 ± 0.01 |
| `lejepa_pred_x` | AB | 0.65 ± 0.04 | 0.97 ± 0.00 | 0.83 ± 0.01 | 0.62 ± 0.04 | 0.97 ± 0.00 |
| | AC | 0.01 ± 0.04 | 0.33 ± 0.08 | 0.12 ± 0.12 | 0.05 ± 0.08 | 0.42 ± 0.16 |
| `lejepa_pred_x_raw` | AB | 0.49 ± 0.01 | 0.88 ± 0.02 | 0.76 ± 0.02 | 0.39 ± 0.02 | 0.85 ± 0.01 |
| | AC | −0.01 ± 0.04 | 0.30 ± 0.04 | 0.06 ± 0.11 | −0.02 ± 0.03 | 0.29 ± 0.04 |
| `contrastive` | AB | 0.86 ± 0.02 | 1.00 ± 0.00 | 0.92 ± 0.01 | 0.74 ± 0.02 | 0.99 ± 0.00 |
| | AC | 0.87 ± 0.00 | 0.99 ± 0.01 | 0.90 ± 0.00 | 0.74 ± 0.01 | 0.96 ± 0.01 |
| `contrastive_x` | AB | 0.86 ± 0.01 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.74 ± 0.01 | 0.98 ± 0.00 |
| | AC | 0.77 ± 0.00 | 0.95 ± 0.00 | 0.85 ± 0.01 | 0.64 ± 0.01 | 0.92 ± 0.00 |
| `contrastive_x_raw` | AB | 0.85 ± 0.01 | 0.97 ± 0.00 | 0.86 ± 0.01 | 0.74 ± 0.01 | 0.94 ± 0.00 |
| | AC | 0.42 ± 0.02 | 0.68 ± 0.01 | 0.68 ± 0.01 | 0.38 ± 0.02 | 0.68 ± 0.02 |

The supervised rows are reference points, not strict bounds: SSL + MLP probe
slightly beats supervised training on log-period (0.91 vs 0.88).

## Projector output vs embedding

What each model keeps in the space its loss acts on (16-d projector output), compared
with what probes normally read (64-d embedding). Probed on survey A's test stars
with MLP probes, using the saved **seed-0** checkpoints in
`results/pairs_unseen/seed0/models` (single seed, λ = 0.02).

| variant | trained on | fine R², embedding | fine R², projector output | logP R², embedding → projector output | cosine: A view vs partner view | effective rank of embedding |
|---|---|---|---|---|---|---|
| `lejepa` | AB | 0.82 | 0.41 | 0.91 → 0.89 | 0.96 | 13.6 |
| | AC | 0.79 | 0.39 | 0.87 → 0.86 | 0.71 | 8.7 |
| `lejepa_pred` | AB | 0.83 | 0.45 | 0.91 → 0.90 | 0.93* | 15.9 |
| | AC | 0.82 | 0.49 | 0.89 → 0.88 | 0.49* | 11.3 |
| `contrastive` | AB | 0.83 | 0.52 | 0.91 → 0.89 | 0.96 | 13.0 |
| | AC | 0.87 | 0.75 | 0.89 → 0.88 | 0.69 | 8.9 |

Cosine: mean cosine similarity between the projector outputs of a test star's
survey-A observation and its partner-survey observation. The average over
*different* stars is 0.00 for every model. \*For `lejepa_pred` the predictor sits between
the two projections during training, so the raw cosine is not the quantity its
loss aligns. Effective rank is the participation ratio of the embedding
covariance spectrum (out of 64).

## λ sweep: does a weaker invariance term close LeJEPA's AC gap?

Hypothesis tested: LeJEPA's invariance term pulls A's projection toward the
partner's, and C cannot see the bump, so with C as partner the bump is erased.
If so, raising λ, which lowers the invariance weight 1 − λ, should raise `lejepa`'s
fine R² on A when trained on AC toward contrastive's 0.87, and close the AB–AC
gap. `lejepa` only, AB and AC, 3 seeds each, MLP probes. The projector-output
column averages all 3 seeds' checkpoints.

| λ | trained on | A: fine R², embedding | A: fine R², projector output | A: class acc | A: logP R² | D: fine R² | D: class acc |
|---|---|---|---|---|---|---|---|
| 0.02 | AB | 0.81 ± 0.01 | 0.36 ± 0.03 | 1.00 ± 0.00 | 0.91 ± 0.00 | 0.67 ± 0.02 | 0.98 ± 0.01 |
| | AC | 0.79 ± 0.01 | 0.38 ± 0.03 | 0.97 ± 0.01 | 0.88 ± 0.00 | 0.60 ± 0.05 | 0.92 ± 0.01 |
| 0.05 | AB | 0.84 ± 0.01 | 0.50 ± 0.03 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.70 ± 0.01 | 0.98 ± 0.00 |
| | AC | 0.82 ± 0.01 | 0.58 ± 0.01 | 0.98 ± 0.00 | 0.89 ± 0.01 | 0.66 ± 0.02 | 0.94 ± 0.01 |
| 0.1 | AB | 0.85 ± 0.01 | 0.62 ± 0.02 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.71 ± 0.01 | 0.98 ± 0.00 |
| | AC | 0.81 ± 0.01 | 0.63 ± 0.04 | 0.99 ± 0.00 | 0.88 ± 0.01 | 0.68 ± 0.01 | 0.96 ± 0.00 |
| 0.2 | AB | 0.87 ± 0.01 | 0.70 ± 0.01 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.73 ± 0.00 | 0.97 ± 0.01 |
| | AC | 0.76 ± 0.01 | 0.63 ± 0.04 | 0.98 ± 0.00 | 0.84 ± 0.01 | 0.64 ± 0.03 | 0.94 ± 0.00 |
| *`contrastive`* | AB | *0.86 ± 0.02* | *0.52 (seed 0)* | *1.00 ± 0.00* | *0.92 ± 0.01* | *0.74 ± 0.02* | *0.99 ± 0.00* |
| | AC | *0.87 ± 0.00* | *0.75 (seed 0)* | *0.99 ± 0.01* | *0.90 ± 0.00* | *0.74 ± 0.01* | *0.96 ± 0.01* |

**The hypothesis is not confirmed.**

* **The prediction fails on the embedding.** On AC, fine R² peaks at 0.82 (λ = 0.05)
  and then falls to 0.76 at λ = 0.2, while AB climbs to 0.87. The AB–AC gap
  widens from 0.02 to 0.11 instead of closing. Log-period on AC also drops at λ = 0.2
  (0.88 → 0.84). No λ reaches contrastive's 0.87 on AC.
* **The detail lost in the projector output is not specific to the partner.** At λ = 0.02,
  `lejepa` keeps the same fine R² in its projector output with B as with C (0.36 vs 0.38),
  even though B sees the bump well (supervised 0.73). A partner that cannot see the bump
  is therefore not what erases it.
* **The invariance weight does control how much detail the loss space keeps.**
  Raising λ lifts projector-output fine R² for both pairs (AB 0.36 → 0.70, AC 0.38 → 0.63),
  so a strong invariance term discards the bump whoever the partner is.
* **Side result: λ = 0.05–0.1 is better than the MINIMAL.md default.** At λ = 0.1,
  AC improves on D (fine 0.60 → 0.68, class 0.92 → 0.96) and AB improves on A (fine
  0.81 → 0.85), with nothing getting worse.

## Reasoning

**1. With within-survey views, the worse partner does not cap survey A.**
Switching the partner from B to C barely changes A's fine R² for `lejepa`
(0.81 → 0.79), `lejepa_pred` (0.83 → 0.82) or `contrastive` (0.86 → 0.87).
All stay near A's own supervised reference (0.87), far above anything
C can see (0.13). The within-survey pairs give A its own training signal, so
an uninformative partner costs little on A itself.

**2. The worse partner still costs something on surveys the model never trained on.**
On the unseen survey D, class accuracy drops 3–6 points from AB to AC, and
`lejepa`'s fine R² drops from 0.67 to 0.60. Training with B, which is about as good
as D, gives a representation that carries over better to D than training with C.
A moderate λ recovers most of this for `lejepa` (D at λ = 0.1: fine 0.68, class 0.96).

**3. With cross-survey pairs only, the partner does set the limit.** Take away
the within-survey term and the AC runs lose most or all of what they learn:
`contrastive_x` 0.86 → 0.77, `contrastive_x_raw` 0.85 → 0.42, and
`lejepa_pred_x` / `lejepa_pred_x_raw` collapse to chance on AC in all 3 seeds.
The predictor loss for `lejepa_pred_x` falls from about 0.29 to 0.04–0.05
on AB but only to 0.21–0.24 on AC, so the C view gives almost nothing to
predict. `lejepa_pred_x_raw` on AC still reaches a loss of about 0.07 (0.01–0.02 on
AB), but only because its representation has collapsed and is easy to
predict, so a low loss does not mean the model learned anything. Contrastive's negatives
keep its representation from collapsing. LeJEPA relies only on SIGReg at weight 0.02,
which is not enough when the alignment target carries no information.

**4. Why contrastive comes out ahead: partly open.** On AB the three main
methods are essentially tied on the embedding (seed 0: 0.82 / 0.83 / 0.83), and
λ = 0.2 brings `lejepa` level with contrastive (0.87 vs 0.86). The lasting
difference is on AC. What we can rule out or support so far:

* **Not alignment strength or rank.** Both methods align the same star's two surveys about
  equally well (cosine 0.71 vs 0.69 on AC), and their embeddings have similar effective rank.
* **Not LeJEPA being pulled toward a partner that can't see the bump.** The λ sweep
  rules this out: `lejepa` loses as much detail in its projector output with B as with C,
  and weakening the pull does not close the AC gap.
* **Supported: LeJEPA's invariance term discards detail regardless of partner.** A strong
  invariance term (0.98 at λ = 0.02) wipes the bump from the projector output whatever
  the partner, and much of it survives only in the embedding, before the projector.
  This is why `lejepa_noproj`, with no projector to absorb the invariance, is clearly
  worse on the bump (0.53 / 0.70).
* **Open: contrastive seems to *gain* from the worse partner.** Its
  projector-output fine R² is higher with C than with B (0.75 vs 0.52, seed 0), whereas
  `lejepa`'s is flat. One untested explanation: when the C view is hard to match, NT-Xent's
  within-survey A–A positives dominate, and telling a star apart from many similar-looking
  negatives using its own A views rewards keeping the bump. `lejepa_noproj`'s
  AC > AB result could have a related cause.

**Open points.** Why `lejepa`'s AC embedding loses fine detail at λ = 0.2 is not
understood. One possibility is that too little alignment pressure lets the noisy C views
pull the shared encoder toward modelling their noise, but this has not been checked.
The contrastive and `lejepa_pred` projector numbers are seed 0 only. The contrastive
temperature of 0.2 is untuned.

## Reproduce

```
sbatch run_pairs.sbatch                                                   # lejepa, lejepa_noproj, contrastive, lejepa_pred
OUT_ROOT=results/pairs_crossonly sbatch --export=ALL,OUT_ROOT=results/pairs_crossonly run_pairs.sbatch --variants contrastive_x,contrastive_x_raw
OUT_ROOT=results/pairs_pred_x sbatch --export=ALL,OUT_ROOT=results/pairs_pred_x run_pairs.sbatch --variants lejepa_pred_x,lejepa_pred_x_raw
python merge_pairs.py results/pairs_final results/pairs_unseen/seed{0,1,2} results/pairs_crossonly/seed{0,1,2} results/pairs_pred_x/seed{0,1,2}

# λ sweep (lejepa only); T = 005, 010, 020 for λ = 0.05, 0.1, 0.2
sbatch --export=ALL,OUT_ROOT=results/pairs_lam$T run_pairs.sbatch --variants lejepa --pairs AB,AC --lam $L --no_supervised
python merge_pairs.py results/pairs_lam$T results/pairs_lam$T/seed{0,1,2}
```

The projector-output probes were computed with a one-off script, not in the repo.
It reloads the saved checkpoints, regenerates each seed's data, and fits the same MLP
probe as `lcsim/probe.py` on the 16-d projector output instead of the embedding.
