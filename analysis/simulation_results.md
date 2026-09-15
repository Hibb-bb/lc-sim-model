# First results (single seed, 30 epochs base / 20 epochs sweep, 2 CPU cores)

Config: 8000 train / 2000 test stars, 5 classes (balanced), fine bump amplitude
up to 0.6·amp. Survey A: σ=0.015, 50 % cadence, 15 % bad nights. Survey B
(base): σ=0.15, 10 % cadence. Survey C (unseen): 800 nm, σ=0.05, 30 %.
Linear probes unless stated. Full tables: `results/*/results.md`.

## 1. The premise holds in the data

Supervised networks trained directly on raw observations recover the fine
bump with R² = 0.82 from survey A but only 0.16 from survey B (base), so B
genuinely cannot see the feature that A can. Coarse latents are recoverable
from both (class 0.98 vs 0.49, log-period 0.84 vs 0.36).

## 2. But the bottleneck does not show up in the SSL representations

Fine-bump R² on **survey A** embeddings (base config):

| method       | fine R² (linear) |
|--------------|------------------|
| contrastive  | 0.61 |
| lejepa       | 0.48 |
| aug_only     | 0.34 |
| single_good  | 0.34 |
| gate_only    | 0.25 |
| ours         | 0.24 |
| split_only   | 0.20 |
| supervised raw A (ceiling) | 0.82 |
| supervised raw B (floor)   | 0.16 |

Two things contradict the fixed-point argument:

* Pairing A with the bad survey **does not reduce** fine information on A.
  LeJEPA trained on (A, B) pairs keeps more of it (0.48) than LeJEPA trained
  on two independent A observations (0.34), and contrastive keeps the most.
* The quality-gap sweep (`results/base/sweep_fine.png`) is flat: as σ_bad goes
  from 2× to 10× σ_good, LeJEPA's fine R² on A stays at 0.45–0.48. The bad
  survey's embedding degrades (right panel), the good survey's does not.

Interpretation: the information bound `I(f(x_A); z) ≤ I(x_B; z)` holds only
at the *exact* optimum of the alignment term. In practice the loss is nowhere
near it: with a noisy partner survey the residual alignment loss is
dominated by the partner's own noise, so a dimension carrying A-only
information is not penalized much more than a dimension carrying anything
else, and the isotropy regularizer still wants 64 dimensions filled. The
encoder therefore keeps the private feature. The bound is a statement about
the limit, not about where training actually sits.

## 3. The split / gated variants do not help here

`ours`, `gate_only` and `split_only` are consistently *below* plain LeJEPA on
the fine latent (0.20–0.25 vs 0.48), at every point of the sweep. In the split
variants the private slice does not hold noticeably more fine information than
the shared slice (`A_zp` 0.18 vs `A_zs` 0.14). Likely causes, in order of
suspicion:

1. Loss weighting. In `ours` the alignment terms sum to weight 2 (L_aug + L_cross)
   against a fixed λ·SIGReg, whereas LeJEPA averages 6 pairs to weight 1. Twice the
   alignment pressure means more invariance, i.e. less fine information. Not
   yet controlled for (`--align_balance` is the obvious next flag).
2. The private slice is trained only by within-observation pairs, which at
   this operating point are the *weaker* signal (see aug_only ≈ single_good <
   lejepa).
3. Single seed, 1.9k steps. Differences of ±0.1 in fine R² between MSE-based
   variants may be within run-to-run noise. Contrastive vs the rest is not.

## 4. Other observations

* **MLP probes matter** (σ_bad = 0.10 run): fine R² on A rises from 0.46 → 0.59
  (lejepa) and 0.20 → 0.42 (ours); log-period from 0.67 → 0.79. The ranking is
  unchanged, but the linear probe under-reports what the embedding holds.
* **Disjoint-epoch views** (two views from non-overlapping halves of the
  window) roughly triple the fine information captured by single-survey SSL
  compared with overlapping crops (0.03 → 0.11 at 12 epochs). Star-level
  pairs beat realization-level pairs; worth keeping regardless of the
  cross-survey question.
* **Unseen survey C**: probes fit on C embeddings give reasonable numbers
  for the cross-survey methods (lejepa fine 0.45, class 0.71) and poor ones
  for single-survey training (class 0.41), so exposure to a second band does
  transfer. Probes fit on A and applied to C do not transfer (large negative
  R²), i.e. the embedding shifts with wavelength even though its content is
  good.
* Class imbalance (`--class_probs 0.05,0.1,0.75,0.05,0.05`) and balanced
  accuracy are implemented but not yet run at full scale.

## 5. What to run next

1. Balance the alignment weight in the split family and rerun `ours` vs
   `lejepa` (3 seeds each) — this is the most likely explanation for §3.
2. Find the regime where the bound bites: a partner survey that is *clean on
   coarse latents but blind to the fine one*. Noise alone does not do it
   (coarse alignment becomes expensive too); a 0.6-day boxcar does not do it
   (linear smoothing at low noise is invertible; supervised on raw B still
   gets 0.78). Candidates: boxcar + moderate noise, or a fine feature that is
   non-periodic and short-lived (a flare) so integration destroys it.
3. Longer training / lower λ, to see whether LeJEPA on A+B *converges toward*
   the B floor as the alignment residual shrinks — the theory predicts the
   bottleneck appears as training approaches the optimum.
4. Report MLP probes alongside linear, and macro/balanced accuracy once the
   imbalanced prior is used.

## 6. Additive test: good+good vs good+good+bad (added after the first delivery)

Same seed and settings as the base run. Fine-bump R² on **survey A**
embeddings, probed linearly / with the MLP:

| views per star            | lejepa        | contrastive   |
|---------------------------|---------------|---------------|
| A + A2 (good + good)      | 0.34 / 0.60   | 0.46 / 0.71   |
| A + A2 + B (+ bad survey) | 0.56 / 0.71   | 0.62 / 0.75   |

Adding the bad survey *raises* the good survey's fine information for both
objectives, and it also raises class accuracy on B (0.39 → 0.49 lejepa) since
B is now seen in training. Coarse latents on A are unchanged (class 0.90 →
0.88, log-period 0.74 → 0.74).

Working hypothesis: in the MSE + isotropy family, how much low-variance
private information survives is set by the ratio of alignment pressure to the
regularizer, not by the partner's quality. Adding B dilutes the good-good
pairs from 6/6 to 6/15 of the averaged alignment term, so A's embedding is
pulled less hard toward invariance and keeps more of the bump. The same story
explains why `ours` (alignment weight 2× LeJEPA's) lost fine information.
Direct test: rerun `lejepa_aa` with λ raised (or the alignment term scaled by
0.4) and check whether fine R² climbs to the `lejepa_aab` value.
