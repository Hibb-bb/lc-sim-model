# lcsim — cross-survey SSL on synthetic variable stars

A small, CPU-runnable simulation for the question: *when two surveys of unequal
quality observe the same star and are used as positive pairs, does the
lower-quality survey bottleneck the representation, and does a split embedding
with a quality-gated, stop-gradiented cross-survey loss avoid it?*

## Layout

```
lcsim/simulate.py   latent generator (5 classes, band-dependent flux, a "fine" bump) + observation model
lcsim/models.py     1D-CNN encoder, band-conditioned predictor, SIGReg, augmentations, all variant losses
lcsim/train.py      training loops, embedding extraction, supervised ceilings
lcsim/probe.py      linear probes (logistic for class, ridge for the rest)
run_experiment.py   end-to-end run -> results/<name>/{results.json, results.md, log.txt}
plot_results.py     bar charts for one run; fine-R² vs quality gap for a sweep
```

## Latents and their tiers

| latent   | what it is                                            | recoverable from       |
|----------|-------------------------------------------------------|------------------------|
| cls      | sinusoid / RR Lyrae / eclipsing / double-mode / DRW    | any observation        |
| logP     | period (timescale for DRW)                             | any observation        |
| amp_band | amplitude in the observed band                         | any observation        |
| fine     | amplitude of a narrow (0.06 phase) periodic bump       | low noise + dense only |
| color    | sets amplitude ratio and lag between bands             | two bands (not probed) |

Survey A is the good survey (σ=0.015, 50 % of grid points, 15 % "bad nights"),
survey B is the bad one (σ=0.12, 15 % of grid points, different band), and
survey C (800 nm, σ=0.05, 30 %) is never used for training. Each observation
has its own epoch, so the two views of a star share the star, not the
realization. The quality score is `q = log(n_points) - log(median σ)`.

## Variants

| name         | cross-survey term                               | slice | notes                          |
|--------------|-------------------------------------------------|-------|--------------------------------|
| lejepa       | symmetric MSE over all view pairs               | all z | + SIGReg                       |
| contrastive  | multi-positive NT-Xent                          | all z |                                |
| aug_only     | none (augmentations of the A observation only)  | all z | realization-level baseline     |
| single_good  | symmetric MSE, A paired with a 2nd A observation | all z | ceiling for private info       |
| split_only   | symmetric MSE                                   | z_s   | + L_aug on z, + SIGReg         |
| gate_only    | quality-gated, stop-grad, band-conditioned pred | all z | + L_aug on z, + SIGReg         |
| ours         | quality-gated, stop-grad, band-conditioned pred | z_s   | + L_aug on z, + SIGReg         |

Gate: `w_{B->A} = sigmoid((q_A - q_B)/tau)`, `w_{A->B} = 1 - w_{B->A}`; both
directions are always computed, each with its own stop-gradient on the target.

## Run

```
# into the repo-root .venv (see ../README.md); CUDA 13.0 build of torch for the Quest A100/H100 nodes
uv pip install --python ../.venv/bin/python "torch==2.14.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install --python ../.venv/bin/python scikit-learn matplotlib

python run_experiment.py --out results/base            # uses CUDA if available (--device cpu to force CPU)
sbatch run_gpu.sbatch --out results/base               # same, as a gengpu batch job

# is the worse partner the limit? three same-band surveys with quality A > B > C,
# LeJEPA trained on each pair (two sources per run), all probed on the same test stars
python run_pairs.py --pairs AB,AC,BC --seeds 0,1,2 --out results/pairs
python plot_results.py results/base

# quality-gap sweep (bad survey noise relative to the good one)
python run_experiment.py --sigma_bad 0.015 --keep_bad 0.5 --variants lejepa,ours,single_good --out results/gap_1
python run_experiment.py --sigma_bad 0.04  --keep_bad 0.3 --variants lejepa,ours,single_good --out results/gap_3
python plot_results.py --sweep results/gap_1 results/gap_3 results/base
```

Useful flags: `--tau` (gate temperature), `--lam` (SIGReg weight), `--d_s`
(shared-slice width), `--bad_prob_good` (fraction of bad nights in A),
`--fine_max` / `--bump_width` (how strong / narrow the fine feature is),
`--smooth_bad` (boxcar integration time of B, a long-cadence survey),
`--class_probs 0.05,0.1,0.75,0.05,0.05` (EW-like class imbalance; balanced
accuracy `cls_bacc` is reported alongside accuracy), `--no_mlp` (skip the MLP
probes; by default every metric is also reported with an `_mlp` suffix from a
256-unit one-hidden-layer probe), `--overlap_views` (overlapping crops instead
of disjoint-epoch views).

See `RESULTS.md` for the first results and their interpretation.

## What to look at

* `supervised` block: ceilings from raw data. If `fine_r2` on raw B is not
  much lower than on raw A, the bottleneck premise does not hold for that
  config and nothing below is informative.
* Survey-A block, `fine_r2`: the headline. lejepa / contrastive should sit
  near the B ceiling, `ours` near `single_good`.
* Survey-B block: nothing should beat lejepa by much; there is nothing extra
  to recover there.
* `A_zs` vs `A_zp`: fine should be readable from z_p and not from z_s.
* `A->C`: probes fit on A applied to the unseen survey C. Large negative R²
  just means the linear readout does not transfer across the band shift;
  the `C` block (probes fit on C) is the representation-quality number.
