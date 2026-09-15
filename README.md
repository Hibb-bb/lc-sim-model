# Cross-survey representation learning for variable stars — working package

Everything from the working session in one place: the theory notes, the
simulation that tested them, the results, and the PHOEBE 2 dataset generator
for the next step.

```
README.md                     this file
analysis/
  theory_notes.md             the math from the discussion: alignment = invariance, the
                              data-processing / common-information bound, the equivariance /
                              predictor escape, and the split + quality-gated design
  simulation_results.md       what the toy simulation actually showed (single seed) and why,
                              including the additive good+good vs good+good+bad test
  quality_gated_lejepa_diagram.html
                              the workflow diagram of the proposed method (open in a browser)
  plots/                      probes_base.png (all variants), sweep_fine.png (quality-gap sweep),
                              probes_add_bad.png (additive test), slices_base.png (z_s vs z_p),
                              lcsim_examples.png (toy light curves), phoebe_examples.png (PHOEBE test output)
lcsim/                        the toy simulation: generator, observation model, 11 training
                              variants (LeJEPA, contrastive, split / gated ablations, good+good(+bad)),
                              linear + MLP probes, class-imbalance option; README.md inside
  results/                    results.json / results.md / log.txt of every run reported
phoebe_dataset/               PHOEBE 2 multi-band eclipsing-binary dataset generator with
                              SLURM template and Hugging Face upload; README.md inside
```

## Reading order

1. `analysis/theory_notes.md` (10 min) — the argument and the proposed method.
2. `analysis/simulation_results.md` (10 min) — the evidence, which does not
   support the bottleneck at practical training lengths; the hypothesis for
   why, and the two experiments that would settle it (balance the alignment
   weight; raise λ on good+good).
3. `lcsim/README.md` to rerun or extend the toy simulation
   (`python run_experiment.py --out results/base`, ~45 min on 2 cores).
4. `phoebe_dataset/README.md` to build the real multi-band dataset on the
   cluster and push it to your private Hub repo.

## Status of each piece

| piece | status |
|---|---|
| theory notes | written up from the discussion |
| toy simulation | runs end to end; results are single-seed |
| PHOEBE generator | `filters`, `generate` (incl. resume, custom passbands), `assemble --save_dir` tested locally with PHOEBE 2.5.4; `--push` not exercised (no network here) |
