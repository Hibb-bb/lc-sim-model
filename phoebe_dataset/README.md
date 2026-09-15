# PHOEBE 2 multi-band eclipsing-binary dataset generator

One script, three sub-commands, produces 20 000 PHOEBE 2 eclipsing-binary
systems observed in several passbands over 2000 days, split 15 000 / 2 500 /
2 500, as a Hugging Face `DatasetDict`, and pushes it to a private repo.

Tested end to end with PHOEBE 2.5.4 (contact, detached and semi-detached
systems, `ck2004` atmospheres and blackbody custom passbands, resume, assemble,
`save_to_disk`). The `push_to_hub` call itself was not exercised here (no
network), it is the standard `datasets` API.

## Files

| file | what it does |
|---|---|
| `phoebe_eb_dataset.py` | the generator (`filters` / `generate` / `assemble`) |
| `run_slurm.sh` | SLURM array template for the `generate` stage |
| `obs_config_example.json` | per-band cadence / noise / season model |
| `example_transmission_fakeR.dat` | example 2-column transmission file for a custom passband |
| `requirements.txt` | `phoebe`, `datasets`, `huggingface_hub`, `pyarrow`, `astropy`, `numpy` |

## Quick start

```bash
pip install -r requirements.txt

# 1) passbands + percentile wavelengths (login node, needs internet once)
python phoebe_eb_dataset.py filters --list_online                      # see what PHOEBE offers
python phoebe_eb_dataset.py filters --bands "Gaia:G,LSST:g,LSST:r,LSST:i,TESS:T" --out data/filters.json

# 2) simulate (SLURM array; or a single machine with --nshards 1 --workers 32)
sbatch run_slurm.sh
#   or, for a smoke test:
python phoebe_eb_dataset.py generate --filters data/filters.json --out data/shards --limit 20 --workers 4

# 3) merge, split, push (login node, HF_TOKEN with write access)
export HF_TOKEN=hf_...
python phoebe_eb_dataset.py assemble --shards data/shards --filters data/filters.json \
       --repo <hf-user>/phoebe-eb-multiband --push --save_dir data/hf_local
```

`generate` is resumable: rerunning a shard skips ids already present in its
parquet checkpoints. Failed systems are retried up to `--max_tries` times with
fresh parameters and logged to `shard*_errors.log`; in the local tests the
failure rate was 0/70.

## What is simulated

* Morphology mix `--morph_fracs contact,detached,semidetached`, default
  0.70 / 0.20 / 0.10 (EW-dominated like real surveys). Contact systems use
  PHOEBE's contact envelope with a fill-out factor in [0.1, 0.9]; semi-detached
  systems use the `semidetached` constraint on the secondary; detached radii are
  main-sequence-like and capped at 85 % of PHOEBE's `requiv_max`.
* Periods: contact 0.22–0.8 d, semi-detached 0.5–12 d, detached 0.8–`--max_period`
  (50 d) log-uniform; eccentric orbits (e ≤ 0.45) for detached systems with P > 3 d.
* Masses 0.6–4 Msun, temperatures from a crude mass–Teff relation clipped to
  3500–15000 K (inside the ck2004 grid); gravity darkening and albedo switch at
  8000 K. Inclinations are drawn so that eclipses occur; systems shallower than
  1 % in every band are rejected.
* Optional third light (`--l3_max`, default up to 0.3 in half of the systems),
  applied after normalisation.
* Each system is computed **once on a phase grid** (201 uniform points plus 80
  extra points around the two eclipses) and interpolated to the observation
  epochs. This assumes strict periodicity, which holds for these models (no
  spots, no apsidal motion, no mass transfer). Reflection is off by default
  (`--irrad_method horvat` turns it on at ~2× cost).
* Observation model per band (`obs_config_example.json`): number of epochs
  over the 2000-day baseline, relative Gaussian noise, and yearly observing
  seasons (`season_len` days out of `year`), so different bands can mimic
  Gaia-like sparse, LSST-like seasonal, or TESS-like dense-but-short sampling.

## Passbands

`filters` writes `data/filters.json` with, for every band, the full
transmission curve and the wavelengths at which the cumulative energy-weighted
transmission reaches 10 %, 50 % and 90 % (`wl_p10_nm`, `wl_p50_nm`, `wl_p90_nm`),
plus the photon-weighted median, transmission-weighted mean, FWHM and PHOEBE's
`effwl`. The same table goes into the dataset card on the Hub.

Two sources of passbands:

* PHOEBE's own tables (`--bands`), downloaded from tables.phoebe-project.org
  with `ck2004` model-atmosphere intensities and interpolated limb darkening.
  Names look like `Gaia:G`, `LSST:g`, `TESS:T`, `Johnson:V`, `Kepler:mean`,
  `SDSS:r`; `--list_online` prints the exact list for your PHOEBE version.
* Custom transmission files (`--custom "Custom:myband=path.dat"`, two columns:
  wavelength in `--custom_wl_unit` (default Å), transmission). PHOEBE cannot
  compute ck2004 tables for these without the atmosphere grids, so they get
  **blackbody** intensities, and the whole run then uses `atm=blackbody` with
  manual logarithmic limb darkening (0.5, 0.2). This is fine for shapes and
  colours to first order; use PHOEBE's tables when you can.

## Dataset layout

One row per system:

| column | meaning |
|---|---|
| `id`, `split`, `morphology` | integer id, `train`/`validation`/`test`, `contact`/`detached`/`semidetached` |
| `period` (d), `t0`, `q`, `incl` (deg), `ecc`, `per0` (deg), `sma` (Rsun) | orbit |
| `mass1/2` (Msun), `teff1/2` (K), `requiv1/2` (Rsun), `logg1/2`, `fillout`, `l3` | stars; `fillout` is NaN unless contact |
| `phase_model` | the phase grid shared by all bands (~280 points) |
| `<band>_time` | observation epochs in days, 0–2000, sorted |
| `<band>_flux`, `<band>_flux_err` | normalised noisy flux and its error |
| `<band>_flux_model` | noise-free model on `phase_model` (max = 1, third light included) |
| `<band>_flux_scale` | PHOEBE flux at maximum before normalisation, so band-to-band ratios are recoverable |

`<band>` is the passband name with `:` replaced by `_` (`LSST_g`). Phase of an
epoch is `((t - t0) / period) % 1`, primary eclipse at 0.

Splits are assigned by a fixed random permutation of the ids at generation
time, so shards can be assembled in any order and any subset gives the same
split labels.

## Cost

Measured on 2 CPU cores at 201 + 80 phases, 800 triangles, 2 bands: contact
~2–3 s, detached/semi-detached ~7–12 s per system per core. Adding bands is
cheaper than linear (the mesh is shared). Expect 50–80 CPU-hours for 20k
systems with 5 bands; `run_slurm.sh` spreads that over a 100-task array.

## Known limitations / knobs

* No spots, pulsations, apsidal motion or period changes; every system is
  strictly periodic. If you want non-periodic effects, PHOEBE can do spots and
  apsidal motion but then the phase-grid shortcut no longer applies and you
  must compute at the actual epochs (100× slower).
* Temperatures are clipped at 15000 K to stay in the ck2004 grid; hot systems
  are under-represented.
* `--irrad_method none` by default; contact and semi-detached systems are the
  ones where reflection matters most.
* The Hub push uses `datasets.DatasetDict.push_to_hub(private=True)` and needs
  a token with write access (`HF_TOKEN` or `huggingface-cli login`).
