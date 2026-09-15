#!/usr/bin/env python3
"""
Synthetic eclipsing-binary light curves with PHOEBE 2, observed through several
passbands, packaged as a Hugging Face dataset and pushed to a private repo.

Pipeline (each step is a sub-command, so the expensive part can run as a SLURM
array and the cheap parts on a login node):

  1. filters   : make sure the passbands are installed (downloads PHOEBE's
                 tables, or builds custom ones from transmission files), compute
                 the 10 / 50 / 90 % transmission-percentile wavelengths and write
                 filters.json
  2. generate  : simulate systems id % nshards == shard, write parquet checkpoints
                 (resumable: rerunning skips finished ids)
  3. assemble  : merge all shards, split train/val/test, push to the Hub together
                 with filters.json and a dataset card

Typical use on a cluster (see run_slurm.sh):

  python phoebe_eb_dataset.py filters  --bands "Gaia:G,LSST:g,LSST:r,LSST:i,TESS:T" --out data/filters.json
  python phoebe_eb_dataset.py generate --shard $SLURM_ARRAY_TASK_ID --nshards 100 --workers 8 \
                                       --filters data/filters.json --out data/shards
  python phoebe_eb_dataset.py assemble --shards data/shards --filters data/filters.json \
                                       --repo <hf-user>/phoebe-eb-multiband --push

Every light curve spans 2000 days. Because PHOEBE systems here are strictly
periodic (no spot evolution, no apsidal motion), each system is computed once
on a phase grid and then sampled at the observation times by interpolation,
which is what makes 20k systems x several bands affordable.
"""
from __future__ import annotations

import argparse, glob, json, math, os, sys, time, traceback
import numpy as np

# --------------------------------------------------------------------------- #
# constants / defaults
# --------------------------------------------------------------------------- #
DEFAULT_BANDS = "Gaia:G,LSST:g,LSST:r,LSST:i,TESS:T"
MORPHOLOGIES = ("contact", "detached", "semidetached")
DEFAULT_MORPH_FRACS = "0.70,0.20,0.10"        # EW-dominated, like real surveys
BASELINE_DAYS = 2000.0
KEPLER_CONST = 4.2084                          # a[Rsun] = K * ((M1+M2)[Msun] * P[d]^2)^(1/3)

# per-band observing model (can be overridden with --obs_config JSON)
DEFAULT_OBS = {
    "default": {"n_points": 800, "sigma": 0.01, "season_len": 240.0, "year": 365.25},
}


def band_key(band: str) -> str:
    """'LSST:g' -> 'LSST_g' for use as a column name."""
    return band.replace(":", "_").replace("-", "_").replace(".", "p")


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #

def transmission_percentiles(wl_m: np.ndarray, tr: np.ndarray) -> dict:
    """Percentile wavelengths of the transmission curve, in nm.

    Energy-weighted (T dλ) 10/50/90 %, plus the photon-weighted (λ T dλ) median
    and the transmission-weighted mean, which is what most people call λ_eff.
    """
    wl = np.asarray(wl_m, float) * 1e9
    tr = np.clip(np.asarray(tr, float), 0, None)
    o = np.argsort(wl); wl, tr = wl[o], tr[o]
    dl = np.gradient(wl)

    def pct(weights, ps):
        c = np.cumsum(weights * dl); c = c / c[-1]
        return [float(np.interp(p, c, wl)) for p in ps]

    p10, p50, p90 = pct(tr, (0.10, 0.50, 0.90))
    p50_ph, = pct(tr * wl, (0.50,))
    return dict(wl_p10_nm=p10, wl_p50_nm=p50, wl_p90_nm=p90, wl_p50_photon_nm=p50_ph,
                wl_mean_nm=float(np.sum(wl * tr * dl) / np.sum(tr * dl)),
                wl_min_nm=float(wl[tr > 0].min()), wl_max_nm=float(wl[tr > 0].max()),
                fwhm_nm=float(wl[tr >= 0.5 * tr.max()].max() - wl[tr >= 0.5 * tr.max()].min()))


def cmd_filters(args):
    import phoebe
    phoebe.logger("WARNING")
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    custom = {}
    for spec in (args.custom or []):
        name, path = spec.split("=", 1)
        custom[name] = path
    installed = set(phoebe.list_installed_passbands())

    # --- custom passbands from transmission files (blackbody intensities) ---
    for name, path in custom.items():
        if name in installed and not args.rebuild_custom:
            print(f"[filters] custom passband {name} already installed"); continue
        pbset, pbname = name.split(":")
        from phoebe.atmospheres.passbands import Passband
        from phoebe.atmospheres import models as M
        print(f"[filters] building {name} from {path} (blackbody intensities)")
        pb = Passband(ptf=path, pbset=pbset, pbname=pbname, wlunits=getattr(phoebe.u, args.custom_wl_unit),
                      calibrated=True, reference=f"custom transmission file {os.path.basename(path)}")
        pb.compute_intensities(atm=M.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False, verbose=False)
        out = os.path.join(args.passband_dir, f"{pbset}_{pbname}.fits")
        os.makedirs(args.passband_dir, exist_ok=True)
        pb.save(out)
        phoebe.install_passband(out, local=True)
        installed.add(name)

    # --- PHOEBE's own passbands (ck2004 etc.) ---
    missing = [b for b in bands if b not in installed and b not in custom]
    if missing:
        print(f"[filters] downloading {missing} from tables.phoebe-project.org ...")
        for b in missing:
            phoebe.download_passband(b, content=args.content, local=True)
        installed = set(phoebe.list_installed_passbands())
        still = [b for b in bands if b not in installed]
        if still:
            online = phoebe.list_online_passbands()
            sys.exit(f"[filters] could not install {still}. Available online: {sorted(online)}")

    all_bands = bands + [c for c in custom if c not in bands]
    atm = "blackbody" if (custom or args.atm == "blackbody") else args.atm
    info = {"bands": all_bands, "atm": atm, "filters": {}}
    for b in all_bands:
        pb = phoebe.get_passband(b)
        tab = pb.ptf_table
        wl, tr = np.asarray(tab["wl"], float), np.asarray(tab["fl"], float)
        d = transmission_percentiles(wl, tr)
        effwl = getattr(pb, "effwl", None)
        try:                                     # Quantity in m in recent PHOEBE, float in older ones
            effwl_nm = float(effwl.to(phoebe.u.nm).value) if hasattr(effwl, "to") else float(effwl) * 1e9
        except Exception:                        # noqa: BLE001
            effwl_nm = None
        d.update(dict(name=b, column=band_key(b), effwl_nm=effwl_nm,
                      custom=b in custom, content=list(pb.content),
                      wavelength_nm=(wl * 1e9).round(3).tolist(), transmission=tr.round(6).tolist()))
        if atm not in {c.split(":")[0] for c in pb.content}:
            sys.exit(f"[filters] passband {b} has no '{atm}' intensities (content={pb.content}); "
                     f"re-download with --content all or use --atm blackbody")
        info["filters"][b] = d
        print(f"[filters] {b:22s} p10={d['wl_p10_nm']:7.1f}  p50={d['wl_p50_nm']:7.1f}  p90={d['wl_p90_nm']:7.1f} nm"
              f"  (λ_eff={d['effwl_nm']})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(info, open(args.out, "w"), indent=1)
    print(f"[filters] wrote {args.out}  (atm={atm})")


# --------------------------------------------------------------------------- #
# parameter sampling
# --------------------------------------------------------------------------- #

def sample_system(rng: np.random.Generator, morph: str, cfg: dict) -> dict:
    """Draw physical parameters for one system. Radii for detached systems are
    finalised inside build_bundle where PHOEBE's requiv_max is available."""
    p = {"morphology": morph}
    m1 = 10 ** rng.uniform(np.log10(cfg["m1_min"]), np.log10(cfg["m1_max"]))
    q = rng.uniform(*cfg["q_range"][morph])
    m2 = q * m1
    # crude main-sequence temperatures with scatter, clipped to the atmosphere grid
    t_of_m = lambda m: 5800.0 * m ** 0.55 * 10 ** rng.normal(0, 0.03)
    t1 = float(np.clip(t_of_m(m1), cfg["teff_min"], cfg["teff_max"]))
    if morph == "contact":
        t2 = t1 * rng.uniform(0.93, 1.03)
    elif morph == "semidetached":
        t2 = t1 * rng.uniform(0.50, 0.90)          # Algol-like cool lobe-filling secondary
    else:
        t2 = float(np.clip(t_of_m(m2), cfg["teff_min"], cfg["teff_max"]))
    t2 = float(np.clip(t2, cfg["teff_min"], cfg["teff_max"]))
    lo, hi = cfg["period_range"][morph]
    period = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
    sma = KEPLER_CONST * ((m1 + m2) * period ** 2) ** (1 / 3)
    ecc, per0 = 0.0, rng.uniform(0, 360)
    if morph == "detached" and period > cfg["ecc_min_period"]:
        ecc = float(rng.uniform(0, cfg["ecc_max"])) if rng.uniform() < cfg["ecc_prob"] else 0.0
    p.update(dict(period=float(period), q=float(q), mass1=float(m1), mass2=float(m2), teff1=t1, teff2=t2,
                  sma=float(sma), ecc=ecc, per0=float(per0), t0=float(rng.uniform(0, period)),
                  fillout=float(rng.uniform(*cfg["fillout_range"])) if morph == "contact" else float("nan"),
                  l3=float(rng.uniform(0, cfg["l3_max"])) if rng.uniform() < cfg["l3_prob"] else 0.0,
                  r_frac=float(rng.uniform(*cfg["detached_rfrac"])),        # detached: radius inflation factor
                  ))
    return p


def phase_grid(n_phases: int, ecc: float, per0_deg: float, n_extra: int = 40) -> np.ndarray:
    """Uniform phase grid plus extra points around both eclipses, so that narrow
    eclipses of long-period detached systems are resolved. The secondary-eclipse
    phase is shifted by ~(2/pi) e cos(omega) for eccentric orbits."""
    ph = np.linspace(0, 1, n_phases)
    shift = (2 / math.pi) * ecc * math.cos(math.radians(per0_deg))
    extra = np.concatenate([np.linspace(-0.04, 0.04, n_extra), 0.5 + shift + np.linspace(-0.06, 0.06, n_extra)])
    ph = np.unique(np.concatenate([ph, np.mod(extra, 1.0)]))
    return ph


def build_bundle(p: dict, bands: list, atm: str, n_phases: int, ntriangles: int, irrad: str, rng):
    import phoebe
    morph = p["morphology"]
    if morph == "contact":
        b = phoebe.default_binary(contact_binary=True)
        b.flip_constraint("pot", solve_for="requiv@primary")
        b.flip_constraint("fillout_factor", solve_for="pot")
    else:
        b = phoebe.default_binary()
    b.set_value("period@binary", p["period"])
    b.set_value("q@binary", p["q"])
    b.set_value("sma@binary", p["sma"])
    b.set_value("ecc@binary", p["ecc"])
    b.set_value("per0@binary", p["per0"])
    b.set_value("t0_supconj@binary", 0.0)
    b.set_value("teff@primary", p["teff1"])
    b.set_value("teff@secondary", p["teff2"])
    for comp, t in (("primary", p["teff1"]), ("secondary", p["teff2"])):
        b.set_value(f"gravb_bol@{comp}", 1.0 if t > 8000 else 0.32)
        b.set_value(f"irrad_frac_refl_bol@{comp}", 1.0 if t > 8000 else 0.6)

    if morph == "contact":
        b.set_value("fillout_factor@contact_envelope", p["fillout"])
        b.run_delayed_constraints()
        r1, r2 = b.get_value("requiv@primary@component"), b.get_value("requiv@secondary@component")
    elif morph == "semidetached":
        b.add_constraint("semidetached", "secondary")
        b.run_delayed_constraints()
        rmax1 = b.get_value("requiv_max@primary@component")
        r1 = min(p["mass1"] ** 0.8 * p["r_frac"], 0.8 * rmax1)
        b.set_value("requiv@primary", r1)
        b.run_delayed_constraints()
        r2 = b.get_value("requiv@secondary@component")
    else:
        b.run_delayed_constraints()
        rmax1 = b.get_value("requiv_max@primary@component")
        rmax2 = b.get_value("requiv_max@secondary@component")
        r1 = min(p["mass1"] ** 0.8 * p["r_frac"], 0.85 * rmax1)
        r2 = min(p["mass2"] ** 0.8 * p["r_frac"] * rng.uniform(0.9, 1.1), 0.85 * rmax2)
        b.set_value("requiv@primary", r1)
        b.set_value("requiv@secondary", r2)
        b.run_delayed_constraints()
    p["requiv1"], p["requiv2"] = float(r1), float(r2)

    # inclination: require eclipses (geometric condition at conjunction)
    rsum = (r1 + r2) / p["sma"] * (1 - p["ecc"] ** 2) / (1 + p["ecc"] * abs(math.sin(math.radians(p["per0"]))))
    cosi = rng.uniform(0, min(1.0, 1.05 * rsum))
    p["incl"] = float(math.degrees(math.acos(cosi)))
    b.set_value("incl@binary", p["incl"])
    b.run_delayed_constraints()
    p["logg1"], p["logg2"] = float(b.get_value("logg@primary@component")), float(b.get_value("logg@secondary@component"))

    phases = phase_grid(n_phases, p["ecc"], p["per0"])
    for band in bands:
        b.add_dataset("lc", compute_phases=phases, passband=band, dataset="lc_" + band_key(band))
    b.set_value_all("atm", atm)
    if atm == "blackbody":
        b.set_value_all("ld_mode", "manual")
        b.set_value_all("ld_func", "logarithmic")
        b.set_value_all("ld_coeffs", [0.5, 0.2])
        b.set_value_all("ld_mode_bol", "manual")
    b.set_value_all("ntriangles", ntriangles)
    b.set_value("irrad_method", irrad)
    return b


# --------------------------------------------------------------------------- #
# observation model
# --------------------------------------------------------------------------- #

def sample_times(rng, n_points, season_len, year, baseline=BASELINE_DAYS):
    """Random epochs inside yearly observing seasons of length season_len."""
    offset = rng.uniform(0, year)
    t = rng.uniform(0, baseline, size=int(n_points * 3))
    t = t[((t + offset) % year) < season_len]
    if len(t) < n_points:                                   # pad if the seasons were unlucky
        t = np.concatenate([t, rng.uniform(0, baseline, size=n_points - len(t))])
    return np.sort(rng.choice(t, size=n_points, replace=False))


def simulate_one(sid: int, seed: int, morph: str, bands: list, atm: str, obs: dict, cfg: dict,
                 n_phases: int, ntriangles: int, irrad: str, max_tries: int = 6):
    """Simulate one system; resample parameters on PHOEBE failure. Returns a row dict."""
    import phoebe
    phoebe.logger("CRITICAL")
    phoebe.interactive_checks_off(); phoebe.interactive_constraints_off()
    last_err = None
    for attempt in range(max_tries):
        rng = np.random.default_rng([seed, sid, attempt])
        try:
            p = sample_system(rng, morph, cfg)
            b = build_bundle(p, bands, atm, n_phases, ntriangles, irrad, rng)
            report = b.run_checks()
            if not report.passed:
                raise RuntimeError("run_checks failed: " + "; ".join(str(i.message) for i in report.items))
            b.run_compute(model="mdl", progressbar=False)
            phases = np.asarray(b.get_value(qualifier="compute_phases", dataset="lc_" + band_key(bands[0]), context="dataset"))
            row = {"id": sid, "seed_attempt": attempt, **{k: v for k, v in p.items() if k != "r_frac"}}
            row["phase_model"] = phases.astype(np.float32)
            depth_ok = False
            for band in bands:
                k = band_key(band)
                f = np.asarray(b.get_value(qualifier="fluxes", dataset="lc_" + k, model="mdl"), float)
                if not np.all(np.isfinite(f)) or f.max() <= 0:
                    raise RuntimeError(f"non-finite fluxes in {band}")
                scale = float(f.max())
                fm = f / scale
                fm = (1 - p["l3"]) * fm + p["l3"]                    # third light as a fraction of max flux
                depth_ok |= (1 - fm.min()) > cfg["min_depth"]
                o = obs.get(band, obs["default"])
                t = sample_times(rng, o["n_points"], o["season_len"], o["year"])
                ph = ((t - p["t0"]) / p["period"]) % 1.0
                flux = np.interp(ph, phases, fm, period=1.0)
                sig = o["sigma"] * rng.uniform(0.8, 1.2, size=len(t))
                flux_obs = flux + rng.normal(0, 1, size=len(t)) * sig
                row[f"{k}_time"] = t.astype(np.float32)
                row[f"{k}_flux"] = flux_obs.astype(np.float32)
                row[f"{k}_flux_err"] = sig.astype(np.float32)
                row[f"{k}_flux_model"] = fm.astype(np.float32)
                row[f"{k}_flux_scale"] = scale
            if not depth_ok:
                raise RuntimeError("no eclipse deeper than min_depth in any band")
            return row
        except Exception as e:                       # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
    return {"id": sid, "error": last_err}


def _worker(task):
    sid, kw = task
    try:
        return simulate_one(sid, **kw)
    except Exception:                                # noqa: BLE001
        return {"id": sid, "error": traceback.format_exc()[-400:]}


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def split_of(sid: int, n_total: int, n_val: int, n_test: int, seed: int) -> str:
    """Fixed random assignment of ids to train / validation / test."""
    perm = np.random.default_rng(seed + 12345).permutation(n_total)
    rank = int(np.argsort(perm)[sid])
    if rank < n_total - n_val - n_test:
        return "train"
    return "validation" if rank < n_total - n_test else "test"


def morph_of(sid: int, fracs: list, seed: int) -> str:
    r = np.random.default_rng([seed, sid, 999]).uniform()
    c = np.cumsum(fracs)
    return MORPHOLOGIES[int(np.searchsorted(c, r))]


def hf_features(bands: list):
    from datasets import Features, Value, Sequence
    f = {"id": Value("int32"), "split": Value("string"), "morphology": Value("string"), "seed_attempt": Value("int8")}
    for k in ("period", "t0", "q", "incl", "ecc", "per0", "sma", "mass1", "mass2", "teff1", "teff2",
              "requiv1", "requiv2", "logg1", "logg2", "fillout", "l3"):
        f[k] = Value("float64")
    f["phase_model"] = Sequence(Value("float32"))
    for band in bands:
        k = band_key(band)
        for c in ("time", "flux", "flux_err", "flux_model"):
            f[f"{k}_{c}"] = Sequence(Value("float32"))
        f[f"{k}_flux_scale"] = Value("float64")
    return Features(f)


def rows_to_parquet(rows: list, features, path: str):
    from datasets import Dataset
    cols = {name: [r[name] for r in rows] for name in features}
    Dataset.from_dict(cols, features=features).to_parquet(path)


def cmd_generate(args):
    import multiprocessing as mp
    finfo = json.load(open(args.filters))
    bands, atm = finfo["bands"], finfo["atm"]
    obs = dict(DEFAULT_OBS)
    if args.obs_config:
        obs.update(json.load(open(args.obs_config)))
    cfg = physics_config(args)
    fracs = [float(x) for x in args.morph_fracs.split(",")]
    assert len(fracs) == 3 and abs(sum(fracs) - 1) < 1e-6, "--morph_fracs must be 3 numbers summing to 1"
    os.makedirs(args.out, exist_ok=True)

    ids = [i for i in range(args.n_total) if i % args.nshards == args.shard]
    if args.limit:
        ids = ids[: args.limit]
    done = set()
    for f in glob.glob(os.path.join(args.out, f"shard{args.shard:04d}_part*.parquet")):
        import pyarrow.parquet as pq
        done.update(pq.read_table(f, columns=["id"]).column("id").to_pylist())
    todo = [i for i in ids if i not in done]
    print(f"[generate] shard {args.shard}/{args.nshards}: {len(ids)} systems, {len(done)} already done, {len(todo)} to go, "
          f"{args.workers} workers, bands={bands}, atm={atm}", flush=True)
    if not todo:
        return

    kw = dict(seed=args.seed, bands=bands, atm=atm, obs=obs, cfg=cfg, n_phases=args.n_phases,
              ntriangles=args.ntriangles, irrad=args.irrad_method, max_tries=args.max_tries)
    tasks = [(sid, {**kw, "morph": morph_of(sid, fracs, args.seed)}) for sid in todo]
    features = hf_features(bands)
    part = len(glob.glob(os.path.join(args.out, f"shard{args.shard:04d}_part*.parquet")))
    buf, n_ok, n_fail, t0 = [], 0, 0, time.time()
    errlog = open(os.path.join(args.out, f"shard{args.shard:04d}_errors.log"), "a")

    def flush():
        nonlocal buf, part
        if buf:
            rows_to_parquet(buf, features, os.path.join(args.out, f"shard{args.shard:04d}_part{part:03d}.parquet"))
            part += 1; buf = []

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, maxtasksperchild=50) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
            if "error" in row:
                n_fail += 1
                errlog.write(f"{row['id']}\t{row['error']}\n"); errlog.flush()
                continue
            row["split"] = split_of(row["id"], args.n_total, args.n_val, args.n_test, args.seed)
            buf.append(row); n_ok += 1
            if len(buf) >= args.checkpoint_every:
                flush()
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                print(f"[generate] {i + 1}/{len(tasks)} done ({n_fail} failed) {el / (i + 1):.1f}s/system, "
                      f"eta {el / (i + 1) * (len(tasks) - i - 1) / 60:.1f} min", flush=True)
    flush()
    print(f"[generate] shard {args.shard} finished: {n_ok} ok, {n_fail} failed after {args.max_tries} attempts each")


def physics_config(args) -> dict:
    return dict(
        m1_min=0.6, m1_max=4.0, teff_min=3500.0, teff_max=15000.0,
        q_range={"contact": (0.10, 0.80), "detached": (0.20, 1.00), "semidetached": (0.15, 0.80)},
        period_range={"contact": (0.22, 0.80), "detached": (0.8, args.max_period), "semidetached": (0.5, 12.0)},
        fillout_range=(0.10, 0.90), detached_rfrac=(0.9, 1.8),
        ecc_min_period=3.0, ecc_max=0.45, ecc_prob=0.6,
        l3_max=args.l3_max, l3_prob=0.5 if args.l3_max > 0 else 0.0,
        min_depth=0.01,
    )


# --------------------------------------------------------------------------- #
# assemble + push
# --------------------------------------------------------------------------- #

def cmd_assemble(args):
    from datasets import load_dataset, DatasetDict
    finfo = json.load(open(args.filters))
    files = sorted(glob.glob(os.path.join(args.shards, "shard*_part*.parquet")))
    if not files:
        sys.exit(f"[assemble] no parquet files under {args.shards}")
    ds = load_dataset("parquet", data_files=files, split="train")
    print(f"[assemble] {len(ds)} systems from {len(files)} files; columns: {len(ds.column_names)}")
    dd = DatasetDict({s: ds.filter(lambda r, s=s: r["split"] == s, num_proc=args.num_proc) for s in ("train", "validation", "test")})
    for s, d in dd.items():
        counts = {m: sum(1 for x in d["morphology"] if x == m) for m in MORPHOLOGIES}
        print(f"[assemble] {s:10s} {len(d):6d}  {counts}")
    card = dataset_card(finfo, dd, args)
    if args.save_dir:
        dd.save_to_disk(args.save_dir)
        open(os.path.join(args.save_dir, "README.md"), "w").write(card)
        json.dump(finfo, open(os.path.join(args.save_dir, "filters.json"), "w"), indent=1)
        print(f"[assemble] saved to {args.save_dir}")
    if args.push:
        from huggingface_hub import HfApi
        token = args.token or os.environ.get("HF_TOKEN")
        if not args.repo:
            sys.exit("[assemble] --push needs --repo <user>/<name> (or $HF_REPO)")
        if not token:
            sys.exit("[assemble] --push needs a write token: --token, $HF_TOKEN, or `huggingface-cli login`")
        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        dd.push_to_hub(args.repo, private=True, token=token, max_shard_size="500MB")
        tmp = "/tmp/_filters.json"; json.dump(finfo, open(tmp, "w"), indent=1)
        api.upload_file(path_or_fileobj=tmp, path_in_repo="filters.json", repo_id=args.repo, repo_type="dataset")
        tmp = "/tmp/_README.md"; open(tmp, "w").write(card)
        api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
        print(f"[assemble] pushed to https://huggingface.co/datasets/{args.repo} (private)")


def dataset_card(finfo, dd, args) -> str:
    rows = "\n".join(
        f"| {b} | {d['column']} | {d['wl_p10_nm']:.1f} | {d['wl_p50_nm']:.1f} | {d['wl_p90_nm']:.1f} | {d['wl_mean_nm']:.1f} | {'blackbody' if d['custom'] else finfo['atm']} |"
        for b, d in finfo["filters"].items())
    sizes = ", ".join(f"{s}: {len(d)}" for s, d in dd.items())
    return f"""---
license: cc-by-4.0
pretty_name: PHOEBE 2 multi-band eclipsing binaries
---
# PHOEBE 2 synthetic eclipsing-binary light curves, multi-band

Generated with `phoebe_eb_dataset.py` (PHOEBE 2, atm=`{finfo['atm']}`). {sizes}.
Every system is observed in every band over a {BASELINE_DAYS:.0f}-day baseline with
seasonal gaps; fluxes are normalised so the maximum of the noise-free model is 1
(third light added after normalisation), with Gaussian noise of the stated error.

## Filters (wavelengths in nm; percentiles of the energy-weighted transmission)

| band | column prefix | 10 % | 50 % | 90 % | mean | intensities |
|---|---|---|---|---|---|---|
{rows}

Full transmission curves are in `filters.json`.

## Columns
`id`, `split`, `morphology` (contact / detached / semidetached), physical
parameters (`period` d, `t0`, `q`, `incl` deg, `ecc`, `per0` deg, `sma` Rsun,
`mass1/2` Msun, `teff1/2` K, `requiv1/2` Rsun, `logg1/2`, `fillout` (contact only),
`l3` third-light fraction), `phase_model` (shared phase grid), and per band
`<band>_time` (days), `<band>_flux`, `<band>_flux_err`, `<band>_flux_model`
(noise-free model on `phase_model`), `<band>_flux_scale` (PHOEBE flux at max, so
absolute band-to-band ratios can be recovered).

Phase of any epoch: `((t - t0) / period) % 1`, primary eclipse at phase 0.
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("filters", help="install passbands, compute percentile wavelengths, write filters.json")
    f.add_argument("--bands", default=DEFAULT_BANDS, help="comma-separated PHOEBE passband names")
    f.add_argument("--custom", action="append", help="Custom:name=path/to/transmission.dat (2 columns: wavelength, transmission); repeatable")
    f.add_argument("--custom_wl_unit", default="AA", help="astropy unit name of the custom file's wavelength column (AA, nm, um)")
    f.add_argument("--passband_dir", default="data/passbands")
    f.add_argument("--rebuild_custom", action="store_true")
    f.add_argument("--atm", default="ck2004", choices=["ck2004", "phoenix", "blackbody"])
    f.add_argument("--content", default="all", help="passband content to download (all, or e.g. ck2004:all)")
    f.add_argument("--list_online", action="store_true")
    f.add_argument("--out", default="data/filters.json")

    g = sub.add_parser("generate", help="simulate one shard of systems")
    g.add_argument("--filters", default="data/filters.json")
    g.add_argument("--obs_config", default=None, help="JSON: {band: {n_points, sigma, season_len, year}} (key 'default' for the rest)")
    g.add_argument("--out", default="data/shards")
    g.add_argument("--shard", type=int, default=0); g.add_argument("--nshards", type=int, default=1)
    g.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    g.add_argument("--n_total", type=int, default=5000); g.add_argument("--n_val", type=int, default=625); g.add_argument("--n_test", type=int, default=625)
    g.add_argument("--morph_fracs", default=DEFAULT_MORPH_FRACS, help="contact,detached,semidetached fractions")
    g.add_argument("--n_phases", type=int, default=201); g.add_argument("--ntriangles", type=int, default=800)
    g.add_argument("--irrad_method", default="none", choices=["none", "horvat", "wilson"], help="reflection; 'horvat' is more realistic and ~2x slower")
    g.add_argument("--max_period", type=float, default=50.0); g.add_argument("--l3_max", type=float, default=0.3)
    g.add_argument("--max_tries", type=int, default=6)
    g.add_argument("--checkpoint_every", type=int, default=100)
    g.add_argument("--limit", type=int, default=0, help="only the first N systems of this shard (testing)")
    g.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("assemble", help="merge shards, split, push to the Hub")
    a.add_argument("--shards", default="data/shards"); a.add_argument("--filters", default="data/filters.json")
    a.add_argument("--repo", default=os.environ.get("HF_REPO", "hibb/phoebe-eb-multiband"), help="<user>/<name>; defaults to $HF_REPO"); a.add_argument("--push", action="store_true")
    a.add_argument("--token", default=None, help="HF token (or set HF_TOKEN / run huggingface-cli login)")
    a.add_argument("--save_dir", default=None, help="also save_to_disk here"); a.add_argument("--num_proc", type=int, default=4)

    args = p.parse_args()
    if args.cmd == "filters":
        if args.list_online:
            import phoebe; print("\n".join(sorted(phoebe.list_online_passbands()))); return
        cmd_filters(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    else:
        if args.push and not args.repo:
            sys.exit("--push needs --repo <user>/<name>")
        cmd_assemble(args)


if __name__ == "__main__":
    main()
