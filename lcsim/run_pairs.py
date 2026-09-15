#!/usr/bin/env python3
"""Is the lower-quality source the limiting factor?

Surveys A, B, C observe the same stars in the same band, with quality A > B > C
(noise and cadence only, nothing else differs). Each method is trained on each
pair of surveys (always two sources), and every model is probed on the same
held-out test stars as observed by each survey, plus survey D, which no model
ever trains on (about as hard as B, but noisier and denser). SSL losses act on
the projector output; probes read the embedding. If the worse partner sets the
limit, training on AB should beat training on AC when probed on A.

methods (see lcsim/models.py):
    lejepa          LeJEPA, invariance + SIGReg on the projector output
    lejepa_noproj   LeJEPA, invariance + SIGReg on the embedding itself (no projector)
    contrastive     multi-positive NT-Xent on the projector output (2 views per survey: augmentation
                    pairs + cross-survey pairs)
    contrastive_x   NT-Xent on cross-survey pairs only: one augmented view per survey per star
    contrastive_x_raw  same, but the observations are not augmented at all
    lejepa_pred     LeJEPA with a predictor on the lower-quality side: within-survey view
                    invariance + p(proj_lo) -> stop-grad[proj_hi]; the SECOND survey of each
                    pair is the lower-quality side (AB: B, AC: C)
    lejepa_pred_x   lejepa_pred on cross-survey pairs only: one augmented view per survey per
                    star, predictor term + SIGReg, no within-survey invariance
    lejepa_pred_x_raw  same, but the observations are not augmented at all

    python run_pairs.py --out results/pairs
    python run_pairs.py --variants lejepa,contrastive --pairs AB,AC --seeds 0,1,2 --epochs 300 --out results/pairs
    sbatch run_pairs.sbatch                      # one array task per seed
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from lcsim.simulate import SimConfig, SurveyConfig, make_dataset
from lcsim.train import train_ssl, embed, train_supervised, supervised_eval, get_device
from lcsim.probe import probe_targets, fit_eval, METRICS

TRAIN_SURVEYS = "ABC"
SURVEYS = "ABCD"          # D is evaluation-only; it is simulated last so A, B, C match runs without it


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=str, default="lejepa,lejepa_noproj,contrastive,lejepa_pred")
    p.add_argument("--pairs", type=str, default="AB,AC", help="training pairs, two of A/B/C each")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=300, help="SSL epochs (31 steps each at batch 256)")
    p.add_argument("--sup_epochs", type=int, default=30, help="epochs for the supervised ceilings")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=5e-2)
    p.add_argument("--lam", type=float, default=0.02)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--wavelength", type=float, default=550.0, help="band shared by all surveys (nm)")
    for s, sig, keep in (("A", 0.015, 0.5), ("B", 0.05, 0.3), ("C", 0.15, 0.1), ("D", 0.08, 0.5)):
        p.add_argument(f"--sigma_{s}", type=float, default=sig, help=f"per-point noise of survey {s}")
        p.add_argument(f"--keep_{s}", type=float, default=keep, help=f"fraction of grid points survey {s} observes")
    p.add_argument("--fine_max", type=float, default=0.6)
    p.add_argument("--bump_width", type=float, default=0.06)
    p.add_argument("--no_supervised", action="store_true")
    p.add_argument("--no_mlp", action="store_true")
    p.add_argument("--no_save", action="store_true", help="do not save model checkpoints")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="results/pairs")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    variants = [v for v in args.variants.split(",") if v]
    pairs = [p for p in args.pairs.split(",") if p]
    assert all(len(p) == 2 and set(p) <= set(TRAIN_SURVEYS) and p[0] != p[1] for p in pairs), pairs
    seeds = [int(s) for s in args.seeds.split(",")]
    surveys = [SurveyConfig(s, args.wavelength, getattr(args, f"sigma_{s}"), getattr(args, f"keep_{s}")) for s in SURVEYS]
    log("surveys: " + "\n         ".join(map(str, surveys)) + "\n         (D is never used for training)")
    log(f"device: {get_device(args.device)}")
    results = {"args": vars(args), "runs": {}}
    t0 = time.time()

    for seed in seeds:
        cfg = SimConfig(seed=seed, fine_range=(0.0, args.fine_max), bump_width=args.bump_width)
        tr = make_dataset(args.n_train, surveys, cfg, seed=seed + 1)
        te = make_dataset(args.n_test, surveys, cfg, seed=seed + 2)     # the same held-out stars for every model
        log(f"\n##### seed {seed}: mean quality q " + " ".join(f"{s}={tr['obs'][s]['q'].mean():.2f}" for s in SURVEYS))
        tgt = {s: (probe_targets(tr["latents"], tr["obs"][s]), probe_targets(te["latents"], te["obs"][s]))
               for s in SURVEYS}
        run = results["runs"][str(seed)] = {"supervised": {}, "probes": {}}

        if not args.no_supervised:
            for s in SURVEYS:
                t_tr, t_te = tgt[s]
                reg_tr = np.stack([t_tr[k] for k in ("logP", "amp_band", "fine")], 1).astype(np.float32)
                reg_te = np.stack([t_te[k] for k in ("logP", "amp_band", "fine")], 1).astype(np.float32)
                mu, sd = reg_tr.mean(0), reg_tr.std(0)
                net = train_supervised(tr["obs"][s]["x"], t_tr["cls"], (reg_tr - mu) / sd, epochs=args.sup_epochs,
                                       batch=args.batch, seed=seed, log=log, width=args.width, device=args.device)
                acc, bacc, r2 = supervised_eval(net, te["obs"][s]["x"], t_te["cls"], (reg_te - mu) / sd)
                run["supervised"][s] = dict(cls_acc=float(acc), cls_bacc=float(bacc), logP_r2=float(r2[0]),
                                            amp_band_r2=float(r2[1]), fine_r2=float(r2[2]))
                log(f"supervised raw {s}: " + " ".join(f"{k}={v:.3f}" for k, v in run["supervised"][s].items()))

        for v in variants:
            for pair in pairs:
                x, y = pair
                if v.startswith("lejepa_pred"):   # the predictor sits on the second survey, which must be the worse one
                    assert tr["obs"][y]["q"].mean() < tr["obs"][x]["q"].mean(), f"{pair}: {y} is not lower quality than {x}"
                log(f"=== {v} trained on {pair} (seed {seed}) ===")
                model = train_ssl(v, tr["obs"][x]["x"], tr["obs"][y]["x"], tr["obs"][x]["q"], tr["obs"][y]["q"],
                                  epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd, seed=seed, log=log,
                                  device=args.device, d=args.d, lam=args.lam, width=args.width, proj_dim=args.proj_dim)
                if not args.no_save:
                    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(args.out, "models", f"{v}_{pair}_seed{seed}.pt"))
                res = run["probes"][f"{v}/{pair}"] = {}
                for s in SURVEYS:   # probes always read the embedding, never the projector output
                    res[s] = fit_eval(embed(model, tr["obs"][s]["x"]), tgt[s][0], embed(model, te["obs"][s]["x"]),
                                      tgt[s][1], mlp=not args.no_mlp)
                    log(f"  probed on {s}: " + " ".join(f"{k}={val:.3f}" for k, val in res[s].items()))
                json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
                write_summary(results, os.path.join(args.out, "results.md"))

    log(f"\ndone in {time.time() - t0:.0f}s -> {args.out}/results.md")


def write_summary(results, path):
    runs = list(results["runs"].values())
    a = results["args"]
    keys = [f"{v}/{p}" for v in a["variants"].split(",") if v for p in a["pairs"].split(",") if p]
    done = [k for k in keys if all(k in r["probes"] for r in runs)]
    surveys = [s for s in SURVEYS if f"sigma_{s}" in a]

    def cell(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"

    lines = ["# Survey-pair experiment", "",
             f"Same band ({a['wavelength']:.0f} nm), quality A > B > C, D never trained on: "
             + ", ".join(f"{s}: σ={a[f'sigma_{s}']}, cadence={a[f'keep_{s}']}" for s in surveys)
             + f". {a['epochs']} SSL epochs. Mean ± std over {len(runs)} seed(s). "
             "Every model is probed (on its embedding) with the same train/test stars.", ""]
    sup = [s for s in surveys if runs and all(s in r["supervised"] for r in runs)]
    if sup:
        lines += ["## Supervised on raw observations (per-survey ceiling)", "",
                  "| survey | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for s in sup:
            lines.append(f"| {s} | " + " | ".join(cell([r["supervised"][s][m] for r in runs]) for m in METRICS) + " |")
        lines.append("")
    for s in surveys:
        for suffix, name in (("", "linear"), ("_mlp", "MLP")):
            metrics = [m + suffix for m in METRICS]
            if not done or s not in runs[0]["probes"][done[0]] or metrics[0] not in runs[0]["probes"][done[0]][s]:
                continue
            title = f"survey {s} (unseen)" if s not in TRAIN_SURVEYS else f"survey {s}"
            lines += [f"## Probed on {title} test observations ({name} probes)", "",
                      "| method | trained on | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
            for k in done:
                v, p = k.split("/")
                lines.append(f"| {v} | {p} | " + " | ".join(cell([r["probes"][k][s][m] for r in runs]) for m in metrics) + " |")
            lines.append("")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
